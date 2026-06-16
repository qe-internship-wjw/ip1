"""Limit order book & queue simulation engine for windowed VWAP execution.

This module isolates the stateful order-matching machinery from the VWAP strategy / metrics layer
in ``src.vwap``. Given a *scheduled* bin frame (one row per 5-minute bin carrying the start-of-bin
quotes, sizes, the volume decomposition and an optional ordered-logit ``pred``), it simulates
resting limit orders across overlapping survival windows and returns their passive/aggressive fills.

Two pieces:

  - ``_sim_session`` — the pure-NumPy matching core for ONE ``[security, date]`` session. It keeps a
    live book of resting orders, tracks queue-ahead per price level, enforces strict price-time
    priority, and attributes every fill / cross back to the bin that placed the order.
  - ``simulate_windowed`` — the Polars driver that maps a scheduled frame onto the core: it builds
    the per-bin arrays, runs the core per session, and stitches the results back onto the frame as
    ``passive_fill / passive_price / agg_qty / agg_price`` (consumed unchanged by the metric
    helpers in ``src.vwap``).
"""

from __future__ import annotations

import numpy as np
import polars as pl


# ── Multi-bin survival windows (stateful order book) ───────────────────────────────

def _sim_session(sched, opp, tp, ts, cp, deep, ext, pred,
                 W, queue_model, qdepth, favourable, adverse, is_buy, has_range):
    """Simulate one [security, date] session with `W`-bin survival windows.

    An order placed at bin i rests over bins [i, i+W-1] and, if unfilled, crosses aggressively at
    the END of its window (the start price of bin i+W). Fills/crosses are attributed to the
    order's OWNER bin (the bin that placed it), so the returned arrays line up with the rows.

    Orders are lightweight records ``[owner, limit, qty, expiry, bundle_id, priority_time]``. The
    queue-ahead depth and the deep-activation flag live in ``bundle_states[bundle_id]`` so several
    order records can SHARE one queue: a favourable re-placement re-tags every resting order onto a
    single deep bundle, which therefore pays the queue-ahead penalty exactly ONCE as a group (no
    structural merging — the records stay independent so their owner attribution survives). Each
    bin the resting book is matched in strict price-time priority: orders are sorted best-price
    first, ties broken by ``priority_time`` (the bin at which the order was placed OR last
    re-priced, since amending a limit forfeits time priority in a real book). Passive fills are
    accumulated as quantity AND cash notional per owner bin, so a partial fill at the touch
    followed by a deeper fill on a favourable downtick yields a correct quantity-weighted
    ``passive_price`` rather than a clobbered scalar.

    Returns (passive_fill, passive_price, agg_qty, agg_price) per bin.
    """
    n = len(sched)
    pf = np.zeros(n); pcash = np.zeros(n); aq = np.zeros(n); ap = np.zeros(n)
    active: list[list] = []
    bundle_states: dict = {}                       # bundle_id -> {"queue_ahead", "is_deep", ...}

    def reached(limit, t):
        if not has_range:
            return True
        e = ext[t]
        return (e <= limit) if is_buy else (e >= limit)

    for t in range(n):
        # 1. Expire orders whose window ends now -> cross the remainder at this bin's start
        if active:
            keep = []
            for o in active:
                if o[3] == t:
                    if o[2] > 0:
                        aq[o[0]] += o[2]; ap[o[0]] = cp[t]; o[2] = 0.0
                else:
                    keep.append(o)
            active = keep

        # 2. Overlay action on the start-of-bin prediction.
        p = pred[t]
        if p == adverse:
            for o in active:
                if o[2] > 0:
                    aq[o[0]] += o[2]; ap[o[0]] = cp[t]; o[2] = 0.0
            if sched[t] > 0:
                aq[t] += sched[t]; ap[t] = cp[t]
            active = []
        elif p == favourable:
            b_id = f"deep_{t}"
            bundle_states[b_id] = {
                "queue_ahead": qdepth * ts[t],
                "is_deep": True,
                "deep_origin": True,
                "placed_at": t,
                "last_touch_price": deep[t]
            }
            for o in active:
                o[1] = deep[t]; o[3] = t + W; o[4] = b_id; o[5] = t
            if sched[t] > 0:
                active.append([t, deep[t], sched[t], t + W, b_id, t])
        else:
            if sched[t] > 0:                       # flat / no prediction -> rest at the touch
                b_id = f"base_{t}"
                bundle_states[b_id] = {
                    "queue_ahead": ts[t],
                    "is_deep": False,
                    "deep_origin": False,
                    "placed_at": t,
                    "last_touch_price": tp[t]
                }
                active.append([t, tp[t], sched[t], t + W, b_id, t])

        # 3. Match the bin's aggressive volume V against the resting book in STRICT price-time
        V = opp[t]
        active.sort(key=lambda o: (-o[1], o[5]) if is_buy else (o[1], o[5]))

        if queue_model:
            # (a) Synchronize bundle queues with live market updates and calculate snapshots
            q_init: dict = {}
            for o in active:
                if o[2] <= 0 or not reached(o[1], t):
                    continue

                b_id = o[4]
                bs = bundle_states[b_id]

                # Dynamic Queue Sync: Adjust queue positions using current market reality
                if o[1] == tp[t]:
                    if bs.get("last_touch_price") != tp[t]:
                        # Market moved away and just returned to our level: reset queue position
                        bs["queue_ahead"] = ts[t]
                    else:
                        # Still at the touch: cap queue ahead by current book size to capture cancellations
                        bs["queue_ahead"] = min(bs["queue_ahead"], ts[t])
                    bs["last_touch_price"] = tp[t]

                if bs["is_deep"]:
                    if t != bs["placed_at"]:
                        bs["queue_ahead"] += ts[t]
                    bs["is_deep"] = False

                if b_id not in q_init:
                    q_init[b_id] = bs["queue_ahead"]
                    # Track remaining queue ahead for the next bin interval
                    bs["queue_ahead"] = max(0.0, bs["queue_ahead"] - V)

            # (b) Walk the volume down the sorted book.
            consumed = 0.0
            for o in active:
                if o[2] <= 0 or not reached(o[1], t):
                    continue
                bs = bundle_states[o[4]]
                qi = q_init[o[4]]

                walk_through = (ext[t] < o[1]) if is_buy else (ext[t] > o[1])
                if has_range and walk_through and not bs["deep_origin"]:
                    consumed += qi + o[2]
                    pf[o[0]] += o[2]; pcash[o[0]] += o[2] * o[1]; o[2] = 0.0
                    continue

                avail = (V - qi) - consumed
                if avail <= 0:
                    continue

                fill = o[2] if o[2] < avail else avail
                o[2] -= fill; consumed += fill
                pf[o[0]] += fill; pcash[o[0]] += fill * o[1]
        else:
            # hilo matching logic
            for o in active:
                if o[2] <= 0 or not reached(o[1], t):
                    continue
                bs = bundle_states[o[4]]
                bs["is_deep"] = False
                pf[o[0]] += o[2]; pcash[o[0]] += o[2] * o[1]; o[2] = 0.0

    # 4. End-of-day cleanup: cross every still-resting order at the last bin
    for o in active:
        if o[2] > 0:
            aq[o[0]] += o[2]; ap[o[0]] = cp[n - 1]

    pp = tp.copy()
    filled = pf > 0
    pp[filled] = pcash[filled] / pf[filled]

    return pf, pp, aq, ap


def simulate_windowed(
    scheduled: pl.DataFrame,
    direction: str = 'buy',
    fill_model: str = 'queue',
    window: int = 1,
    ticks: int = 2,
    queue_depth_levels: float = 0.7,
    pred_col: str | None = None,
    high_col: str = 'high',
    low_col: str = 'low',
) -> pl.DataFrame:
    """General VWAP execution with `window`-bin order survival (overlapping windows).

    `window` is the survival length in bins: an order rests for `window` bins, then crosses at
    the end of its window (the next bin's start). `window=1` is the naive non-overlapping
    framework (place, rest one bin, cross next bin). Larger windows let a resting order sit
    across several bins so it can fill against more opposing flow — important in tick-constrained
    books where one bin rarely clears the queue.

    Fill models (per resting order, over its whole window):
      - 'queue': the cumulative opposing flow over the window must first clear the queue resting
        ahead at placement (the touch size; ``queue_depth_levels``x it for a deeper overlay
        order), then fills the order.
      - 'hilo' : full fill once the bin range reaches the limit anywhere in the window.

    With `pred_col`, the ordered-logit overlay is active (requires a `tick` column): on a
    favourable prediction the resting orders are cancelled and merged with the current schedule
    into one deeper order on a fresh window; on an adverse prediction everything resting is
    crossed immediately. `window=1` + `pred_col` reproduces the single-bin overlay.

    Emits the per-bin columns ``passive_fill / passive_price / agg_qty / agg_price`` (attributed
    to each order's owner bin) plus ``mid``, so the metric helpers consume it unchanged.
    """
    if direction not in ('buy', 'sell'):
        raise ValueError(f"direction must be 'buy' or 'sell', got {direction!r}")
    if fill_model not in ('queue', 'hilo'):
        raise ValueError(f"fill_model must be 'queue' or 'hilo', got {fill_model!r}")
    if int(window) < 1:
        raise ValueError(f"window must be a positive integer (bins), got {window!r}")
    window = int(window)
    overlay = pred_col is not None and pred_col in scheduled.columns
    has_range = high_col in scheduled.columns and low_col in scheduled.columns
    if fill_model == 'hilo' and not has_range:
        raise ValueError(
            f"fill_model='hilo' needs '{high_col}'/'{low_col}' — add HIGH/LOW to "
            f"pipeline.BINNED_COLS and rebuild. Present: {scheduled.columns}"
        )
    if overlay and 'tick' not in scheduled.columns:
        raise ValueError("overlay (pred_col) needs a `tick` column — call attach_tick_sizes() first.")

    is_buy = direction == 'buy'
    favourable = -2 if is_buy else 2
    adverse = 2 if is_buy else -2

    bid, ask = pl.col('bid_start'), pl.col('ask_start')
    df = scheduled.sort(['security', 'date', 'bin_start_time']).with_columns(mid=(bid + ask) / 2)
    if is_buy:
        df = df.with_columns(
            _opp=((pl.col('volume') - pl.col('signed_volume')) / 2).clip(lower_bound=0),
            _tp=bid, _ts=pl.col('bid_size_start'), _cp=ask,
        )
    else:
        df = df.with_columns(
            _opp=((pl.col('volume') + pl.col('signed_volume')) / 2).clip(lower_bound=0),
            _tp=ask, _ts=pl.col('ask_size_start'), _cp=bid,
        )
    if overlay:
        step = ticks * pl.col('tick')
        df = df.with_columns(_deep=(pl.col('_tp') - step) if is_buy else (pl.col('_tp') + step))
    else:
        df = df.with_columns(_deep=pl.col('_tp'))
    df = df.with_columns(
        _ext=(pl.col(low_col) if is_buy else pl.col(high_col)) if has_range else pl.lit(None, dtype=pl.Float64),
        _pred=pl.col(pred_col).fill_null(0).cast(pl.Int64) if overlay else pl.lit(0, dtype=pl.Int64),
    )

    arrs = ['scheduled_qty', '_opp', '_tp', '_ts', '_cp', '_deep', '_ext', '_pred']
    queue_model = fill_model == 'queue'
    pf_all, pp_all, aq_all, ap_all = [], [], [], []
    for _key, g in df.group_by(['security', 'date'], maintain_order=True):
        a = {c: g.get_column(c).to_numpy() for c in arrs}
        pf, pp, aq, ap = _sim_session(
            a['scheduled_qty'].astype(float), a['_opp'].astype(float), a['_tp'].astype(float),
            a['_ts'].astype(float), a['_cp'].astype(float), a['_deep'].astype(float),
            a['_ext'].astype(float), a['_pred'].astype(int),
            window, queue_model, queue_depth_levels, favourable, adverse, is_buy, has_range,
        )
        pf_all.append(pf); pp_all.append(pp); aq_all.append(aq); ap_all.append(ap)

    df = df.with_columns(
        passive_fill=pl.Series('passive_fill', np.concatenate(pf_all) if pf_all else np.zeros(0)),
        passive_price=pl.Series('passive_price', np.concatenate(pp_all) if pp_all else np.zeros(0)),
        agg_qty=pl.Series('agg_qty', np.concatenate(aq_all) if aq_all else np.zeros(0)),
        agg_price=pl.Series('agg_price', np.concatenate(ap_all) if ap_all else np.zeros(0)),
    )
    return df.drop(['_opp', '_tp', '_ts', '_cp', '_deep', '_ext', '_pred'])
