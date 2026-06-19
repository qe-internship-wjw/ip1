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

def compute_survival_window(
    df: pl.DataFrame,
    window_fraction: float = 0.3,
    direction: str = 'buy',
    group_col: str = 'security',
) -> pl.DataFrame:
    """Per-group order survival window, sized from the group's own queue dynamics.

    The touch-size-to-trade-volume ratio (mean resting depth on OUR side / mean bin trade volume)
    estimates queue time — how many bins of opposing flow are needed to clear the depth ahead. The
    window is `window_fraction` of that ratio, rounded and floored at 1 bin:

        W = max(1, round(window_fraction * mean(touch_size) / mean(volume)))

    The touch side follows `direction` (the bid for a buy, the ask for a sell). A group with no
    traded volume falls back to a single-bin window. ``simulate_windowed`` calls this with
    ``group_col='security'`` (the per-security window it simulates); pass ``group_col='qcode'`` to
    inspect the window at the qcode level. Returns one row per group with `avg_touch_size`,
    `avg_volume` and the integer `window`.
    """
    if direction not in ('buy', 'sell'):
        raise ValueError(f"direction must be 'buy' or 'sell', got {direction!r}")
    touch = 'bid_size_start' if direction == 'buy' else 'ask_size_start'
    return (
        df.group_by(group_col, maintain_order=True)
        .agg(avg_touch_size=pl.col(touch).mean(), avg_volume=pl.col('volume').mean())
        .with_columns(
            window=pl.when(pl.col('avg_volume') > 0)
            .then((window_fraction * pl.col('avg_touch_size') / pl.col('avg_volume')).round())
            .otherwise(1.0)
            .clip(lower_bound=1.0)
            .cast(pl.Int64)
        )
    )


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

    Two extra arrays, ``pf_x`` / ``aq_x``, attribute the SAME fills/crosses to the bin where they
    ACTUALLY trade (the execution bin) rather than the owner bin — the schedule is owner-attributed
    and so identical across strategies, but realised execution timing is not (an overlay reschedules
    fills by resting deeper / crossing early). These drive the per-bin traded-volume visualisation;
    metrics keep using the owner-attributed arrays.

    Returns (passive_fill, passive_price, agg_qty, agg_price, passive_traded, agg_traded) per bin.
    """
    n = len(sched)
    pf = np.zeros(n); pcash = np.zeros(n); aq = np.zeros(n); ap = np.zeros(n)
    pf_x = np.zeros(n); aq_x = np.zeros(n)         # traded volume by EXECUTION bin (for viz)
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
                        aq[o[0]] += o[2]; aq_x[t] += o[2]; ap[o[0]] = cp[t]; o[2] = 0.0
                else:
                    keep.append(o)
            active = keep

        # 2. Overlay action on the start-of-bin prediction.
        p = pred[t]
        if p == adverse:
            for o in active:
                if o[2] > 0:
                    aq[o[0]] += o[2]; aq_x[t] += o[2]; ap[o[0]] = cp[t]; o[2] = 0.0
            if sched[t] > 0:
                aq[t] += sched[t]; aq_x[t] += sched[t]; ap[t] = cp[t]
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

                # Dynamic Queue Sync (BASELINE touch orders only): track cancellations at the
                # touch. Deep overlay bundles are excluded so the 0.7 placement-depth fraction is
                # never capped/clobbered by touch-tracking when the touch falls to their level.
                if not bs["deep_origin"] and o[1] == tp[t]:
                    if bs.get("last_touch_price") != tp[t]:
                        # Market moved away and just returned to our level: reset queue position
                        bs["queue_ahead"] = ts[t]
                    else:
                        # Still at the touch: cap queue ahead by current book size to capture cancellations
                        bs["queue_ahead"] = min(bs["queue_ahead"], ts[t])
                    bs["last_touch_price"] = tp[t]

                if bs["is_deep"]:
                    # Downtick reached: the cumulative-flow requirement is the downtick bin's
                    # top-of-book (ts[t]) ON TOP of the 0.7 placement-depth fraction -- always,
                    # even when the downtick happens in the very bin the order was placed.
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

                # Trade-through: if the bin prints STRICTLY through our limit the depth ahead must
                # have cleared, so the order fills in full at its limit -- for deep overlay orders
                # just as for baseline touch orders.
                walk_through = (ext[t] < o[1]) if is_buy else (ext[t] > o[1])
                if has_range and walk_through:
                    consumed += qi + o[2]
                    pf[o[0]] += o[2]; pf_x[t] += o[2]; pcash[o[0]] += o[2] * o[1]; o[2] = 0.0
                    continue

                avail = (V - qi) - consumed
                if avail <= 0:
                    continue

                fill = o[2] if o[2] < avail else avail
                o[2] -= fill; consumed += fill
                pf[o[0]] += fill; pf_x[t] += fill; pcash[o[0]] += fill * o[1]
        else:
            # hilo matching logic
            for o in active:
                if o[2] <= 0 or not reached(o[1], t):
                    continue
                bs = bundle_states[o[4]]
                bs["is_deep"] = False
                pf[o[0]] += o[2]; pf_x[t] += o[2]; pcash[o[0]] += o[2] * o[1]; o[2] = 0.0

    # 4. End-of-day cleanup: cross every still-resting order at the last bin
    for o in active:
        if o[2] > 0:
            aq[o[0]] += o[2]; aq_x[n - 1] += o[2]; ap[o[0]] = cp[n - 1]

    pp = tp.copy()
    filled = pf > 0
    pp[filled] = pcash[filled] / pf[filled]

    return pf, pp, aq, ap, pf_x, aq_x


def simulate_windowed(
    scheduled: pl.DataFrame,
    direction: str = 'buy',
    fill_model: str = 'queue',
    window_fraction: float = 0.3,
    ticks: int = 2,
    queue_depth_levels: float = 0.7,
    pred_col: str | None = None,
    overlay_actions: str = 'both',
    high_col: str = 'high',
    low_col: str = 'low',
) -> pl.DataFrame:
    """General VWAP execution with a PER-SECURITY order survival window (overlapping windows).

    Rather than a single survival length for every security, the window is sized from each
    security's own queue dynamics. The touch-size-to-trade-volume ratio (mean resting depth on
    our side / mean bin trade volume) is a simple estimator of queue time — how many bins of
    opposing flow are needed to clear the depth ahead. The survival window is `window_fraction`
    of that ratio, rounded and floored at 1 bin:

        W(security) = max(1, round(window_fraction * mean(touch_size) / mean(volume)))

    e.g. a security whose touch is ~20x its per-bin volume, at the default ``window_fraction=0.3``,
    rests an order for ~6 bins. An order rests for `W` bins, then crosses at the end of its window
    (the next bin's start); `W=1` is the naive non-overlapping framework. Larger windows let a
    resting order sit across several bins so it can fill against more opposing flow — important in
    tick-constrained books where one bin rarely clears the queue. The window is computed off the
    `scheduled` frame, so the SAME per-security window applies to the baseline and any overlay run
    over the same frame (keeping a comparison like-for-like).

    Fill models (per resting order, over its whole window):
      - 'queue': the cumulative opposing flow over the window must first clear the queue resting
        ahead at placement (the touch size; ``queue_depth_levels``x it for a deeper overlay
        order), then fills the order.
      - 'hilo' : full fill once the bin range reaches the limit anywhere in the window.

    With `pred_col`, the ordered-logit overlay is active (requires a `tick` column): on a
    favourable prediction the resting orders are cancelled and merged with the current schedule
    into one deeper order on a fresh window; on an adverse prediction everything resting is
    crossed immediately. A small enough `window_fraction` (single-bin windows) + `pred_col`
    reproduces the single-bin overlay.

    `overlay_actions` selects WHICH of the two overlay actions is enabled, for studying each in
    isolation (only meaningful when `pred_col` is set):
      - 'both'        (default): both actions — the full overlay (current behaviour).
      - 'favourable'  : act only on favourable calls (rest deeper); adverse calls are neutralised
                        so those bins execute exactly like the baseline (rest at the touch).
      - 'adverse'     : act only on adverse calls (cross immediately); favourable calls are
                        neutralised so those bins rest at the touch like the baseline.
    Each partial overlay is realised by zeroing the predictions of the disabled action, so the
    unaffected bins are bit-for-bit identical to the baseline.

    Emits the per-bin columns ``passive_fill / passive_price / agg_qty / agg_price`` (attributed
    to each order's owner bin) plus ``mid``, so the metric helpers consume it unchanged. Two
    further columns ``passive_traded / agg_traded`` give the same fills/crosses attributed to the
    bin where they actually trade (execution timing), for visualising realised intraday execution.
    """
    if direction not in ('buy', 'sell'):
        raise ValueError(f"direction must be 'buy' or 'sell', got {direction!r}")
    if fill_model not in ('queue', 'hilo'):
        raise ValueError(f"fill_model must be 'queue' or 'hilo', got {fill_model!r}")
    if window_fraction <= 0:
        raise ValueError(f"window_fraction must be positive, got {window_fraction!r}")
    if overlay_actions not in ('both', 'favourable', 'adverse'):
        raise ValueError(
            f"overlay_actions must be 'both', 'favourable' or 'adverse', got {overlay_actions!r}"
        )
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

    # Partial overlay: neutralise the disabled action's predictions to 0 so those bins fall back to
    # the baseline (rest at the touch). 'favourable' keeps only favourable calls, 'adverse' only
    # adverse calls; 'both' leaves `_pred` untouched.
    if overlay and overlay_actions != 'both':
        drop_class = adverse if overlay_actions == 'favourable' else favourable
        df = df.with_columns(
            _pred=pl.when(pl.col('_pred') == drop_class).then(0).otherwise(pl.col('_pred'))
        )

    # Per-security survival window: window_fraction * (mean touch size / mean trade volume),
    # rounded and floored at 1 bin (see ``compute_survival_window``). A security with no traded
    # volume falls back to a single-bin window.
    win = compute_survival_window(df, window_fraction, direction, 'security')
    window_by_sec = dict(zip(win['security'].to_list(), win['window'].to_list()))

    arrs = ['scheduled_qty', '_opp', '_tp', '_ts', '_cp', '_deep', '_ext', '_pred']
    queue_model = fill_model == 'queue'
    pf_all, pp_all, aq_all, ap_all, pfx_all, aqx_all = [], [], [], [], [], []
    for _key, g in df.group_by(['security', 'date'], maintain_order=True):
        a = {c: g.get_column(c).to_numpy() for c in arrs}
        W = window_by_sec.get(_key[0], 1)
        pf, pp, aq, ap, pf_x, aq_x = _sim_session(
            a['scheduled_qty'].astype(float), a['_opp'].astype(float), a['_tp'].astype(float),
            a['_ts'].astype(float), a['_cp'].astype(float), a['_deep'].astype(float),
            a['_ext'].astype(float), a['_pred'].astype(int),
            W, queue_model, queue_depth_levels, favourable, adverse, is_buy, has_range,
        )
        pf_all.append(pf); pp_all.append(pp); aq_all.append(aq); ap_all.append(ap)
        pfx_all.append(pf_x); aqx_all.append(aq_x)

    _cat = lambda parts: np.concatenate(parts) if parts else np.zeros(0)
    df = df.with_columns(
        passive_fill=pl.Series('passive_fill', _cat(pf_all)),
        passive_price=pl.Series('passive_price', _cat(pp_all)),
        agg_qty=pl.Series('agg_qty', _cat(aq_all)),
        agg_price=pl.Series('agg_price', _cat(ap_all)),
        passive_traded=pl.Series('passive_traded', _cat(pfx_all)),
        agg_traded=pl.Series('agg_traded', _cat(aqx_all)),
    )
    return df.drop(['_opp', '_tp', '_ts', '_cp', '_deep', '_ext', '_pred'])
