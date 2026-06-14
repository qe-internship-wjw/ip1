"""Baseline VWAP execution algorithm for rolling calendar spreads.

The institutional roller's problem: migrate a fixed quantity (e.g. 10,000 lots) from the near
to the far contract over a roll period, as cheaply as possible. This module implements the
passive-then-aggressive baseline that the predictive ordered-logit model (see
``src/backtest.py``) must later beat:

    1. SCHEDULE — slice the target quantity across the roll's 5-minute bins in proportion to the
       historical volume curve (``src.backtest.compute_volume_curve`` /
       ``historical_volume_curve``), so more is worked when the market is liquid. Each roll
       period is a distinct ``security``; the curve is built from PRIOR rolls of the ``qcode``.

    2. EXECUTE (per bin, buying spreads) — rest a passive limit at the best bid for the bin's
       scheduled quantity. Whatever fills passively saves the spread. Whatever is unfilled at
       the END of the bin is crossed aggressively at the START price of the NEXT bin (a buy
       lifts the next bin's ask). Selling is the mirror image (rest at ask; cross to the next
       bin's bid).

Two interchangeable FIFO fill models decide how much of a resting order fills in a bin
(``fill_model=``):

    - 'queue'  (default; needs no extra columns): the order joins the BACK of the touch queue.
      The opposing aggressive flow over the bin is, for a bid, the sell-initiated volume
      ``(volume - signed_volume) / 2`` (since volume = buy+sell, signed_volume = buy-sell).
      It first consumes the depth resting ahead of us (``bid_size_start``); the passive fill is
      the overflow ``max(0, sell_volume - bid_size_start)``, capped at the scheduled quantity.

    - 'hilo'   (needs HIGH / LOW, added to ``pipeline.BINNED_COLS``): a binary check — a resting
      bid fills in full iff the bin's LOW trades down to or through it (``low <= bid``); a
      resting ask fills iff ``high >= ask``. Coarser, ignores queue position.

Metrics (``per_security_metrics`` / ``summarize``) quantify execution quality against two
benchmarks — the realised passive-fill rate (spread saved), the schedule-weighted MID
(unrealistic: assumes every lot trades at mid), and the market VWAP — with costs reported both
in price units and in basis points of the near-leg future price (the project convention,
matching ``utils.add_microstructure_signals``'s ``delta_p_bp``).

    from src.vwap import run_vwap_backtest
    per_sec, summary = run_vwap_backtest(df_cs, target_qty=10_000, direction='buy')
"""

from __future__ import annotations

import polars as pl

from src.backtest import compute_volume_curve, historical_volume_curve

# Columns the simulator reads off each bin (besides the schedule).
_PRICE_COLS = ['bid_start', 'ask_start', 'bid_size_start', 'ask_size_start', 'volume', 'signed_volume']


# ── 1. Scheduling ────────────────────────────────────────────────────────────────

def attach_volume_schedule(
    roll_bins: pl.DataFrame,
    curve: pl.DataFrame,
    target_qty: float,
    qcode_col: str = 'qcode',
    position_cols: tuple[str, ...] = ('days_until', 'bin_start_time'),
) -> pl.DataFrame:
    """Attach ``scheduled_qty`` to each bin by spreading `target_qty` across a security's bins
    in proportion to `curve`'s volume fractions.

    The curve carries ``volume_fraction`` per (qcode, *position_cols). We join it on, fill
    missing buckets with 0, then renormalise WITHIN each security so the schedule sums exactly
    to `target_qty` (the roll's realised bins rarely cover every curve bucket). A security with
    no overlapping history (e.g. the first roll of a qcode) gets an all-zero schedule and is
    skipped by the metrics.
    """
    pos = list(position_cols)
    c = curve.select([qcode_col, *pos, 'volume_fraction'])
    out = roll_bins.join(c, on=[qcode_col, *pos], how='left')

    f = pl.col('volume_fraction').fill_null(0.0)
    f_sum = f.sum().over('security')
    return out.with_columns(
        scheduled_qty=pl.when(f_sum > 0).then(target_qty * f / f_sum).otherwise(0.0)
    ).drop('volume_fraction')


# ── 2. Execution simulation ──────────────────────────────────────────────────────

def simulate_passive_aggressive(
    scheduled: pl.DataFrame,
    direction: str = 'buy',
    fill_model: str = 'queue',
    high_col: str = 'high',
    low_col: str = 'low',
) -> pl.DataFrame:
    """Simulate the passive-then-aggressive fill of each bin's ``scheduled_qty``.

    Per bin: rest a limit on our side (bid for a buy, ask for a sell); the passive fill is
    decided by `fill_model`; the unfilled remainder crosses at the NEXT bin's start price
    (the next ask for a buy, the next bid for a sell). The last bin of a roll has no next bin,
    so its remainder crosses at the same bin's opposing start price (flagged via `agg_is_next`).

    Adds per-bin columns: ``mid``, ``passive_price``, ``passive_fill``, ``agg_qty``,
    ``agg_price``, ``agg_is_next``. Returns the frame sorted by [security, date, bin_start_time].
    """
    if direction not in ('buy', 'sell'):
        raise ValueError(f"direction must be 'buy' or 'sell', got {direction!r}")
    if fill_model not in ('queue', 'hilo'):
        raise ValueError(f"fill_model must be 'queue' or 'hilo', got {fill_model!r}")
    if fill_model == 'hilo' and (high_col not in scheduled.columns or low_col not in scheduled.columns):
        raise ValueError(
            f"fill_model='hilo' needs '{high_col}'/'{low_col}' columns — add HIGH/LOW to "
            f"pipeline.BINNED_COLS and rebuild. Present: {scheduled.columns}"
        )

    df = scheduled.sort(['security', 'date', 'bin_start_time'])

    # Decompose volume into aggressor sides: volume = buy_init + sell_init,
    # signed_volume = buy_init - sell_init. Clip tiny negatives from data noise.
    df = df.with_columns(
        mid=(pl.col('bid_start') + pl.col('ask_start')) / 2,
        sell_init=((pl.col('volume') - pl.col('signed_volume')) / 2).clip(lower_bound=0),
        buy_init=((pl.col('volume') + pl.col('signed_volume')) / 2).clip(lower_bound=0),
    )

    if direction == 'buy':
        passive_price = pl.col('bid_start')               # rest at the best bid
        queue_ahead = pl.col('bid_size_start')            # depth resting in front of us
        opposing_flow = pl.col('sell_init')               # sell-initiated trades hit the bid
        next_cross = pl.col('ask_start').shift(-1).over('security')  # lift next bin's ask
        cross_fallback = pl.col('ask_start')
        touched = (pl.col(low_col) <= pl.col('bid_start')) if fill_model == 'hilo' else None
    else:
        passive_price = pl.col('ask_start')               # rest at the best ask
        queue_ahead = pl.col('ask_size_start')
        opposing_flow = pl.col('buy_init')                # buy-initiated trades lift the ask
        next_cross = pl.col('bid_start').shift(-1).over('security')  # hit next bin's bid
        cross_fallback = pl.col('bid_start')
        touched = (pl.col(high_col) >= pl.col('ask_start')) if fill_model == 'hilo' else None

    if fill_model == 'queue':
        raw_fill = (opposing_flow - queue_ahead).clip(lower_bound=0)  # overflow past the queue
    else:  # hilo: binary — full fill iff the bin's range reaches our resting price
        raw_fill = pl.when(touched).then(pl.col('scheduled_qty')).otherwise(0.0)

    df = df.with_columns(
        passive_price=passive_price,
        passive_fill=pl.min_horizontal(raw_fill, pl.col('scheduled_qty')),
        _next_cross=next_cross,
        _cross_fallback=cross_fallback,
    )
    return df.with_columns(
        agg_qty=(pl.col('scheduled_qty') - pl.col('passive_fill')).clip(lower_bound=0),
        agg_price=pl.coalesce([pl.col('_next_cross'), pl.col('_cross_fallback')]),
        agg_is_next=pl.col('_next_cross').is_not_null(),
    ).drop(['_next_cross', '_cross_fallback'])


# ── 3. Metrics ───────────────────────────────────────────────────────────────────

def _side(direction: str) -> int:
    """+1 for a buy (paying above a benchmark is a cost), -1 for a sell (receiving below it is)."""
    return 1 if direction == 'buy' else -1


def _cost_columns(df: pl.DataFrame, side: int) -> pl.DataFrame:
    """Derive realised average price, benchmarks, and signed costs (price units + bp) from the
    notional/quantity aggregates already on `df`."""
    return df.with_columns(
        total_qty=pl.col('passive_qty') + pl.col('agg_qty'),
    ).with_columns(
        passive_rate=pl.col('passive_qty') / pl.col('total_qty'),
        exec_avg_price=pl.col('exec_notional') / pl.col('total_qty'),
        mid_benchmark=pl.col('mid_sched_notional') / pl.col('scheduled_qty'),
        mkt_vwap=pl.col('mkt_vwap_num') / pl.col('mkt_vol'),
    ).with_columns(
        # Signed so positive = unfavourable vs the benchmark, for either direction.
        cost_vs_mid=side * (pl.col('exec_avg_price') - pl.col('mid_benchmark')),
        cost_vs_vwap=side * (pl.col('exec_avg_price') - pl.col('mkt_vwap')),
        cost_vs_arrival=side * (pl.col('exec_avg_price') - pl.col('arrival_mid')),
    ).with_columns(
        cost_vs_mid_bp=pl.col('cost_vs_mid') * 1e4 / pl.col('avg_futures'),
        cost_vs_vwap_bp=pl.col('cost_vs_vwap') * 1e4 / pl.col('avg_futures'),
        cost_vs_arrival_bp=pl.col('cost_vs_arrival') * 1e4 / pl.col('avg_futures'),
        half_spread_bp=pl.col('avg_half_spread') * 1e4 / pl.col('avg_futures'),
    )


_AGG_EXPRS = [
    pl.col('qcode').first().alias('qcode'),
    pl.col('scheduled_qty').sum().alias('scheduled_qty'),
    pl.col('passive_fill').sum().alias('passive_qty'),
    pl.col('agg_qty').sum().alias('agg_qty'),
    (pl.col('passive_fill') * pl.col('passive_price')
     + pl.col('agg_qty') * pl.col('agg_price')).sum().alias('exec_notional'),
    (pl.col('scheduled_qty') * pl.col('mid')).sum().alias('mid_sched_notional'),
    (pl.col('mid') * pl.col('volume')).sum().alias('mkt_vwap_num'),
    pl.col('volume').sum().alias('mkt_vol'),
    pl.col('mid').first().alias('arrival_mid'),
    pl.col('futures_price').mean().alias('avg_futures'),
    ((pl.col('ask_start') - pl.col('bid_start')) / 2).mean().alias('avg_half_spread'),
    pl.len().alias('n_bins'),
]


def per_security_metrics(sim: pl.DataFrame, direction: str = 'buy') -> pl.DataFrame:
    """One row of execution metrics per roll (security). Securities with an all-zero schedule
    (no historical curve) are dropped. Key columns: ``passive_rate`` (fraction filled passively,
    saving the spread), ``exec_avg_price``, and signed costs vs the mid / VWAP / arrival
    benchmarks in both price units and bp of the future price."""
    g = sim.group_by('security', maintain_order=True).agg(_AGG_EXPRS).filter(pl.col('scheduled_qty') > 0)
    return _cost_columns(g, _side(direction)).sort('security')


def summarize(sim: pl.DataFrame, direction: str = 'buy') -> dict:
    """Portfolio-level execution summary (quantity-weighted across all scheduled rolls)."""
    tot = sim.filter(
        pl.col('scheduled_qty').sum().over('security') > 0
    ).select(
        pl.col('scheduled_qty').sum().alias('scheduled_qty'),
        pl.col('passive_fill').sum().alias('passive_qty'),
        pl.col('agg_qty').sum().alias('agg_qty'),
        (pl.col('passive_fill') * pl.col('passive_price')
         + pl.col('agg_qty') * pl.col('agg_price')).sum().alias('exec_notional'),
        (pl.col('scheduled_qty') * pl.col('mid')).sum().alias('mid_sched_notional'),
        (pl.col('mid') * pl.col('volume')).sum().alias('mkt_vwap_num'),
        pl.col('volume').sum().alias('mkt_vol'),
        # Schedule-weighted arrival mid / future price across rolls.
        ((pl.col('scheduled_qty') * pl.col('mid')).sum()
         / pl.col('scheduled_qty').sum()).alias('arrival_mid'),
        ((pl.col('scheduled_qty') * pl.col('futures_price')).sum()
         / pl.col('scheduled_qty').sum()).alias('avg_futures'),
        ((pl.col('ask_start') - pl.col('bid_start')) / 2).mean().alias('avg_half_spread'),
        pl.col('security').n_unique().alias('n_rolls'),
    )
    summary = _cost_columns(tot, _side(direction)).row(0, named=True)
    summary['direction'] = direction
    return summary


# ── 4. Driver ────────────────────────────────────────────────────────────────────

def run_vwap_backtest(
    df_cs: pl.DataFrame,
    target_qty: float = 10_000,
    direction: str = 'buy',
    fill_model: str = 'queue',
    leakage_safe: bool = True,
    curve: pl.DataFrame | None = None,
    qcode_col: str = 'qcode',
    position_cols: tuple[str, ...] = ('days_until', 'bin_start_time'),
    target_col: str = 'target_date',
) -> tuple[pl.DataFrame, dict]:
    """Run the baseline VWAP over every roll in `df_cs` and return (per-security metrics, summary).

    `df_cs` is the spread frame from ``pipeline.build_datasets`` (one row per bin, carrying
    `qcode`, `security`, `days_until`, `target_date`, the *_start quotes, volume / signed_volume,
    and `futures_price`).

    Scheduling curve:
      - `leakage_safe=True` (default): each roll is scheduled against a curve built only from
        PRIOR rolls of its qcode (``historical_volume_curve(before=target_date)``) — backtest-
        honest, and the first roll per qcode is unscheduled (no history). O(rolls) curve builds.
      - `leakage_safe=False`: schedule every roll against one full-sample curve (`curve`, or one
        computed here). Quicker, but peeks at the whole sample — diagnostics only.
    """
    if leakage_safe:
        keys = df_cs.select('security', qcode_col, target_col).unique()
        parts = []
        for row in keys.iter_rows(named=True):
            qc, tgt, sec = row[qcode_col], row[target_col], row['security']
            hist = historical_volume_curve(
                df_cs.filter(pl.col(qcode_col) == qc), before=tgt,
                group_col=qcode_col, position_cols=position_cols,
            )
            bins = df_cs.filter(pl.col('security') == sec)
            parts.append(attach_volume_schedule(bins, hist, target_qty, qcode_col, position_cols))
        scheduled = pl.concat(parts) if parts else df_cs.head(0)
    else:
        if curve is None:
            curve = compute_volume_curve(df_cs, group_col=qcode_col, position_cols=position_cols)
        scheduled = attach_volume_schedule(df_cs, curve, target_qty, qcode_col, position_cols)

    sim = simulate_passive_aggressive(scheduled, direction=direction, fill_model=fill_model)
    return per_security_metrics(sim, direction), summarize(sim, direction)
