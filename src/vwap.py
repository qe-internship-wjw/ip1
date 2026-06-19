"""Baseline VWAP execution algorithm for rolling calendar spreads.

The institutional roller's problem: migrate a quantity from the near to the far contract over a
roll period, as cheaply as possible. The quantity is sized as a ``participation_rate`` of the
qcode's average total roll volume (e.g. 5% of a typical 100k-lot roll = 5k lots), so each
security's target scales with its own liquidity. This module implements the passive-then-
aggressive baseline that the predictive ordered-logit model (see ``src/backtest.py``) must beat:

    1. SCHEDULE — set each roll's target to ``participation_rate`` of its qcode's average roll
       volume (``src.backtest.compute_roll_volume`` / ``historical_roll_volume``), then slice
       that target across the roll's 5-minute bins in proportion to the historical volume curve
       (``src.backtest.compute_volume_curve`` / ``historical_volume_curve``), so more is worked
       when the market is liquid. Each roll period is a distinct ``security``; both the curve and
       the roll-volume average are built from PRIOR rolls of the ``qcode``.

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

Metrics (``per_security_metrics`` / ``summarize``) quantify execution quality against the
benchmarks — the realised passive-fill rate (spread saved), the schedule-weighted MID
(unrealistic: assumes every lot trades at mid), the market VWAP, and arrival mid. Costs are
reported in two evaluation frameworks:

  - TICKS PER LOT (the default): price cost / tick size. E.g. always crossing a 1-tick-wide
    spread costs half a tick per lot over mid. This is the natural unit for tick-constrained
    rolls and is robust to spreads whose outright price is near zero or negative.
  - BASIS POINTS: price cost * 1e4 / near-leg future price (the project convention, matching
    ``utils.add_microstructure_signals``'s ``delta_p_bp``).

Both are emitted for every benchmark; the tick framework requires a `tick` column on the bins
(``attach_tick_sizes``). ``run_improved_vwap_backtest(framework=...)`` picks the headline unit.

    from src.vwap import run_vwap_backtest
    per_sec, summary = run_vwap_backtest(df_cs, participation_rate=0.05, direction='buy')
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.backtest import (
    compute_volume_curve,
    historical_volume_curve,
    compute_roll_volume,
    historical_roll_volume,
)
# The stateful limit-order-book / queue matching engine lives in its own module; re-exported here
# so existing ``from src.vwap import simulate_windowed`` call sites keep working.
from src.lob_simulation import simulate_windowed, compute_survival_window

# Columns the simulator reads off each bin (besides the schedule).
_PRICE_COLS = ['bid_start', 'ask_start', 'bid_size_start', 'ask_size_start', 'volume', 'signed_volume']


# ── 1. Scheduling ────────────────────────────────────────────────────────────────

def attach_volume_schedule(
    roll_bins: pl.DataFrame,
    curve: pl.DataFrame,
    roll_volume: pl.DataFrame,
    participation_rate: float = 0.05,
    qcode_col: str = 'qcode',
    position_cols: tuple[str, ...] = ('days_until', 'bin_start_time'),
) -> pl.DataFrame:
    """Attach ``scheduled_qty`` to each bin.

    Each roll's target quantity is ``participation_rate`` of its qcode's average total roll
    volume (`roll_volume`'s ``avg_roll_volume`` — see ``src.backtest.compute_roll_volume``); that
    target is then spread across the security's bins in proportion to `curve`'s volume fractions.

    The curve carries ``volume_fraction`` per (qcode, *position_cols). We join it and the
    per-qcode roll volume on, fill missing buckets with 0, then renormalise WITHIN each security
    so the schedule sums exactly to the security's target (the roll's realised bins rarely cover
    every curve bucket). A security with no overlapping history (e.g. the first roll of a qcode,
    which has neither a curve nor a roll-volume estimate) gets an all-zero schedule and is
    skipped by the metrics.
    """
    pos = list(position_cols)
    c = curve.select([qcode_col, *pos, 'volume_fraction'])
    rv = roll_volume.select([qcode_col, 'avg_roll_volume'])
    out = roll_bins.join(c, on=[qcode_col, *pos], how='left').join(rv, on=qcode_col, how='left')

    target = participation_rate * pl.col('avg_roll_volume').fill_null(0.0)
    f = pl.col('volume_fraction').fill_null(0.0)
    f_sum = f.sum().over('security')
    return out.with_columns(
        scheduled_qty=pl.when(f_sum > 0).then(target * f / f_sum).otherwise(0.0)
    ).drop(['volume_fraction', 'avg_roll_volume'])


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
    (the next ask for a buy, the next bid for a sell).

    NO OVERNIGHT CROSSING: the next-bin lookup is confined to the SAME [security, date]
    session, and the LAST bin of each session is always executed fully aggressively (at that
    bin's own opposing start price). A remainder is therefore never carried across the close to
    be crossed at the next morning's open.

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

    # Last bin of the session: forced fully aggressive (overnight-crossing guard).
    is_last = pl.col('bin_start_time') == pl.col('bin_start_time').max().over(['security', 'date'])

    if direction == 'buy':
        passive_price = pl.col('bid_start')               # rest at the best bid
        queue_ahead = pl.col('bid_size_start')            # depth resting in front of us
        opposing_flow = pl.col('sell_init')               # sell-initiated trades hit the bid
        next_cross = pl.col('ask_start').shift(-1).over(['security', 'date'])  # next bin's ask, same day
        cross_fallback = pl.col('ask_start')
        touched = (pl.col(low_col) <= pl.col('bid_start')) if fill_model == 'hilo' else None
    else:
        passive_price = pl.col('ask_start')               # rest at the best ask
        queue_ahead = pl.col('ask_size_start')
        opposing_flow = pl.col('buy_init')                # buy-initiated trades lift the ask
        next_cross = pl.col('bid_start').shift(-1).over(['security', 'date'])  # next bin's bid, same day
        cross_fallback = pl.col('bid_start')
        touched = (pl.col(high_col) >= pl.col('ask_start')) if fill_model == 'hilo' else None

    if fill_model == 'queue':
        raw_fill = (opposing_flow - queue_ahead).clip(lower_bound=0)  # overflow past the queue
    else:  # hilo: binary — full fill iff the bin's range reaches our resting price
        raw_fill = pl.when(touched).then(pl.col('scheduled_qty')).otherwise(0.0)

    df = df.with_columns(
        passive_price=passive_price,
        # Force a zero passive fill on the session's last bin so its whole quantity crosses now.
        passive_fill=pl.when(is_last).then(0.0)
                       .otherwise(pl.min_horizontal(raw_fill, pl.col('scheduled_qty'))),
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
    """Derive realised average price, benchmarks, and signed costs in BOTH evaluation
    frameworks from the notional/quantity aggregates already on `df`:

      - ticks per lot  : price cost / tick size (the default; requires an `avg_tick` aggregate),
      - basis points   : price cost * 1e4 / future price.

    Each baseline (mid, market VWAP, arrival) gets a cost in both units. Costs are signed so
    positive = unfavourable for either direction. Tick columns are emitted only when `avg_tick`
    is present (i.e. a `tick` column reached the aggregation); bp columns only when `avg_futures`
    is present (a `futures_price` column reached the aggregation — see ``attach_futures_price``)."""
    df = df.with_columns(
        total_qty=pl.col('passive_qty') + pl.col('agg_qty'),
    ).with_columns(
        passive_rate=pl.col('passive_qty') / pl.col('total_qty'),
        exec_avg_price=pl.col('exec_notional') / pl.col('total_qty'),
        mid_benchmark=pl.col('mid_sched_notional') / pl.col('scheduled_qty'),
        mkt_vwap=pl.col('mkt_vwap_num') / pl.col('mkt_vol'),
    ).with_columns(
        cost_vs_mid=side * (pl.col('exec_avg_price') - pl.col('mid_benchmark')),
        cost_vs_vwap=side * (pl.col('exec_avg_price') - pl.col('mkt_vwap')),
        cost_vs_arrival=side * (pl.col('exec_avg_price') - pl.col('arrival_mid')),
    )

    exprs = []
    if 'avg_futures' in df.columns:
        exprs += [
            (pl.col('cost_vs_mid') * 1e4 / pl.col('avg_futures')).alias('cost_vs_mid_bp'),
            (pl.col('cost_vs_vwap') * 1e4 / pl.col('avg_futures')).alias('cost_vs_vwap_bp'),
            (pl.col('cost_vs_arrival') * 1e4 / pl.col('avg_futures')).alias('cost_vs_arrival_bp'),
            (pl.col('avg_half_spread') * 1e4 / pl.col('avg_futures')).alias('half_spread_bp'),
        ]
    if 'avg_tick' in df.columns:
        exprs += [
            (pl.col('cost_vs_mid') / pl.col('avg_tick')).alias('cost_vs_mid_ticks'),
            (pl.col('cost_vs_vwap') / pl.col('avg_tick')).alias('cost_vs_vwap_ticks'),
            (pl.col('cost_vs_arrival') / pl.col('avg_tick')).alias('cost_vs_arrival_ticks'),
            (pl.col('avg_half_spread') / pl.col('avg_tick')).alias('half_spread_ticks'),
        ]
    return df.with_columns(exprs)


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
    ((pl.col('ask_start') - pl.col('bid_start')) / 2).mean().alias('avg_half_spread'),
    pl.len().alias('n_bins'),
]


def per_security_metrics(sim: pl.DataFrame, direction: str = 'buy') -> pl.DataFrame:
    """One row of execution metrics per roll (security). Securities with an all-zero schedule
    (no historical curve) are dropped. Key columns: ``passive_rate`` (fraction filled passively,
    saving the spread), ``exec_avg_price``, and signed costs vs the mid / VWAP / arrival
    benchmarks in BOTH ticks per lot (default) and bp of the future price. Tick costs require a
    `tick` column on `sim` (e.g. via ``attach_tick_sizes``)."""
    aggs = list(_AGG_EXPRS)
    if 'futures_price' in sim.columns:
        aggs.append(pl.col('futures_price').mean().alias('avg_futures'))
    if 'tick' in sim.columns:
        aggs.append(pl.col('tick').mean().alias('avg_tick'))
    g = sim.group_by('security', maintain_order=True).agg(aggs).filter(pl.col('scheduled_qty') > 0)
    return _cost_columns(g, _side(direction)).sort('security')


def summarize(sim: pl.DataFrame, direction: str = 'buy') -> dict:
    """Portfolio-level execution summary (quantity-weighted across all scheduled rolls). 
    Aggregates per-security metrics first to eliminate cross-asset price and tick 
    distortion while fully preserving all original column names.
    """
    # 1. Compute metrics at the security level first to eliminate aggregation bias
    ps = per_security_metrics(sim, direction)
    if ps.height == 0:
        return {'direction': direction}

    total_qty_col = pl.col('total_qty')
    
    # 2. Reconstruct portfolio aggregates preserving all original column names
    selects = [
        # Quantities and raw cash notionals aggregate cleanly as global portfolio sums
        pl.col('scheduled_qty').sum().alias('scheduled_qty'),
        pl.col('passive_qty').sum().alias('passive_qty'),
        pl.col('agg_qty').sum().alias('agg_qty'),
        pl.col('total_qty').sum().alias('total_qty'),
        pl.col('exec_notional').sum().alias('exec_notional'),
        pl.col('mid_sched_notional').sum().alias('mid_sched_notional'),
        pl.col('mkt_vwap_num').sum().alias('mkt_vwap_num'),
        pl.col('mkt_vol').sum().alias('mkt_vol'),
        pl.col('security').n_unique().alias('n_rolls'),
    ]

    # Helper function to apply true lot-weighted means for prices, spreads, and costs
    def lot_weighted(col_name: str):
        return ((pl.col(col_name) * total_qty_col).sum() / total_qty_col.sum()).alias(col_name)

    # Core normalized benchmarks and costs
    lot_weighted_cols = [
        'passive_rate', 'exec_avg_price', 'mid_benchmark', 'mkt_vwap', 'arrival_mid',
        'avg_half_spread', 'cost_vs_mid', 'cost_vs_vwap', 'cost_vs_arrival',
    ]
    for c in lot_weighted_cols:
        selects.append(lot_weighted(c))

    # Conditional basis-point framework columns (only when a futures price was available)
    if 'avg_futures' in ps.columns:
        bp_cols = [
            'avg_futures', 'cost_vs_mid_bp', 'cost_vs_vwap_bp',
            'cost_vs_arrival_bp', 'half_spread_bp'
        ]
        for c in bp_cols:
            selects.append(lot_weighted(c))

    # Conditional tick framework columns
    if 'avg_tick' in ps.columns:
        tick_cols = [
            'avg_tick', 'cost_vs_mid_ticks', 'cost_vs_vwap_ticks', 
            'cost_vs_arrival_ticks', 'half_spread_ticks'
        ]
        for c in tick_cols:
            selects.append(lot_weighted(c))

    # 3. Execute the aggregation profile and return as a flat dictionary
    summary = ps.select(selects).row(0, named=True)
    summary['direction'] = direction
    return summary


# ── 4. Driver ────────────────────────────────────────────────────────────────────

def build_schedules(
    df_bins: pl.DataFrame,
    df_history: pl.DataFrame,
    participation_rate: float = 0.05,
    leakage_safe: bool = True,
    curve: pl.DataFrame | None = None,
    roll_volume: pl.DataFrame | None = None,
    qcode_col: str = 'qcode',
    position_cols: tuple[str, ...] = ('days_until', 'bin_start_time'),
    target_col: str = 'target_date',
) -> pl.DataFrame:
    """Attach ``scheduled_qty`` to every bin in `df_bins`. Each roll's target is
    ``participation_rate`` of its qcode's average roll volume, sliced by the volume curve; both
    the curve and the roll-volume average are sourced from `df_history` (the full roll history —
    usually the same `df_cs`).

      - `leakage_safe=True` (default): each roll is scheduled against a curve AND a roll-volume
        average built only from PRIOR rolls of its qcode (``historical_volume_curve`` /
        ``historical_roll_volume`` with ``before=target_date``) — backtest-honest, so the first
        roll per qcode is unscheduled (no history). O(rolls) builds.
      - `leakage_safe=False`: schedule every roll against one full-sample curve and roll volume
        (`curve` / `roll_volume`, or ones computed from `df_history`). Quicker, but peeks at the
        whole sample — diagnostics only.
    """
    if leakage_safe:
        keys = df_bins.select('security', qcode_col, target_col).unique()
        parts = []
        for row in keys.iter_rows(named=True):
            qc, tgt, sec = row[qcode_col], row[target_col], row['security']
            qc_hist = df_history.filter(pl.col(qcode_col) == qc)
            hist = historical_volume_curve(
                qc_hist, before=tgt, group_col=qcode_col, position_cols=position_cols,
            )
            rv = historical_roll_volume(qc_hist, before=tgt, group_col=qcode_col, target_col=target_col)
            bins = df_bins.filter(pl.col('security') == sec)
            parts.append(attach_volume_schedule(bins, hist, rv, participation_rate, qcode_col, position_cols))
        return pl.concat(parts) if parts else df_bins.head(0)

    if curve is None:
        curve = compute_volume_curve(df_history, group_col=qcode_col, position_cols=position_cols)
    if roll_volume is None:
        roll_volume = compute_roll_volume(df_history, group_col=qcode_col)
    return attach_volume_schedule(df_bins, curve, roll_volume, participation_rate, qcode_col, position_cols)


def run_vwap_backtest(
    df_cs: pl.DataFrame,
    participation_rate: float = 0.05,
    direction: str = 'buy',
    fill_model: str = 'queue',
    window_fraction: float = 0.3,
    leakage_safe: bool = True,
    curve: pl.DataFrame | None = None,
    roll_volume: pl.DataFrame | None = None,
    df_signals: pl.DataFrame | None = None,
    qcode_col: str = 'qcode',
    position_cols: tuple[str, ...] = ('days_until', 'bin_start_time'),
    target_col: str = 'target_date',
) -> tuple[pl.DataFrame, dict]:
    """Run the baseline VWAP over every roll in `df_cs` and return (per-security metrics, summary).

    `df_cs` is the spread frame from ``pipeline.build_datasets`` (one row per bin, carrying
    `qcode`, `security`, `days_until`, `target_date`, the *_start quotes and volume /
    signed_volume). See ``build_schedules`` for the `leakage_safe` curve semantics.

    `participation_rate` sizes each roll as that fraction of its qcode's average roll volume.
    `window_fraction` sets the per-security order survival window to that fraction of the
    security's touch-size-to-trade-volume ratio (see ``lob_simulation.simulate_windowed``); a
    larger window lets a passive order rest across more bins, raising the passive-fill rate. Pass
    `df_signals` to report costs in ticks per lot (the default framework, from its `_tick`) and,
    when it carries a non-null `futures_price` (i.e. ``generate_signals`` was given a `df_fut`), in
    basis points too; without it only the ticks framework is emitted.
    """
    if df_signals is not None:
        df_cs = attach_tick_sizes(df_cs, df_signals)       # adds `tick`  -> ticks framework
        df_cs = attach_futures_price(df_cs, df_signals)    # adds `futures_price` -> bp framework
    scheduled = build_schedules(
        df_cs, df_cs, participation_rate, leakage_safe, curve, roll_volume,
        qcode_col, position_cols, target_col,
    )
    sim = simulate_windowed(scheduled, direction=direction, fill_model=fill_model,
                            window_fraction=window_fraction)
    return per_security_metrics(sim, direction), summarize(sim, direction)


# ── 5. Improved VWAP: ordered-logit overlay ────────────────────────────────────────

def attach_tick_sizes(df_cs: pl.DataFrame, df_signals: pl.DataFrame, tick_col: str = '_tick') -> pl.DataFrame:
    """Join the per-BBG_CODE price tick (``generate_signals``' `_tick`) onto `df_cs` as `tick`."""
    ticks = df_signals.group_by('bbg_code').agg(tick=pl.col(tick_col).first())
    return df_cs.join(ticks, on='bbg_code', how='left')


def attach_futures_price(df_cs: pl.DataFrame, df_signals: pl.DataFrame) -> pl.DataFrame:
    """Join the per-bin near-leg ``futures_price`` from ``generate_signals`` output onto `df_cs`,
    enabling the basis-point cost framework in ``per_security_metrics`` / ``summarize``.

    ``build_datasets``' `df_cs` no longer carries `futures_price` — it is computed in
    ``generate_signals`` (only when a `df_fut` is supplied). This brings it back onto the bins via
    the same `df_signals` conduit ``attach_tick_sizes`` uses. A no-op if `df_signals` lacks the
    column, `df_cs` already has it, or the column is entirely null (``generate_signals`` was run
    without a `df_fut`, so there is no real futures price) — in which case the bp framework is
    simply not emitted and only the ticks framework is reported."""
    if (
        'futures_price' not in df_signals.columns
        or 'futures_price' in df_cs.columns
        or df_signals['futures_price'].null_count() == df_signals.height
    ):
        return df_cs
    keys = ['security', 'date', 'bin_start_time']
    return df_cs.join(df_signals.select([*keys, 'futures_price']), on=keys, how='left')


def simulate_improved_vwap(
    scheduled: pl.DataFrame,
    direction: str = 'buy',
    fill_model: str = 'hilo',
    tick_col: str = 'tick',
    pred_col: str = 'pred',
    ticks: int = 2,
    queue_depth_levels: float = 1.7,
    overlay_actions: str = 'both',
    high_col: str = 'high',
    low_col: str = 'low',
) -> pl.DataFrame:
    """Ordered-logit overlay on the baseline VWAP. The model's predicted price move for the
    target bin (`pred_col` ∈ {-2, 0, +2}, known at the bin's start) reshapes execution. For a BUY:

      - pred == -2 (favourable, price expected DOWN): rest the passive limit `ticks` ticks BELOW
        the best bid (``bid_start - ticks*tick``). If filled, we buy cheaper than the normal bid;
        the unfilled remainder still crosses at the next bin's start. Saves ~`ticks` ticks when
        right; when wrong the limit simply does not fill and the bin is crossed (the modelled cost).
      - pred == +2 (adverse, price expected UP): skip resting entirely and cross at the START of
        the target bin (``ask_start``) — beating the higher ask the baseline would have paid later.
      - pred == 0 / null: behave exactly like the baseline (rest at the touch).

    Selling mirrors this (favourable = +2 -> rest `ticks` above the ask; adverse = -2 -> cross now).
    The session's last bin is still forced aggressive (overnight guard), as in the baseline.

    `overlay_actions` enables only one of the two actions for studying each in isolation: 'both'
    (default), 'favourable' (rest-deeper only — adverse calls fall back to the baseline touch), or
    'adverse' (cross-now only — favourable calls fall back to the baseline touch).

    Fill of the deeper passive level is genuinely uncertain with only top-of-book depth, so:
      - `fill_model='hilo'` (default, recommended): the limit fills in full iff the bin's range
        reaches it (``low <= bid - ticks*tick`` for a buy) — the most defensible call from the
        data, and the reason HIGH/LOW were added to the pull.
      - `fill_model='queue'`: a discretionary proxy — the order must clear ~`queue_depth_levels`x
        the visible touch depth before filling (``opposing_flow - queue_depth_levels*depth``),
        gated on the range reaching the level when HIGH/LOW are available.

    Emits the same columns as ``simulate_passive_aggressive`` so the metric helpers are reused.
    """
    if direction not in ('buy', 'sell'):
        raise ValueError(f"direction must be 'buy' or 'sell', got {direction!r}")
    if fill_model not in ('queue', 'hilo'):
        raise ValueError(f"fill_model must be 'queue' or 'hilo', got {fill_model!r}")
    if overlay_actions not in ('both', 'favourable', 'adverse'):
        raise ValueError(
            f"overlay_actions must be 'both', 'favourable' or 'adverse', got {overlay_actions!r}"
        )
    has_range = high_col in scheduled.columns and low_col in scheduled.columns
    if fill_model == 'hilo' and not has_range:
        raise ValueError(
            f"fill_model='hilo' needs '{high_col}'/'{low_col}' — add HIGH/LOW to "
            f"pipeline.BINNED_COLS and rebuild. Present: {scheduled.columns}"
        )
    if tick_col not in scheduled.columns:
        raise ValueError(f"missing tick column '{tick_col}' — call attach_tick_sizes() first.")

    favourable = -2 if direction == 'buy' else 2   # price moves our way -> rest deeper to capture it
    adverse = 2 if direction == 'buy' else -2       # price moves against us -> take liquidity now

    df = scheduled.sort(['security', 'date', 'bin_start_time']).with_columns(
        mid=(pl.col('bid_start') + pl.col('ask_start')) / 2,
        sell_init=((pl.col('volume') - pl.col('signed_volume')) / 2).clip(lower_bound=0),
        buy_init=((pl.col('volume') + pl.col('signed_volume')) / 2).clip(lower_bound=0),
    )

    is_last = pl.col('bin_start_time') == pl.col('bin_start_time').max().over(['security', 'date'])
    sched = pl.col('scheduled_qty')
    tick = pl.col(tick_col)

    if direction == 'buy':
        touch = pl.col('bid_start')
        deep_level = touch - ticks * tick
        depth = pl.col('bid_size_start')
        opposing = pl.col('sell_init')
        now_price = pl.col('ask_start')                                  # cross immediately (adverse / last)
        next_cross = pl.col('ask_start').shift(-1).over(['security', 'date'])
        cross_fallback = pl.col('ask_start')
        reach_touch = (pl.col(low_col) <= touch) if has_range else pl.lit(True)
        reach_deep = (pl.col(low_col) <= deep_level) if has_range else pl.lit(True)
    else:
        touch = pl.col('ask_start')
        deep_level = touch + ticks * tick
        depth = pl.col('ask_size_start')
        opposing = pl.col('buy_init')
        now_price = pl.col('bid_start')
        next_cross = pl.col('bid_start').shift(-1).over(['security', 'date'])
        cross_fallback = pl.col('bid_start')
        reach_touch = (pl.col(high_col) >= touch) if has_range else pl.lit(True)
        reach_deep = (pl.col(high_col) >= deep_level) if has_range else pl.lit(True)

    if fill_model == 'hilo':
        touch_fill = pl.when(reach_touch).then(sched).otherwise(0.0)
        deep_fill = pl.when(reach_deep).then(sched).otherwise(0.0)
    else:  # queue (discretionary for the deeper level)
        touch_fill = (opposing - depth).clip(lower_bound=0)
        deep_fill = pl.when(reach_deep).then((opposing - queue_depth_levels * depth).clip(lower_bound=0)).otherwise(0.0)

    pred = pl.col(pred_col)
    # Partial overlay: drop the disabled action so those bins fall back to the baseline touch.
    act_adverse = overlay_actions in ('both', 'adverse')
    act_favourable = overlay_actions in ('both', 'favourable')
    force_agg = is_last | ((pred == adverse) if act_adverse else pl.lit(False))
    is_fav = ((pred == favourable) & ~is_last) if act_favourable else pl.lit(False)

    passive_price = pl.when(is_fav).then(deep_level).otherwise(touch)
    passive_fill = (
        pl.when(force_agg).then(0.0)
        .when(is_fav).then(deep_fill)
        .otherwise(touch_fill)                      # flat or no prediction -> baseline touch fill
    )

    df = df.with_columns(
        passive_price=passive_price,
        passive_fill=pl.min_horizontal(passive_fill, sched),
        _force_agg=force_agg,
        _next_cross=next_cross,
        _cross_fallback=cross_fallback,
        _now_price=now_price,
    )
    return df.with_columns(
        agg_qty=(sched - pl.col('passive_fill')).clip(lower_bound=0),
        # adverse / last -> cross now (this bin's start); otherwise the remainder crosses next bin.
        agg_price=pl.when(pl.col('_force_agg')).then(pl.col('_now_price'))
                    .otherwise(pl.coalesce([pl.col('_next_cross'), pl.col('_cross_fallback')])),
        agg_is_next=(~pl.col('_force_agg')) & pl.col('_next_cross').is_not_null(),
    ).drop(['_force_agg', '_next_cross', '_cross_fallback', '_now_price'])


# Evaluation frameworks: the cost-column suffix each one uses.
_FRAMEWORK_UNIT = {'ticks': '_ticks', 'bp': '_bp'}


def run_improved_vwap_backtest(
    df_cs: pl.DataFrame,
    predictions: pl.DataFrame,
    df_signals: pl.DataFrame,
    participation_rate: float = 0.05,
    direction: str = 'buy',
    fill_model: str = 'hilo',
    ticks: int = 2,
    window_fraction: float = 0.3,
    leakage_safe: bool = True,
    framework: str = 'ticks',
    overlay_actions: str = 'both',
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compare the ordered-logit VWAP to the baseline on the SAME rolls and schedule.

    `predictions` is ``src.backtest.predict_target_bins`` output — one row per OOS target bin
    with the (start-of-bin) predicted class `pred`. `df_signals` supplies the per-BBG tick.
    Execution is restricted to the securities (tick-constrained, out-of-sample rolls) that the
    model actually predicts, so the two strategies are compared like-for-like; the volume-curve
    schedule is identical for both. The session's last bin is forced aggressive in both.

    `framework` selects the headline evaluation unit — 'ticks' (per lot, the default) or 'bp'.
    The headline `improvement_vs_baseline` / `improvement_vs_vwap` columns are in that unit
    (positive = overlay cheaper); per-strategy cost columns are reported in BOTH units so the
    other framework is always one subtraction away.

    `overlay_actions` selects which overlay action the compared strategy uses — 'both' (default,
    the full overlay), 'favourable' (rest-deeper only), or 'adverse' (cross-now only) — so the two
    modifications can be evaluated in isolation against the same baseline (see
    ``lob_simulation.simulate_windowed``).

    `participation_rate` sizes each roll as that fraction of its qcode's average roll volume.
    `window_fraction` sets each security's order survival window to that fraction of its
    touch-size-to-trade-volume ratio (see ``lob_simulation.simulate_windowed``); it is computed
    off the shared schedule, so the same per-security window applies to both strategies and the
    comparison stays like-for-like.

    Returns (summary, per_security):
      - `summary`: two rows (baseline_vwap, ordered_logit_vwap) of portfolio metrics + the
        overlay's `improvement_vs_baseline` over the baseline (in `framework` units).
      - `per_security`: per-roll passive rates and mid/VWAP costs (both units) for both
        strategies side by side, plus `improvement_vs_vwap`.
    """
    if framework not in _FRAMEWORK_UNIT:
        raise ValueError(f"framework must be one of {list(_FRAMEWORK_UNIT)}, got {framework!r}")
    if predictions is None or predictions.height == 0:
        raise ValueError('no predictions — predict_target_bins returned None/empty (every window skipped?).')
    unit = _FRAMEWORK_UNIT[framework]
    vwap_cost = f'cost_vs_vwap{unit}'

    secs = predictions.select('security').unique()
    df_use = (
        df_cs.join(secs, on='security', how='inner')
        .pipe(attach_tick_sizes, df_signals)      # adds `tick` -> enables the ticks framework
        .pipe(attach_futures_price, df_signals)   # adds `futures_price` -> enables the bp framework
        .join(
            predictions.select('security', 'date', 'bin_start_time', 'pred'),
            on=['security', 'date', 'bin_start_time'], how='left',
        )
    )

    scheduled = build_schedules(df_use, df_cs, participation_rate, leakage_safe=leakage_safe)

    base_sim = simulate_windowed(scheduled, direction=direction, fill_model=fill_model,
                                 window_fraction=window_fraction)
    impr_sim = simulate_windowed(scheduled, direction=direction, fill_model=fill_model,
                                 window_fraction=window_fraction, ticks=ticks, pred_col='pred',
                                 overlay_actions=overlay_actions)

    base_sum = {'strategy': 'baseline_vwap', **summarize(base_sim, direction)}
    impr_sum = {'strategy': 'ordered_logit_vwap', **summarize(impr_sim, direction)}
    if vwap_cost not in base_sum:
        raise ValueError(f"framework={framework!r} needs '{vwap_cost}' — was a `tick` column available?")
    summary = pl.DataFrame([base_sum, impr_sum]).with_columns(
        # Lower cost is better; positive = overlay beat the baseline (per the signed cost).
        improvement_vs_baseline=pl.lit(base_sum[vwap_cost] - impr_sum[vwap_cost]),
        framework=pl.lit(framework),
    )

    # `total_qty` (lots executed per roll, = the roll's participation-rate target by conservation)
    # is the correct per-lot weight for aggregating per-roll improvement across rolls — it is what
    # `summarize` pools over. With a participation rate this varies by roll (larger rolls weigh more).
    base_full = per_security_metrics(base_sim, direction)
    # bp cost columns only exist when a futures price was available (see `per_security_metrics`).
    cost_cols = [c for c in ('cost_vs_vwap_ticks', 'cost_vs_vwap_bp', 'cost_vs_mid_ticks',
                             'cost_vs_mid_bp') if c in base_full.columns]
    base_ps = base_full.select(
        'security', 'qcode', 'total_qty', 'passive_rate', 'exec_avg_price', *cost_cols,
    )
    impr_ps = per_security_metrics(impr_sim, direction).select(
        'security',
        pl.col('passive_rate').alias('passive_rate_impr'),
        pl.col('exec_avg_price').alias('exec_avg_price_impr'),
        *[pl.col(c).alias(f'{c}_impr') for c in cost_cols],
    )
    per_security = base_ps.join(impr_ps, on='security', how='left').with_columns(
        improvement_vs_vwap=pl.col(vwap_cost) - pl.col(f'{vwap_cost}_impr'),
    )
    return summary, per_security
