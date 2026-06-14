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

def build_schedules(
    df_bins: pl.DataFrame,
    df_history: pl.DataFrame,
    target_qty: float = 10_000,
    leakage_safe: bool = True,
    curve: pl.DataFrame | None = None,
    qcode_col: str = 'qcode',
    position_cols: tuple[str, ...] = ('days_until', 'bin_start_time'),
    target_col: str = 'target_date',
) -> pl.DataFrame:
    """Attach ``scheduled_qty`` to every bin in `df_bins`, sourcing the volume curve from
    `df_history` (the full roll history — usually the same `df_cs`).

      - `leakage_safe=True` (default): each roll is scheduled against a curve built only from
        PRIOR rolls of its qcode (``historical_volume_curve(before=target_date)``) — backtest-
        honest, so the first roll per qcode is unscheduled (no history). O(rolls) curve builds.
      - `leakage_safe=False`: schedule every roll against one full-sample curve (`curve`, or one
        computed from `df_history`). Quicker, but peeks at the whole sample — diagnostics only.
    """
    if leakage_safe:
        keys = df_bins.select('security', qcode_col, target_col).unique()
        parts = []
        for row in keys.iter_rows(named=True):
            qc, tgt, sec = row[qcode_col], row[target_col], row['security']
            hist = historical_volume_curve(
                df_history.filter(pl.col(qcode_col) == qc), before=tgt,
                group_col=qcode_col, position_cols=position_cols,
            )
            bins = df_bins.filter(pl.col('security') == sec)
            parts.append(attach_volume_schedule(bins, hist, target_qty, qcode_col, position_cols))
        return pl.concat(parts) if parts else df_bins.head(0)

    if curve is None:
        curve = compute_volume_curve(df_history, group_col=qcode_col, position_cols=position_cols)
    return attach_volume_schedule(df_bins, curve, target_qty, qcode_col, position_cols)


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
    and `futures_price`). See ``build_schedules`` for the `leakage_safe` curve semantics.
    """
    scheduled = build_schedules(
        df_cs, df_cs, target_qty, leakage_safe, curve, qcode_col, position_cols, target_col,
    )
    sim = simulate_passive_aggressive(scheduled, direction=direction, fill_model=fill_model)
    return per_security_metrics(sim, direction), summarize(sim, direction)


# ── 5. Improved VWAP: ordered-logit overlay ────────────────────────────────────────

def attach_tick_sizes(df_cs: pl.DataFrame, df_signals: pl.DataFrame, tick_col: str = '_tick') -> pl.DataFrame:
    """Join the per-BBG_CODE price tick (``generate_signals``' `_tick`) onto `df_cs` as `tick`."""
    ticks = df_signals.group_by('bbg_code').agg(tick=pl.col(tick_col).first())
    return df_cs.join(ticks, on='bbg_code', how='left')


def simulate_improved_vwap(
    scheduled: pl.DataFrame,
    direction: str = 'buy',
    fill_model: str = 'hilo',
    tick_col: str = 'tick',
    pred_col: str = 'pred',
    ticks: int = 2,
    queue_depth_levels: float = 2.0,
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
    force_agg = is_last | (pred == adverse)
    is_fav = (pred == favourable) & ~is_last

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


def run_improved_vwap_backtest(
    df_cs: pl.DataFrame,
    predictions: pl.DataFrame,
    df_signals: pl.DataFrame,
    target_qty: float = 10_000,
    direction: str = 'buy',
    fill_model: str = 'hilo',
    ticks: int = 2,
    leakage_safe: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compare the ordered-logit VWAP to the baseline on the SAME rolls and schedule.

    `predictions` is ``src.backtest.predict_target_bins`` output — one row per OOS target bin
    with the (start-of-bin) predicted class `pred`. `df_signals` supplies the per-BBG tick.
    Execution is restricted to the securities (tick-constrained, out-of-sample rolls) that the
    model actually predicts, so the two strategies are compared like-for-like; the volume-curve
    schedule is identical for both. The session's last bin is forced aggressive in both.

    Returns (summary, per_security):
      - `summary`: two rows (baseline_vwap, ordered_logit_vwap) of portfolio metrics plus the
        bp improvement of the overlay over the baseline.
      - `per_security`: per-roll passive rates and costs for both strategies, side by side.
    """
    if predictions is None or predictions.height == 0:
        raise ValueError('no predictions — predict_target_bins returned None/empty (every window skipped?).')
    secs = predictions.select('security').unique()
    df_use = (
        df_cs.join(secs, on='security', how='inner')
        .pipe(attach_tick_sizes, df_signals)
        .join(
            predictions.select('security', 'date', 'bin_start_time', 'pred'),
            on=['security', 'date', 'bin_start_time'], how='left',
        )
    )

    scheduled = build_schedules(df_use, df_cs, target_qty, leakage_safe=leakage_safe)

    base_sim = simulate_passive_aggressive(scheduled, direction=direction, fill_model=fill_model)
    impr_sim = simulate_improved_vwap(scheduled, direction=direction, fill_model=fill_model, ticks=ticks)

    base_sum = {'strategy': 'baseline_vwap', **summarize(base_sim, direction)}
    impr_sum = {'strategy': 'ordered_logit_vwap', **summarize(impr_sim, direction)}
    summary = pl.DataFrame([base_sum, impr_sum]).with_columns(
        # Lower cost is better; positive = overlay beat the baseline (per the signed cost).
        improvement_vs_baseline_bp=pl.lit(base_sum['cost_vs_vwap_bp'] - impr_sum['cost_vs_vwap_bp']),
    )

    base_ps = per_security_metrics(base_sim, direction).select(
        'security', 'qcode', 'passive_rate', 'exec_avg_price', 'cost_vs_vwap_bp', 'cost_vs_mid_bp',
    )
    impr_ps = per_security_metrics(impr_sim, direction).select(
        'security', pl.col('passive_rate').alias('passive_rate_impr'),
        pl.col('exec_avg_price').alias('exec_avg_price_impr'),
        pl.col('cost_vs_vwap_bp').alias('cost_vs_vwap_bp_impr'),
        pl.col('cost_vs_mid_bp').alias('cost_vs_mid_bp_impr'),
    )
    per_security = base_ps.join(impr_ps, on='security', how='left').with_columns(
        improvement_vs_vwap_bp=pl.col('cost_vs_vwap_bp') - pl.col('cost_vs_vwap_bp_impr'),
    )
    return summary, per_security
