"""Data pipeline for Order Book Signals on Bond & Equity Calendar Spreads.

Extracts the preprocessing and signal-generation code from
notebooks/02_regressions.ipynb into importable functions:

    from src.pipeline import build_datasets, generate_signals

    df_cs, df_combined = build_datasets(env_path='../.env')
    df_signals = generate_signals(df_cs)

Stages (each also callable individually for inspection / iteration):
  1. load_raw_tables          - pull BINNED_DATA / QCODE_MAPPING / SECURITY_META from Snowflake
  2. preprocess               - join metadata, parse tickers, attach roll calendar, filter years
  3. split_spreads_futures    - roll-window filters -> (calendar spreads, futures)
  4. apply_roll_volume_filter - volume-based roll-period filter + futures reference price
  5. generate_signals         - microstructure signals + leakage-safe normalisation
  6. generate_futures_signals - the SAME signals computed on the outright futures, and
     attach_outright_signals    buy-/sell-leg outright OBI/STV joined onto each spread bin
"""

import operator
from functools import reduce

import polars as pl
from snowflake.snowpark import functions as F
from snowflake.snowpark.session import Session

from src.utils import (
    add_microstructure_signals,
    build_contract_calendar,
    create_snowpark_session,
    parse_security,
    read_table,
    retrieve_polars_from_snowpark,
)

# ── Configuration ──────────────────────────────────────────────────────────────

DB = 'LISTED_INTERN_PROJECT'
SCHEMA = 'PROJECT_5'

BINNED_COLS = [
    'QCODE', 'SECURITY', 'BIN_START_TIME', 'PUBLICATION_DATE',
    'BID_SIZE_START', 'ASK_SIZE_START', 'BID_START', 'ASK_START',
    'VOLUME', 'SIGNED_VOLUME',
    # HIGH / LOW of the bin: used by the VWAP hi-lo fill model (was the bin's price range
    # deep enough to fill a resting limit?). Verify these match the BINNED_DATA column names.
    'HIGH', 'LOW',
]
QCODE_COLS = ['QCODE', 'BBG_CODE', 'YELLOW_KEY', 'DELIVERY', 'IS_CONVENTION_BUY_NEAR']
SEC_META_COLS = ['SECURITY', 'LAST_TRADE_DATE', 'FIRST_NOTICE_DATE']

# Data subset: trading years 2021-2025, feeding the chronological OOS split downstream.
YEARS = [2021, 2022, 2023, 2024, 2025]

# Chronological out-of-sample split (by trading-data year). Defined here so every
# downstream transform references the SAME train partition - in particular the
# within-train z-score padding in the normalisation step, which must not pull
# statistics from validation/test rows.
TRAIN_YEARS = [2021, 2022, 2023]
VAL_YEARS = [2024]
TEST_YEARS = [2025]
YEAR_LABEL = {'train': '2021-2023', 'val': '2024', 'test': '2025'}

ROLL_DAYS = 10              # business days in each contract's roll window
ROLL_VOLUME_THRESHOLD = 0.25  # daily volume >= 25% of the security's max daily volume
ROLL_WIN = 60               # rolling-window length for z-scores (bins, within each session)
SESSION = ['security', 'date']


# ── 1. Load raw tables ─────────────────────────────────────────────────────────

def load_raw_tables(session: Session, verbose: bool = True):
    """Pull QCODE_MAPPING, SECURITY_META and the filtered BINNED_DATA subset.

    BINNED_DATA is filtered in Snowflake to:
      - futures: QCODE in the Phys/Cash subset AND SECURITY has no slash, OR
      - calendar spreads: SECURITY matches '{BBG_CODE}%/% {YELLOW_KEY}' for any subset product
        (spreads pulled via the name pattern may have QCODEs outside the subset).

    Returns (binned, qmap, sec_meta) as Polars DataFrames.
    """
    qmap = read_table(session, DB, SCHEMA, 'QCODE_MAPPING', QCODE_COLS)
    sec_meta = read_table(session, DB, SCHEMA, 'SECURITY_META', SEC_META_COLS)

    phys_qcodes = qmap.filter(pl.col('delivery') == 'Phys')['qcode'].unique().to_list()
    cash_qcodes = qmap.filter(pl.col('delivery') == 'Cash')['qcode'].unique().to_list()
    subset_qcodes = phys_qcodes + cash_qcodes

    # BBG_CODE + YELLOW_KEY - used to construct the spread filter.
    subset_products = (
        qmap.filter(pl.col('qcode').is_in(subset_qcodes))
        .select('bbg_code', 'yellow_key')
        .unique()
    )

    # Futures: QCODE in our subset AND SECURITY has no slash (LIKE '%/%' negated).
    futures_filter = (
        F.col('QCODE').isin(subset_qcodes) &
        ~F.col('SECURITY').like('%/%')
    )

    # Spreads: SECURITY matches pattern {BBG_CODE}.../{...} {YELLOW_KEY}.
    # Using Snowflake LIKE: '%' = zero-or-more wildcard characters.
    spread_clauses = [
        F.col('SECURITY').like(f'{row["bbg_code"]}%/% {row["yellow_key"]}')
        for row in subset_products.iter_rows(named=True)
    ]
    spread_filter = reduce(operator.or_, spread_clauses)

    snow_binned = (
        session.table(f'{DB}.{SCHEMA}.BINNED_DATA')
        .select(*BINNED_COLS)
        .filter(futures_filter | spread_filter)
    )
    binned = retrieve_polars_from_snowpark(snow_binned)

    if verbose:
        n_fut = binned.filter(~pl.col('security').str.contains('/')).height
        n_spr = binned.filter(pl.col('security').str.contains('/')).height
        print(f'Phys qcodes : {phys_qcodes}')
        print(f'Cash qcodes : {cash_qcodes}')
        print(f'binned   : {binned.shape}  (futures: {n_fut:,}  |  spreads: {n_spr:,})')
        print(f'qmap     : {qmap.shape}')
        print(f'sec_meta : {sec_meta.shape}')
    return binned, qmap, sec_meta


# ── 2. Preprocess ───────────────────────────────────────────────────────────────

def preprocess(
    binned: pl.DataFrame,
    qmap: pl.DataFrame,
    sec_meta: pl.DataFrame,
    roll_days: int = ROLL_DAYS,
    years: list[int] = YEARS,
) -> pl.DataFrame:
    """Join metadata, parse tickers, attach the roll calendar and filter to `years`.

    Steps:
      1. merge BINNED_DATA with QCODE_MAPPING on QCODE,
      2. parse SECURITY -> is_spread, near/far identifiers, meta_key,
      3-4. build the per-future roll calendar (target date + own/previous roll windows)
         and join it on meta_key,
      5. trading date from PUBLICATION_DATE (BIN_START_TIME is time-only, not a date),
      6. keep rows whose trading date (PUBLICATION_DATE year) is in `years`.
    """
    data = binned.join(qmap, on='qcode', how='left')
    data = parse_security(data, col='security')

    calendar = build_contract_calendar(sec_meta, qmap, roll_days=roll_days)

    # Drop qmap's 'delivery' before joining the calendar so the calendar's 'delivery'
    # (correctly resolved from BBG_CODE+YELLOW_KEY for every security) is the sole authority.
    data = data.drop('delivery')
    data = data.join(
        calendar.rename({'security': 'meta_key'}),
        on='meta_key',
        how='left',
    )

    data = data.with_columns(date=pl.col('publication_date').cast(pl.Date))
    return data.filter(pl.col('publication_date').dt.year().is_in(years))


# ── 3. Roll-window split ────────────────────────────────────────────────────────

def split_spreads_futures(data: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split the preprocessed frame into (calendar spreads, futures) by roll window.

      - Calendar spreads: own (near-leg) roll period only; adds `days_until` =
        business days from the bin's date to the target date.
      - Futures: own roll period OR previous contract's roll period.
    """
    # A bin sits inside a [start, end] window (inclusive). Null bounds (missing metadata) -> False.
    def _in_window(start: str, end: str) -> pl.Expr:
        return pl.col('date').is_between(pl.col(start), pl.col(end), closed='both').fill_null(False)

    in_own_roll = _in_window('roll_start', 'roll_end')
    in_prev_roll = _in_window('prev_roll_start', 'prev_roll_end')

    df_cs = data.filter(pl.col('is_spread') & in_own_roll)
    df_cs = df_cs.with_columns(days_until=pl.business_day_count(pl.col('date'), pl.col('target_date')))

    df_fut = data.filter(~pl.col('is_spread') & (in_own_roll | in_prev_roll))
    return df_cs, df_fut


# ── 4. Volume-based roll filter + futures reference price ──────────────────────

def apply_roll_volume_filter(
    df_cs: pl.DataFrame,
    df_fut: pl.DataFrame,
    threshold: float = ROLL_VOLUME_THRESHOLD,
) -> pl.DataFrame:
    """Keep spread bins in the volume-defined roll period and attach `futures_price`.

    Roll period: daily volume >= `threshold` x the same security's max daily volume.
    `futures_price`: the near-leg future's daily average mid price, as-of joined strictly
    *before* the spread row's date (backward, no exact matches) - used downstream for the
    basis-point price change.
    """
    daily_volume = df_cs.group_by('security', 'date').agg(daily_vol=pl.col('volume').sum())
    max_volume = daily_volume.group_by('security').agg(max_daily_vol=pl.col('daily_vol').max())

    df_cs = df_cs.join(daily_volume, on=['security', 'date'], how='left')
    df_cs = df_cs.join(max_volume, on='security', how='left')
    df_cs = df_cs.with_columns(
        is_roll_period=(pl.col('daily_vol') >= threshold * pl.col('max_daily_vol'))
    ).drop(['daily_vol', 'max_daily_vol'])

    df_fut_avg = (
        df_fut
        .with_columns(mid_price=pl.mean_horizontal(['bid_start', 'ask_start']))
        .group_by(['security', 'date'])
        .agg(futures_price=pl.col('mid_price').mean())
    )

    df_cs = df_cs.sort('date')
    df_fut_avg = df_fut_avg.sort('date')

    df_cs = df_cs.join_asof(
        df_fut_avg,
        on='date',
        by_left='near_identifier',
        by_right='security',
        strategy='backward',
        allow_exact_matches=False,  # forces it to look strictly *before* the current row's date
    )

    return df_cs.filter(pl.col('is_roll_period'))


# ── 5. Signal generation + normalisation ───────────────────────────────────────

# Flow signals carried as z-score-able / vol-scalable quantities (the others - delta_p and
# obi - get their own scale normalisation in the transform stage).
SIGNAL_NORM_COLS = ['ofi', 'stv', 'noi']

# Time-of-roll volatility: a 1-hour centred window on 5-minute bins.
TOR_VOL_WIN = 12
TOR_VOL_MIN_SAMPLES = 6


def _apply_signal_transforms(df_cs: pl.DataFrame) -> pl.DataFrame:
    """Microstructure signals + the *scale* transforms that are independent of any split.

    Produces (all overwriting the raw columns):
      - delta_p : in ticks (per-BBG_CODE tick size, kept as `_tick`),
      - obi     : fraction of total quoted size at the touch,
      - stv/noi : signed square root (variance-stabilising, sign-preserving),
      - ofi     : left RAW here (its location/scale normalisation happens downstream).

    No within-session rolling statistics and no train-restricted means are computed here, so
    this stage is identical for every normalisation strategy. ``fill_nan(None)`` converts
    floating-point 0/0 = NaN for inactive bins to null rather than propagating NaN.
    """
    df_signals = add_microstructure_signals(df_cs, security_col='security', time_col='bin_start_time')

    # Tick size per BBG_CODE: minimum strictly-positive |delta_p| (drop zero / near-zero
    # changes so noise is not mistaken for the tick).
    tick_sizes = (
        df_signals
        .filter(pl.col('delta_p').abs() > 10e-6)
        .group_by('bbg_code')
        .agg(_tick=pl.col('delta_p').abs().min())
    )
    df_signals = df_signals.join(tick_sizes, on='bbg_code', how='left')

    return df_signals.with_columns(
        delta_p=(pl.col('delta_p') / pl.col('_tick')).fill_nan(None),
        obi=(pl.col('obi') / (pl.col('bid_size_start') + pl.col('ask_size_start'))).fill_nan(None),
        stv=(pl.col('stv').cast(pl.Float64).abs().sqrt() * pl.col('stv').sign()).fill_nan(None),
        noi=(pl.col('noi').cast(pl.Float64).abs().sqrt() * pl.col('noi').sign()).fill_nan(None),
    )


def _normalize_rolling_zscore(
    df_signals: pl.DataFrame,
    train_years: list[int],
    roll_win: int,
) -> pl.DataFrame:
    """Original normalisation: rolling z-score of OFI / STV / NOI within each [security, date]
    session.

    LEAKAGE NOTE: degenerate (near-zero) STV/NOI rolling stds are imputed with a typical std
    whose mean is taken over TRAIN ROWS ONLY (`train_years`). This fallback still references a
    train partition (the motivation for the time-of-roll alternative below) and the rolling std
    is noisy at the start of each session where the window has few samples.
    """
    df_signals = df_signals.with_columns(
        _ofi_mu=pl.col('ofi').rolling_mean(window_size=roll_win, min_samples=2).over(SESSION),
        _ofi_sd=pl.col('ofi').rolling_std(window_size=roll_win, min_samples=2).over(SESSION),
    ).with_columns(
        ofi=((pl.col('ofi') - pl.col('_ofi_mu')) / pl.col('_ofi_sd')).fill_nan(None),
    ).drop(['_ofi_mu', '_ofi_sd'])

    df_signals = df_signals.with_columns(
        _stv_mu=pl.col('stv').rolling_mean(window_size=roll_win, min_samples=2).over(SESSION),
        _stv_sd=pl.col('stv').rolling_std(window_size=roll_win, min_samples=2).over(SESSION),
        _noi_mu=pl.col('noi').rolling_mean(window_size=roll_win, min_samples=2).over(SESSION),
        _noi_sd=pl.col('noi').rolling_std(window_size=roll_win, min_samples=2).over(SESSION),
    )

    _is_train = pl.col('date').dt.year().is_in(train_years)
    df_signals = df_signals.with_columns(
        _stv_sd=pl.when(pl.col('_stv_sd') < 1e-6)
                .then(pl.col('_stv_sd').filter((pl.col('_stv_sd') >= 1e-6) & _is_train).mean())
                .otherwise(pl.col('_stv_sd')),
        _noi_sd=pl.when(pl.col('_noi_sd') < 1e-6)
                .then(pl.col('_noi_sd').filter((pl.col('_noi_sd') >= 1e-6) & _is_train).mean())
                .otherwise(pl.col('_noi_sd')),
    )

    return df_signals.with_columns(
        stv=((pl.col('stv') - pl.col('_stv_mu')) / pl.col('_stv_sd')),
        noi=((pl.col('noi') - pl.col('_noi_mu')) / pl.col('_noi_sd')),
    ).drop(['_stv_mu', '_stv_sd', '_noi_mu', '_noi_sd'])


def _normalize_time_of_roll(
    df_signals: pl.DataFrame,
    signal_cols: list[str] = SIGNAL_NORM_COLS,
    vol_win: int = TOR_VOL_WIN,
    min_samples: int = TOR_VOL_MIN_SAMPLES,
    qcode_col: str = 'qcode',
) -> pl.DataFrame:
    """Time-of-roll volatility normalisation (leakage-free, no start-of-session noise blow-up).

    Each signal is divided by a volatility that depends only on the *stage of the roll*
    - (`days_until`, `bin_start_time`) - estimated from PAST rolls of the same ``qcode``:

      1. Local intraday vol: within each [security, date] session, the centred `vol_win`-bin
         (1-hour) rolling std of the signal. Centring uses neighbouring bins of the *same*
         completed roll, so it is look-ahead only within already-historical rolls.
      2. Time-of-roll profile: for each (qcode, days_until, bin_start_time) stage, the mean of
         that local vol across all STRICTLY PRIOR rolls (securities ordered by `target_date`,
         current roll excluded). A roll is normalised purely by rolls that finished before it,
         so there is no train/val/test leakage and no train-restricted fallback.
      3. Normalise: signal / profile-vol (scale only - the flow signals are ~zero-mean).

    The earliest roll of each ``qcode`` (no prior history) and stages with a degenerate profile
    vol yield null - dropped downstream rather than imputed, by design.
    """
    df = df_signals.sort(['security', 'date', 'bin_start_time'])

    # 1. Local centred intraday vol per signal, never crossing a day boundary.
    df = df.with_columns([
        pl.col(c)
        .rolling_std(window_size=vol_win, min_samples=min_samples, center=True)
        .over(SESSION)
        .alias(f'_locvol_{c}')
        for c in signal_cols
    ])

    # 2. Expanding mean of local vol over prior rolls within each (qcode, stage) cell. Each
    # security appears once per stage and rolls have distinct target_dates, so the cumulative
    # sum/count ordered by target_date (current row excluded) is exactly the past-rolls mean.
    stage = [qcode_col, 'days_until', 'bin_start_time']
    df = df.sort([*stage, 'target_date', 'security'])
    tor_exprs = []
    for c in signal_cols:
        lv = pl.col(f'_locvol_{c}')
        present = lv.is_not_null().cast(pl.Int64)
        prior_sum = lv.fill_null(0).cum_sum().over(stage) - lv.fill_null(0)
        prior_cnt = present.cum_sum().over(stage) - present
        # Guard prior_cnt == 0 explicitly: 0/0 would yield NaN (not null), which then slips
        # past the > 1e-12 scale guard below. No prior roll -> no profile -> null.
        tor_exprs.append(
            pl.when(prior_cnt > 0).then(prior_sum / prior_cnt).otherwise(None).alias(f'_torvol_{c}')
        )
    df = df.with_columns(tor_exprs)

    # 3. Scale-normalise; guard against a degenerate (≈0) profile vol producing inf.
    df = df.with_columns([
        pl.when(pl.col(f'_torvol_{c}') > 1e-12)
        .then(pl.col(c) / pl.col(f'_torvol_{c}'))
        .otherwise(None)
        .alias(c)
        for c in signal_cols
    ])

    drop = [f'_locvol_{c}' for c in signal_cols] + [f'_torvol_{c}' for c in signal_cols]
    return df.drop(drop).sort(['security', 'date', 'bin_start_time'])


def generate_signals(
    df_cs: pl.DataFrame,
    train_years: list[int] = TRAIN_YEARS,
    roll_win: int = ROLL_WIN,
    normalization: str = 'rolling_zscore',
) -> pl.DataFrame:
    """Microstructure signals (partitioned by SESSION [security, date]) + normalisation.

    Normalised values overwrite the raw columns so downstream sections use them automatically:

      1. delta_p in ticks (per-BBG_CODE tick size, kept as `_tick`),
      2. OBI as fraction of total quoted size at the touch,
      3-5. OFI / STV / NOI scaled by the chosen `normalization` (STV & NOI signed-sqrt first).

    `normalization` selects the scale applied to OFI / STV / NOI:
      - 'rolling_zscore' (default): within-session rolling z-score - the original method; see
        ``_normalize_rolling_zscore`` (train-restricted std fallback, noisy at session start).
      - 'time_of_roll': divide by a volatility profiled by roll stage from prior rolls only;
        see ``_normalize_time_of_roll`` (leakage-free, no start-of-session blow-up). Requires
        `qcode`, `days_until` and `target_date` columns (present on the spread frame).
    """
    df_signals = _apply_signal_transforms(df_cs)

    if normalization == 'rolling_zscore':
        return _normalize_rolling_zscore(df_signals, train_years, roll_win)
    if normalization == 'time_of_roll':
        return _normalize_time_of_roll(df_signals)
    raise ValueError(
        f"normalization must be 'rolling_zscore' or 'time_of_roll', got {normalization!r}"
    )


# ── 6. Outright-futures signals + buy/sell-leg augmentation ─────────────────────

def generate_futures_signals(
    df: pl.DataFrame,
    train_years: list[int] = TRAIN_YEARS,
    roll_win: int = ROLL_WIN,
    normalization: str = 'rolling_zscore',
) -> pl.DataFrame:
    """Compute the normalised OBI / STV signals for the OUTRIGHT FUTURES, using exactly the
    same machinery (`generate_signals`) that produces the calendar-spread signals.

    `df` may be the standalone futures frame (`df_fut`) or the combined frame (`df_combined`);
    futures are selected with ``~is_spread`` when that flag is present. Two columns that the
    spread path supplies upstream are reconstructed here so the shared signal code runs
    unchanged on the futures:
      - `futures_price`: undefined for an outright leg -> null. It only feeds the spread-only
        `delta_p_bp`, which is not used for futures, so a null column is harmless.
      - `days_until`: business days from the bin's date to the contract's own target date
        (mirrors `split_spreads_futures` for spreads; required by the 'time_of_roll' option).

    Returns one row per surviving future bin with the normalised signal columns. Callers
    typically keep ``[security, date, bin_start_time, obi, stv]`` for the leg join below.
    """
    df_fut = df.filter(~pl.col('is_spread')) if 'is_spread' in df.columns else df

    if 'futures_price' not in df_fut.columns:
        df_fut = df_fut.with_columns(futures_price=pl.lit(None, dtype=pl.Float64))

    # Recompute days_until for the future's OWN contract (null on combined-frame futures, which
    # inherit the spread-only column as null from the diagonal concat).
    df_fut = df_fut.with_columns(
        days_until=pl.business_day_count(pl.col('date'), pl.col('target_date')),
    )

    return generate_signals(
        df_fut, train_years=train_years, roll_win=roll_win, normalization=normalization,
    )


def attach_outright_signals(
    df_spread_signals: pl.DataFrame,
    df_futures_signals: pl.DataFrame,
    signal_cols: tuple[str, ...] = ('obi', 'stv'),
) -> pl.DataFrame:
    """Attach the buy-leg and sell-leg outright signals to each calendar-spread bin.

    Using IS_CONVENTION_BUY_NEAR (carried on the spread frame from QCODE_MAPPING) together with
    the parsed `near_identifier` / `far_identifier`, resolve which leg is bought vs. sold under
    the spread's quoting convention:

        is_convention_buy_near = True  -> buy = near leg, sell = far leg
        is_convention_buy_near = False -> buy = far  leg, sell = near leg
        null  (qcode outside the subset) -> buy/sell undefined -> null leg identifiers

    Then left-join the futures' normalised `signal_cols` for each leg on
    ``[<leg>_identifier, date, bin_start_time]`` (both sides share the same 5-minute binning),
    producing `buy_<sig>` / `sell_<sig>` columns. Economic prior for the downstream model: a
    price improvement on the BUY leg pushes the calendar-spread price up, and vice versa for the
    sell leg.

    Bins whose leg is absent from the futures frame (e.g. a leg outside its roll window) get
    null leg signals and are left in place for the modelling step to drop.
    """
    is_buy_near = pl.col('is_convention_buy_near').cast(pl.Boolean)
    buy_id = (
        pl.when(is_buy_near).then(pl.col('near_identifier'))
        .when(~is_buy_near).then(pl.col('far_identifier'))
        .otherwise(None)
    )
    sell_id = (
        pl.when(is_buy_near).then(pl.col('far_identifier'))
        .when(~is_buy_near).then(pl.col('near_identifier'))
        .otherwise(None)
    )
    df = df_spread_signals.with_columns(buy_identifier=buy_id, sell_identifier=sell_id)

    fut = df_futures_signals.select(['security', 'date', 'bin_start_time', *signal_cols])
    for side in ('buy', 'sell'):
        renamed = fut.rename({
            'security': f'{side}_identifier',
            **{c: f'{side}_{c}' for c in signal_cols},
        })
        df = df.join(renamed, on=[f'{side}_identifier', 'date', 'bin_start_time'], how='left')

    return df


def attach_leg_signals(
    df_signals: pl.DataFrame,
    df_combined: pl.DataFrame,
    train_years: list[int] = TRAIN_YEARS,
    roll_win: int = ROLL_WIN,
    normalization: str = 'rolling_zscore',
    signal_cols: tuple[str, ...] = ('obi', 'stv'),
) -> pl.DataFrame:
    """Add the buy-/sell-leg outright `signal_cols` to an already-built spread-signals frame.

    One-call orchestrator over `generate_futures_signals` (compute the outright OBI/STV with the
    SAME normalisation used for the spread signals) and `attach_outright_signals` (resolve the
    buy/sell legs from IS_CONVENTION_BUY_NEAR and join each leg's signals on
    ``[<leg>_identifier, date, bin_start_time]``).

    `df_signals` is the output of `generate_signals(df_cs)`; `df_combined` is the second output
    of `build_datasets` (it carries the outright futures alongside the spreads). Returns
    `df_signals` with `buy_<sig>` / `sell_<sig>` columns added; rows whose leg is outside its
    roll window get null leg signals and are dropped by the modelling step's `drop_nulls`.
    """
    df_fut_signals = generate_futures_signals(
        df_combined, train_years=train_years, roll_win=roll_win, normalization=normalization,
    )
    return attach_outright_signals(df_signals, df_fut_signals, signal_cols=signal_cols)


# ── Top-level entry point ───────────────────────────────────────────────────────

def build_datasets(
    env_path: str = '../.env',
    session: Session | None = None,
    roll_days: int = ROLL_DAYS,
    years: list[int] = YEARS,
    volume_threshold: float = ROLL_VOLUME_THRESHOLD,
    verbose: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run the full preprocessing pipeline and return (df_cs, df_combined).

      - df_cs       : filtered calendar spreads only (roll-window AND volume-defined roll
                      period, with `futures_price` attached) - ready for generate_signals().
      - df_combined : df_cs plus the filtered futures, diagonally concatenated (futures get
                      nulls in the spread-only columns: days_until, is_roll_period,
                      futures_price).

    Pass an existing Snowpark `session` to reuse a connection; otherwise one is created
    from `env_path`.
    """
    if session is None:
        session = create_snowpark_session(env_path)

    binned, qmap, sec_meta = load_raw_tables(session, verbose=verbose)
    data = preprocess(binned, qmap, sec_meta, roll_days=roll_days, years=years)
    df_cs, df_fut = split_spreads_futures(data)
    df_cs = apply_roll_volume_filter(df_cs, df_fut, threshold=volume_threshold)

    df_combined = pl.concat([df_cs, df_fut], how='diagonal')

    if verbose:
        print(f'df_cs       : {df_cs.shape}')
        print(f'df_fut      : {df_fut.shape}')
        print(f'df_combined : {df_combined.shape}')
    return df_cs, df_combined
