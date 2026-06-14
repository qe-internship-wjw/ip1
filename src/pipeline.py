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
      1. merge BINNED_DATA with QCODE_MAPPING on QCODE (spreads pulled via the name-pattern
         filter may have QCODEs outside the subset, so their bbg_code / yellow_key /
         delivery / is_convention_buy_near will be null here),
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

def generate_signals(
    df_cs: pl.DataFrame,
    train_years: list[int] = TRAIN_YEARS,
    roll_win: int = ROLL_WIN,
) -> pl.DataFrame:
    """Microstructure signals (partitioned by SESSION [security, date]) + normalisation.

    Normalised values overwrite the raw columns so downstream sections (H1, H2, H3)
    use normalised signals automatically:

      1. delta_p in ticks (per-BBG_CODE tick size, kept as `_tick`),
      2. OBI as fraction of total quoted size at the touch,
      3. STV as signed square root, then rolling z-score within session,
      4. OFI as rolling z-score within session,
      5. NOI as signed square root, then rolling z-score within session.

    LEAKAGE FIX: degenerate (near-zero) rolling stds are imputed with a typical std whose
    mean is computed over TRAIN ROWS ONLY (`train_years`) - an un-grouped mean across all
    years would let validation/test variance bleed into the normalisation of train rows
    (and vice versa). The padded rows themselves may sit in any split; only the *mean*
    is restricted.

    fill_nan(None) converts floating-point 0/0 = NaN to null for inactive bins rather
    than propagating NaN into the regression.
    """
    df_signals = add_microstructure_signals(df_cs, security_col='security', time_col='bin_start_time')

    # 1. Tick size per BBG_CODE: minimum strictly-positive |delta_p|
    # (filter out zero and near-zero price changes to avoid noise/tick size confusion).
    tick_sizes = (
        df_signals
        .filter(pl.col('delta_p').abs() > 10e-6)
        .group_by('bbg_code')
        .agg(_tick=pl.col('delta_p').abs().min())
    )
    df_signals = df_signals.join(tick_sizes, on='bbg_code', how='left')

    # Pre-compute rolling OFI mean and std within each session.
    # df_signals is already sorted by [security, date, bin_start_time].
    df_signals = df_signals.with_columns(
        _ofi_mu=pl.col('ofi').rolling_mean(window_size=roll_win, min_samples=2).over(SESSION),
        _ofi_sd=pl.col('ofi').rolling_std(window_size=roll_win, min_samples=2).over(SESSION),
    )

    # All five expressions are evaluated against the INPUT state of df_signals in this
    # single with_columns call - Polars never sees intermediate results within the same
    # call. This means pl.col('ofi') in the NOI denominator is the PRE-Z-SCORE raw OFI.
    df_signals = df_signals.with_columns(
        delta_p=(pl.col('delta_p') / pl.col('_tick')).fill_nan(None),
        obi=(pl.col('obi') / (pl.col('bid_size_start') + pl.col('ask_size_start'))).fill_nan(None),
        stv=(pl.col('stv').cast(pl.Float64).abs().sqrt() * pl.col('stv').sign()).fill_nan(None),
        ofi=((pl.col('ofi') - pl.col('_ofi_mu')) / pl.col('_ofi_sd')).fill_nan(None),
        noi=(pl.col('noi').cast(pl.Float64).abs().sqrt() * pl.col('noi').sign()).fill_nan(None),
    ).drop(['_ofi_mu', '_ofi_sd'])

    df_signals = df_signals.with_columns(
        _stv_mu=pl.col('stv').rolling_mean(window_size=roll_win, min_samples=2).over(SESSION),
        _stv_sd=pl.col('stv').rolling_std(window_size=roll_win, min_samples=2).over(SESSION),
        _noi_mu=pl.col('noi').rolling_mean(window_size=roll_win, min_samples=2).over(SESSION),
        _noi_sd=pl.col('noi').rolling_std(window_size=roll_win, min_samples=2).over(SESSION),
    )

    # Impute degenerate (near-zero) rolling stds with a typical std from TRAIN rows only.
    _is_train = pl.col('date').dt.year().is_in(train_years)
    df_signals = df_signals.with_columns(
        _stv_sd=pl.when(pl.col('_stv_sd') < 1e-6)
                .then(pl.col('_stv_sd').filter((pl.col('_stv_sd') >= 1e-6) & _is_train).mean())
                .otherwise(pl.col('_stv_sd')),
        _noi_sd=pl.when(pl.col('_noi_sd') < 1e-6)
                .then(pl.col('_noi_sd').filter((pl.col('_noi_sd') >= 1e-6) & _is_train).mean())
                .otherwise(pl.col('_noi_sd')),
    )

    # STV and NOI as rolling z-scores within session.
    return df_signals.with_columns(
        stv=((pl.col('stv') - pl.col('_stv_mu')) / pl.col('_stv_sd')),
        noi=((pl.col('noi') - pl.col('_noi_mu')) / pl.col('_noi_sd')),
    ).drop(['_stv_mu', '_stv_sd', '_noi_mu', '_noi_sd'])


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
