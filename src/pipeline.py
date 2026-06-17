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
  4. apply_roll_volume_filter - volume-based roll-period flag (is_roll_period, no filter)
  5. generate_signals         - microstructure signals + leakage-safe normalisation
  6. generate_futures_signals - the SAME signals computed on the outright futures, and
     attach_outright_signals    buy-/sell-leg outright OBI/STV joined onto each spread bin
"""

import polars as pl
from snowflake.snowpark.session import Session

from src.utils import (
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
    'VOLUME', 'SIGNED_VOLUME', 'TWA_ASK_SIZE', 'TWA_BID_SIZE',
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

ROLL_DAYS = 10              # business days in each contract's rough roll window
ROLL_VOLUME_THRESHOLD = 0.25  # percentage of max volume to be considered roll
SESSION = ['security', 'date']


# ── 1. Load raw tables ─────────────────────────────────────────────────────────

def load_raw_tables(session: Session, verbose: bool = True):
    """Pull QCODE_MAPPING, SECURITY_META and BINNED_DATA from Snowflake.

    Assumes all QCODEs are Phys or Cash delivery and all BINNED_DATA rows are
    either outright futures (no slash in SECURITY) or calendar spreads (slash present).

    Returns (binned, qmap, sec_meta) as Polars DataFrames.
    """
    qmap = read_table(session, DB, SCHEMA, 'QCODE_MAPPING', QCODE_COLS)
    sec_meta = read_table(session, DB, SCHEMA, 'SECURITY_META', SEC_META_COLS)
    binned = retrieve_polars_from_snowpark(
        session.table(f'{DB}.{SCHEMA}.BINNED_DATA').select(*BINNED_COLS)
    )

    if verbose:
        n_fut = binned.filter(~pl.col('security').str.contains('/')).height
        n_spr = binned.filter(pl.col('security').str.contains('/')).height
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
      2. parse SECURITY -> is_spread, near/far identifiers,
      3-4. build the per-future roll calendar (target date + own/previous roll windows)
         and join it on near_identifier,
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
        calendar.rename({'security': 'near_identifier'}),
        on='near_identifier',
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
    threshold: float = ROLL_VOLUME_THRESHOLD,
) -> pl.DataFrame:
    """Tag spread bins with `is_roll_period` without filtering.

    Roll period: daily volume >= `threshold` x the same security's max daily volume.
    The column is retained for downstream use; rows outside the volume roll are not
    dropped because roll timing can differ across contracts sharing the same qcode.
    """
    daily_volume = df_cs.group_by('security', 'date').agg(daily_vol=pl.col('volume').sum())
    max_volume = daily_volume.group_by('security').agg(max_daily_vol=pl.col('daily_vol').max())

    df_cs = df_cs.join(daily_volume, on=['security', 'date'], how='left')
    df_cs = df_cs.join(max_volume, on='security', how='left')
    return df_cs.with_columns(
        is_roll_period=(pl.col('daily_vol') >= threshold * pl.col('max_daily_vol'))
    ).drop(['daily_vol', 'max_daily_vol'])


# ── 5. Signal generation + normalisation ───────────────────────────────────────

# Forward-looking columns that are null on the last bin of each session and must be dropped.
SIGNAL_LAG_COLS = ["delta_p", "delta_lb", "delta_la"]

# Valid choices for the obi_method parameter of add_microstructure_signals.
OBI_METHODS = ("end_of_bin", "twa", "twa_or_eob")


def add_microstructure_signals(
    df: pl.DataFrame,
    security_col: str = "security",
    time_col: str = "bin_start_time",
    date_col: str = "date",
    drop_lag_nulls: bool = True,
    df_fut: pl.DataFrame | None = None,
    obi_method: str = "end_of_bin",
) -> pl.DataFrame:
    """Compute the order-book / flow signals from §3 of the research plan.

    CRITICAL — strictly intraday: all forward-looking quantities are partitioned by the trading
    session ``[security_col, date_col]`` (not just the contract) and ordered by `time_col`, so a
    forward step never spans an overnight gap, weekend, or day boundary. `date_col` is derived from
    `time_col` if absent. The last bin of every (security, date) session therefore has null values.

    Columns added (using the *_START fields as each bin's representative quote, per the spec):

      date              : calendar date of the bin (session key, created if missing)
      mid_price (P_t)   : (bid_start + ask_start) / 2
      delta_p  (ΔP_t)   : P_{t+1} - P_t   (within session)
      obi      (OBI_t)  : bid_size - ask_size, sized by `obi_method` (see below)
      _obi_denom        : total quoted size (bid + ask) for the same method — denominator for
                          fractional OBI normalisation in _apply_signal_transforms
      delta_lb (ΔL_t^b) : 3-case bid liquidity change from t to t+1 (within session)
      delta_la (ΔL_t^a) : 3-case ask liquidity change from t to t+1 (inequalities mirrored vs. the bid:
                          an ask *improvement* is a price decrease)
      ofi      (OFI_t)  : ΔL_t^b - ΔL_t^a   (this is the Cont et al. order-flow imbalance)
      stv      (STV_t)  : signed_volume from the dataset; empty bins (null) -> 0 trades
      noi      (NOI_t)  : OFI_t - STV_t

    `obi_method` controls which quote sizes are used for OBI and its denominator:
      "end_of_bin"  (default) : start-of-next-bin sizes — consistent with the forward-looking
                                delta_p / OFI, null on the last bin of each session.
      "twa"                   : time-weighted average sizes (twa_bid_size / twa_ask_size) within
                                the bin — available for all bins including the last.
      "twa_or_eob"            : TWA sizes when mid-price is flat (delta_p == 0), end-of-bin sizes
                                when the price moves — captures state-change accurately while
                                using the smoother TWA estimate in quiet periods.

    With `drop_lag_nulls=True` the last bin of every session (null forward quantities) is removed.
    """
    if obi_method not in OBI_METHODS:
        raise ValueError(f"obi_method must be one of {OBI_METHODS}, got {obi_method!r}")

    # Session key: derive the calendar date from the bin timestamp if not already present.
    if date_col not in df.columns:
        df = df.with_columns(pl.col(time_col).cast(pl.Date).alias(date_col))

    session = [security_col, date_col]
    df = df.sort([*session, time_col])

    bid, ask = pl.col("bid_start"), pl.col("ask_start")
    bid_sz, ask_sz = pl.col("bid_size_start"), pl.col("ask_size_start")
    twa_bid_sz, twa_ask_sz = pl.col("twa_bid_size"), pl.col("twa_ask_size")

    # mid_price must be materialised before the forward shift that produces next_mid.
    df = df.with_columns(mid_price=(bid + ask) / 2)

    # Next-bin quotes WITHIN the same session — forward-looking, never across day/overnight boundaries.
    next_bid = bid.shift(-1).over(session)
    next_ask = ask.shift(-1).over(session)
    next_bid_sz = bid_sz.shift(-1).over(session)
    next_ask_sz = ask_sz.shift(-1).over(session)
    next_mid = pl.col("mid_price").shift(-1).over(session)

    df = df.with_columns(
        delta_p=next_mid - pl.col("mid_price"),
        # Bid: next price up -> all new liquidity; next price down -> old liquidity gone; flat -> net change.
        delta_lb=pl.when(next_bid == bid).then(next_bid_sz - bid_sz)
        .when(next_bid > bid).then(next_bid_sz)
        .otherwise(-bid_sz),
        # Ask: mirrored — next ask price down (improvement) -> all new; price up -> old gone; flat -> net.
        delta_la=pl.when(next_ask == ask).then(next_ask_sz - ask_sz)
        .when(next_ask < ask).then(next_ask_sz)
        .otherwise(-ask_sz),
    )

    # OBI signed imbalance and its normalisation denominator, chosen by obi_method.
    if obi_method == "end_of_bin":
        obi_expr = next_bid_sz - next_ask_sz
        obi_denom_expr = next_bid_sz + next_ask_sz
    elif obi_method == "twa":
        obi_expr = twa_bid_sz - twa_ask_sz
        obi_denom_expr = twa_bid_sz + twa_ask_sz
    else:  # "twa_or_eob"
        price_moved = pl.col("delta_p").is_not_null() & (pl.col("delta_p") != 0)
        obi_expr = pl.when(price_moved).then(next_bid_sz - next_ask_sz).otherwise(twa_bid_sz - twa_ask_sz)
        obi_denom_expr = pl.when(price_moved).then(next_bid_sz + next_ask_sz).otherwise(twa_bid_sz + twa_ask_sz)

    df = df.with_columns(
        obi=obi_expr,
        _obi_denom=obi_denom_expr,
    )

    if df_fut is not None:
        df_fut_avg = (
            df_fut
            .with_columns(mid_price=pl.mean_horizontal(['bid_start', 'ask_start']))
            .group_by([security_col, date_col])
            .agg(futures_price=pl.col('mid_price').mean())
        )
        df = df.sort(date_col)
        df_fut_avg = df_fut_avg.sort(date_col)
        df = df.join_asof(
            df_fut_avg,
            on=date_col,
            by_left='near_identifier',
            by_right=security_col,
            strategy='backward',
            allow_exact_matches=False,
        )
    elif 'futures_price' not in df.columns:
        df = df.with_columns(futures_price=pl.lit(None, dtype=pl.Float64))

    df = df.with_columns(
        delta_p_bp=pl.col('delta_p') * 10000 / pl.col('futures_price')
    )

    df = df.with_columns(
        ofi=pl.col("delta_lb") - pl.col("delta_la"),
        stv=pl.col("signed_volume").fill_null(0),  # no trades in a bin -> signed volume 0
    ).with_columns(
        noi=pl.col("ofi") - pl.col("stv"),
    )

    df = df.with_columns(
        days_before=pl.business_day_count(
            pl.col("date").cast(pl.Date),
            pl.col("target_date").cast(pl.Date),
        )
    )

    if drop_lag_nulls:
        df = df.drop_nulls(subset=SIGNAL_LAG_COLS)

    return df


# Flow signals carried as z-score-able / vol-scalable quantities (the others - delta_p and
# obi - get their own scale normalisation in the transform stage).
SIGNAL_NORM_COLS = ['ofi', 'stv', 'noi']

# Time-of-roll volatility: a 1-hour centred window on 5-minute bins.
TOR_VOL_WIN = 12
TOR_VOL_MIN_SAMPLES = 6


def _apply_signal_transforms(
    df_cs: pl.DataFrame,
    df_fut: pl.DataFrame | None = None,
    obi_method: str = "end_of_bin",
) -> pl.DataFrame:
    """Microstructure signals + the *scale* transforms that are independent of any split.

    Produces (all overwriting the raw columns):
      - delta_p : in ticks (per-BBG_CODE tick size, kept as `_tick`),
      - obi     : fraction of total quoted size at the touch (denominator chosen by `obi_method`),
      - stv/noi : signed square root (variance-stabilising, sign-preserving),
      - ofi     : left RAW here (its location/scale normalisation happens downstream).

    No within-session rolling statistics and no train-restricted means are computed here, so
    this stage is identical for every normalisation strategy. ``fill_nan(None)`` converts
    floating-point 0/0 = NaN for inactive bins to null rather than propagating NaN.
    """
    df_signals = add_microstructure_signals(
        df_cs, security_col='security', time_col='bin_start_time', df_fut=df_fut, obi_method=obi_method,
    )

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
        obi=(pl.col('obi') / pl.col('_obi_denom')).fill_nan(None),
        stv=(pl.col('stv').cast(pl.Float64).abs().sqrt() * pl.col('stv').sign()).fill_nan(None),
        noi=(pl.col('noi').cast(pl.Float64).abs().sqrt() * pl.col('noi').sign()).fill_nan(None),
    ).drop('_obi_denom')


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
    df_fut: pl.DataFrame | None = None,
    obi_method: str = "end_of_bin",
) -> pl.DataFrame:
    """Microstructure signals (partitioned by SESSION [security, date]) + time-of-roll normalisation.

    Normalised values overwrite the raw columns so downstream sections use them automatically:

      1. delta_p in ticks (per-BBG_CODE tick size, kept as `_tick`),
      2. OBI as fraction of total quoted size at the touch (denominator set by `obi_method`),
      3-5. OFI / STV / NOI divided by a volatility profiled by roll stage from prior rolls only
           (see ``_normalize_time_of_roll``); requires `qcode`, `days_until`, `target_date`.
    """
    return _normalize_time_of_roll(_apply_signal_transforms(df_cs, df_fut=df_fut, obi_method=obi_method))


# ── 6. Outright-futures signals + buy/sell-leg augmentation ─────────────────────

def generate_futures_signals(df: pl.DataFrame, obi_method: str = "end_of_bin") -> pl.DataFrame:
    """Compute the normalised OBI / STV signals for the OUTRIGHT FUTURES.

    `df` may be the standalone futures frame (`df_fut`) or the combined frame (`df_combined`);
    futures are selected with ``~is_spread`` when that flag is present. `days_until` is
    recomputed from the future's own target date (required by `_normalize_time_of_roll`).

    Returns one row per surviving future bin. Callers typically keep
    ``[security, date, bin_start_time, obi, stv]`` for the leg join below.
    """
    df_fut = df.filter(~pl.col('is_spread')) if 'is_spread' in df.columns else df
    df_fut = df_fut.with_columns(
        days_until=pl.business_day_count(pl.col('date'), pl.col('target_date')),
    )
    return generate_signals(df_fut, obi_method=obi_method)


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
    signal_cols: tuple[str, ...] = ('obi', 'stv'),
    obi_method: str = "end_of_bin",
) -> pl.DataFrame:
    """Add the buy-/sell-leg outright `signal_cols` to an already-built spread-signals frame.

    One-call orchestrator over `generate_futures_signals` and `attach_outright_signals`.
    `df_signals` is the output of `generate_signals(df_cs)`; `df_combined` is the second output
    of `build_datasets`. Returns `df_signals` with `buy_<sig>` / `sell_<sig>` columns added;
    rows whose leg is outside its roll window get null leg signals.
    """
    return attach_outright_signals(
        df_signals, generate_futures_signals(df_combined, obi_method=obi_method), signal_cols=signal_cols,
    )


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

      - df_cs       : calendar spreads within the coarse roll window, with `is_roll_period`
                      flag — pass with `df_fut` to generate_signals() to compute futures_price.
      - df_combined : df_cs plus the filtered futures, diagonally concatenated (futures get
                      nulls in the spread-only columns: days_until, is_roll_period).

    Pass an existing Snowpark `session` to reuse a connection; otherwise one is created
    from `env_path`.
    """
    if session is None:
        session = create_snowpark_session(env_path)

    binned, qmap, sec_meta = load_raw_tables(session, verbose=verbose)
    data = preprocess(binned, qmap, sec_meta, roll_days=roll_days, years=years)
    df_cs, df_fut = split_spreads_futures(data)
    df_cs = apply_roll_volume_filter(df_cs, threshold=volume_threshold)

    df_combined = pl.concat([df_cs, df_fut], how='diagonal')

    if verbose:
        print(f'df_cs       : {df_cs.shape}')
        print(f'df_fut      : {df_fut.shape}')
        print(f'df_combined : {df_combined.shape}')
    return df_cs, df_combined
