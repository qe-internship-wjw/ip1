"""Utility functions for Snowflake + Polars workflows."""

import base64
import os

import numpy as np
import polars as pl
from dotenv import load_dotenv
from snowflake import connector as snowflake_connector
from snowflake.snowpark import functions as F
from snowflake.snowpark.session import Session
from snowflake.snowpark.window import Window


# ── Connection ────────────────────────────────────────────────────────────────

def _build_connection_params(env_path: str = "../.env") -> dict:
    load_dotenv(env_path)
    return {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "role": os.getenv("SNOWFLAKE_ROLE"),
        "user": os.getenv("SNOWFLAKE_USERNAME"),
        "private_key": base64.b64decode(os.getenv("SNOWFLAKE_PRIVATE_KEY")),
    }


def connect_snowflake(env_path: str = "../.env") -> snowflake_connector.SnowflakeConnection:
    return snowflake_connector.connect(**_build_connection_params(env_path))


def create_snowpark_session(env_path: str = "../.env") -> Session:
    return Session.builder.configs(_build_connection_params(env_path)).create()


# ── Snowpark helpers ──────────────────────────────────────────────────────────

def unpack_kwargs(**kwargs) -> dict:
    """Make .with_columns readable: .with_columns(**unpack_kwargs(col=expr))."""
    return {"col_names": list(kwargs.keys()), "values": list(kwargs.values())}


def unpack_kwargs_for_agg(**kwargs) -> list:
    """Make .agg readable: .agg(*unpack_kwargs_for_agg(col=expr))."""
    return [value.alias(key) for key, value in kwargs.items()]


def cumulative_sum(column, order_by, partition_by=None):
    if partition_by is None:
        spec = Window.orderBy(order_by).rowsBetween(Window.UNBOUNDED_PRECEDING, Window.CURRENT_ROW)
    else:
        spec = Window.partitionBy(partition_by).orderBy(order_by).rowsBetween(Window.UNBOUNDED_PRECEDING, Window.CURRENT_ROW)
    return F.sum(column).over(spec)


def forward_fill(column, order_by, partition_by=None):
    if partition_by is None:
        spec = Window.orderBy(order_by).rowsBetween(Window.UNBOUNDED_PRECEDING, Window.CURRENT_ROW)
    else:
        spec = Window.partitionBy(partition_by).orderBy(order_by).rowsBetween(Window.UNBOUNDED_PRECEDING, Window.CURRENT_ROW)
    return F.last_value(column, ignore_nulls=True).over(spec)


def retrieve_polars_from_snowpark(df) -> pl.DataFrame:
    return pl.from_pandas(df.to_pandas()).select(pl.all().name.to_lowercase())


def read_table(session: Session, database: str, schema: str, table: str, columns: list[str]) -> pl.DataFrame:
    """Select only `columns` from `database.schema.table` via Snowpark and return a Polars
    DataFrame with lower-cased column names.

    Pushing the column projection into Snowflake keeps the wire transfer minimal.
    """
    fqn = f"{database}.{schema}.{table}"
    snow_df = session.table(fqn).select(*columns)
    return retrieve_polars_from_snowpark(snow_df)


# ── Contract ticker parsing ─────────────────────────────────────────────────────

# Standard futures month codes
_MC = "FGHJKMNQUVXZ"

# Naming conventions (whitespace-separated yellow key suffix):
#   Future        : BBG_CODE + YY + MONTH_CODE                      e.g. "AB2024M Comdty"
#   Calendar spread: BBG_CODE + YY + MC + "/" + YY + MC             e.g. "AB2024M/2024N Comdty"
_FUTURE_RE = rf"^(?P<bbg>[^/]*?)(?P<yy>\d{{4}})(?P<mc>[{_MC}])\s(?P<yk>\S+)$"
_SPREAD_RE = rf"^(?P<bbg>[^/]*?)(?P<ny>\d{{4}})(?P<nmc>[{_MC}])/(?P<fy>\d{{4}})(?P<fmc>[{_MC}])\s(?P<yk>\S+)$"


def _build_identifier(bbg: pl.Expr, yy: pl.Expr, mc: pl.Expr, yk: pl.Expr) -> pl.Expr:
    """Reconstruct a single-future ticker string, e.g. ('ABC','2024','M','Comdty') -> 'ABC2024M Comdty'."""
    return pl.concat_str([bbg, yy, mc, pl.lit(" "), yk])


def parse_security(df: pl.DataFrame, col: str = "security") -> pl.DataFrame:
    """Parse the contract ticker in `col` and add structured future/spread columns."""
    
    # Step 1: Materialise the regex structs and is_spread flag as real columns.
    df = df.with_columns(
        is_spread=pl.col(col).str.contains("/"),
        _fut=pl.col(col).str.extract_groups(_FUTURE_RE),
        _sp=pl.col(col).str.extract_groups(_SPREAD_RE),
    )

    # Step 2: Extract fields using .struct.field() and apply fill_null() 
    # to protect the eager evaluation graph from crashing on non-matching rows.
    sp_bbg = pl.col("_sp").struct.field("bbg").fill_null("")
    sp_ny = pl.col("_sp").struct.field("ny").fill_null("2000")
    sp_nmc = pl.col("_sp").struct.field("nmc").fill_null("U")
    sp_fy = pl.col("_sp").struct.field("fy").fill_null("2000")
    sp_fmc = pl.col("_sp").struct.field("fmc").fill_null("U")
    sp_yk = pl.col("_sp").struct.field("yk").fill_null("")

    # Step 3: Derive all parsed fields safely.
    df = df.with_columns(
        near_identifier=pl.when(pl.col("is_spread"))
            .then(_build_identifier(sp_bbg, sp_ny, sp_nmc, sp_yk))
            .otherwise(pl.col(col)),

        far_identifier=pl.when(pl.col("is_spread"))
            .then(_build_identifier(sp_bbg, sp_fy, sp_fmc, sp_yk))
            .otherwise(None),
    )

    return df.drop(["_fut", "_sp"])


# ── Roll-period (business-day) math ─────────────────────────────────────────────

def add_roll_window(df: pl.DataFrame, target_col: str, roll_days: int = 10) -> pl.DataFrame:
    """Add `roll_start` / `roll_end` Date columns spanning the `roll_days` business days
    (weekends excluded) immediately *preceding* `target_col` — i.e. the target date itself
    is excluded. Rows with a null target get null windows.
    """
    return df.with_columns(
        pl.col(target_col)
        .cast(pl.Date)
        .dt.add_business_days(-roll_days, roll="backward")
        .alias("roll_start"),
        
        pl.col(target_col)
        .cast(pl.Date)
        .dt.add_business_days(-1, roll="backward")
        .alias("roll_end")
    )


def build_contract_calendar(
    security_meta: pl.DataFrame,
    qcode_mapping: pl.DataFrame,
    roll_days: int = 10,
) -> pl.DataFrame:
    """Build a per-future roll calendar from SECURITY_META.

    SECURITY_META contains only individual futures. For each future we:
      1. parse its ticker to extract bbg_code,
      2. attach DELIVERY from QCODE_MAPPING on bbg_code (unique per product),
      3. pick the target date (LAST_TRADE_DATE for 'Cash', FIRST_NOTICE_DATE for 'Phys'),
      4. compute the contract's own 10-business-day roll window, and
      5. attach the immediately-previous chronological contract's roll window (per product).

    Returns one row per future keyed by `security`, with columns:
      security, delivery, target_date,
      roll_start, roll_end, prev_roll_start, prev_roll_end.
    """
    delivery_map = qcode_mapping.select("bbg_code", "delivery").unique()

    cal = parse_security(security_meta, col="security").with_columns(
        bbg_code=pl.col("security").str.extract_groups(_FUTURE_RE).struct["bbg"],
    )

    cal = cal.join(delivery_map, on="bbg_code", how="left")

    cal = cal.with_columns(
        target_date=pl.when(pl.col("delivery") == "Cash")
        .then(pl.col("last_trade_date"))
        .otherwise(pl.col("first_notice_date"))
        .cast(pl.Date),
    )

    cal = add_roll_window(cal, "target_date", roll_days=roll_days)

    cal = cal.sort(["bbg_code", "target_date"]).with_columns(
        prev_roll_start=pl.col("roll_start").shift(1).over("bbg_code"),
        prev_roll_end=pl.col("roll_end").shift(1).over("bbg_code"),
    )

    return cal.select(
        "security", "delivery", "target_date",
        "roll_start", "roll_end", "prev_roll_start", "prev_roll_end",
    )


# ── Microstructure signals (Research Plan §3) ───────────────────────────────────

# Lag-derived columns that are null on the first bin of each security and must be dropped.
SIGNAL_LAG_COLS = ["delta_p", "delta_lb", "delta_la"]


def add_microstructure_signals(
    df: pl.DataFrame,
    security_col: str = "security",
    time_col: str = "bin_start_time",
    date_col: str = "date",
    drop_lag_nulls: bool = True,
    df_fut: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Compute the order-book / flow signals from §3 of the research plan.

    CRITICAL — strictly intraday: all lag-dependent quantities are partitioned by the trading
    session ``[security_col, date_col]`` (not just the contract) and ordered by `time_col`, so a
    lag never spans an overnight gap, weekend, or day boundary. `date_col` is derived from
    `time_col` if absent. The first bin of every (security, date) session therefore has null lags.

    Columns added (using the *_START fields as each bin's representative quote, per the spec):

      date              : calendar date of the bin (session key, created if missing)
      mid_price (P_t)   : (bid_start + ask_start) / 2
      delta_p  (ΔP_t)   : P_t - P_{t-1}   (within session)
      obi      (OBI_t)  : bid_size_start - ask_size_start
      delta_lb (ΔL_t^b) : 3-case bid liquidity change (within session)
      delta_la (ΔL_t^a) : 3-case ask liquidity change (inequalities mirrored vs. the bid:
                          an ask *improvement* is a price decrease)
      ofi      (OFI_t)  : ΔL_t^b - ΔL_t^a   (this is the Cont et al. order-flow imbalance)
      stv      (STV_t)  : signed_volume from the dataset; empty bins (null) -> 0 trades
      noi      (NOI_t)  : OFI_t - STV_t

    With `drop_lag_nulls=True` the first bin of every session (null lags) is removed.
    """
    # Session key: derive the calendar date from the bin timestamp if not already present.
    if date_col not in df.columns:
        df = df.with_columns(pl.col(time_col).cast(pl.Date).alias(date_col))

    session = [security_col, date_col]
    df = df.sort([*session, time_col])

    bid, ask = pl.col("bid_start"), pl.col("ask_start")
    bid_sz, ask_sz = pl.col("bid_size_start"), pl.col("ask_size_start")

    # Static, non-lagged signals.
    df = df.with_columns(
        mid_price=(bid + ask) / 2,
        obi=bid_sz - ask_sz,
    )

    # Previous-bin quotes WITHIN the same session — never across day/overnight boundaries.
    prev_bid = bid.shift(1).over(session)
    prev_ask = ask.shift(1).over(session)
    prev_bid_sz = bid_sz.shift(1).over(session)
    prev_ask_sz = ask_sz.shift(1).over(session)
    prev_mid = pl.col("mid_price").shift(1).over(session)

    df = df.with_columns(
        delta_p=pl.col("mid_price") - prev_mid,
        # Bid: price up -> all new liquidity; price down -> old liquidity gone; flat -> net change.
        delta_lb=pl.when(bid == prev_bid).then(bid_sz - prev_bid_sz)
        .when(bid > prev_bid).then(bid_sz)
        .otherwise(-prev_bid_sz),
        # Ask: mirrored — price down (improvement) -> all new; price up -> old gone; flat -> net.
        delta_la=pl.when(ask == prev_ask).then(ask_sz - prev_ask_sz)
        .when(ask < prev_ask).then(ask_sz)
        .otherwise(-prev_ask_sz),
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


# ── Autocorrelation diagnostics (Research Plan §5.1, Hypothesis H1) ──────────────

def acf_by_security(
    df: pl.DataFrame,
    value_col: str,
    group_cols=("security", "date"),
    time_col: str = "bin_start_time",
    nlags: int = 20,
    min_obs: int = 50,
) -> tuple:
    """Sample ACF of `value_col` within each session (default `group_cols=(security, date)`,
    ordered by `time_col`), then averaged across sessions. Grouping by the intraday session keeps
    lags from crossing overnight / weekend / day boundaries.

    Returns ``(lags, mean_acf, per_session_acf, n_series)`` where ``lags`` is 1..nlags,
    ``mean_acf`` is the cross-sectional mean ACF at those lags, and ``per_session_acf`` is the
    (n_series, nlags) matrix. Sessions shorter than `min_obs` or with zero variance are skipped.
    """
    from statsmodels.tsa.stattools import acf

    group_cols = list(group_cols)
    acfs = []
    for _key, g in df.sort([*group_cols, time_col]).group_by(group_cols, maintain_order=True):
        x = g[value_col].drop_nulls().to_numpy()
        if len(x) >= max(min_obs, nlags + 1) and np.nanstd(x) > 0:
            acfs.append(acf(x, nlags=nlags, fft=True)[1:])  # drop lag-0 (==1)

    if not acfs:
        raise ValueError(f"No session had >= {min_obs} valid observations for '{value_col}'.")

    per_session = np.vstack(acfs)
    lags = np.arange(1, nlags + 1)
    return lags, per_session.mean(axis=0), per_session, per_session.shape[0]


def ljungbox_by_security(
    df: pl.DataFrame,
    value_col: str,
    group_cols=("security", "date"),
    time_col: str = "bin_start_time",
    lag: int = 20,
    min_obs: int = 50,
) -> pl.DataFrame:
    """Ljung-Box Q-test (cumulative through `lag`) of `value_col` per intraday session
    (default `group_cols=(security, date)`), so the test never spans a day boundary.

    Returns one row per tested session: the group columns plus n_obs, lb_stat, lb_pvalue.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    group_cols = list(group_cols)
    rows = []
    for key, g in df.sort([*group_cols, time_col]).group_by(group_cols, maintain_order=True):
        key = key if isinstance(key, tuple) else (key,)
        x = g[value_col].drop_nulls().to_numpy()
        if len(x) >= max(min_obs, lag + 1) and np.nanstd(x) > 0:
            lb = acorr_ljungbox(x, lags=[lag], return_df=True)
            row = dict(zip(group_cols, key))
            row.update(
                n_obs=len(x),
                lb_stat=float(lb["lb_stat"].iloc[0]),
                lb_pvalue=float(lb["lb_pvalue"].iloc[0]),
            )
            rows.append(row)

    return pl.DataFrame(rows)