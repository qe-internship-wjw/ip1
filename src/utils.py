"""Utility functions for Snowflake + Polars workflows."""

import base64
import os

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


# ── Ticker / contract helpers ─────────────────────────────────────────────────

# Standard CME/Bloomberg month codes
MONTH_CODES: dict[str, int] = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}


def parse_expiry(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add expiry_year, expiry_month, expiry_date columns from Bloomberg security tickers.

    Expected format: '{QCODE}{YEAR}{MONTH_CODE} {ASSET_CLASS}'
    Example: 'TY2024H Govt' → year=2024, month=3, expiry_date=2024-03-01
    """
    return (
        df
        .with_columns(
            base_ticker=pl.col("security").str.split(" ").list.first()
        )
        .with_columns(
            expiry_year=pl.col("base_ticker").str.slice(-5, 4).cast(pl.Int32),
            expiry_month_code=pl.col("base_ticker").str.slice(-1),
        )
        .with_columns(
            expiry_month=pl.col("expiry_month_code").map_elements(
                lambda x: MONTH_CODES.get(x, 0), return_dtype=pl.Int32
            )
        )
        .with_columns(
            expiry_date=pl.date(pl.col("expiry_year"), pl.col("expiry_month"), 1)
        )
    )


def rank_contracts(df: pl.DataFrame, group_cols: list[str] | None = None) -> pl.DataFrame:
    """
    Add contract_rank column: 1 = nearest expiry (front month), 2 = next, etc.
    Groups by (qcode, date, bin_start_time) by default.
    """
    if group_cols is None:
        group_cols = ["qcode", "date", "bin_start_time"]
    return df.with_columns(
        contract_rank=pl.col("expiry_date")
            .rank("dense", descending=False)
            .over(group_cols)
    )


def build_calendar_spread(df: pl.DataFrame, roll_days: int = 20) -> pl.DataFrame:
    """
    Join front (rank=1) and back (rank=2) contract rows into a single spread row.

    Filters to the roll period: date is within `roll_days` trading days before
    the front contract's expiry (i.e. 0 < days_to_expiry <= roll_days).
    """
    near = df.filter(pl.col("contract_rank") == 1)
    far = df.filter(pl.col("contract_rank") == 2)

    join_cols = ["qcode", "date", "bin_start_time"]
    spread = near.join(far, on=join_cols, suffix="_far", how="inner")

    spread = (
        spread
        .with_columns(
            days_to_expiry=(pl.col("expiry_date") - pl.col("date")).dt.total_days().cast(pl.Int32)
        )
        .filter(
            (pl.col("days_to_expiry") > 0) &
            (pl.col("days_to_expiry") <= roll_days)
        )
    )
    return spread


# ── Order book signal construction ───────────────────────────────────────────

def add_mid_prices(df: pl.DataFrame, suffix: str = "") -> pl.DataFrame:
    """Compute mid prices from bid/ask columns. Pass suffix='' for near, '_far' for far leg."""
    s = suffix
    return df.with_columns(
        **{
            f"mid_start{s}": (pl.col(f"bid_start{s}") + pl.col(f"ask_start{s}")) / 2,
            f"mid_end{s}": (pl.col(f"bid_end{s}") + pl.col(f"ask_end{s}")) / 2,
            f"twa_mid{s}": (pl.col(f"twa_bid{s}") + pl.col(f"twa_ask{s}")) / 2,
        }
    )


def add_ofi_signals(df: pl.DataFrame, suffix: str = "") -> pl.DataFrame:
    """
    Add three order flow imbalance signals for a given leg (near='' or far='_far').

    - ofi_vol:   signed_volume / volume  (trade-side imbalance, range [-1, 1])
    - ofi_quote: (Δbid_size - Δask_size) / avg_total_size  (quote-level OFI)
    - ofi_size:  (twa_bid_size - twa_ask_size) / (twa_bid_size + twa_ask_size)  (static imbalance)
    """
    s = suffix
    eps = 1e-10
    return df.with_columns(
        **{
            f"ofi_vol{s}": pl.col(f"signed_volume{s}") / (pl.col(f"volume{s}") + eps),
            f"ofi_quote{s}": (
                (pl.col(f"bid_size_end{s}") - pl.col(f"bid_size_start{s}"))
                - (pl.col(f"ask_size_end{s}") - pl.col(f"ask_size_start{s}"))
            ) / (
                (pl.col(f"twa_bid_size{s}") + pl.col(f"twa_ask_size{s}")) / 2 + eps
            ),
            f"ofi_size{s}": (
                pl.col(f"twa_bid_size{s}") - pl.col(f"twa_ask_size{s}")
            ) / (pl.col(f"twa_bid_size{s}") + pl.col(f"twa_ask_size{s}") + eps),
        }
    )


def add_spread_signals(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add calendar-spread-level signals (near minus far) and the spread mid-price.
    Requires both near-leg and far-leg columns to already be present.
    """
    return df.with_columns(
        cs_mid=pl.col("twa_mid") - pl.col("twa_mid_far"),
        cs_ofi_vol=pl.col("ofi_vol") - pl.col("ofi_vol_far"),
        cs_ofi_quote=pl.col("ofi_quote") - pl.col("ofi_quote_far"),
        cs_ofi_size=pl.col("ofi_size") - pl.col("ofi_size_far"),
    )


# ── Return / IC helpers ───────────────────────────────────────────────────────

def add_forward_cs_return(df: pl.DataFrame, n_periods: int = 1) -> pl.DataFrame:
    """
    Add forward calendar-spread return: Δcs_mid over the next n_periods bins,
    partitioned by (qcode, security, security_far) and ordered by (date, bin_start_time).
    """
    return df.with_columns(
        cs_mid_fwd=pl.col("cs_mid").shift(-n_periods).over(
            ["qcode", "security", "security_far"],
            order_by=["date", "bin_start_time"],
        )
    ).with_columns(
        cs_return_fwd=(pl.col("cs_mid_fwd") - pl.col("cs_mid")) / pl.col("cs_mid").abs().clip(lower_bound=1e-10)
    )


def rank_ic(signal: pl.Series, forward_return: pl.Series) -> float:
    """Spearman rank IC between a signal and its forward return."""
    import scipy.stats as stats
    mask = signal.is_not_nan() & forward_return.is_not_nan()
    s = signal.filter(mask).to_numpy()
    r = forward_return.filter(mask).to_numpy()
    if len(s) < 5:
        return float("nan")
    return stats.spearmanr(s, r).statistic
