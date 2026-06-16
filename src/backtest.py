"""Rolling-window backtesting groundwork for the predictive ordered-logit model.

The overarching goal is to backtest the 1-step-ahead ordered logit (notebook
``02b_regressions_tc.ipynb``) as an execution signal that improves on VWAP, lowering roll
costs for institutional rollers. This module lays the groundwork for that by replacing the
notebook's single fixed 2021-23 / 2024 / 2025 split with a *walk-forward* protocol and by
pre-computing the per-``qcode`` volume curves that a VWAP scheduler will consume.

Two independent pieces, both importable:

    from src.backtest import (
        generate_rolling_windows, run_rolling_backtest,   # walk-forward regression
        compute_volume_curve, historical_volume_curve,    # VWAP volume profiles
    )

Model-specific code (fitting, threshold tuning, reporting) lives in ``src.ordered_logit``:

    from src.ordered_logit import (
        fit_weighted_olr, tune_thresholds, assign_classes,   # fit + per-tail F-beta thresholds
        pr_threshold_df, report_all_splits, confusion_tile,  # validation/test reporting
    )

1. Walk-forward splits
   ``generate_rolling_windows`` emits a sequence of (train, val, test) date ranges:
     - train   : 4 years   (sliding, fixed length; first window starts 2016-01-01)
     - val     : 1 year     (threshold tuning)
     - test    : 3 months   (the *predictive period* — most tick-constrained groups roll
                             monthly or quarterly, so a quarter covers one full roll cycle)
   Each step advances every boundary by the 3-month test length, so successive test windows
   tile the timeline without overlap or gaps.

   ``run_rolling_backtest`` drives the model machinery (adapted here to take explicit date
   ranges rather than the hard-coded TRAIN/VAL/TEST_YEARS) once per window and collects
   out-of-sample metrics into one tidy Polars frame.

2. Volume curves
   ``compute_volume_curve`` builds, per ``qcode``, the average fraction of a roll's total
   volume traded in each (days-to-target, time-of-day) bucket — i.e. the VWAP execution
   curve. Each roll period is a distinct ``security``, so the curve is the cross-roll mean of
   the per-security normalised profiles. ``historical_volume_curve`` is the leakage-safe
   variant that only uses rolls completed strictly before a cut-off date (use the test
   window's start), so a backtested VWAP never peeks at its own roll.

Inputs are produced by ``src.pipeline``; pull the full history first, e.g.

    from src.pipeline import build_datasets, generate_signals
    df_cs, _ = build_datasets(env_path='../.env', years=list(range(2016, 2026)))
    df_signals = generate_signals(df_cs, normalization='time_of_roll')  # leakage-free scaling
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, asdict
from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl

from src.ordered_logit import (
    TC_FEATURES,
    DEFAULT_BETA,
    CATS,
    fit_ordered_logit,
    tune_thresholds,
    assign_classes,
    split_tick_constrained,
    clean_delta_p_tc,
)

# ── Configuration ────────────────────────────────────────────────────────────────

# Walk-forward geometry (see module docstring). Lengths in calendar months.
DATA_START = date(2016, 1, 1)
TRAIN_MONTHS = 48   # 4-year sliding train window
VAL_MONTHS = 12     # 1-year validation window (threshold tuning)
TEST_MONTHS = 3     # 3-month predictive period (one monthly/quarterly roll cycle)
STEP_MONTHS = TEST_MONTHS  # advance by the test length -> non-overlapping test tiles


# ── 1. Walk-forward window generation ─────────────────────────────────────────────

@dataclass(frozen=True)
class RollingWindow:
    """One walk-forward split. All bounds are INCLUSIVE calendar dates, matching the
    ``is_between(..., closed='both')`` filters used elsewhere in the pipeline."""
    index: int
    train_start: date
    train_end: date
    val_start: date
    val_end: date
    test_start: date
    test_end: date

    def label(self) -> str:
        return f"W{self.index:02d} test {self.test_start:%Y-%m}..{self.test_end:%Y-%m}"


def _add_months(d: date, months: int) -> date:
    """Shift `d` by `months` calendar months, clamping the day to the target month length."""
    m0 = d.month - 1 + months
    year = d.year + m0 // 12
    month = m0 % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def generate_rolling_windows(
    data_end: date,
    data_start: date = DATA_START,
    train_months: int = TRAIN_MONTHS,
    val_months: int = VAL_MONTHS,
    test_months: int = TEST_MONTHS,
    step_months: int = STEP_MONTHS,
) -> list[RollingWindow]:
    """Build the sequence of walk-forward (train, val, test) windows.

    The first window's train starts at `data_start`; every subsequent window slides all
    boundaries forward by `step_months`. A window is emitted only if its *full* test period
    fits within `data_end` (inclusive), so we never report a metric on a truncated quarter.

    Boundaries are computed on month edges and stored as inclusive [start, end] dates:
      train : [t,                       val_start  - 1 day]   (`train_months` long)
      val   : [t + train_months,        test_start - 1 day]   (`val_months`   long)
      test  : [t + train+val_months,    test_end          ]   (`test_months`  long)
    """
    windows: list[RollingWindow] = []
    one_day = timedelta(days=1)
    i = 0
    train_start = data_start
    while True:
        val_start = _add_months(train_start, train_months)
        test_start = _add_months(val_start, val_months)
        test_end_excl = _add_months(test_start, test_months)
        test_end = test_end_excl - one_day

        if test_end > data_end:
            break

        windows.append(RollingWindow(
            index=i,
            train_start=train_start,
            train_end=val_start - one_day,
            val_start=val_start,
            val_end=test_start - one_day,
            test_start=test_start,
            test_end=test_end,
        ))
        i += 1
        train_start = _add_months(train_start, step_months)

    return windows


def windows_to_frame(windows: list[RollingWindow]) -> pl.DataFrame:
    """Tidy overview of the generated windows (one row each), for quick inspection/plotting."""
    return pl.DataFrame([asdict(w) for w in windows])


# ── 2. Volume curves for VWAP ──────────────────────────────────────────────────────

def compute_volume_curve(
    df_cs: pl.DataFrame,
    group_col: str = 'qcode',
    position_cols: tuple[str, ...] = ('days_until', 'bin_start_time'),
    volume_col: str = 'volume',
) -> pl.DataFrame:
    """Per-``qcode`` VWAP volume curve over the roll period.

    Each roll period is a distinct ``security`` (a calendar spread in one roll window). For
    every security we normalise its bin volumes by that security's total roll volume, then
    average those fractions across all securities of the same ``qcode`` at each position
    (days-to-target, time-of-day). The result is renormalised to sum to 1 per group, giving
    the expected fraction of a roll's volume to execute in each bucket — the schedule a VWAP
    slicer follows.

    Averaging *fractions* (not raw volumes) weights every roll equally, so a single
    unusually large roll cannot dominate the curve.

    Returns one row per (group_col, *position_cols) with:
      volume_fraction : VWAP weight (sums to 1 within each group),
      n_securities    : number of distinct rolls contributing to the bucket,
      mean_frac       : pre-renormalisation cross-roll mean fraction,
      total_vol       : raw volume summed across contributing rolls (diagnostic).
    """
    pos = list(position_cols)

    sec_total = df_cs.group_by('security').agg(_sec_total=pl.col(volume_col).sum())

    per_bucket = (
        df_cs.group_by([group_col, 'security', *pos])
        .agg(_vol=pl.col(volume_col).sum())
        .join(sec_total, on='security', how='left')
        # Skip rolls with zero total volume (fraction undefined).
        .filter(pl.col('_sec_total') > 0)
        .with_columns(_frac=pl.col('_vol') / pl.col('_sec_total'))
    )

    curve = (
        per_bucket.group_by([group_col, *pos])
        .agg(
            mean_frac=pl.col('_frac').mean(),
            n_securities=pl.col('security').n_unique(),
            total_vol=pl.col('_vol').sum(),
        )
        .with_columns(
            volume_fraction=pl.col('mean_frac') / pl.col('mean_frac').sum().over(group_col)
        )
    )
    return curve.sort([group_col, *pos])


def historical_volume_curve(
    df_cs: pl.DataFrame,
    before: date,
    group_col: str = 'qcode',
    position_cols: tuple[str, ...] = ('days_until', 'bin_start_time'),
    volume_col: str = 'volume',
    target_col: str = 'target_date',
) -> pl.DataFrame:
    """Leakage-safe ``compute_volume_curve``: only rolls that *completed* before `before`.

    A roll's reference date is its ``target_date`` (last-trade / first-notice). Restricting to
    ``target_date < before`` means a VWAP backtested over a test window only ever uses curves
    estimated from rolls that finished before that window began. Pass the test window's
    ``test_start`` as `before`.
    """
    past = df_cs.filter(pl.col(target_col).cast(pl.Date) < before)
    return compute_volume_curve(past, group_col, position_cols, volume_col)


# ── 3. Walk-forward driver ──────────────────────────────────────────────────────────

def run_rolling_backtest(
    df_signals_tc_clean: pl.DataFrame,
    windows: list[RollingWindow],
    target_col: str = 'delta_p_fwd',
    feature_cols: list[str] = TC_FEATURES,
    beta: float = DEFAULT_BETA,
    min_train_rows: int = 500,
    min_test_rows: int = 50,
    verbose: bool = True,
) -> pl.DataFrame:
    """Run the predictive ordered logit walk-forward across `windows` and collect OOS metrics.

    For each window: fit on train, tune the -2 and +2 thresholds INDEPENDENTLY on val (each
    maximising its tail's F-beta — see ``tune_thresholds`` / ``beta``), then score test at those
    thresholds. Windows whose train/test slices are too small (early history may lack
    tick-constrained data) are skipped with a logged reason rather than erroring, so the whole
    walk-forward never aborts on a thin window.

    Returns one row per attempted window: the window bounds, fitted class weight, the two tuned
    tail thresholds, per-tail validation F-beta, and test accuracy / macro-F-beta / per-class
    support, plus a `status` column ('ok' or the skip reason). `beta` is recorded so ablation
    sweeps over beta stay self-describing.
    """
    from sklearn.metrics import fbeta_score

    rows = []
    for w in windows:
        rec: dict = {**asdict(w), 'label': w.label()}
        try:
            res, data, class_weight = fit_ordered_logit(
                df_signals_tc_clean,
                train=(w.train_start, w.train_end),
                val=(w.val_start, w.val_end),
                test=(w.test_start, w.test_end),
                target_col=target_col,
                feature_cols=feature_cols,
            )
        except Exception as exc:  # singular fit, empty slice, optimiser failure, ...
            rec['status'] = f'fit_error: {type(exc).__name__}: {exc}'
            rows.append(rec)
            if verbose:
                print(f'{w.label()}: SKIP ({rec["status"]})')
            continue

        n_train = len(data['train'][0])
        n_test = len(data['test'][0])
        if n_train < min_train_rows or n_test < min_test_rows:
            rec['status'] = f'thin (train={n_train}, test={n_test})'
            rec['n_train'], rec['n_test'] = n_train, n_test
            rows.append(rec)
            if verbose:
                print(f'{w.label()}: SKIP ({rec["status"]})')
            continue

        y_val, probs_val = data['val']
        thr_down, thr_up, val_fb_down, val_fb_up = tune_thresholds(y_val, probs_val, beta=beta)

        y_test, probs_test = data['test']
        pred_test = assign_classes(probs_test, thr_down, thr_up)
        test_counts = pd.Series(y_test).value_counts().reindex(CATS).fillna(0).astype(int)

        rec.update(
            status='ok',
            class_weight=float(class_weight),
            beta=float(beta),
            thr_down=float(thr_down),
            thr_up=float(thr_up),
            val_fbeta_down=float(val_fb_down),
            val_fbeta_up=float(val_fb_up),
            n_train=int(n_train),
            n_val=int(len(y_val)),
            n_test=int(n_test),
            test_accuracy=float((pred_test == y_test).mean()),
            test_macro_fbeta=float(fbeta_score(y_test, pred_test, beta=beta, labels=CATS,
                                               average='macro', zero_division=0)),
            test_n_down=int(test_counts[-2]),
            test_n_flat=int(test_counts[0]),
            test_n_up=int(test_counts[2]),
        )
        rows.append(rec)
        if verbose:
            print(f'{w.label()}: thr=(-2:{thr_down:.3f}, +2:{thr_up:.3f})  '
                  f'val_Fb=(-2:{val_fb_down:.4f}, +2:{val_fb_up:.4f})  '
                  f'test_Fb={rec["test_macro_fbeta"]:.4f}  acc={rec["test_accuracy"]:.4f}  '
                  f'(n_train={n_train:,}, n_test={n_test:,})')

    return pl.DataFrame(rows)


# ── 4. Per-bin OOS predictions (for the VWAP overlay) ───────────────────────────────

def predict_target_bins(
    df_signals_tc_clean: pl.DataFrame,
    windows: list[RollingWindow],
    target_col: str = 'delta_p_fwd',
    feature_cols: list[str] = TC_FEATURES,
    beta: float = DEFAULT_BETA,
    min_train_rows: int = 500,
    min_val_rows: int = 50,
    verbose: bool = True,
) -> pl.DataFrame | None:
    """Walk-forward OOS predictions of the next bin's move, aligned to each TARGET bin.

    For each window: fit on train, tune the threshold on val, then predict ``delta_p_fwd`` for
    every test bin with non-null features. The prediction made at bin t is about bin t+1, so it
    is shifted one step forward WITHIN each [security, date] session to land on the target bin it
    describes (``pred``). This is exactly the information an executor has at the *start* of the
    target bin (the previous bin's features are already known), so there is no look-ahead.

    For each window the -2 and +2 thresholds are tuned independently on val by F-beta (``beta``).
    Returns one row per OOS target bin: ``security, date, bin_start_time, pred, pred_made,
    window, thr_down, thr_up`` (``pred`` is null on each session's first bin). None if no window
    produced predictions. Consumed by ``src.vwap.run_improved_vwap_backtest``.
    """
    out = []
    for w in windows:
        try:
            res, data, _ = fit_ordered_logit(
                df_signals_tc_clean,
                train=(w.train_start, w.train_end),
                val=(w.val_start, w.val_end),
                test=(w.test_start, w.test_end),
                target_col=target_col,
                feature_cols=feature_cols,
            )
        except Exception as exc:
            if verbose:
                print(f'{w.label()}: SKIP ({type(exc).__name__}: {exc})')
            continue

        if len(data['train'][0]) < min_train_rows or len(data['val'][0]) < min_val_rows:
            if verbose:
                print(f'{w.label()}: SKIP (thin train/val)')
            continue

        thr_down, thr_up, *_ = tune_thresholds(*data['val'], beta=beta)

        # Predict on ALL test bins with present features (do not require a non-null forward
        # target, so the last bin of each session is still scored), keeping the row keys.
        # 1. Keep the full continuous timeline intact first
        test_full = (
            df_signals_tc_clean
            .filter(pl.col('date').is_between(pl.lit(w.test_start), pl.lit(w.test_end), closed='both'))
            .select(['security', 'date', 'bin_start_time', *feature_cols])
            .sort(['security', 'date', 'bin_start_time'])
        )

        # 2. Isolate rows that have valid features to generate predictions safely
        test_valid = test_full.drop_nulls(feature_cols)
        if test_valid.height == 0:
            continue

        probs = np.asarray(res.predict(exog=test_valid.select(feature_cols).to_pandas()))
        pred_made = assign_classes(probs, thr_down, thr_up)
        test_valid = test_valid.with_columns(pred_made=pl.Series('pred_made', pred_made))

        # 3. Join the predictions back to the continuous timeline, THEN shift chronologically
        test_shifted = (
            test_full.join(
                test_valid.select(['security', 'date', 'bin_start_time', 'pred_made']),
                on=['security', 'date', 'bin_start_time'],
                how='left'
            )
            .with_columns(
                pred=pl.col('pred_made').shift(1).over(['security', 'date']),
                window=pl.lit(w.index),
                thr_down=pl.lit(float(thr_down)),
                thr_up=pl.lit(float(thr_up)),
            )
            # Drop rows where no prediction can be applied to the execution engine
            .drop_nulls('pred')
        )

        out.append(test_shifted.select('security', 'date', 'bin_start_time', 'pred', 'pred_made',
                                       'window', 'thr_down', 'thr_up'))
        if verbose:
            print(f'{w.label()}: thr=(-2:{thr_down:.3f}, +2:{thr_up:.3f})  test bins={test_shifted.height:,}')

    return pl.concat(out) if out else None


# ── Top-level entry point ───────────────────────────────────────────────────────────

def build_backtest_inputs(
    env_path: str = '../.env',
    data_start_year: int = 2016,
    data_end_year: int = 2025,
    session=None,
):
    """Convenience wiring: pull the full history, generate signals, clean the tick-constrained
    target, and build both the walk-forward windows and per-``qcode`` volume curves.

    Returns ``(df_signals_tc_clean, volume_curves, windows)``. Requires a live Snowflake
    connection; the regression/volume utilities above are usable standalone on any cached
    ``df_cs`` / ``df_signals`` without calling this.
    """
    from src.pipeline import build_datasets, generate_signals

    years = list(range(data_start_year, data_end_year + 1))
    df_cs, _ = build_datasets(env_path=env_path, session=session, years=years)
    # 'time_of_roll' normalisation profiles each signal's volatility from PRIOR rolls only, so
    # it needs no `train_years` and leaks nothing across the walk-forward splits.
    df_signals = generate_signals(df_cs, normalization='time_of_roll')

    _, df_signals_tc = split_tick_constrained(df_signals)
    df_signals_tc_clean = clean_delta_p_tc(df_signals_tc)

    volume_curves = compute_volume_curve(df_cs)

    data_end = df_cs['date'].max()
    windows = generate_rolling_windows(data_end=data_end, data_start=date(data_start_year, 1, 1))

    return df_signals_tc_clean, volume_curves, windows


if __name__ == '__main__':
    # Smoke test the pure (no-Snowflake) pieces: window geometry is deterministic.
    wins = generate_rolling_windows(data_end=date(2025, 12, 31))
    print(f'{len(wins)} walk-forward windows from {DATA_START}:')
    print(windows_to_frame(wins))
