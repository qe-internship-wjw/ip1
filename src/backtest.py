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

It also hosts the fixed-split model/threshold/reporting helpers shared with notebook
``02b_regressions_tc.ipynb`` (kept here, not duplicated in the notebook, to avoid drift):

    from src.backtest import (
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

   ``run_rolling_backtest`` drives the notebook's regression machinery (adapted here to take
   explicit date ranges rather than the hard-coded TRAIN/VAL/TEST_YEARS) once per window and
   collects out-of-sample metrics into one tidy Polars frame.

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

# ── Configuration ────────────────────────────────────────────────────────────────

# Walk-forward geometry (see module docstring). Lengths in calendar months.
DATA_START = date(2016, 1, 1)
TRAIN_MONTHS = 48   # 4-year sliding train window
VAL_MONTHS = 12     # 1-year validation window (threshold tuning)
TEST_MONTHS = 3     # 3-month predictive period (one monthly/quarterly roll cycle)
STEP_MONTHS = TEST_MONTHS  # advance by the test length -> non-overlapping test tiles

# Tick-constrained curve groups: trade on a near-discrete price grid, so delta_p is cleaned
# to the ordered set {-2, 0, +2} and modelled with the class-weighted ordered logit. Same
# list as notebook 02b.
TICK_CONSTRAINED_BBG = ['OE', 'XP', 'DU', 'IK', 'RX', 'QZ', 'FV', 'TU', 'VG', 'UB', 'OAT']

# Ordered-logit target categories and explanatory variables (OFI omitted: NOI = OFI - STV
# makes the three collinear — see Research Plan §5.2).
CATS = [-2, 0, 2]
TC_FEATURES = ['obi', 'noi', 'stv']

# Default F-beta for tail-threshold tuning. beta < 1 weights PRECISION above recall — the
# execution use-case cares more about a tail signal being right than about catching every tail.
# Override per call (e.g. beta=1.0, 2.0) for the backtest ablation studies.
DEFAULT_BETA = 0.5


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


# ── 2. Tick-constrained target preparation ────────────────────────────────────────

def split_tick_constrained(
    df_signals: pl.DataFrame,
    tick_bbg: list[str] = TICK_CONSTRAINED_BBG,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Partition signalled spreads into (tick-unconstrained, tick-constrained) by BBG_CODE."""
    is_tc = pl.col('bbg_code').is_in(tick_bbg)
    return df_signals.filter(~is_tc), df_signals.filter(is_tc)


def clean_delta_p_tc(df_signals_tc: pl.DataFrame, cats: list[int] = CATS) -> pl.DataFrame:
    """Clean tick-constrained ``delta_p`` to the ordered set {-2, 0, +2} and add the
    1-step-ahead target ``delta_p_fwd``.

    This is the cleaning step from notebook 02b, extracted verbatim. It is applied over the
    *full* sample within each ``[security, date]`` session — the rules below are local
    (depend only on the previous/next bin in the same session), so this is split-independent
    and introduces no train/test leakage. The forward target is built here, before any
    chronological split, so ``.shift(-1)`` never crosses a session or window boundary
    (last bin of each session -> null target).

    Neighbour-context rules (applied simultaneously to the original tick values):
      1. 0, ±2: keep as-is.
      2. ±1:  a. prev was ±1 -> 0 (consumed second of a prior pair)
              b. next is ±1, same sign -> ±2 (full-tick move)
              c. next is ±1, opposite sign -> 0 (immediate reversal / noise)
              d. isolated ±1 -> promote to ±2
      3. |delta_p| > 2 -> clip to sign * 2.
      4. Keep only {-2, 0, +2}.
    """
    cleaned = (
        df_signals_tc
        .sort(['security', 'date', 'bin_start_time'])
        .with_columns(delta_p=pl.col('delta_p').cast(pl.Int32))
        .with_columns(
            _prev_dp=pl.col('delta_p').shift(1).over(['security', 'date']),
            _next_dp=pl.col('delta_p').shift(-1).over(['security', 'date']),
        )
        .with_columns(
            delta_p=(
                pl.when(pl.col('delta_p').abs() > 2)
                .then(pl.col('delta_p').sign() * 2)                       # rule 3
                .when(pl.col('delta_p').abs() == 1)
                .then(
                    pl.when(pl.col('_prev_dp').abs() == 1)
                    .then(pl.lit(0))                                      # 2a
                    .when(pl.col('_next_dp').abs() == 1)
                    .then(
                        pl.when(
                            ((pl.col('_next_dp') > 0) & (pl.col('delta_p') > 0)) |
                            ((pl.col('_next_dp') < 0) & (pl.col('delta_p') < 0))
                        )
                        .then(pl.col('delta_p').sign() * 2)               # 2b
                        .otherwise(pl.lit(0))                             # 2c
                    )
                    .otherwise(pl.col('delta_p') * 2)                     # 2d
                )
                .otherwise(pl.col('delta_p'))                             # rule 1
            )
        )
        .drop(['_prev_dp', '_next_dp'])
        .with_columns(
            delta_p_fwd=pl.col('delta_p').shift(-1).over(['security', 'date']),
        )
    )
    # Keep only the valid categories on the contemporaneous target (rule 4); the forward
    # target inherits cleaned values and may be null on the last bin of each session.
    return cleaned.filter(pl.col('delta_p').is_in(cats))


# ── 3. Volume curves for VWAP ──────────────────────────────────────────────────────

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


# ── 4. Class-weighted ordered logit (date-range adapted from notebook 02b) ──────────

# Imported lazily inside the fitter so that the window/volume utilities above stay importable
# in environments without statsmodels (e.g. a lightweight VWAP-only context).

def _make_weighted_ordered_model():
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    class WeightedOrderedModel(OrderedModel):
        """OrderedModel with per-observation class weights. Overriding ``loglikeobs``
        propagates the weights into both the optimisation objective and the ``score_obs``
        used by the robust sandwich covariance."""

        def __init__(self, endog, exog, obs_weights=None, **kwargs):
            super().__init__(endog, exog, **kwargs)
            n = len(np.asarray(endog))
            self._obs_weights = (
                np.ones(n) if obs_weights is None else np.asarray(obs_weights, dtype=float)
            )

        def loglikeobs(self, params):
            return super().loglikeobs(params) * self._obs_weights

    return WeightedOrderedModel


def _date_slice(
    df: pl.DataFrame,
    target_col: str,
    feature_cols: list[str],
    lo: date,
    hi: date,
) -> pd.DataFrame:
    """Materialise the rows with ``lo <= date <= hi`` to pandas ONCE, carrying the target, the
    features and a ``cluster`` key = security|date (the intraday session). Row order
    [security, date, bin_start_time] is preserved for the cluster-robust covariance."""
    pdf = (
        df
        .filter(pl.col('date').is_between(pl.lit(lo), pl.lit(hi), closed='both'))
        .select([target_col, *feature_cols, 'security', 'date'])
        .drop_nulls()
        .to_pandas()
    )
    pdf['cluster'] = pdf['security'].astype(str) + '|' + pdf['date'].astype(str)
    return pdf


def assign_classes(probs: np.ndarray, thr_down: float, thr_up: float) -> np.ndarray:
    """Tail decision rule with INDEPENDENT per-tail thresholds: predict +2 where P(+2) >= `thr_up`
    and -2 where P(-2) >= `thr_down`, else 0. When both tails fire on the same row the more
    probable tail wins (so the two thresholds need not be jointly calibrated). ``probs`` columns
    are ordered as CATS = [-2, 0, +2]."""
    p_down, p_up = probs[:, 0], probs[:, 2]
    up_fires, down_fires = p_up >= thr_up, p_down >= thr_down
    take_up = up_fires & (~down_fires | (p_up >= p_down))
    take_down = down_fires & (~up_fires | (p_down > p_up))
    return np.where(take_up, 2, np.where(take_down, -2, 0))


def tune_thresholds(
    y_val: np.ndarray,
    probs_val: np.ndarray,
    beta: float = DEFAULT_BETA,
    grid: np.ndarray | None = None,
):
    """Choose the -2 and +2 decision thresholds INDEPENDENTLY on the VALIDATION set, each
    maximising that tail's one-vs-rest F-beta. ``beta`` trades precision (beta<1) against recall
    (beta>1) — see ``DEFAULT_BETA`` — and is the knob varied across the backtest ablations.

    Tuning each tail one-vs-rest (P(tail) >= thr) decouples the two thresholds; the final
    multiclass assignment in ``assign_classes`` then resolves the rare both-fire row by
    probability. Returns ``(thr_down, thr_up, fbeta_down, fbeta_up)``.
    """
    from sklearn.metrics import fbeta_score

    if grid is None:
        grid = np.round(np.linspace(0.10, 0.95, 171), 4)

    def _best_tail(col: int, cls: int):
        y_bin = (y_val == cls).astype(int)
        best_thr, best_f = float(grid[0]), -1.0
        for thr in grid:
            f = fbeta_score(y_bin, (probs_val[:, col] >= thr).astype(int),
                            beta=beta, zero_division=0)
            if f > best_f:
                best_thr, best_f = float(thr), float(f)
        return best_thr, best_f

    thr_down, fbeta_down = _best_tail(0, -2)
    thr_up, fbeta_up = _best_tail(2, 2)
    return thr_down, thr_up, fbeta_down, fbeta_up


def fit_ordered_logit(
    df: pl.DataFrame,
    window: RollingWindow,
    target_col: str = 'delta_p_fwd',
    feature_cols: list[str] = TC_FEATURES,
):
    """Fit the class-weighted ordered logit on the window's TRAIN range, with cluster-robust
    SE (clustered on the [security, date] session). Returns
    ``(result, {split: (y_int, probs)}, class_weight)`` with predicted class probabilities for
    train / val / test.

    The ±2 minority class weight is the train inverse-frequency relative to the 0 class,
    computed on THIS window's train slice only — deriving it from the full sample (or from
    val/test) would leak class balance into a training hyperparameter. The feature
    normalisation is likewise leakage-free: ``generate_signals(normalization='time_of_roll')``
    scales each signal by a volatility profiled from strictly prior rolls, so signals can be
    generated once over the full history and sliced per window without contamination.
    """
    WeightedOrderedModel = _make_weighted_ordered_model()

    tr = _date_slice(df, target_col, feature_cols, window.train_start, window.train_end)
    y_tr = tr[target_col].astype(int).values

    counts = pd.Series(y_tr).value_counts()
    n0 = counts.get(0, 0)
    n_pos = counts.get(2, 0)
    class_weight = (n0 / n_pos) if n_pos else 1.0
    weights = np.where(y_tr == 0, 1.0, class_weight)

    endog = pd.Categorical(y_tr, categories=CATS, ordered=True)
    res = WeightedOrderedModel(endog, tr[feature_cols], obs_weights=weights, distr='logit').fit(
        method='bfgs', cov_type='cluster', cov_kwds={'groups': tr['cluster'].values}, disp=False,
    )

    splits = {
        'train': (window.train_start, window.train_end),
        'val': (window.val_start, window.val_end),
        'test': (window.test_start, window.test_end),
    }
    out = {}
    for name, (lo, hi) in splits.items():
        fr = _date_slice(df, target_col, feature_cols, lo, hi)
        y = fr[target_col].astype(int).values
        probs = np.asarray(res.predict(exog=fr[feature_cols])) if len(fr) else np.empty((0, 3))
        out[name] = (y, probs)
    return res, out, class_weight


# ── 4b. Fixed-split helpers (used by notebook 02b_regressions_tc) ───────────────────
#
# The notebook reports the single fixed 2021-23 / 2024 / 2025 split (rather than the
# walk-forward windows above). These helpers live here — not duplicated in the notebook — so
# the model definition, threshold tuning and reporting cannot drift between the two. They share
# ``assign_classes`` / ``tune_thresholds`` with the walk-forward driver.


def _year_slice(
    df: pl.DataFrame,
    target_col: str,
    feature_cols: list[str],
    years: list[int],
) -> pd.DataFrame:
    """Materialise the rows whose trading-date year is in `years` to pandas, carrying the target,
    the features and a ``cluster`` key = security|date. Mirrors ``_date_slice`` but selects by
    calendar year (the notebook's chronological split)."""
    pdf = (
        df
        .filter(pl.col('date').dt.year().is_in(years))
        .select([target_col, *feature_cols, 'security', 'date'])
        .drop_nulls()
        .to_pandas()
    )
    pdf['cluster'] = pdf['security'].astype(str) + '|' + pdf['date'].astype(str)
    return pdf


def fit_weighted_olr(
    df: pl.DataFrame,
    target_col: str,
    feature_cols: list[str],
    train_years: list[int],
    val_years: list[int],
    test_years: list[int],
    label: str | None = None,
    verbose: bool = True,
):
    """Fit the class-weighted ordered logit on TRAIN `train_years` with cluster-robust SE
    (clustered on the [security, date] session), then predict class probabilities for each split.

    Returns ``(result, {split: (y_int, probs)})``. The ±2 minority class weight is the train
    inverse-frequency relative to the 0 class, computed on the train years ONLY (deriving it from
    val/test would leak class balance into a training hyperparameter). With ``verbose`` the model
    summary and train class counts are printed.
    """
    WeightedOrderedModel = _make_weighted_ordered_model()

    tr = _year_slice(df, target_col, feature_cols, train_years)
    y_tr = tr[target_col].astype(int).values

    counts = pd.Series(y_tr).value_counts()
    n0, n_pos = counts.get(0, 0), counts.get(2, 0)
    class_weight = (n0 / n_pos) if n_pos else 1.0
    weights = np.where(y_tr == 0, 1.0, class_weight)

    endog = pd.Categorical(y_tr, categories=CATS, ordered=True)
    res = WeightedOrderedModel(endog, tr[feature_cols], obs_weights=weights, distr='logit').fit(
        method='bfgs', cov_type='cluster', cov_kwds={'groups': tr['cluster'].values}, disp=False,
    )

    if verbose:
        lbl = label or target_col
        print(f'=== {lbl}:  {target_col} ~ {" + ".join(feature_cols)}  '
              f'(train n={len(tr):,}, clusters={tr["cluster"].nunique():,}, '
              f'class_weight on ±2 = {class_weight:.1f}x) ===')
        print('train class counts:', dict(pd.Series(y_tr).value_counts().reindex(CATS).items()))
        print(res.summary())

    out = {}
    for name, yrs in (('train', train_years), ('val', val_years), ('test', test_years)):
        fr = _year_slice(df, target_col, feature_cols, yrs)
        y = fr[target_col].astype(int).values
        probs = np.asarray(res.predict(exog=fr[feature_cols])) if len(fr) else np.empty((0, 3))
        out[name] = (y, probs)
    return res, out


def pr_threshold_df(y_val: np.ndarray, probs_val: np.ndarray) -> pd.DataFrame:
    """Tidy precision/recall-vs-threshold for the two tail classes (long form), for plotting the
    validation threshold sweep that ``tune_thresholds`` optimises over."""
    from sklearn.metrics import precision_recall_curve

    frames = []
    for cls, col in ((-2, 0), (2, 2)):
        prec, rec, thr = precision_recall_curve((y_val == cls).astype(int), probs_val[:, col])
        frames.append(pd.DataFrame({
            'Threshold': thr, 'Precision': prec[:-1], 'Recall': rec[:-1], 'Class': f'Class {cls:+d}',
        }))
    return pd.concat(frames, ignore_index=True).melt(
        id_vars=['Threshold', 'Class'], value_vars=['Precision', 'Recall'],
        var_name='Metric', value_name='Value',
    )


def report_all_splits(
    data: dict,
    thr_down: float,
    thr_up: float,
    label: str,
    split_labels: dict | None = None,
):
    """Print per-split classification reports (precision / recall / F1 / support) at the tuned
    ``(thr_down, thr_up)`` tail thresholds. ``split_labels`` maps split name -> display string
    (e.g. {'train': '2021-2023', ...}); the raw name is used if omitted."""
    from sklearn.metrics import classification_report

    for name in ('train', 'val', 'test'):
        y, probs = data[name]
        pred = assign_classes(probs, thr_down, thr_up)
        disp = (split_labels or {}).get(name, name)
        print(f'\n--- {label} | {name} ({disp}) | thr=(-2:{thr_down:.3f}, +2:{thr_up:.3f}) | '
              f'accuracy={(pred == y).mean():.4f} | n={len(y):,} ---')
        print(classification_report(y, pred, labels=CATS,
                                    target_names=['-2', '0', '+2'], zero_division=0))


def confusion_tile(y_true: np.ndarray, y_pred: np.ndarray, title: str):
    """plotnine confusion-matrix heatmap with raw counts and row percentages."""
    from plotnine import (
        ggplot, aes, geom_tile, geom_text, scale_fill_gradient, labs, theme_bw, theme,
    )
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=CATS)
    cm_long = (
        pd.DataFrame(cm, index=CATS, columns=CATS)
        .rename_axis('Actual').reset_index()
        .melt(id_vars='Actual', var_name='Predicted', value_name='n')
        .assign(
            row_pct=lambda d: 100 * d['n'] / d.groupby('Actual')['n'].transform('sum'),
            label=lambda d: d.apply(lambda r: f"{int(r['n'])}\n({r['row_pct']:.1f}%)", axis=1),
        )
    )
    for col, order in [('Predicted', ['-2', '0', '2']), ('Actual', ['2', '0', '-2'])]:
        cm_long[col] = pd.Categorical(cm_long[col].astype(str), categories=order)
    return (
        ggplot(cm_long, aes(x='Predicted', y='Actual', fill='row_pct'))
        + geom_tile(color='white', size=0.5)
        + geom_text(aes(label='label'), size=9)
        + scale_fill_gradient(low='#f0f4ff', high='#2166ac', name='Row %')
        + labs(title=title, x='Predicted ΔP', y='Actual ΔP')
        + theme_bw(base_size=11) + theme(figure_size=(5, 4))
    )


# ── 5. Walk-forward driver ──────────────────────────────────────────────────────────

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
            res, data, class_weight = fit_ordered_logit(df_signals_tc_clean, w, target_col, feature_cols)
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


# ── 6. Per-bin OOS predictions (for the VWAP overlay) ───────────────────────────────

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
    import numpy as np

    out = []
    for w in windows:
        try:
            res, data, _ = fit_ordered_logit(df_signals_tc_clean, w, target_col, feature_cols)
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
        test = (
            df_signals_tc_clean
            .filter(pl.col('date').is_between(pl.lit(w.test_start), pl.lit(w.test_end), closed='both'))
            .select([*feature_cols, 'security', 'date', 'bin_start_time'])
            .drop_nulls(feature_cols)
            .sort(['security', 'date', 'bin_start_time'])
        )
        if test.height == 0:
            continue

        probs = np.asarray(res.predict(exog=test.select(feature_cols).to_pandas()))
        pred_made = assign_classes(probs, thr_down, thr_up)
        test = test.with_columns(pred_made=pl.Series('pred_made', pred_made)).with_columns(
            pred=pl.col('pred_made').shift(1).over(['security', 'date']),
            window=pl.lit(w.index),
            thr_down=pl.lit(float(thr_down)),
            thr_up=pl.lit(float(thr_up)),
        )
        out.append(test.select('security', 'date', 'bin_start_time', 'pred', 'pred_made',
                               'window', 'thr_down', 'thr_up'))
        if verbose:
            print(f'{w.label()}: thr=(-2:{thr_down:.3f}, +2:{thr_up:.3f})  test bins={test.height:,}')

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
