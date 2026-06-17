"""Weighted ordered logit model for tick-constrained calendar spreads.

All code specific to the model itself lives here: the WeightedOrderedModel
subclass, data-slicing helpers, fitting routines, threshold tuning, and
validation/reporting utilities.

Public API consumed by ``src.backtest`` (walk-forward) and notebook
``02b_regressions_tc.ipynb`` (fixed split):

    from src.ordered_logit import (
        split_tick_constrained, clean_delta_p_tc,     # target preparation
        fit_ordered_logit,                             # walk-forward fitter
        fit_weighted_olr,                              # fixed-split fitter
        assign_classes, tune_thresholds,               # threshold logic
        pr_threshold_df, report_all_splits,            # validation reporting
        confusion_tile,                                # plotnine heatmap
    )
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import polars as pl

# ── Constants ─────────────────────────────────────────────────────────────────

# Tick-constrained curve groups: trade on a near-discrete price grid, so delta_p is cleaned
# to the ordered set {-2, 0, +2} and modelled with the class-weighted ordered logit. Same
# list as notebook 02b.
TICK_CONSTRAINED_BBG = ['OE', 'DU', 'IK', 'RX', 'UB', 'OAT'
                        'XP', 'QZ', 'VG', 
                        # 'FV', 'TU',
                        ]

# Ordered-logit target categories and explanatory variables (OFI omitted: NOI = OFI - STV
# makes the three collinear — see Research Plan §5.2).
CATS = [-2, 0, 2]
TC_FEATURES = ['obi', 'noi', 'stv']

# Default F-beta for tail-threshold tuning. beta < 1 weights PRECISION above recall — the
# execution use-case cares more about a tail signal being right than about catching every tail.
# Override per call (e.g. beta=1.0, 2.0) for the backtest ablation studies.
DEFAULT_BETA = 0.5


# ── Target preparation ────────────────────────────────────────────────────────

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


# ── WeightedOrderedModel factory ──────────────────────────────────────────────

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


# ── Data-slicing helpers ──────────────────────────────────────────────────────

def _date_slice(
    df: pl.DataFrame,
    target_col: str,
    feature_cols: list[str],
    lo,
    hi,
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


# ── Decision rule and threshold tuning ───────────────────────────────────────

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


# ── Walk-forward fitter ───────────────────────────────────────────────────────

def fit_ordered_logit(
    df: pl.DataFrame,
    train: tuple[date, date],
    val: tuple[date, date],
    test: tuple[date, date],
    target_col: str = 'delta_p_fwd',
    feature_cols: list[str] = TC_FEATURES,
    verbose: bool = False,
    label: str | None = None,
):
    """Fit the class-weighted ordered logit on the TRAIN date range, with cluster-robust
    SE (clustered on the [security, date] session). Returns
    ``(result, {split: (y_int, probs)}, class_weight)`` with predicted class probabilities for
    train / val / test.

    `train`, `val`, `test` are ``(start_date, end_date)`` inclusive pairs. For walk-forward use
    pass ``(w.train_start, w.train_end)`` etc.; for a fixed calendar split pass
    ``(date(2021, 1, 1), date(2023, 12, 31))`` etc.

    The ±2 minority class weight is the train inverse-frequency relative to the 0 class,
    computed on THIS train slice only — deriving it from the full sample (or from val/test)
    would leak class balance into a training hyperparameter.

    With ``verbose`` the model summary and train class counts are printed; ``label`` prefixes
    the header line (defaults to ``target_col``).
    """
    WeightedOrderedModel = _make_weighted_ordered_model()

    tr = _date_slice(df, target_col, feature_cols, *train)
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

    if verbose:
        lbl = label or target_col
        print(f'=== {lbl}:  {target_col} ~ {" + ".join(feature_cols)}  '
              f'(train n={len(tr):,}, clusters={tr["cluster"].nunique():,}, '
              f'class_weight on ±2 = {class_weight:.1f}x) ===')
        print('train class counts:', dict(pd.Series(y_tr).value_counts().reindex(CATS).items()))
        print(res.summary())

    out = {}
    for name, (lo, hi) in (('train', train), ('val', val), ('test', test)):
        fr = _date_slice(df, target_col, feature_cols, lo, hi)
        y = fr[target_col].astype(int).values
        probs = np.asarray(res.predict(exog=fr[feature_cols])) if len(fr) else np.empty((0, 3))
        out[name] = (y, probs)
    return res, out, class_weight


# ── Reporting helpers ─────────────────────────────────────────────────────────

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
