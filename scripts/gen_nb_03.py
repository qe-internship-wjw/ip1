"""Generate notebooks/03_vwap_backtest.ipynb (demonstration + plotnine visualisations)."""
import json
import os

cells = []
def md(src): cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
def code(src): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src})

md("""# Improving VWAP Roll Execution with a Predictive Ordered Logit

This notebook exposits and demonstrates the **rolling-window backtest** built in `src/backtest.py`
and `src/vwap.py`. The institutional roller's problem is to migrate a fixed quantity from the near
to the far contract over a roll period as cheaply as possible. The benchmark is **VWAP**; the goal is
to beat it using the predictive ordered-logit signal.

The pipeline, end to end:

1. **Data** - pull 2016-2025 spread bins and generate microstructure signals with the leakage-free
   *time-of-roll* volatility normalisation.
2. **Walk-forward windows** - 4y train / 1y validation / 3m test, sliding by one quarter.
3. **Volume curves** - the per-`qcode` VWAP execution schedule, profiled from prior rolls.
4. **Predictive ordered logit** - walk-forward out-of-sample classification of the next bin's move.
5. **Baseline VWAP** - passive-at-touch, cross the remainder; measure the spread captured.
6. **Ordered-logit overlay** - rest deeper when the model predicts a favourable move, cross
   immediately when it predicts an adverse one; compare head-to-head with the baseline.

All inference is strictly out-of-sample and leakage-controlled: signals are scaled only by prior
rolls, the model trains/tunes/tests on disjoint chronological slices, and volume curves use only
rolls that completed before the one being executed.""")

code("""%load_ext autoreload
%autoreload 2
import sys
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
import polars as pl
from plotnine import *
from plotnine.themes import theme_bw

from src.pipeline import build_datasets, generate_signals
from src.backtest import (
    generate_rolling_windows, windows_to_frame, split_tick_constrained,
    clean_delta_p_tc, run_rolling_backtest, predict_target_bins,
    compute_volume_curve, TICK_CONSTRAINED_BBG, TC_FEATURES,
)
from src.vwap import (
    run_vwap_backtest, run_improved_vwap_backtest, build_schedules, attach_tick_sizes,
)

pl.Config.set_tbl_rows(20)
pl.Config.set_tbl_cols(30)
TARGET_QTY = 10_000          # lots to roll per security
DIRECTION  = 'buy'           # buying calendar spreads
FILL_MODEL = 'hilo'          # price-range fills (needs HIGH/LOW)""")

md("""## 1. Data

`build_datasets` pulls the full history (2016-2025) of calendar-spread bins restricted to their roll
periods, with the near-leg `futures_price` attached. `generate_signals(..., normalization='time_of_roll')`
adds the microstructure signals (OBI / OFI / STV / NOI, tick-scaled `delta_p`) and normalises each flow
signal by a volatility profiled from **prior rolls only** - no train/test leakage, and no start-of-session
variance blow-up.""")

code("""df_cs, df_combined = build_datasets(env_path='../.env', years=list(range(2016, 2026)))
df_signals = generate_signals(df_cs, normalization='time_of_roll')

print('df_cs     :', df_cs.shape)
print('df_signals:', df_signals.shape)
print('date range:', df_cs['date'].min(), '->', df_cs['date'].max())
print('qcodes    :', df_cs['qcode'].n_unique(), '| securities (rolls):', df_cs['security'].n_unique())
df_signals.select('security', 'date', 'bin_start_time', 'qcode', 'bbg_code',
                  'days_until', 'obi', 'ofi', 'stv', 'noi', 'delta_p').head()""")

md("""## 2. Walk-forward windows

Each window trains on 4 years, tunes the decision threshold on the next 1 year, and tests on the
following 3 months - then everything slides forward by that 3-month test length, so the test
quarters tile the timeline without overlap. The first train window starts 2016-01-01.""")

code("""windows = generate_rolling_windows(data_end=df_cs['date'].max())
print(f'{len(windows)} walk-forward windows')
windows_to_frame(windows)""")

code("""# Walk-forward layout: train / validation / test spans per window.
wf = windows_to_frame(windows).to_pandas()
seg = pd.DataFrame([
    {'window': int(r['index']), 'phase': ph, 'start': r[ph + '_start'], 'end': r[ph + '_end']}
    for _, r in wf.iterrows() for ph in ['train', 'val', 'test']
])
seg['start'] = pd.to_datetime(seg['start']); seg['end'] = pd.to_datetime(seg['end'])
seg['phase'] = pd.Categorical(seg['phase'], categories=['train', 'val', 'test'], ordered=True)

(
    ggplot(seg, aes(x='start', xend='end', y='window', yend='window', color='phase'))
    + geom_segment(size=3)
    + scale_color_manual(values={'train': '#4575b4', 'val': '#f6c700', 'test': '#d73027'})
    + scale_y_reverse()
    + labs(title='Rolling walk-forward windows (4y train / 1y val / 3m test, step 3m)',
           x='Date', y='Window index', color='Split')
    + theme_bw(base_size=11) + theme(figure_size=(11, 6))
)""")

md("""## 3. Volume curves - the VWAP schedule

For each `qcode`, `compute_volume_curve` averages, across its rolls, the fraction of total roll volume
traded in each (days-to-target, time-of-day) bucket. This is the schedule a VWAP slicer follows: work
more where the market historically trades more. In the backtest the curve is leakage-safe (built only
from rolls that completed before the one being executed); here we show the full-sample curve for
exposition.""")

code("""vol_curve = compute_volume_curve(df_cs)

# Focus on the most-traded tick-constrained qcode for the heatmap.
top_q = (
    df_cs.filter(pl.col('bbg_code').is_in(TICK_CONSTRAINED_BBG))
    .group_by('qcode').len().sort('len', descending=True)['qcode'][0]
)
q_curve = (
    vol_curve.filter(pl.col('qcode') == top_q)
    .with_columns(hour=pl.col('bin_start_time').dt.hour() + pl.col('bin_start_time').dt.minute() / 60)
    .to_pandas()
)
print('heatmap qcode:', top_q)

(
    ggplot(q_curve, aes(x='hour', y='days_until', fill='volume_fraction'))
    + geom_tile()
    + scale_fill_gradient(low='#f7fbff', high='#08306b', name='Vol. frac')
    + scale_y_reverse()
    + labs(title=f'Volume curve for {top_q}: fraction of roll volume by stage',
           x='Time of day (hour)', y='Business days until target (roll progresses downward)')
    + theme_bw(base_size=11) + theme(figure_size=(10, 5))
)""")

code("""# Marginal curves: intraday shape and day-of-roll shape, for the busiest qcodes.
busy = (df_cs.group_by('qcode').len().sort('len', descending=True)['qcode'].head(4).to_list())

intraday = (
    vol_curve.filter(pl.col('qcode').is_in(busy))
    .with_columns(hour=pl.col('bin_start_time').dt.hour() + pl.col('bin_start_time').dt.minute() / 60)
    .group_by('qcode', 'hour').agg(frac=pl.col('volume_fraction').sum())
    .to_pandas()
)
p_intraday = (
    ggplot(intraday, aes('hour', 'frac', color='qcode'))
    + geom_line(size=1)
    + labs(title='Intraday volume profile (summed over roll days)', x='Time of day (hour)',
           y='Volume fraction', color='qcode')
    + theme_bw(base_size=11) + theme(figure_size=(10, 4))
)

by_day = (
    vol_curve.filter(pl.col('qcode').is_in(busy))
    .group_by('qcode', 'days_until').agg(frac=pl.col('volume_fraction').sum())
    .to_pandas()
)
p_byday = (
    ggplot(by_day, aes('days_until', 'frac', color='qcode'))
    + geom_line(size=1) + scale_x_reverse()
    + labs(title='Day-of-roll volume profile (summed over time of day)',
           x='Business days until target (roll progresses left)', y='Volume fraction', color='qcode')
    + theme_bw(base_size=11) + theme(figure_size=(10, 4))
)
display(p_intraday); display(p_byday)""")

md("""## 4. Predictive ordered logit, walk-forward

We restrict to the **tick-constrained** curve groups (which trade on a near-discrete grid), clean
`delta_p` to the ordered set {-2, 0, +2}, and run the class-weighted ordered logit walk-forward: fit on
train, tune the macro-F1 threshold on validation, evaluate on the held-out test quarter. Each window's
out-of-sample macro-F1 is plotted over time.""")

code("""_, df_signals_tc = split_tick_constrained(df_signals)
df_clean = clean_delta_p_tc(df_signals_tc)
print('tick-constrained cleaned rows:', df_clean.shape,
      '| class balance:', df_clean['delta_p'].value_counts(sort=True).to_dict())

results = run_rolling_backtest(df_clean, windows, verbose=False)
results.filter(pl.col('status') == 'ok').select(
    'label', 'threshold', 'val_macro_f1', 'test_macro_f1', 'test_accuracy', 'n_train', 'n_test'
)""")

code("""ok = results.filter(pl.col('status') == 'ok')
f1 = ok.select('test_start', 'val_macro_f1', 'test_macro_f1').to_pandas()
f1['test_start'] = pd.to_datetime(f1['test_start'])
f1m = f1.melt(id_vars='test_start', var_name='split', value_name='macro_f1')

(
    ggplot(f1m, aes('test_start', 'macro_f1', color='split'))
    + geom_line(size=1) + geom_point(size=2)
    + scale_color_manual(values={'val_macro_f1': '#f6c700', 'test_macro_f1': '#d73027'})
    + labs(title='Walk-forward macro-F1 over time (predicting next-bin move)',
           x='Test quarter', y='Macro-F1', color='Split')
    + theme_bw(base_size=11) + theme(figure_size=(11, 4.5))
)""")

md("""## 5. Baseline VWAP execution

The baseline rests a passive limit at the touch (best bid when buying) for each bin's scheduled
quantity; whatever fills passively **saves the spread**, and the unfilled remainder crosses at the next
bin's start price. The last bin of each session is always crossed (no overnight risk). Costs are signed
so positive = unfavourable, in basis points of the future price.""")

code("""df_tc = df_cs.filter(pl.col('bbg_code').is_in(TICK_CONSTRAINED_BBG))
per_base, summ_base = run_vwap_backtest(
    df_tc, target_qty=TARGET_QTY, direction=DIRECTION, fill_model=FILL_MODEL,
)
print(f'rolls executed      : {per_base.height}')
print(f'passive fill rate   : {summ_base["passive_rate"]:.3f}')
print(f'cost vs market VWAP : {summ_base["cost_vs_vwap_bp"]:+.3f} bp')
print(f'cost vs mid (ideal) : {summ_base["cost_vs_mid_bp"]:+.3f} bp')
print(f'avg half-spread     : {summ_base["half_spread_bp"]:.3f} bp')

pb = per_base.to_pandas()
(
    ggplot(pb, aes('cost_vs_vwap_bp'))
    + geom_histogram(bins=40, fill='#4575b4', color='white')
    + geom_vline(xintercept=0, linetype='dashed', color='grey')
    + labs(title='Baseline VWAP: per-roll cost vs market VWAP', x='Cost vs VWAP (bp)', y='Rolls')
    + theme_bw(base_size=11) + theme(figure_size=(10, 4))
)""")

md("""## 6. Ordered-logit VWAP overlay

The overlay uses the model's prediction for each target bin (known at its start). When buying:

- predicted **down** (favourable) -> rest the limit **2 ticks below** the bid to capture a better price;
- predicted **up** (adverse) -> **cross immediately** at the bin's start, beating the higher ask later;
- predicted **flat** -> behave like the baseline.

`predict_target_bins` produces the out-of-sample, session-shifted predictions; `run_improved_vwap_backtest`
schedules once and runs both strategies on the **same** rolls for a like-for-like comparison.""")

code("""preds = predict_target_bins(df_clean, windows, verbose=False)
print('predicted target bins:', preds.shape,
      '| class mix:', preds['pred'].value_counts(sort=True).to_dict())

summary, per_sec = run_improved_vwap_backtest(
    df_cs, preds, df_signals,
    target_qty=TARGET_QTY, direction=DIRECTION, fill_model=FILL_MODEL,
)
summary.select('strategy', 'passive_rate', 'exec_avg_price',
               'cost_vs_vwap_bp', 'cost_vs_mid_bp', 'improvement_vs_baseline_bp')""")

code("""# Head-to-head: passive rate and cost vs VWAP, baseline vs overlay.
sm = summary.select('strategy', 'passive_rate', 'cost_vs_vwap_bp', 'cost_vs_mid_bp').to_pandas()
sm_long = sm.melt(id_vars='strategy', var_name='metric', value_name='value')
sm_long['strategy'] = pd.Categorical(sm_long['strategy'], ['baseline_vwap', 'ordered_logit_vwap'])

p_cmp = (
    ggplot(sm_long, aes('strategy', 'value', fill='strategy'))
    + geom_col(show_legend=False)
    + facet_wrap('~metric', scales='free_y')
    + scale_fill_manual(values={'baseline_vwap': '#4575b4', 'ordered_logit_vwap': '#d73027'})
    + labs(title='Baseline VWAP vs ordered-logit overlay', x='', y='Value')
    + theme_bw(base_size=11) + theme(figure_size=(11, 4), axis_text_x=element_text(rotation=15))
)
display(p_cmp)

# Per-roll improvement of the overlay over the baseline (positive = overlay cheaper).
ps = per_sec.drop_nulls('improvement_vs_vwap_bp').to_pandas()
mean_impr = ps['improvement_vs_vwap_bp'].mean()
p_impr = (
    ggplot(ps, aes('improvement_vs_vwap_bp'))
    + geom_histogram(bins=40, fill='#1a9850', color='white')
    + geom_vline(xintercept=0, linetype='dashed', color='grey')
    + geom_vline(xintercept=mean_impr, color='#d73027', size=1)
    + labs(title=f'Overlay improvement over baseline per roll (mean = {mean_impr:+.2f} bp)',
           x='Improvement vs baseline (bp, positive = overlay cheaper)', y='Rolls')
    + theme_bw(base_size=11) + theme(figure_size=(10, 4))
)
display(p_impr)""")

code("""# Where does the overlay help or hurt? Improvement by qcode.
by_q = (
    per_sec.drop_nulls('improvement_vs_vwap_bp')
    .group_by('qcode').agg(
        mean_improvement_bp=pl.col('improvement_vs_vwap_bp').mean(),
        rolls=pl.len(),
    ).sort('mean_improvement_bp')
    .to_pandas()
)
by_q['qcode'] = pd.Categorical(by_q['qcode'], categories=by_q['qcode'].tolist(), ordered=True)
(
    ggplot(by_q, aes('qcode', 'mean_improvement_bp', fill='mean_improvement_bp > 0'))
    + geom_col(show_legend=False)
    + geom_hline(yintercept=0, color='grey')
    + scale_fill_manual(values={True: '#1a9850', False: '#d73027'})
    + coord_flip()
    + labs(title='Mean overlay improvement by qcode', x='qcode', y='Improvement vs baseline (bp)')
    + theme_bw(base_size=11) + theme(figure_size=(9, 6))
)""")

md("""## 7. A single roll, illustrated

To make the schedule concrete, take one out-of-sample roll and show how the target quantity is spread
across the roll by the historical volume curve (aggregated to business-days-until-target).""")

code("""sec = preds['security'].unique()[0]
one = df_cs.filter(pl.col('security') == sec)
sched1 = build_schedules(one, df_cs, target_qty=TARGET_QTY)
by_day1 = (
    sched1.group_by('days_until').agg(scheduled=pl.col('scheduled_qty').sum())
    .sort('days_until', descending=True).to_pandas()
)
print('illustrated roll:', sec)
(
    ggplot(by_day1, aes('days_until', 'scheduled'))
    + geom_col(fill='#4575b4') + scale_x_reverse()
    + labs(title=f'Scheduled quantity across the roll - {sec}',
           x='Business days until target (roll progresses left)', y='Scheduled lots')
    + theme_bw(base_size=11) + theme(figure_size=(10, 4))
)""")

md("""## Summary & caveats

- The **walk-forward** design keeps every number out-of-sample: time-of-roll signal scaling, the
  ordered-logit train/val/test split, and the volume curves all use only information available before
  the bin being executed.
- The **baseline VWAP** quantifies the spread it captures passively; the **overlay** reshapes execution
  around the model's directional call.
- Whether the overlay beats the baseline is governed by the model's precision/recall on the +/-2 tails and
  by the fill probability of the 2-tick-deeper passive order. With modest precision (~0.3) the favourable
  signal misfires often, so the gains from resting deeper trade off against the cost of crossing when the
  limit misses - exactly the cost trade-off the strategy is designed around. The per-qcode chart shows
  where the edge is real.
- The hi-lo fill model assumes a resting limit fills in full once the bin's range reaches it; the queue
  model is a more conservative alternative. Real fill behaviour at the deeper level is uncertain with only
  top-of-book depth.""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = os.path.join(os.path.dirname(__file__), '..', 'notebooks', '03_vwap_backtest.ipynb')
out = os.path.abspath(out)
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('wrote', out, 'with', len(cells), 'cells')
