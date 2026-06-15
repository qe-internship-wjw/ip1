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
TARGET_QTY  = 10_000         # lots to roll per security
DIRECTION   = 'buy'          # buying calendar spreads
FILL_MODEL  = 'hilo'         # price-range fills (needs HIGH/LOW)
WINDOW      = 3              # order survival window in bins (1 = naive single-bin)
WINDOW_GRID = [1, 2, 3, 4, 6]  # 5 min .. 30 min, for the window-size sweep""")

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
train, tune the **-2 and +2 thresholds independently** on validation (each maximising its tail's
F-beta, β=0.5 → precision-weighted), evaluate on the held-out test quarter. Each window's
out-of-sample macro-F-beta is plotted over time.""")

code("""_, df_signals_tc = split_tick_constrained(df_signals)
df_clean = clean_delta_p_tc(df_signals_tc)
print('tick-constrained cleaned rows:', df_clean.shape,
      '| class balance:', df_clean['delta_p'].value_counts(sort=True).to_dict())

results = run_rolling_backtest(df_clean, windows, verbose=False)
results.filter(pl.col('status') == 'ok').select(
    'label', 'thr_down', 'thr_up', 'val_fbeta_down', 'val_fbeta_up',
    'test_macro_fbeta', 'test_accuracy', 'n_train', 'n_test'
)""")

code("""ok = results.filter(pl.col('status') == 'ok').with_columns(
    val_fbeta_mean=(pl.col('val_fbeta_down') + pl.col('val_fbeta_up')) / 2
)
fb = ok.select('test_start', 'val_fbeta_mean', 'test_macro_fbeta').to_pandas()
fb['test_start'] = pd.to_datetime(fb['test_start'])
fbm = fb.melt(id_vars='test_start', var_name='split', value_name='fbeta')

(
    ggplot(fbm, aes('test_start', 'fbeta', color='split'))
    + geom_line(size=1) + geom_point(size=2)
    + scale_color_manual(values={'val_fbeta_mean': '#f6c700', 'test_macro_fbeta': '#d73027'})
    + labs(title='Walk-forward F-beta over time (predicting next-bin move)',
           x='Test quarter', y='Macro F-beta (β=0.5)', color='Split')
    + theme_bw(base_size=11) + theme(figure_size=(11, 4.5))
)""")

md("""## 5. Baseline VWAP execution

The baseline rests a passive limit at the touch (best bid when buying) for each bin's scheduled
quantity; whatever fills passively **saves the spread**. With a survival **window** of `W` bins an
order rests for `W` bins before its unfilled remainder crosses at the end of the window — `W=1` is the
naive single-bin scheme, while larger `W` lets the order sit across several bins and fill against more
opposing flow (essential in tick-constrained books where one bin rarely clears the queue). Orders whose
window reaches the session close are cleaned up at the last bin (no overnight risk). Costs are signed so
positive = unfavourable, reported in two frameworks: **ticks per lot** (the default — e.g. always
crossing a 1-tick spread costs half a tick over mid) and basis points of the future price.""")

code("""df_tc = df_cs.filter(pl.col('bbg_code').is_in(TICK_CONSTRAINED_BBG))
per_base, summ_base = run_vwap_backtest(
    df_tc, target_qty=TARGET_QTY, direction=DIRECTION, fill_model=FILL_MODEL,
    window=WINDOW, df_signals=df_signals,
)
print(f'rolls executed       : {per_base.height}')
print(f'passive fill rate    : {summ_base["passive_rate"]:.3f}')
print(f'cost vs market VWAP  : {summ_base["cost_vs_vwap_ticks"]:+.3f} ticks/lot  ({summ_base["cost_vs_vwap_bp"]:+.2f} bp)')
print(f'cost vs mid (ideal)  : {summ_base["cost_vs_mid_ticks"]:+.3f} ticks/lot  ({summ_base["cost_vs_mid_bp"]:+.2f} bp)')
print(f'avg half-spread      : {summ_base["half_spread_ticks"]:.3f} ticks       ({summ_base["half_spread_bp"]:.2f} bp)')

pb = per_base.to_pandas()
(
    ggplot(pb, aes('cost_vs_vwap_ticks'))
    + geom_histogram(bins=40, fill='#4575b4', color='white')
    + geom_vline(xintercept=0, linetype='dashed', color='grey')
    + labs(title='Baseline VWAP: per-roll cost vs market VWAP',
           x='Cost vs VWAP (ticks per lot)', y='Rolls')
    + theme_bw(base_size=11) + theme(figure_size=(10, 4))
)""")

md("""## 6. Ordered-logit VWAP overlay

The overlay uses the model's prediction for each target bin (known at its start). When buying:

- predicted **down** (favourable) -> rest the limit **2 ticks below** the bid to capture a better price;
- predicted **up** (adverse) -> **cross immediately** at the bin's start, beating the higher ask later;
- predicted **flat** -> behave like the baseline.

`predict_target_bins` produces the out-of-sample, session-shifted predictions; `run_improved_vwap_backtest`
schedules once and runs both strategies on the **same** rolls for a like-for-like comparison.

Aggregate improvement is computed **per lot**: each roll is weighted by the lots it executes (`total_qty`,
which follows the volume curve and equals `TARGET_QTY` by conservation), exactly as
`summary['improvement_vs_baseline']` pools lots. A plain mean of per-roll ratios, or a market-*volume*
weighting (a different quantity), is not per-lot and can disagree in sign with the portfolio number.""")

code("""preds = predict_target_bins(df_clean, windows, verbose=False)
print('predicted target bins:', preds.shape,
      '| class mix:', preds['pred'].value_counts(sort=True).to_dict())

summary, per_sec = run_improved_vwap_backtest(
    df_cs, preds, df_signals,
    target_qty=TARGET_QTY, direction=DIRECTION, fill_model=FILL_MODEL, window=WINDOW, framework='ticks',
)
# `per_sec.total_qty` = lots executed per roll (follows the volume curve; = TARGET_QTY by
# conservation). It is the correct PER-LOT weight for aggregating per-roll improvement -- the same
# lots that summary['improvement_vs_baseline'] pools over. Weighting by *market* volume instead is
# a different quantity that over-weights high-volume rolls and can flip the aggregate's sign.

# Headline framework is ticks per lot; bp columns are present too.
summary.select('strategy', 'framework', 'passive_rate', 'exec_avg_price',
               'cost_vs_vwap_ticks', 'cost_vs_mid_ticks', 'cost_vs_vwap_bp', 'improvement_vs_baseline')""")

code("""# Head-to-head: passive rate and cost vs VWAP / mid (ticks per lot), baseline vs overlay.
sm = summary.select('strategy', 'passive_rate', 'cost_vs_vwap_ticks', 'cost_vs_mid_ticks').to_pandas()
sm_long = sm.melt(id_vars='strategy', var_name='metric', value_name='value')
sm_long['strategy'] = pd.Categorical(sm_long['strategy'], ['baseline_vwap', 'ordered_logit_vwap'])

p_cmp = (
    ggplot(sm_long, aes('strategy', 'value', fill='strategy'))
    + geom_col(show_legend=False)
    + facet_wrap('~metric', scales='free_y')
    + scale_fill_manual(values={'baseline_vwap': '#4575b4', 'ordered_logit_vwap': '#d73027'})
    + labs(title='Baseline VWAP vs ordered-logit overlay (ticks per lot)', x='', y='Value')
    + theme_bw(base_size=11) + theme(figure_size=(11, 4), axis_text_x=element_text(rotation=15))
)
display(p_cmp)

# Per-roll improvement (positive = overlay cheaper), ticks/lot. PER-LOT basis: the histogram and the
# red mean line are weighted by lots executed per roll ('total_qty'), so the mean agrees in sign with
# summary['improvement_vs_baseline'] (NOT a market-volume weighting, which flipped the sign).
ps = per_sec.drop_nulls('improvement_vs_vwap').to_pandas()
lot_mean = np.average(ps['improvement_vs_vwap'], weights=ps['total_qty'])
p_impr = (
    ggplot(ps, aes('improvement_vs_vwap', weight='total_qty'))
    + geom_histogram(bins=40, fill='#1a9850', color='white')
    + geom_vline(xintercept=0, linetype='dashed', color='grey')
    + geom_vline(xintercept=lot_mean, color='#d73027', size=1)
    + labs(title=f'Overlay improvement over baseline, lot-weighted (mean = {lot_mean:+.3f} ticks/lot)',
           x='Improvement vs baseline (ticks per lot, positive = overlay cheaper)',
           y='Lots executed (weight)')
    + theme_bw(base_size=11) + theme(figure_size=(10, 4))
)
display(p_impr)""")

code("""# Where does the overlay help or hurt? Per-qcode improvement, weighted by lots executed (per-lot):
# within each qcode, rolls are weighted by their executed lots ('total_qty'), matching summary.
by_q = (
    per_sec.drop_nulls('improvement_vs_vwap')
    .group_by('qcode').agg(
        lot_improvement_ticks=(pl.col('improvement_vs_vwap') * pl.col('total_qty')).sum()
                              / pl.col('total_qty').sum(),
        rolls=pl.len(),
        lots=pl.col('total_qty').sum(),
    ).sort('lot_improvement_ticks')
    .to_pandas()
)
by_q['qcode'] = pd.Categorical(by_q['qcode'], categories=by_q['qcode'].tolist(), ordered=True)
(
    ggplot(by_q, aes('qcode', 'lot_improvement_ticks', fill='lot_improvement_ticks > 0'))
    + geom_col(show_legend=False)
    + geom_hline(yintercept=0, color='grey')
    + scale_fill_manual(values={True: '#1a9850', False: '#d73027'})
    + coord_flip()
    + labs(title='Lot-weighted overlay improvement by qcode', x='qcode',
           y='Improvement vs baseline (ticks per lot, lot-weighted)')
    + theme_bw(base_size=11) + theme(figure_size=(9, 6))
)""")

md("""## 6b. Sweeping the survival window

The survival window trades off **fill rate against adverse selection**: a longer window lets passive
orders fill more often (saving the spread), but an order that sits longer is more likely to be run over
by an adverse move. We sweep `WINDOW_GRID` and track the passive-fill rate and cost vs market VWAP for
the baseline and the overlay. Predictions are window-independent, so they are reused across the sweep.""")

code("""sweep_rows = []
for W in WINDOW_GRID:
    s_w, _ = run_improved_vwap_backtest(
        df_cs, preds, df_signals,
        target_qty=TARGET_QTY, direction=DIRECTION, fill_model=FILL_MODEL, window=W, framework='ticks',
    )
    for r in s_w.iter_rows(named=True):
        sweep_rows.append(dict(window=W, strategy=r['strategy'], passive_rate=r['passive_rate'],
                               cost_vs_vwap_ticks=r['cost_vs_vwap_ticks'],
                               cost_vs_mid_ticks=r['cost_vs_mid_ticks']))
sweep = pl.DataFrame(sweep_rows)
sweep""")

code("""sw = sweep.to_pandas()

p_pr = (
    ggplot(sw, aes('window', 'passive_rate', color='strategy'))
    + geom_line(size=1) + geom_point(size=2)
    + scale_color_manual(values={'baseline_vwap': '#4575b4', 'ordered_logit_vwap': '#d73027'})
    + labs(title='Passive-fill rate vs survival window', x='Survival window (bins)',
           y='Passive fill rate', color='Strategy')
    + theme_bw(base_size=11) + theme(figure_size=(10, 4))
)
p_cost = (
    ggplot(sw, aes('window', 'cost_vs_vwap_ticks', color='strategy'))
    + geom_line(size=1) + geom_point(size=2)
    + geom_hline(yintercept=0, linetype='dashed', color='grey')
    + scale_color_manual(values={'baseline_vwap': '#4575b4', 'ordered_logit_vwap': '#d73027'})
    + labs(title='Cost vs market VWAP vs survival window', x='Survival window (bins)',
           y='Cost vs VWAP (ticks per lot)', color='Strategy')
    + theme_bw(base_size=11) + theme(figure_size=(10, 4))
)
display(p_pr); display(p_cost)""")

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
- The **survival window** is the main lever on passive-fill rate: in tick-constrained books one bin of
  flow rarely clears the queue, so orders must rest across several bins to fill. The sweep shows the
  fill-rate vs adverse-selection trade-off; pick the window where cost vs VWAP bottoms out.
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
