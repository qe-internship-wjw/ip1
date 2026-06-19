# Order Book Signals on Bond & Equity Calendar Spreads

This repository contains the code behind the report *Alpha-Driven Execution for Large-Tick
Calendar Spreads*, Individal Project 1 of WJW supervised by HAL. It investigates whether
**order-book imbalance (OBI)** predicts short-horizon returns on **large-tick (tick-constrained)
bond and equity futures calendar spreads** during roll periods, and whether that signal can be
used as a **tactical overlay on a VWAP roll-execution algorithm** to reduce transaction cost.

This is a map from the **report's sections** to the **code that produces them**. It does
not restate the findings — read the report for results and interpretation, and read this to find
the function, hyperparameter, or notebook cell behind each figure or claim.

---

## Repository layout

```
src/                 importable library — the entire methodology lives here
  utils.py           Snowflake connection, ticker parsing, roll-calendar math, ACF diagnostics
  pipeline.py        data ingestion, preprocessing, roll filtering, signal generation, normalisation
  ordered_logit.py   the predictive model: weighted ordered logit, CRSE, F-beta threshold tuning
  backtest.py        leakage-free rolling walk-forward split + walk-forward driver, volume curves
  vwap.py            baseline VWAP execution + ordered-logit tactical overlay + cost accounting
  lob_simulation.py  limit-order-book / queue-position simulator (fill engine for the backtest)

notebooks/           result walkthroughs (see "Notebooks" below)
  01_exploration.ipynb        data, roll periods, large-tick stylised facts, signal exploration
  02a_regressions_tu.ipynb    hypothesis tests + small-tick (tick-unconstrained) curves [appendix]
  02b_regressions_tc.ipynb    the predictive ordered-logit model on large-tick curves
  03_vwap_backtest.ipynb      the execution backtest and tactical overlay

environment.yml      conda environment
.env                 Snowflake credentials (not committed; see Setup)
```

The `src/` modules are arranged as a pipeline: **`utils` → `pipeline` → `ordered_logit` →
`backtest` → `vwap`/`lob_simulation`**. Each notebook is a thin, narrated driver over these modules.

---

## Setup

### 1. Install dependencies

The project targets Python 3.11. Create the conda environment from the spec:

```bash
conda env create -f environment.yml
conda activate ip1
```

Key libraries: **polars** (all data wrangling), **statsmodels** (the ordered logit + cluster-robust
standard errors), **snowflake-connector-python** / **snowflake-snowpark-python** (data access),
**plotnine** (figures), and **scipy** / **numpy**.

### 2. Provide Snowflake credentials in `.env`

Data is pulled live from Snowflake (database `LISTED_INTERN_PROJECT`, schema `PROJECT_5`). Create a
`.env` file in the project root (it is git-ignored) with:

```
SNOWFLAKE_ACCOUNT=...
SNOWFLAKE_WAREHOUSE=...
SNOWFLAKE_ROLE=...
SNOWFLAKE_USERNAME=...
SNOWFLAKE_PRIVATE_KEY=...      # base64-encoded private key (key-pair auth)
```

---

## Description of Source Code

### `src/pipeline.py` — data cleaning & signal generation

The full preprocessing and feature pipeline. Entry point
[`build_datasets`](src/pipeline.py#L613) runs stages 1–4; [`generate_signals`](src/pipeline.py#L478)
runs stages 5–6.

1. **`add_microstructure_signals`** — computes every signal in the report, strictly intraday
   (partitioned by `(security, date)` so no forward step spans an overnight gap):
   - `mid_price`, `delta_p` (the return to predict),
   - **OBI** (Order Book Imbalance) — the predictive signal,
   - **OFI** (Order Flow Imbalance), the 3-case `delta_lb`/`delta_la` liquidity-change definition
     from the report's *Contemporaneous Signals*,
   - **STV** (Signed Trade Volume) and **NOI** (Net Order Inflow = OFI − STV, the appendix's
     negative-result signal).
   - `obi_method` selects how OBI is measured (`end_of_bin`, `twa`, or `twa_or_eob`).
2. **`_apply_signal_transforms` + normalisation** — the report's **Normalization** section:
   - returns → **half ticks** (per-`BBG_CODE` tick size),
   - OBI → divided by **touch size** (total quoted size at the touch),
   - OFI/STV/NOI → scaled by an intraday volatility, choosing between
     [`_normalize_time_of_roll`](src/pipeline.py#L372) (**Time-of-Roll Normalization** — a fixed
     train-set volatility profile indexed by `(qcode, days_until)`) and
     [`_normalize_rolling`](src/pipeline.py#L427) (a **rolling intraday z-score** within each
     session). `generate_signals` defaults to `normalization="time_of_roll"`, and the walk-forward
     backtest ([`backtest.py`](src/backtest.py#L314)) uses it too — note this differs from the
     report draft, which discusses time-of-roll introducing outliers; the `"rolling"` scheme is the
     leakage-free alternative also implemented here.
   - **Leakage control:** the time-of-roll profile is estimated only on rolls with
     `target_date <= train_end`, then applied to all rows.
3. **Outright-leg augmentation** — `generate_futures_signals` + `attach_outright_signals` /
   `attach_leg_signals` compute OBI/STV on the outright legs and join `buy_*` / `sell_*` columns
   onto each spread bin. This backs the appendix's **Outright OBI** experiment and the 6-feature
   model in notebook 02b.

### `src/ordered_logit.py` — predictive model

Everything in the report's **Model Specification** and **Model Results** sections:

- **`WeightedOrderedModel`** / **`fit_ordered_logit`** — the ordered logit itself:
  - latent-variable formulation `y* = β·OBI + ε` with two fitted **boundary constants**
    (κ₋₂/₀, κ₀/₂), estimated by **maximum likelihood** (BFGS);
  - **adaptive class weights** `w₊₂ = N₀ / N₊₂` applied per observation, computed only on the
    training slice, to counter the >10× dominance of the no-change class;
  - **cluster-robust standard errors (CRSE)** with each `(security, date)` session as a cluster,
    addressing the OBI autocorrelation that otherwise inflates t-statistics.
  - Default features `TC_FEATURES = ['obi', 'noi', 'stv']`; default target `delta_p_fwd`.
- **`tune_thresholds`** — the report's **Probability Threshold Tuning**: independently picks a
  probability threshold for the `+2` and `−2` tails to maximise a one-vs-rest **F-beta** objective
  (`DEFAULT_BETA = 0.5`, weighting precision over recall because every directional signal triggers
  an execution adjustment whose false positives cost crossing). `pr_threshold_df` produces the
  Figure 10 precision/recall sweep.

### `src/backtest.py` — data split & walk-forward infrastructure
The **leakage-free rolling-window** framework (report Figure 1):

- **`RollingWindow`** dataclass + **`generate_rolling_windows`** — implement the
  **4-year train / 1-year validation / 3-month test, 3-month stride** scheme. Validation is used
  *only* for threshold tuning, never to fit model parameters — exactly as the report stipulates.
  Geometry is set by module constants `TRAIN_MONTHS=48`, `VAL_MONTHS=12`, `TEST_MONTHS=3`,
  `STEP_MONTHS=3`, `DATA_START=2016-01-01`.
- **`run_rolling_backtest`** — orchestrates the per-window loop: time-of-roll normalise within the
  window's train rolls → `fit_ordered_logit` → `tune_thresholds` on validation → evaluate on the
  3-month test set. Returns one row of metrics per window (accuracy, macro F-beta, thresholds,
  class counts). Undersized or singular windows are skipped.
- **`predict_target_bins`** — produces per-bin out-of-sample predictions aligned to test windows;
  this is the hand-off into the execution backtest (it feeds `vwap.run_improved_vwap_backtest`).

### `src/vwap.py` — execution algorithms

Both the baseline and the alpha-driven overlay:

- **`simulate_passive_aggressive` / `run_vwap_backtest`** — the **Baseline VWAP**: rest a child
  limit order at the touch (best bid when buying); if unfilled after the **survival window W**
  bins, cross aggressively with a market order. The last bin of a session is forced to fully cross
  (no overnight carry).
- **`simulate_improved_vwap` / `run_improved_vwap_backtest`** — the **Tactical Overlay** driven by
  the ordered-logit prediction:
  - **Favourable** prediction (predict price drop when buying) → re-rest the merged order **deeper**
    at best − `ticks` (passive, low-risk, captures incoming flow);
  - **Adverse** prediction (predict price rise when buying) → **cross immediately** with a market
    order (guards against worse future prices);
  - **Neutral / no signal** → behave exactly like the baseline.
  - `overlay_actions ∈ {'both','favourable','adverse'}` reproduces the report's **Per-Action Cost
    Attribution** (Figure 17) by enabling only one action at a time.
- **Cost accounting** — `per_security_metrics` / `summarize` compute the execution cost per lot in
  **ticks** (and basis points), and `improvement_vs_baseline = baseline_cost − overlay_cost`
  (positive = overlay cheaper). This produces the Figure 14/15 per-lot and per-curve improvement
  distributions.

### `src/lob_simulation.py` — order book & queue simulation

The fill engine that `vwap.py` calls.

- **`compute_survival_window`** — turns `window_fraction` into the per-curve window W =
  `max(1, round(window_fraction × mean(touch_size) / mean(volume)))` bins. This is Figure 13 in the report.
- **`_sim_session` / `simulate_windowed`** — the stateful per-`(security, date)` engine enforcing:
  - **Price-time priority (FIFO):** orders sorted best-price-first, ties by placement time;
    re-pricing (the favourable action) forfeits time priority.
  - **No cancellations** ahead of our order, **no market impact**, **instant execution** at the
    start-of-bin snapshot — the four assumptions listed in the report.
  - Queue-ahead consumption from estimated opposing sell-initiated flow `(V − STV)/2`, trade-through
    detection from OHLC, and aggressive crossing at window expiry.
  - **`deep_price_fraction`** (passed as `queue_depth_levels`, default 0.7) sets the estimated
    resting depth at the best − 1 level, since L2 depth is not in the dataset.

> The report flags **US Treasury Notes (FV, TU)** as outliers because their matching is mainly
> **pro-rata**, which this price-time-priority simulator misrepresents — the same caveat applies to
> any use of `lob_simulation.py` on those curves.

---

## Hyperparameters

All end-to-end hyperparameters from the report's summary table, and where they live in code:

| Hyperparameter | Default | Location | Controls |
|---|---|---|---|
| `roll_volume_frac` | 0.25 | `pipeline.ROLL_VOLUME_THRESHOLD` → `apply_roll_volume_filter` | Volume threshold defining a roll period |
| `obi_method` | `end_of_bin` | `pipeline.generate_signals` | How OBI is measured within a bin |
| `normalization` | `rolling` | `pipeline.generate_signals` (and `backtest.py`) | OFI/STV/NOI scaling scheme (`time_of_roll` vs. `rolling`) |
| `beta` | 0.5 | `ordered_logit.DEFAULT_BETA` → `tune_thresholds` | Precision weight in the F-beta threshold objective (< 1 favours precision) |
| `participation_rate` | 0.01 | `vwap.build_schedules` | Child order size as a fraction of historical roll volume; **set per-curve** in production |
| `window_fraction` | 0.3 | `lob_simulation.compute_survival_window` | Survival window W = passive-rest duration before crossing |
| `deep_price_fraction` | 0.7 | `lob_simulation` (`queue_depth_levels`) | Estimated depth at best − 1 relative to the touch |

The train/validation/test split geometry is also configurable, via the constants at the top of
[`backtest.py`](src/backtest.py) (`TRAIN_MONTHS`, `VAL_MONTHS`, `TEST_MONTHS`, `STEP_MONTHS`).
Some of these parameters are only relevant if you adopt the report's specific method — e.g.
`roll_volume_frac` matters only if you use this roll-period identification, and `deep_price_fraction`
only if you lack live L2 depth.

---

## Notebooks

The notebooks are the narrated walkthrough of the report; each is a driver over `src/`.

- **`01_exploration.ipynb` — Data and Preprocessing, Large-Tick Curves, Signal Generation.**
  Connects to Snowflake, runs `build_datasets`, and establishes the report's stylised facts:
  the **roll-period** definition and volume seasonality (Figures 1–2), the **large-tick**
  classification (discrete one-tick return distribution Figure 3, one-tick-spread % Figure 4,
  touch-to-volume ratio Figure 5), and exploratory views of the OBI/OFI/STV signals including the
  OBI-vs-returns ACF (Figure 7).

- **`02b_regressions_tc.ipynb` — Model Specification, Results, OOS Evaluation (main model).**
  The core modelling notebook for the **tick-constrained** curves. Fits the contemporaneous
  baseline, then the **predictive 1-step-ahead ordered logit** (`fit_ordered_logit`), tunes
  thresholds, and reports cross-`qcode` performance — producing Figures 9–12 (model parameters,
  threshold tuning, confusion matrix, per-class precision/recall and the 5×–7× precision gain). The
  final section adds the **outright-leg** features (6-feature model) for the appendix comparison.

- **`03_vwap_backtest.ipynb` — Backtest (the entire execution section).**
  The execution walkthrough. Builds the walk-forward windows and volume curves, generates
  walk-forward predictions (`predict_target_bins`), runs the **baseline VWAP** and the
  **ordered-logit overlay**, then decomposes the overlay by action and by curve (Figures 14–17),
  sweeps the survival window (Figure 16), and illustrates a single roll bin-by-bin. Regenerable via
  `python scripts/gen_nb_03.py`.

- **`02a_regressions_tu.ipynb` — Appendix (hypothesis tests + small-tick / tick-unconstrained
  curves).** Supports the appendix: the H1/H2/H3 research-plan hypothesis tests, residual
  diagnostics, and the linear-regression treatment of small-tick equity curves (where returns are
  less discrete). Not part of the main large-tick result.

---

## Adapting this codebase for live production

Most of `src/` is research scaffolding (data pulls, walk-forward evaluation, cost accounting). For a
production deployment, two pieces carry over directly, and the rest informs how to wire them in.

### 1. The model — `ordered_logit.py`

A fitted model from `fit_ordered_logit` returns a standard statsmodels result object plus the tuned
class weight. To serve it live:

1. **Train and freeze.** Fit once on the latest available train window and tune thresholds on the
   most recent validation roll. Persist the fitted result object (pickle/joblib) together with the
   tuned `(thr_down, thr_up)` and the per-`BBG_CODE` tick sizes and normalisation statistics from
   `pipeline`.
2. **Score live bins.** Compute the same features (`obi`, `noi`, `stv`) on the incoming 5-minute
   bin **with identical normalisation** (critically, OBI ÷ touch size; flow z-scored with the same
   rolling scheme), then `result.predict(exog=...)` → class probabilities →
   `assign_classes(probs, thr_down, thr_up)` → a `{-2, 0, +2}` directional signal.
3. **Refit cadence.** Re-fit and re-tune on a schedule (e.g. quarterly, matching the test stride)
   by appending newly completed rolls — `generate_rolling_windows` already expresses exactly this
   loop; in production the latest window is "live" rather than "test".

Keep the feature engineering in `pipeline.py` byte-for-byte identical between training and serving;
any change to OBI measurement or normalisation invalidates the frozen coefficients.

### 2. The execution overlay — `vwap.py`

`simulate_improved_vwap` encodes the **decision logic** that ports to a live execution algorithm,
independent of the backtest's simulated fills:

- On each bin, take the frozen model's signal and apply the **favourable → rest deeper** /
  **adverse → cross now** / **neutral → baseline** rule against your live order book.
- The report's per-action result (favourable helps, adverse hurts) suggests deploying the
  **favourable action only** first (`overlay_actions='favourable'`) — `vwap.py` supports running
  that subset directly.
- Replace the simulated fill engine (`lob_simulation.py`) with real fills from the venue. The
  simulator's assumptions (price-time priority, no cancellations, no market impact, `deep_price_fraction`
  for best − 1 depth) exist only because the backtest lacks L2 and tick data; live, you have the
  real queue and **do not need `deep_price_fraction`** (set it aside if L2 depth is available).

### Caveats carried from the report

- The overlay's benefit is **conditional on the baseline execution algorithm**. If Quantedge's
  current roll algorithm differs materially from the `vwap.py` baseline, re-establish the overlay's
  improvement against the *actual* baseline before deploying.
- **Pro-rata curves (FV, TU)** are not handled correctly by the price-time-priority queue model;
  treat them separately (see report Future Improvements §3).
- Several hyperparameters are deployment-conditional (`roll_volume_frac`, `window_fraction`,
  `deep_price_fraction`) — they matter only if you adopt the corresponding method from the report
  rather than an existing production equivalent.
