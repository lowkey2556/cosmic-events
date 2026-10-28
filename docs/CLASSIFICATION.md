# Classification: windows, splits, models, metrics and ablations

*(Review 2, "Classification" slides. Plain-English source.)*

## 1. The samples: time windows (Spark, Step 5)

Each star's cleaned light curve is cut into **100-day windows that start every 50 days**
(window *k* covers `[t0 + 50k, t0 + 50k + 100)` days, `t0` = the star's first detection).
Windows with **fewer than 20 observations are dropped**. Every window inherits its star's
label (anomalous = 1, normal = 0) and keeps the star's `auid`. Window length, stride and the
minimum count are set in `config/settings.yaml` (`windows`).

All of this runs in Spark: the membership is a Spark SQL `explode(sequence(...))`, the summary
features are Spark SQL aggregates, and the Lomb–Scargle periodogram and the token sequences
are computed per window inside Spark with `applyInPandas`. The results are written to HDFS as
Parquet and registered in Hive (`windows`, `window_features`, `window_sequences`, `splits`).

The **same windows** feed both kinds of model, so the comparison is like-for-like:

| For | What each window becomes |
|---|---|
| **RGA-Net** (PyTorch) | A time-sorted sequence of up to L = 200 observations (an evenly spaced subsample if longer, padding plus a mask if shorter). Each observation: `mag_norm`, days since window start, `log1p` gap since the previous observation, band, observation type, uncertainty + missing flag, observer's A_k, observer's offset. Exported from Spark to `work/windows_seq.npz`. |
| **The 4 ML models** (Spark MLlib) | 13 summary numbers. Light curve: amplitude (p95 − p5), std, skewness, kurtosis, Lomb–Scargle period (log10 days) and power. Cadence: median gap, largest gap, observation count, time span. Observers: reliability-weighted mean A_k (each observation weighted equally, so observers who contributed more count more), number of distinct observers, mean \|offset\| of the window's scored observers. |

Window counts per class and per split are logged (`window_counts_by_class.csv`,
`window_counts_by_split.csv`). `star_attrition.csv` lists **every star** in the raw data with
the rows it had after each preprocessing step, its number of windows, and — if it is not
used — the exact reason (no VSX match, type not in the label map, all rows removed at a named
step, or no window with ≥ 20 observations).

### ZTF windows (Phase 2, `ztf_windows`)

Labelled ZTF light curves (VSX match within 2″; **every ZTF object within 10″ of any AAVSO star is excluded**) are cut
into the same windows with the same code (`windows.assign_windows_from`). ZTF is one professional survey, so each token
has A_k = 1 and offset = 0. Light curves and then windows are sampled class-balanced over all fields, at most 50,000
windows (`ztf.pretrain_windows`). They are used for RGA-Net pre-training and, in the ML models' `aavso_ztf` data
setting, as extra **training** windows only. ZTF windows never enter any validation or test set.

## 2. Splits: by star, 10 times

- **70 % train / 15 % validation / 15 % test, by star** (`auid`), stratified by class, so every set has both
  classes. All windows of a star go to the same set: no star is ever in two sets (checked in code).
- **10 seeds → 10 different splits** (`split.seeds`). The same splits (saved once as `work/splits.csv` / Hive table
  `splits`) are used by RGA-Net and by all four ML models.
- **The test set is never used to choose anything.** Hyperparameters, data settings, early stopping and the decision
  threshold all use the validation set.

Why by star: windows of the same star look alike. If a star had windows in train and test, the test score would reward
recognising that star, not recognising a new one.

## 3. The models

| Model | Library | Inputs | Chosen on validation (best AUPRC) |
|---|---|---|---|
| **RGA-Net (proposed)** | PyTorch (Colab GPU) | token sequences (+ optional summary features) | ZTF pre-training; lr × dropout × hybrid head on seed 1's validation stars; early-stopping epoch |
| Gaussian Naive Bayes (`modelType="gaussian"`) | Spark MLlib | summary features | data setting only |
| Decision Tree | Spark MLlib | summary features | data setting × maxDepth ∈ {5, 8} (minInstancesPerNode 5) |
| Random Forest | Spark MLlib | summary features | data setting × (numTrees 100, maxDepth 8) |
| Logistic Regression | Spark MLlib | summary features | data setting × regParam ∈ {0.01, 0.1} |

The grids are in `config/settings.yaml` → `classify`. For the ML models, **two data settings** are tried and chosen per
model, feature set and seed on validation AUPRC: `aavso` (AAVSO training windows only) and `aavso_ztf` (plus 10,000
class-balanced ZTF training windows, `ztf.ml_ztf_windows`). Median imputation and standardisation are **fitted on the
training windows only** (including the ZTF ones in `aavso_ztf`); class imbalance is handled with `weightCol`
(weight = n_train / (2 · n_class) on the training windows). The chosen setting and the training time for every seed are
in `ml_selected_params.csv`. Details of RGA-Net: [PROPOSED_MODEL.md](PROPOSED_MODEL.md).

### Who is a "prior approach" (Fix 1)

| Role | Rows | Used for |
|---|---|---|
| **Prior approaches** | NB, DT, RF, LR **without** observer features (10 light-curve and cadence features) | the comparison with RGA-Net (`prior_vs_proposed_*`) |
| **Our framework applied to classic models** | NB, DT, RF, LR **with** the 3 observer features (`w_a`, `n_observers`, `mean_abs_offset`) | reported separately (`observer_feature_gain.csv`); never called prior approaches and never used to claim an improvement over prior work |
| **Proposed** | RGA-Net full (ZTF pre-trained, gate active) | the headline row |
| Ablations | RGA-Net no_pretrain, no_gate, observer_blind | `ablation_table.csv` |

## 4. Metrics (Phase 3)

Every model writes the predicted probability of "anomalous" for each validation and test window; **one shared
implementation** (`src/cosmic/evaluation.py`) computes every metric, from the saved prediction files only.

- **Main metrics:** **MCC** (Matthews correlation coefficient, the headline), **ROC-AUC**, **AUPRC** (average
  precision), **Cohen's κ** and the **Brier score** (mean squared error of the probability; lower is better).
- **Threshold (MCC, κ):** the probability cut-off with the **highest validation MCC** (ties: closest to 0.5), chosen
  separately at window and star level; the test set never moves it (tested). ROC-AUC, AUPRC and Brier need no threshold.
- **Star level:** a star's probability = the mean of its windows' probabilities; the stricter view, because windows of
  one star are correlated. Headline level: `metrics.main_level` (window).
- **Reported as mean ± std over the 10 splits.** Precision, recall, accuracy, F1 and balanced accuracy, at the same
  threshold, go to the appendix (`metrics_appendix.csv`).
- **Paired tests:** for each prior approach, the per-seed differences RGA-Net − prior, the number of seeds RGA-Net is
  ahead, and a two-sided **Wilcoxon signed-rank** p-value (SciPy) over the 10 paired seeds
  (`prior_vs_proposed_paired.csv`); the same for each ablation (`ablation_paired.csv`) and for each classic model
  without → with observer features (`observer_feature_gain.csv`).

MCC uses all four cells of the confusion matrix, so it is not flattered by class imbalance; κ corrects accuracy for
chance agreement; Brier checks the probabilities themselves, not just their ranking.

## 5. Distributed inference

After training, Spark applies the saved RGA-Net models (one per split) to **every** window in parallel (`mapInPandas`,
`infer`) and writes the Hive table `window_predictions`. `distributed_inference_check.csv` compares the Spark and
PyTorch probabilities on the validation/test windows.

## 6. Outputs (in `v3_5GB/outputs/`, all made by `compare` from the saved predictions)

`per_seed_metrics.csv` (every model × variant × seed × level, with threshold and confusion counts),
`comparison_table.csv` (both levels, mean ± std), `prior_vs_proposed_table.csv`, `prior_vs_proposed_paired.csv`,
`observer_feature_gain.csv`, `ablation_table.csv`, `ablation_paired.csv`, `rganet_lambda.csv`, `metrics_appendix.csv`
and `COMPARISON.md`, written from the numbers: the best model per metric and whether the gap exceeds the seed spread,
RGA-Net vs each prior approach (seeds ahead, Wilcoxon p), the effect of ZTF pre-training and of the gate, and an honest
line about the classic models with observer features. The `report` step turns these into figures 08–13 with generated
"Inference:" captions (`outputs/figures/`, `figure_captions.md`).

## 7. Honest limits

- About 50 labelled AAVSO stars have windows, so each test set holds only a few stars per class. One star can move a
  metric a lot; that is why every result is a mean ± std over 10 splits and the comparisons are paired by seed.
- With 10 seeds the Wilcoxon test can only reach p < 0.05 when one model wins on most seeds; a larger p means the
  difference is not reliable, and `COMPARISON.md` says so.
- The 10 test sets overlap (each is 15 % of the same ~50 stars), so the seeds are not independent samples; the
  Wilcoxon p-values are a guide, not an exact significance level.
- If RGA-Net is not the best model, the tables and `COMPARISON.md` say so; nothing is re-run or re-tuned to change that.
