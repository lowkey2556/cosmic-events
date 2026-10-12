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

## 2. Splits: by star, 5 times

- **70 % train / 15 % validation / 15 % test, by star** (`auid`), stratified by class, so
  every set has both classes. All windows of a star go to the same set: no star is ever in two
  sets (checked in code).
- **5 seeds → 5 different splits.** The same 5 splits (saved once as `splits.csv` / Hive table
  `splits`) are used by RGA-Net and by all four ML models.
- **The test set is never used to choose anything.** Hyperparameters, early stopping and the
  decision threshold all use the validation set.

Why by star: windows of the same star look alike. If a star had windows in train and test, the
test score would reward recognising that star, not recognising a new one.

## 3. The models

| Model | Library | Inputs | Chosen on validation |
|---|---|---|---|
| **RGA-Net (proposed)** | PyTorch (Colab GPU) | token sequences | early-stopping epoch (best validation PR-AUC); threshold |
| Gaussian Naive Bayes (`modelType="gaussian"`) | Spark MLlib | 13 summary features | threshold (no hyperparameters) |
| Decision Tree | Spark MLlib | 13 summary features | maxDepth ∈ {3, 5, 8} × minInstancesPerNode ∈ {1, 10}; threshold |
| Random Forest | Spark MLlib | 13 summary features | numTrees = 100, maxDepth ∈ {5, 10}; threshold |
| Logistic Regression | Spark MLlib | 13 summary features | regParam ∈ {0.001, 0.01, 0.1}; threshold |

For the ML models: median imputation and standardisation (mean and std) are **fitted on the
training windows only**; class imbalance is handled with `weightCol` (weight = n_train /
(2 · n_class) on the training windows); the grid setting with the best **validation PR-AUC** is
kept. RGA-Net uses the equivalent class weighting in its loss. Details of RGA-Net:
[PROPOSED_MODEL.md](PROPOSED_MODEL.md).

## 4. Metrics

Every model writes the predicted probability of "anomalous" for each validation and test
window; **one shared implementation** (`src/cosmic/evaluation.py`) then computes all metrics.

- **Threshold:** the probability cut-off with the best F1 on the **validation** windows.
- **On the test windows, for the anomalous class:** precision, recall, F1, balanced accuracy
  ((recall + specificity) / 2), ROC-AUC and PR-AUC (average precision). Plain accuracy is not
  headlined because the classes are not balanced.
- **Star level:** a star's probability = the mean of its windows' probabilities; threshold
  chosen on the validation stars; the same metrics on the test stars. This is the stricter view,
  because windows of one star are correlated.
- **Reported as mean ± std over the 5 splits**, plus training time.

## 5. Ablations and supplementary runs

| Run | Question it answers |
|---|---|
| RGA-Net full vs **no gate** (λ = 0, A_k still a token feature) | Does gating attention by reliability help, beyond giving A_k as an input? |
| no gate vs **observer-blind** (A_k and offset removed) | Does observer information help at all, as a plain input? |
| Each ML model **with vs without** the 3 observer features | Does observer information help the classical models? |

All runs use the same windows, splits, seeds and settings.

## 6. Outputs (in `outputs/review2/`)

`comparison_table.csv` (window level) and `comparison_table_star_level.csv`; `ablation_table.csv`;
`ml_observer_features_table.csv`; `metrics_per_seed.csv` (every number before averaging);
`rganet_learned_lambda.csv`; figures `fig_comparison_metrics`, `fig_roc_pr_curves`,
`fig_confusion_matrices`, `fig_rganet_training_curves`, `fig_ablation` (each `.png` + `.html`);
and `COMPARISON.md`, written from the numbers, which states which model is best on which metric,
by how much, and whether the gap is larger than the seed-to-seed spread.

## 7. Honest limits

- About 60 stars in total, so each test set holds only a few stars per class. One star can move
  recall or precision a lot; that is why every result is a mean ± std over 5 splits.
- "Larger than the seed-to-seed spread" is a rough check (gap in means vs the larger standard
  deviation, plus how many of the 5 seeds the better model wins), not a formal significance test.
- If RGA-Net is not the best model, the tables and `COMPARISON.md` say so; nothing is re-run
  or re-tuned to change that.
