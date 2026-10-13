# Review 2 checklist: every requirement → exact files

All result files are produced **from the real AAVSO data** by the notebook
(`notebooks/cosmic_colab.ipynb`, one cell per step; `python run.py <step> --data real`). They are
saved in **`MyDrive/cosmic_roja/outputs/review2/`** and shown inline at the end of the notebook.
Every graph is a **Plotly** figure saved as `fig_*.png` (slides, via kaleido), `fig_*.html`
(interactive) and `fig_*.json`; `figure_export_log.csv` records any PNG export failure.
Nothing in `review2/` comes from synthetic data: synthetic runs write to `synthetic_run/`, and
the code refuses to write synthetic output to any `review2/` folder.

## Requirement 1 — Preprocessing results on the full dataset (tables and graphs)

| Item | File(s) | Produced by |
|---|---|---|
| Dataset summary: size in MB, rows, stars, observers, bands, observation types, date range | `dataset_summary.csv` (+ `audit_summary.csv`) | Step 1 `ingest.dataset_summary`, Step 0 `audit` |
| Rows remaining after each cleaning step (thinning shown as its own step, with % removed) | `preprocess_row_counts.csv`, `fig_preprocessing_funnel.png/.html` | Step 2 `preprocess` |
| Rows per star after each step | `preprocess_rows_per_star.csv` (Hive `preprocess_star_log`) | Step 2 |
| Every star: used or not, and exactly why | `star_attrition.csv` | Step 5 `windows` |
| Partition-pruning comparison | `partition_pruning.csv`, `fig_partition_pruning.png/.html` | Step 1 |
| Observer-score distribution | `fig_observer_score_hist.png/.html`, `observer_score_summary.csv`, `observer_scores.csv`, `observer_top_bottom10.csv`, `spearman_score_vs_uncertainty.csv` | Step 3 `observer_score` |
| One light curve before/after normalisation (magnitude axis reversed) | `fig_lightcurve_before_after.png/.html`, `lightcurve_star.csv` | Step 2 |
| Observations per star; rows per band and type | `fig_obs_per_star.png/.html`, `rows_per_band.csv`, `rows_per_type.csv` | Step 1 |
| Windows per class and per split | `window_summary.csv`, `window_counts_by_class.csv`, `window_counts_by_split.csv` | Step 5 |
| Plain-English description of every step | `docs/PREPROCESSING.md` | — |

## Requirement 2 — A novel deep learning model, implemented, trained and evaluated

| Item | File(s) |
|---|---|
| Model description, equations, why RGA-Net, related work, references | `docs/PROPOSED_MODEL.md` |
| Implementation (PyTorch) | `src/cosmic/rganet.py` (model, training, ablations) |
| Training runs: best epoch, time, parameters, learned λ | `rganet_runs.csv`, `rganet_learned_lambda.csv` |
| Training and validation curves | `rganet_history.csv`, `fig_rganet_training_curves.png/.html` |
| Ablation: full vs no-gate vs observer-blind | `ablation_table.csv`, `ablation_table_star_level.csv`, `fig_ablation.png/.html` |
| Predictions (validation and test windows, every seed and variant) | `predictions_rganet.csv` |
| Trained models | `work/models/rganet_<variant>_seed<seed>.pt` |
| Distributed inference: Spark applies the trained model to all windows | `src/cosmic/infer.py`; Hive table `window_predictions`; `window_predictions_summary.csv`; `distributed_inference_check.csv` (Spark vs PyTorch) |

## Requirement 3 — Comparison with 4 ML algorithms (table and graphs)

| Item | File(s) |
|---|---|
| The 4 ML models (Spark MLlib: Gaussian NB, DT, RF, LR) | `src/cosmic/classify.py`; chosen settings + training time in `ml_selected_params.csv`; predictions in `predictions_ml.csv` |
| Comparison table, all metrics, mean ± std over 5 splits, training time | `comparison_table.csv` (window level), `comparison_table_star_level.csv` |
| Every per-seed number | `metrics_per_seed.csv` |
| Grouped metric bars with error bars | `fig_comparison_metrics.png/.html`, `fig_comparison_metrics_star_level.png/.html` |
| ROC and PR curves, all 5 models, one seed, shared axes | `fig_roc_pr_curves.png/.html` |
| Confusion-matrix grid, all 5 models | `fig_confusion_matrices.png/.html` |
| ML with vs without observer features (supplementary) | `ml_observer_features_table.csv` |
| Written comparison: best per metric, margin, vs seed-to-seed spread | `COMPARISON.md` (generated from the numbers) |
| Method in plain English | `docs/CLASSIFICATION.md` |

## Requirement 4 — At least 50 MB of real AAVSO data

| Item | File(s) |
|---|---|
| Size per file and in total; PASS/FAIL against 50 MB, both total and after removing repeated observations | `audit_summary.csv`, `audit_files.csv` |
| Per star rows, observers, dates, coverage, windows; class balance | `audit_stars.csv`, `audit_class_balance.csv` |
| Skipped files (bad header) and unmatched VSX types | `audit_skipped_files.csv`, `audit_unmatched_vsx_types.csv` |

## Requirement 5 — Everything before visualisation complete

| Stage | Code | Hive tables / files |
|---|---|---|
| 0 Audit | `src/cosmic/audit.py`, `rawfiles.py` | `audit_*.csv` |
| 1 Ingestion (HDFS, Spark, Parquet, Hive) | `ingest.py`, `labels.py` | `observations_raw`, `observations_flat`, `star_labels` |
| 2 Preprocessing (Spark SQL) | `preprocess.py` | `obs_clean`, `obs_upper_limits`, `preprocess_log`, `preprocess_star_log` |
| 3 Observer score (Spark SQL) | `observer_score.py` | `observer_residuals`, `observer_scores` |
| 4 Per-star features | `features.py` | `star_features` |
| 5 Windowing (Spark) | `windows.py`, `windowing.py`, `splits.py` | `windows`, `window_features`, `window_sequences`, `splits`; `work/windows_seq.npz` |
| 6 K-Means (MLlib, supporting) | `cluster.py` | `star_clusters` |
| 7 ML models (MLlib) | `classify.py` | `ml_predictions` |
| 8 RGA-Net (PyTorch) | `rganet.py` | `predictions_rganet.csv`, `work/models/*.pt` |
| 8b Distributed inference (Spark `mapInPandas`) | `infer.py` | `window_predictions` |
| 9 Comparison | `compare.py`, `evaluation.py` | tables, figures, `COMPARISON.md` |
| 10 Scalability (YARN, 1/2/4 executors × 3 runs) | `scalability.py` | `scalability.csv`, `scalability_runs.csv`, `fig_scalability.png/.html` |
| Architecture diagram (Graphviz) | `architecture.py` | `architecture.png/.svg/.dot` (copies in `docs/figures/`) |

Not built yet, by design: Tableau exports, dashboards and final visualisations.

## The rules, and where each is enforced

| Rule | Where |
|---|---|
| Split by star everywhere; no star in two sets | `splits.py` (`star_split`, `check_disjoint`); one `splits.csv` shared by every model; tests `test_star_split_*`, `test_classify_fit_predict_on_windows`, `test_rganet_trains_and_ablations_run` |
| Nothing tuned on test | ML grids and RGA-Net early stopping use validation PR-AUC; thresholds from `evaluation.choose_threshold` on validation; test `test_threshold_comes_from_validation_only` |
| Same windows, splits and seeds for every model | `windows.py` builds them once; `classify.py` and `rganet.py` both read them |
| Synthetic data never in results | `config.py` refuses `review2` for synthetic runs; test `test_synthetic_outputs_never_go_to_review2` |
| No duplicated, padded or synthetic rows in the dataset | the audit reports repeated rows and the size **after** removing them |
| Big Data stack does ingestion, preprocessing, windowing, features and the 4 ML models | `ingest.py`, `preprocess.py`, `observer_score.py`, `windows.py`, `classify.py` (Spark, Spark SQL, Hive, MLlib on YARN) |
| PyTorch only for the deep learning model | `rganet.py` (training); `infer.py` runs the same model inside Spark executors |
| Plotly only (no Matplotlib/Seaborn) | `plots.py`; test `test_no_matplotlib_or_seaborn_anywhere` |

Smoke-test evidence (notebook section 3): HDFS put/list, YARN ResourceManager up, SparkPi on YARN,
Hive metastore table create/query, and a PySpark-on-YARN table read back through Hive.
