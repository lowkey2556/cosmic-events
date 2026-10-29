# Review 2 checklist (v3): every requirement → exact files

This is an expected-output checklist, not a record of a verified real-data run. Real model results still require
execution and inspection in Colab. When run on **real AAVSO and ZTF data**, the notebook (`notebooks/cosmic_colab.ipynb`;
`python run.py <step> --data real`) writes the files under **`MyDrive/cosmic_roja/v3_5GB/`** (`outputs/`, `outputs/figures/`,
`dashboard_data/`). The notebook's last section, **▶️ RESULTS — start here**, shows all of them and ends with
`ALL RESULTS SHOWN`. Charts are Plotly (PNG + HTML + JSON); **no bar charts or histograms** (enforced by a test). Old v2
files are moved to `outputs/retired_v2/` by the `report` step and are not results. Synthetic data never reaches these
folders: synthetic runs write to `synthetic_run/`.

## 1. Data at Big Data scale (Phase 1–2)

| Item | File(s) | Step |
|---|---|---|
| AAVSO ≥ 50 MB (total and after removing repeats) | `audit_summary.csv`, `audit_files.csv`, `audit_stars.csv` | `audit` |
| ZTF: 117 files, 8.53 GB, 4.1 M light curves, 841 M epochs; 25 of 60 AAVSO stars in ZTF with ≥ 20 unsaturated epochs | `ztf_audit_summary.csv`, `ztf_audit_files.csv`, `ztf_aavso_stars.csv` | `ztf_audit` |
| ZTF in HDFS + partitioned Hive table (field × filter) | Hive `ztf_lightcurves_raw`; DATA_FLOW stage 05 | `ztf_ingest` |
| ZTF cleaned, step by step; ≥ 5 GB check; Drive copy | `ztf_clean_steps.csv`, `ztf_clean_size_check.csv`, `data_quality_ztf.csv`; Hive `ztf_lightcurves_clean`; `v3_5GB/clean/` | `ztf_clean` |
| ZTF labels from VSX (2″; VizieR, else HEASARC `aavsovsx`); AAVSO overlaps excluded | `ztf_label_counts.csv`, `composition_ztf.csv`; Hive `ztf_labels` | `ztf_label` |
| Class-balanced ZTF windows (≤ 50,000) | `ztf_window_counts.csv`; Hive `ztf_window_features`, `ztf_window_sequences`; `work/ztf_windows_seq.npz` | `ztf_windows` |
| A_k validated against ZTF (Spearman ρ, n) | `ak_validation_summary.csv`, `ak_vs_ztf_observers.csv`; figure 06 | `akval` |
| Where the data is at every stage, with measured sizes and times | `DATA_FLOW.txt`, `step_timings.csv`; explained in [DATA_FLOW_EXPLAINED.md](DATA_FLOW_EXPLAINED.md) | `report` |

## 2. Preprocessing results (tables and charts)

| Item | File(s) |
|---|---|
| Raw vs cleaned samples (AAVSO, ZTF) | figure 01 (four tables); `sample_aavso_*.csv`, `sample_ztf_*.csv` |
| Data quality before vs after (size, rows, nulls, duplicates, flags, bands, stars, observers) | figure 02; `data_quality_aavso.csv`, `data_quality_ztf.csv` |
| Rows after every cleaning step, AAVSO and ZTF | figure 03 (Sankey); `preprocess_row_counts.csv`, `ztf_clean_steps.csv` |
| Every star: used or not, and why | `star_attrition.csv` |
| Partition pruning | `partition_pruning.csv` |
| Observer reliability A_k | figures 04 (light curve coloured by A_k) and 05 (violin by observation type); `observer_scores.csv`, `ak_summary_by_type.csv`, `spearman_score_vs_uncertainty.csv` |
| Composition source → class → VSX type | figure 07 (sunburst); `composition_aavso.csv`, `composition_ztf.csv` |
| Method in plain English | [PREPROCESSING.md](PREPROCESSING.md) |

## 3. The proposed deep learning model (RGA-Net)

| Item | File(s) |
|---|---|
| Model, equations, why RGA-Net, related work | [PROPOSED_MODEL.md](PROPOSED_MODEL.md) |
| Implementation | `src/cosmic/rganet.py` |
| ZTF pre-training (hold-out AUPRC, epochs, time) | `rganet_pretraining.csv`; `work/models/pretrained_{full,blind}.pt` |
| Validation-grid selection separately within each split | `rganet_grid.csv`; tuning excludes that split's test stars |
| 4 variants × 10 seeds: best epoch, time, parameters, learned λ | `rganet_runs.csv`, `rganet_history.csv`, `rganet_lambda.csv` |
| Ablation | figure 12; `ablation_table.csv`, `ablation_paired.csv` |
| Trained models; Spark applies them to all windows | `work/models/rganet_<variant>_seed<n>.pt`; Hive `window_predictions`; `distributed_inference_check.csv` |

## 4. Comparison with the four ML algorithms (Phase 3)

| Item | File(s) |
|---|---|
| NB, DT, RF, LR in Spark MLlib; with/without observer features; AAVSO-only vs AAVSO + ZTF chosen on validation | `src/cosmic/classify.py`; `ml_selected_params.csv`; `work/predictions_ml.csv`; Hive `ml_predictions` |
| MCC (threshold = max validation MCC), ROC-AUC, AUPRC, Cohen's κ, Brier; window and star level; mean ± std over 10 seeds | figure 08; `comparison_table.csv`, `per_seed_metrics.csv`; appendix `metrics_appendix.csv` |
| Proposed vs prior approaches (classic models **without** observer features only; Fix 1), paired Wilcoxon | figures 09, 10, 13; `prior_vs_proposed_table.csv`, `prior_vs_proposed_paired.csv` |
| Classic models with observer features (our framework, not prior work) | figure 11; `observer_feature_gain.csv` |
| Written comparison, generated from saved predictions only | `COMPARISON.md` |
| Method | [CLASSIFICATION.md](CLASSIFICATION.md) |

## 5. Phase 4 deliverables

| Item | File(s) |
|---|---|
| 13 figures, each with a generated "Inference:" caption | `outputs/figures/01…13_*.png/.html/.json`; `figure_captions.md`, `figure_captions.json` |
| One colour/naming config for every chart, table and the dashboard | `config/style.yaml` (copied to `dashboard_data/`) |
| Dashboard-ready tidy tables + data dictionary | `dashboard_data/*.csv`, `*.parquet`; Hive `dash_*`; `dashboard_data/DATA_DICTIONARY.md` |
| Scalability (YARN 1/2/4 executors × 3 runs) — table only | `scalability.csv`, `scalability_runs.csv` |
| Architecture diagram | `architecture.png/.svg/.dot` |

## The rules, and where each is enforced

| Rule | Where (code; test) |
|---|---|
| Split by star; no star in two sets; 10 splits shared by every model | `splits.py`; `test_star_split_disjoint_stratified_reproducible` |
| Nothing chosen on test (grids, data setting, early stopping, thresholds) | `classify.py`, `rganet.choose_config`, `evaluation.choose_threshold`; `test_threshold_comes_from_validation_only` |
| ZTF never in validation/test; no ZTF object near an AAVSO star in training | `classify.fit_predict`, `ztf_spark.match_labels`; `test_leakage_ztf_windows_only_ever_train_the_ml_models`, `test_leakage_ztf_objects_near_aavso_stars_are_flagged_and_match_radius` |
| Augmentation and token scaling from training windows only | `rganet.fit`, `rganet.train_one`; `test_leakage_rganet_statistics_and_augmentation_use_training_windows_only`. Observer scores are archive-wide and transductive; see PROPOSED_MODEL. |
| "Prior approach" = classic model without observer features only (Fix 1) | `compare.role`, `compare.prior_vs_proposed`; `test_fix1_prior_approaches_are_classic_models_without_observer_features` |
| Comparison only from saved predictions | `compare.load_predictions`; `test_compare_end_to_end_on_synthetic_predictions` |
| Metrics correct | `evaluation.py`; `test_mcc_kappa_brier_by_hand_and_against_sklearn`, `test_metrics_by_hand`, `test_paired_wilcoxon_matches_scipy` |
| Every chart has a generated caption; missing inputs listed, never faked | `report.py`, `plots.py`; `test_every_chart_has_a_generated_inference_caption`, `test_report_makes_figures_and_captions_from_saved_files_and_retires_v2` |
| DATA_FLOW from measurements only | `dataflow.py`; `test_data_flow_is_generated_from_recorded_measurements` |
| Fresh machine: missing tables rebuilt, every path from PROJECT_ROOT | `pipeline.py`, `config.py`; `test_fresh_machine_*`, `test_every_path_derives_from_project_root` |
| No bar charts/histograms; no v2 results in the notebook; notebook up to date | `test_no_bar_charts_histograms_or_v2_results_anywhere`, `test_notebook_runs_every_step_in_order_and_ends_with_the_results` |
| Plotly only | `test_no_matplotlib_or_seaborn_anywhere` |
