# Review 2 checklist: slide → exact files

All result files are produced **from the real AAVSO data** by `python run.py milestone1 --data real`
(one notebook cell per step). They are saved in `MyDrive/cosmic/outputs/review2/` and shown inline in
`notebooks/cosmic_colab.ipynb`. Nothing in `review2/` comes from synthetic data: synthetic runs
write to `synthetic_run/`, and the code refuses to write synthetic output to `review2/`.

| Slide | Item | File(s) | Produced by |
|---|---|---|---|
| **1. Dataset** | Total rows, stars, observers, date range, label counts | `dataset_summary.csv`, `label_counts.csv` | Step 1 (`ingest.dataset_summary`) |
| | Rows per band / per observation type | `rows_per_band.csv`, `rows_per_type.csv` | Step 1 |
| | Bar chart of observations per star | `fig_obs_per_star.png` (+ `obs_per_star.csv`) | Step 1 |
| | Raw data in HDFS (`hdfs dfs -ls`) | notebook cell output under "Step 1" (also printed by `ingest`) | notebook |
| | Hive `SHOW PARTITIONS` | notebook cell output (Hive CLI) + `hive_partitions.csv` (Spark SQL) | notebook / Step 1 |
| **2. Proposed Model** | Plain-English description + equations | `docs/PROPOSED_MODEL.md` | — |
| **3. Architecture** | Diagram, 300 dpi, Matplotlib | `architecture.png` (also `docs/figures/architecture.png`) | `run.py architecture` |
| **4. Preprocessing info** | Each Big Data step in plain English | `docs/PREPROCESSING.md` | — |
| **5. Preprocessing results** | Row counts after each step: table + funnel | `preprocess_row_counts.csv`, `fig_preprocessing_funnel.png` | Step 2 |
| | Partition pruning: files and bytes scanned | `partition_pruning.csv`, `fig_partition_pruning.png` | Step 1 |
| | One star's light curve before/after band normalisation (y inverted, by observer) | `fig_lightcurve_before_after.png`, `lightcurve_star.csv` | Step 2 |
| | Observer-score histogram coloured by type | `fig_observer_score_hist.png` (+ `observer_scores.csv`, `observer_score_summary.csv`) | Step 3 |
| | Top and bottom 10 observers | `observer_top_bottom10.csv` | Step 3 |
| | Spearman: score vs self-reported uncertainty, with n | `spearman_score_vs_uncertainty.csv` | Step 3 |
| **6. Classification** | Model, features, split, metrics in plain English | `docs/CLASSIFICATION.md` | — |
| **7. Proposed model results** | Metrics table | `proposed_model_metrics.csv` (+ `split_summary.csv`, `split_stars.csv`, `feature_log.csv`) | Step 5 |
| | Confusion matrix figure | `fig_confusion_proposed.png` | Step 5 |
| | K-Means silhouette plot + cluster-vs-label crosstab | `fig_kmeans_silhouette.png`, `kmeans_silhouette.csv`, `kmeans_crosstab_label.csv`, `kmeans_crosstab_type.csv` | Step 6 |
| **8. Comparison** | One table: NB, DT, LR, RF (baseline) + NB, DT (+ observer) — precision, recall, F1, balanced accuracy | `model_metrics.csv` | Step 5 |
| | Grouped bar chart | `fig_model_comparison.png` | Step 5 |
| | Side-by-side confusion matrices: DT baseline vs proposed | `fig_confusion_dt_vs_proposed.png` | Step 5 |
| **12. Colab link** | Notebook run top to bottom with outputs saved | `notebooks/cosmic_colab.ipynb` (shared from Colab after **Run all** + **File → Save**) | you |

Smoke-test evidence (cells under "3. Smoke tests" in the notebook): HDFS put/list, YARN RM up,
SparkPi on YARN, Hive metastore table create and query, and a PySpark-on-YARN table read back
through Hive.

## Honest status of the Milestone 1 results

- The classification results use **one fixed seed (42) on a small test set** (about 30% of the
  labelled stars: 10 stars, 5 of them anomalous). A single star moves recall by 20 points.
  They are **preliminary** and must be presented that way.
- **Milestone 2** adds the 5-seed robustness (mean ± std), model interpretation (feature
  importances, `toDebugString`, whether observer features are used in splits), error analysis,
  the YARN scalability test, the Tableau exports and `RESULTS.md`.
- If the observer features do not improve on the baseline, the comparison table will show that,
  and we report it as it is.
