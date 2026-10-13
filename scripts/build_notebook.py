"""Builds notebooks/cosmic_colab.ipynb (the Colab deliverable) from the cell list below.

Why a generator: the notebook stays reviewable in git as plain Python, and every cell
is a thin call into the repository's scripts (no pipeline logic lives in the notebook).
Run:  python scripts/build_notebook.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CELLS = [
    ("md", """# Cosmic Event Crowdsourcing Platform: Review 2

Detects anomalous (eruptive) variable stars in crowdsourced **AAVSO** observations. The proposed model,
**RGA-Net** (Reliability-Gated Attention Network), is a Transformer encoder (Vaswani et al., 2017) whose attention
is gated by each observer's measured reliability (the gate is our contribution). It is compared with four Spark MLlib models on the same windows and
the same 5 star-grouped splits.

**Stack inside this Colab VM:** Hadoop 3.3.6 (HDFS + YARN, pseudo-distributed) · Spark 3.5.9 on YARN · Spark SQL ·
Hive 3.1.3 metastore · Spark MLlib · PyTorch (RGA-Net only, on the GPU). **Every graph is a Plotly figure**, shown
inline and saved as PNG (slides) + interactive HTML. The architecture diagram is drawn with Graphviz.

Use a **GPU runtime** (Runtime → Change runtime type → T4 GPU). Step-by-step guide: `docs/COLAB_GUIDE.md`.
Every path comes from one setting, `PROJECT_ROOT` (`MyDrive/cosmic_roja`): CSVs in `data/`, results in
`outputs/review2/` (shown at the end of the notebook), models, labels and the tarball cache in `work/`."""),
    ("md", "## 0. Mount Google Drive and get the code"),
    ("code", """from google.colab import drive
drive.mount('/content/drive')"""),
    ("code", """REPO_URL = "https://github.com/lowkey2556/cosmic-events.git"
import os
if not os.path.isdir('/content/cosmic/.git'):
    !git clone -q "$REPO_URL" "/content/cosmic"
else:
    !cd "/content/cosmic" && git pull -q
%cd /content/cosmic
!git log --oneline | head -5"""),
    ("md", """### Project folder and path check
Every path is derived from `PROJECT_ROOT`. The AAVSO CSVs must be **directly inside** `PROJECT_ROOT/data/`
(only that folder is read). This cell prints the resolved paths and **stops** if `data/` is missing, if any path points
at the retired folders (`MyDrive/cosmic`, `cosmic_OLD_do_not_use`), if CSVs are in sub-folders of `data/`, or if two
files have the same name. Stars are identified by `auid`, never by file name."""),
    ("code", """PROJECT_ROOT = "/content/drive/MyDrive/cosmic_roja"
import os, sys
os.environ["COSMIC_PROJECT_ROOT"] = PROJECT_ROOT      # every `!python run.py` and `!bash setup/...` below reads this
sys.path.insert(0, "/content/cosmic/src")
from cosmic.config import check_paths, load_config
from cosmic.rawfiles import list_csv_files
cfg = load_config("real")
print("Resolved paths:")
check_paths(cfg)                          # stops on a retired-folder path or a missing data/ folder
files = list_csv_files(cfg.data_dir)      # stops on CSVs in sub-folders or duplicate file names
assert files, f"No CSV files directly in {cfg.data_dir}"
print()
print(f"{len(files)} CSV files directly in {cfg.data_dir}, {sum(os.path.getsize(f) for f in files) / 2**20:.1f} MB")
OUT = cfg.paths()["outputs"]"""),
    ("code", """# GPU check (RGA-Net trains on the GPU if there is one; everything else uses the CPU)
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE - choose Runtime > Change runtime type > T4 GPU")
!nproc; free -g | head -2"""),
    ("md", """## Step 0: Dataset audit (plain Python, runs before the Big Data stack)
Per file and in total: size in MB, rows, stars, observers and date range; PASS/FAIL against 50 MB, both as stored
and after removing observations repeated across overlapping downloads; per star: rows, observers, dates, coverage,
windows; class balance; skipped files; unmatched VSX types. (1 MB = 1,048,576 bytes.)"""),
    ("code", """!pip -q install pyyaml astropy astroquery "plotly==5.24.1" "kaleido==0.2.1"
!python run.py audit --data real"""),
    ("md", """## 1. Install the stack
Java 8 comes from apt. Hadoop, Spark, Hive and Derby tarballs are cached in `PROJECT_ROOT/work/cache/`, so a reinstall
after a Colab reset only extracts them. The first run downloads about 1.5 GB."""),
    ("code", """%%time
!bash setup/install_stack.sh"""),
    ("md", "## 2. Start HDFS, YARN, the Derby network server and the Hive metastore"),
    ("code", """%%time
!bash setup/start_stack.sh"""),
    ("code", """# Load the stack's environment (JAVA_HOME, HADOOP_HOME, SPARK_HOME, HIVE_HOME, PATH...) into this kernel,
# so every later `!command` cell can call hdfs, yarn, spark-submit and hive directly.
import os, re
env_file = open('setup/ENV_LOCATION').read().strip()
for line in open(env_file):
    m = re.match(r'export (\\w+)=(.*)', line.strip())
    if m:
        os.environ[m.group(1)] = os.path.expandvars(m.group(2))
!java -version 2>&1 | head -1; hadoop version | head -1; spark-submit --version 2>&1 | grep -m1 version; hive --version 2>/dev/null | head -1"""),
    ("md", "## 3. Smoke tests (each must print PASS before the pipeline runs)"),
    ("code", "!bash setup/smoke_tests.sh 1   # hdfs dfs put + list"),
    ("code", "!bash setup/smoke_tests.sh 2   # YARN ResourceManager up"),
    ("code", "!bash setup/smoke_tests.sh 3   # SparkPi submitted to YARN"),
    ("code", "!bash setup/smoke_tests.sh 4   # Hive metastore: table created and queried"),
    ("code", "!bash setup/smoke_tests.sh 5   # PySpark-on-YARN Hive table read back through Hive"),
    ("md", """### Optional: open the web UIs
YARN ResourceManager is on 8088, the HDFS NameNode on 9870, and the Spark UI on 4040 while a job runs."""),
    ("code", """from google.colab import output
for port in (8088, 9870):
    output.serve_kernel_port_as_window(port)"""),
    ("md", "## 4. Architecture diagram"),
    ("code", """!which dot >/dev/null || apt-get -qq install -y graphviz   # Graphviz draws the diagram
!python run.py architecture --data real"""),
    ("md", """## 5. Step 1: Ingestion and storage (HDFS, Spark, Parquet, Hive)
`hdfs dfs -put` to `/cosmic/raw/` → explicit-schema read → Parquet + Snappy partitioned by `obs_year`/`sky_region` →
Hive external table `observations_raw` → VSX labels → partition-pruning test → dataset summary."""),
    ("code", "!python run.py ingest --data real"),
    ("code", """# The raw data in HDFS, and the partitions as Hive itself sees them
!hdfs dfs -ls -R -h "/cosmic/raw" | head -40
!hdfs dfs -du -s -h "/cosmic/raw" "/cosmic/warehouse/observations_raw" "/cosmic/warehouse/observations_flat"
!hive -S -e "SHOW PARTITIONS cosmic.observations_raw;" 2>/dev/null"""),
    ("md", """## 6. Step 2: Preprocessing in Spark SQL
Deduplication → null keys → comment codes → fainter-than split → **thinning** (≤ 10 points per observer per star per band
per night) → band normalisation. Rows are counted after every step, in total and per star."""),
    ("code", "!python run.py preprocess --data real"),
    ("md", "## 7. Step 3: Observer agreement score A_k"),
    ("code", "!python run.py score --data real"),
    ("md", "## 8. Step 4: Per-star features (used by K-Means)"),
    ("code", "!python run.py features --data real"),
    ("md", """## 9. Step 5: Windowing in Spark (the samples for every model)
100-day windows (stride 50 days, ≥ 20 observations) → per-window ML features (Spark SQL + Lomb–Scargle in
applyInPandas) and RGA-Net token sequences (L = 200) → Parquet on HDFS + Hive tables → 5 star-grouped 70/15/15
splits → export to `work/windows_seq.npz` for PyTorch. Also writes `star_attrition.csv` (every star: used or not, and why)."""),
    ("code", "!python run.py windows --data real"),
    ("code", """# The window tables in Hive
!hive -S -e "USE cosmic; SHOW TABLES;" 2>/dev/null
!hive -S -e "SELECT label, COUNT(*) FROM cosmic.window_features GROUP BY label;" 2>/dev/null"""),
    ("md", "## 10. Step 6: K-Means on per-star features (supporting analysis)"),
    ("code", "!python run.py cluster --data real"),
    ("md", """## 11. Step 7: The four ML models (Spark MLlib) on the windows
Naive Bayes (gaussian), Decision Tree, Random Forest, Logistic Regression; 5 splits; class weights via `weightCol`;
scaling fitted on train; grids tuned on validation; with and without observer features."""),
    ("code", """%%time
!python run.py classify --data real"""),
    ("md", """## 12. Step 8: RGA-Net and its ablations (PyTorch, GPU)
Full model, no-gate (λ = 0) and observer-blind, each on the same 5 splits. Early stopping on validation PR-AUC."""),
    ("code", """%%time
!python run.py rganet --data real"""),
    ("md", """## 12b. Step 8b: Distributed inference with Spark
Spark broadcasts the trained RGA-Net models (one per split seed) to the executors and scores **every** window in
parallel with `mapInPandas` (a pandas UDF), writing the Hive table `window_predictions`. The probabilities on the
validation/test windows are checked against PyTorch's own predictions."""),
    ("code", """%%time
!python run.py infer --data real"""),
    ("code", """# The predictions as Hive sees them
!hive -S -e "SELECT variant, seed, \\`set\\`, COUNT(*), ROUND(AVG(prob), 3) FROM cosmic.window_predictions GROUP BY variant, seed, \\`set\\` ORDER BY seed, \\`set\\`;" 2>/dev/null
!hive -S -e "SELECT window_id, auid, label, seed, \\`set\\`, ROUND(prob, 4) FROM cosmic.window_predictions LIMIT 10;" 2>/dev/null"""),
    ("md", "## 13. Step 9: Comparison (tables, figures, COMPARISON.md)"),
    ("code", "!python run.py compare --data real"),
    ("md", """## 14. Step 10: Scalability (YARN with 1, 2 and 4 executors, 3 runs each)
Re-runs preprocessing + windowing 9 times, so it takes a while. Run it last."""),
    ("code", """%%time
!python run.py scalability --data real"""),
    ("md", "# Review 2 results (all from the real data above)"),
    ("code", f"""import os
import pandas as pd
import plotly.io as pio
from IPython.display import Image, Markdown, display
OUT = os.path.join(os.environ["COSMIC_PROJECT_ROOT"], "outputs", "review2")   # from PROJECT_ROOT (cell at the top)
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 120)
def csv(name, title=None):
    display(Markdown(f'**{{title or name}}**'))
    path = f'{{OUT}}/{{name}}'
    display(pd.read_csv(path)) if os.path.exists(path) else display(Markdown(f'*{{name}}: not produced yet*'))
def fig(name):
    # show a saved Plotly figure inline (interactive); its PNG and HTML copies are next to it in OUT
    path = f'{{OUT}}/{{name}}.json'
    pio.read_json(path).show() if os.path.exists(path) else display(Markdown(f'*{{name}}: not produced yet*'))
def png(name, width=900):
    display(Image(f'{{OUT}}/{{name}}', width=width))
if os.path.exists(f'{{OUT}}/figure_export_log.csv'):
    log = pd.read_csv(f'{{OUT}}/figure_export_log.csv')
    bad = log[~log['png'].astype(bool)]
    print('PNG export: all OK' if bad.empty else f'PNG export FAILED for {{len(bad)}} figures (HTML kept):')
    if not bad.empty: display(bad)"""),
    ("md", "## Dataset"),
    ("code", """csv('audit_summary.csv', 'Dataset audit: summary (50 MB check)')
csv('dataset_summary.csv', 'Dataset summary after ingestion')
csv('audit_class_balance.csv', 'Class balance'); csv('audit_skipped_files.csv', 'Skipped files (bad header)')
csv('audit_unmatched_vsx_types.csv', 'Unmatched VSX types')
csv('rows_per_band.csv', 'Rows per band'); csv('rows_per_type.csv', 'Rows per observation type')
fig('fig_obs_per_star')"""),
    ("md", "## Architecture"),
    ("code", "png('architecture.png', 1100)   # Graphviz PNG (SVG and DOT source saved next to it)"),
    ("md", "## Preprocessing results"),
    ("code", """csv('preprocess_row_counts.csv', 'Rows after each step (thinning is step 5)'); fig('fig_preprocessing_funnel')
csv('partition_pruning.csv', 'Partition pruning: partitioned vs unpartitioned'); fig('fig_partition_pruning')
csv('lightcurve_star.csv', 'Light-curve star'); fig('fig_lightcurve_before_after')
csv('observer_score_summary.csv', 'Observer scores: summary'); fig('fig_observer_score_hist')
csv('observer_top_bottom10.csv', 'Top and bottom 10 observers by agreement score')
csv('spearman_score_vs_uncertainty.csv', 'Spearman: score vs self-reported uncertainty')"""),
    ("md", "## Windows, splits and star attrition"),
    ("code", """csv('window_summary.csv', 'Windows'); csv('window_counts_by_class.csv', 'Windows per class')
csv('window_counts_by_split.csv', 'Windows per split and class (every seed)')
att = pd.read_csv(f'{OUT}/star_attrition.csv')
print(att['status'].value_counts().to_string())
display(att[att['status'] != 'used'][['auid', 'name', 'vsx_type', 'status', 'reason']])"""),
    ("md", "## Main result: RGA-Net vs the four ML models (5 star-grouped splits)"),
    ("code", """csv('comparison_table.csv', 'Test windows, mean ± std over 5 splits (positive class = anomalous)')
csv('comparison_table_star_level.csv', 'Test stars (mean window probability), mean ± std over 5 splits')
fig('fig_comparison_metrics'); fig('fig_comparison_metrics_star_level')
fig('fig_roc_pr_curves'); fig('fig_confusion_matrices')"""),
    ("md", "## RGA-Net: training curves, ablation, learned λ"),
    ("code", """fig('fig_rganet_training_curves')
csv('ablation_table.csv', 'Ablation, test windows (mean ± std)'); fig('fig_ablation')
csv('rganet_learned_lambda.csv', 'Learned gate strengths λ (full model)')"""),
    ("md", "## Distributed inference (Spark applies RGA-Net to all windows)"),
    ("code", """csv('distributed_inference_check.csv', 'Spark vs PyTorch probabilities (validation/test windows)')
csv('window_predictions_summary.csv', 'Hive table window_predictions: windows and mean probability per seed and set')"""),
    ("md", "## Supplementary: ML models with vs without observer features"),
    ("code", "csv('ml_observer_features_table.csv', 'ML models, test windows (mean ± std)'); csv('ml_selected_params.csv', 'Chosen settings and training time')"),
    ("md", "## COMPARISON.md (generated from the numbers above)"),
    ("code", """p = f'{OUT}/COMPARISON.md'
display(Markdown(open(p, encoding='utf-8').read())) if os.path.exists(p) else print('run the compare step first')"""),
    ("md", "## Scalability"),
    ("code", "csv('scalability.csv', 'Preprocessing + windowing through YARN'); fig('fig_scalability')"),
    ("md", "## Supporting: K-Means"),
    ("code", """csv('kmeans_silhouette.csv', 'K-Means silhouette'); fig('fig_kmeans_silhouette')
csv('kmeans_crosstab_label.csv', 'Cluster vs label')"""),
    ("code", '!ls -la "$OUT"'),
]


def build():
    cells = []
    for kind, src in CELLS:
        lines = src.split("\n")
        source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source})
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}, "accelerator": "GPU",
                                       "colab": {"provenance": [], "gpuType": "T4"}},
          "nbformat": 4, "nbformat_minor": 0}
    path = os.path.join(ROOT, "notebooks", "cosmic_colab.ipynb")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
    print(f"wrote {path} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
