"""Builds notebooks/cosmic_colab.ipynb (the Colab deliverable) from the cell list below.

Why a generator: the notebook stays reviewable in git as plain Python, and every cell
is a thin call into the repository's scripts (no pipeline logic lives in the notebook).
Run:  python scripts/build_notebook.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CELLS = [
    ("md", """# Cosmic Event Crowdsourcing Platform: Review 2 (v3)

Detects anomalous (eruptive) variable stars in crowdsourced **AAVSO** observations. The proposed model, **RGA-Net**
(Reliability-Gated Attention Network), is a Transformer encoder (Vaswani et al., 2017) whose attention is gated by each
observer's measured reliability (the gate is our contribution). **ZTF** public light curves add Big Data scale,
pre-training data and an independent check of observer reliability.

**Stack inside this Colab VM:** Hadoop 3.3.6 (HDFS + YARN) · Spark 3.5.9 on YARN · Spark SQL · Hive 3.1.3 · Spark MLlib ·
PyTorch (RGA-Net only). Charts: Plotly. Diagram: Graphviz.

**How to run:** use a GPU runtime (Runtime → Change runtime type → T4 GPU), run the **SETUP** cell first, then the
sections in order. Guide: `docs/COLAB_GUIDE.md`. Everything new is written under `cosmic_roja/v3_5GB/`; `data/` and
`work/cache/` are only read."""),
    ("md", """## SETUP — run this first, every session
Mounts Drive, sets **every** path and environment variable, gets or updates the code, and installs Python: a Python 3.12
environment for the Spark steps (PySpark 3.5 does not support Python 3.13) and the notebook's own packages. It then prints
the resolved paths and **stops** if `data/` is missing or holds CSVs in sub-folders or duplicate names, or if a path points
at a retired folder (`MyDrive/cosmic`, `cosmic_OLD_do_not_use`, `_OLD_100MB`). Takes 2–4 minutes."""),
    ("code", """PROJECT_ROOT = "/content/drive/MyDrive/cosmic_roja"
REPO_URL = "https://github.com/lowkey2556/cosmic-events.git"
import os, subprocess, sys
from google.colab import drive
drive.mount("/content/drive")
os.environ.update(COSMIC_PROJECT_ROOT=PROJECT_ROOT, V3_ROOT=f"{PROJECT_ROOT}/v3_5GB", CACHE_DIR=f"{PROJECT_ROOT}/work/cache",
                  COSMIC_SPARK_PY="/content/cosmic_py/bin/python", PYTHON="/content/cosmic_py/bin/python")

def sh(cmd):
    # run a shell command, show its output live, stop the notebook if it fails
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="")
    if p.wait():
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}")

if os.path.isdir("/content/cosmic/.git"):
    sh("git -C /content/cosmic pull -q")
else:
    sh(f"git clone -q {REPO_URL} /content/cosmic")
os.chdir("/content/cosmic")
sys.path.insert(0, "/content/cosmic/src")
sh("git log --oneline | head -3")
sh("bash setup/python_env.sh")
from cosmic.config import check_paths, load_config
from cosmic.rawfiles import list_csv_files
cfg = load_config("real")
print("\\nResolved paths:")
check_paths(cfg)
files = list_csv_files(cfg.data_dir)
assert files, f"No CSV files directly in {cfg.data_dir}"
print(f"{len(files)} AAVSO CSV files in data/, {sum(os.path.getsize(f) for f in files) / 2**20:.1f} MB")
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE (Runtime > Change runtime type > T4 GPU)")
print("Python", sys.version.split()[0], "| CPU cores:", os.cpu_count())
print("SETUP OK")"""),
    ("md", """## Phase 1 — ZTF light curves and new AAVSO candidates
No Big Data stack needed for these four cells. **Run them in order**; each can be re-run safely.

1. **Choose ZTF files**: cone-searches ZTF at every AAVSO star, then picks the light-curve files (tiles) that contain
   our stars plus neighbouring tiles rich in labelled VSX variables, up to 8–10 GB. Logs every choice and the reason.
2. **Download**: resumable and MD5-verified; files already in Drive are skipped, so after a disconnect just run it again.
3. **ZTF audit**: files, GB, rows, objects, filters, date range, and our AAVSO stars found in ZTF (unsaturated epochs).
4. **Candidate stars** for more AAVSO downloads (VSX only)."""),
    ("code", "!python run.py ztf_select     # about 5-15 min (one ZTF query per star, file sizes, VSX counts)"),
    ("code", "!python run.py ztf_download   # about 10-40 min for 8-10 GB; prints progress and time left"),
    ("code", "!python run.py ztf_audit      # about 5-15 min (reads every downloaded file)"),
    ("code", "!python run.py candidates     # about 5-10 min (VSX queries); writes v3_5GB/outputs/aavso_candidate_stars.csv"),
    ("md", """## Step 0: AAVSO dataset audit (plain Python, runs before the Big Data stack)
Per file and in total: size in MB, rows, stars, observers and date range; PASS/FAIL against 50 MB, both as stored
and after removing observations repeated across overlapping downloads; per star: rows, observers, dates, coverage,
windows; class balance; skipped files; unmatched VSX types. (1 MB = 1,048,576 bytes.)"""),
    ("code", "!python run.py audit --data real"),
    ("md", """## 1. Install the stack
Java 8 comes from apt. Hadoop, Spark, Hive and Derby tarballs are read from `PROJECT_ROOT/work/cache/` (the cell prints
`using cached <file>` for each), so they are only downloaded if they are not there yet."""),
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
OUT = os.path.join(os.environ.get("V3_ROOT", "/content/drive/MyDrive/cosmic_roja/v3_5GB"), "outputs")
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
