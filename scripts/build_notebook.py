"""Builds notebooks/cosmic_colab.ipynb (the Colab deliverable) from the cell list below.

Why a generator: the notebook stays reviewable in git as plain Python, and every cell
is a thin call into the repository's scripts (no pipeline logic lives in the notebook).
Run:  python scripts/build_notebook.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CELLS = [
    ("md", """# Cosmic Event Crowdsourcing Platform: Milestone 1 (Review 2)

A Spark pipeline that detects anomalous celestial events in crowdsourced **AAVSO** variable-star
observations, using each observer's computed reliability as an explicit classifier feature.

**Stack running inside this Colab VM:** Hadoop 3.3.6 (HDFS + YARN, pseudo-distributed) · Spark 3.5.9 on YARN ·
Hive 3.1.3 metastore on a Derby network server · Spark SQL · Spark MLlib · PyTorch (deep learning). **Every graph is a Plotly figure**, shown inline and saved as PNG (slides) + interactive HTML. The architecture diagram is drawn with Graphviz.

Run top to bottom with **Runtime → Run all**. Outputs are written to `MyDrive/cosmic/outputs/review2/` and shown inline below."""),
    ("md", "## 0. Mount Google Drive and get the code"),
    ("code", """from google.colab import drive
drive.mount('/content/drive')"""),
    ("code", """REPO_URL = "https://github.com/lowkey2556/cosmic-events.git"
import os
if not os.path.isdir('/content/cosmic/.git'):
    !git clone -q $REPO_URL /content/cosmic
else:
    !cd /content/cosmic && git pull -q
%cd /content/cosmic
!git log --oneline | head -15"""),
    ("md", """### Data folder
Set `DATA_ROOT` to the Drive folder that holds the AAVSO CSVs. It is searched recursively
(`EXISTING/`, `NEW_2024/`, `NEW_2025/` ...). ZIPs, other files and `DOWNLOAD_LOG/` are ignored."""),
    ("code", """DATA_ROOT = "/content/drive/MyDrive/AAVSO_DATA"   # <- the folder that contains EXISTING/, NEW_2025/ ...
import os
os.environ["COSMIC_DATA_ROOT"] = DATA_ROOT        # every `!python run.py ...` below reads this
assert os.path.isdir(DATA_ROOT), f"Drive folder not found: {DATA_ROOT}"
!find "$DATA_ROOT" -maxdepth 2 -type d | sort"""),
    ("md", """## Step 0: Dataset audit (plain Python, runs before the Big Data stack)
Per file and in total: size in MB, rows, stars, observers and date range. The total size gets a PASS/FAIL
against the 50 MB target, both as stored and after removing observations repeated across overlapping downloads.
Per star: rows, observers, dates, longest continuous coverage and the number of 100-day windows.
Class balance, skipped files and unmatched VSX types. (1 MB = 1,048,576 bytes.)"""),
    ("code", """!pip -q install pyyaml astropy astroquery
!python run.py audit --data real"""),
    ("md", """## 1. Install the stack
Java 8 comes from apt. Hadoop, Spark, Hive and Derby tarballs are cached in `MyDrive/cosmic/cache/`, so a reinstall after a Colab reset only extracts them.
The first run downloads about 1.5 GB."""),
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
    ("md", "## 4. Raw data in Drive"),
    ("code", """!find "$DATA_ROOT" -type f -iname '*.csv' -not -path '*/DOWNLOAD_LOG/*' | sort | head -100
!ls -la /content/drive/MyDrive/cosmic/labels 2>/dev/null || echo "no labels file yet - Step 1 will fetch VSX rows from VizieR" """),
    ("md", "## 5. Architecture diagram (slide 3)"),
    ("code", """!which dot >/dev/null || apt-get -qq install -y graphviz   # Graphviz draws the diagram
!python run.py architecture --data real"""),
    ("md", """## 6. Step 1: Ingestion and storage
`hdfs dfs -put` to `/cosmic/raw/` → explicit-schema read → Parquet + Snappy partitioned by `obs_year`/`sky_region` →
Hive external table `observations_raw` → partition-pruning test → dataset summary."""),
    ("code", "!python run.py ingest --data real"),
    ("code", """# The raw data in HDFS, and the partitions as Hive itself sees them
!hdfs dfs -ls -R -h /cosmic/raw | head -60
!hdfs dfs -du -s -h /cosmic/raw /cosmic/warehouse/observations_raw /cosmic/warehouse/observations_flat
!hive -S -e "SHOW PARTITIONS cosmic.observations_raw;" 2>/dev/null"""),
    ("md", "## 7. Step 2: Preprocessing in Spark SQL"),
    ("code", "!python run.py preprocess --data real"),
    ("md", "## 8. Step 3: Observer agreement score"),
    ("code", "!python run.py score --data real"),
    ("md", "## 9. Step 4: Features"),
    ("code", "!python run.py features --data real"),
    ("md", "## 10. Step 5: Classification, one fixed seed"),
    ("code", "!python run.py classify --data real"),
    ("md", "## 11. Step 6: K-Means"),
    ("code", "!python run.py cluster --data real"),
    ("code", """# Hive tables produced by the pipeline (listed through the Hive CLI)
!hive -S -e "USE cosmic; SHOW TABLES;" 2>/dev/null"""),
    ("md", "# Review 2 deliverables (all from the real data above)"),
    ("code", """import pandas as pd
import os
import plotly.io as pio
from IPython.display import Image, Markdown, display
OUT = '/content/drive/MyDrive/cosmic/outputs/review2'
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)
def csv(name, title=None):
    display(Markdown(f'**{title or name}**')); display(pd.read_csv(f'{OUT}/{name}'))
def fig(name):
    # show a saved Plotly figure inline (interactive); its PNG and HTML copies are next to it in OUT
    path = f'{OUT}/{name}.json'
    if os.path.exists(path):
        pio.read_json(path).show()
    else:
        display(Markdown(f'*{name}: not produced yet*'))
def png(name, width=900):
    display(Image(f'{OUT}/{name}', width=width))
if os.path.exists(f'{OUT}/figure_export_log.csv'):
    log = pd.read_csv(f'{OUT}/figure_export_log.csv')
    bad = log[~log['png'].astype(bool)]
    print('PNG export: all OK' if bad.empty else f'PNG export FAILED for {len(bad)} figures (HTML kept):')
    if not bad.empty: display(bad)"""),
    ("md", "## Slide 1: Dataset"),
    ("code", """csv('audit_summary.csv', 'Dataset audit: summary (50 MB check)')
csv('audit_files.csv', 'Dataset audit: per file'); csv('audit_class_balance.csv', 'Class balance')
csv('audit_skipped_files.csv', 'Skipped files (bad header)'); csv('audit_unmatched_vsx_types.csv', 'Unmatched VSX types')
csv('dataset_summary.csv', 'Dataset summary')
csv('rows_per_band.csv', 'Rows per band'); csv('rows_per_type.csv', 'Rows per observation type')
csv('label_counts.csv', 'Stars per label')
fig('fig_obs_per_star')"""),
    ("md", "## Slide 3: Architecture"),
    ("code", "png('architecture.png', 1100)   # Graphviz PNG (SVG and DOT source saved next to it)"),
    ("md", "## Slide 5: Preprocessing results"),
    ("code", """csv('preprocess_row_counts.csv', 'Row counts after each step'); fig('fig_preprocessing_funnel')
csv('partition_pruning.csv', 'Partition pruning: partitioned vs unpartitioned'); fig('fig_partition_pruning')
csv('lightcurve_star.csv', 'Light-curve star'); fig('fig_lightcurve_before_after')
csv('observer_score_summary.csv', 'Observer scores: summary'); fig('fig_observer_score_hist')
csv('observer_top_bottom10.csv', 'Top and bottom 10 observers by agreement score')
csv('spearman_score_vs_uncertainty.csv', 'Spearman: score vs self-reported uncertainty')"""),
    ("md", "## Slide 7: Proposed model results"),
    ("code", """csv('split_summary.csv', 'Split by star (auid)')
csv('proposed_model_metrics.csv', 'Proposed model (DT + observer features), seed 42')
fig('fig_confusion_proposed')
csv('feature_log.csv', 'Feature log')
csv('kmeans_silhouette.csv', 'K-Means silhouette'); fig('fig_kmeans_silhouette')
csv('kmeans_crosstab_label.csv', 'Cluster vs label'); csv('kmeans_crosstab_type.csv', 'Cluster vs VSX type')"""),
    ("md", "## Slide 8: Comparison with existing classifiers"),
    ("code", """csv('model_metrics.csv', 'All models, same split (seed 42); positive class = anomalous')
fig('fig_model_comparison')
fig('fig_confusion_dt_vs_proposed')"""),
    ("md", """**Caveat:** these are preliminary results from **one seed on a small test set**. Milestone 2 adds the 5-seed robustness (mean ± std)."""),
    ("code", "!ls -la /content/drive/MyDrive/cosmic/outputs/review2"),
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
                                       "language_info": {"name": "python"}, "colab": {"provenance": []}},
          "nbformat": 4, "nbformat_minor": 0}
    path = os.path.join(ROOT, "notebooks", "cosmic_colab.ipynb")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
    print(f"wrote {path} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
