"""Builds notebooks/cosmic_colab.ipynb (the Colab deliverable) from the cell list below.

Why a generator: the notebook stays reviewable in git as plain Python, and every cell is a thin call into the
repository's scripts (no pipeline logic lives in the notebook). The pipeline cells run the steps in the order
of docs/COLAB_GUIDE.md; the last section, "▶️ RESULTS — start here", only READS the files the steps saved
(tables, the 13 figures with their generated "Inference:" captions, DATA_FLOW.txt, COMPARISON.md) and ends with
"ALL RESULTS SHOWN". No v2 results and no bar charts or histograms are shown anywhere.
Run:  python scripts/build_notebook.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_TITLE = "▶️ RESULTS — start here"
END_MARKER = "ALL RESULTS SHOWN"

# The 13 result items made by `run.py report` (outputs/figures/<name>.png|html|json); item 01 has four tables.
FIGURES = [
    ("01_sample_aavso_raw", "01_sample_aavso_clean", "01_sample_ztf_raw", "01_sample_ztf_clean"),
    ("02_data_quality",), ("03_cleaning_sankey",), ("04_lightcurve_ak",), ("05_ak_violin",), ("06_ak_validation",),
    ("07_composition_sunburst",), ("08_comparison_table",), ("09_prior_vs_proposed",), ("10_radar",),
    ("11_observer_feature_dumbbell",), ("12_ablation",), ("13_confusion_grid",),
]


def run(step, note="", data=True):
    """A cell that runs one or more run.py steps (their wall time is saved to outputs/step_timings.csv)."""
    cmd = f"python run.py {step}" + (" --data real" if data else "")
    return f"sh({cmd!r})" + (f"   # {note}" if note else "")


CELLS = [
    ("md", f"""# Cosmic Event Crowdsourcing Platform: Review 2 (v3)

Detects anomalous (eruptive) variable stars in crowdsourced **AAVSO** observations. The proposed model, **RGA-Net**
(Reliability-Gated Attention Network), is a Transformer encoder whose attention is gated by each observer's measured
reliability A_k (the gate is our contribution). **ZTF** public light curves (117 files, 8.53 GB, 4.1 million light curves,
841 million epochs) add Big Data scale, pre-training data and an independent check of observer reliability.

**Stack inside this Colab VM:** Hadoop 3.3.6 (HDFS + YARN) · Spark 3.5.9 on YARN · Spark SQL · Hive 3.1.3 · Spark MLlib ·
PyTorch (RGA-Net only). Charts: Plotly (no bar charts, no histograms). Diagram: Graphviz.

**Only want the results?** Run the first cell under **{RESULTS_TITLE}** (at the end): it mounts Drive and shows every
saved table and figure; nothing is recomputed.

**To run the pipeline:** GPU runtime (Runtime → Change runtime type → T4 GPU), run **SETUP**, then the sections in order.
Guide: `docs/COLAB_GUIDE.md`; data journey: `docs/DATA_FLOW_EXPLAINED.md`. Everything new is written under
`cosmic_roja/v3_5GB/`; `data/` and `work/cache/` are only read."""),
    ("md", """## SETUP — run this first, every session
Mounts Drive, sets **every** path and environment variable, gets or updates the code, and installs Python: a Python 3.12
environment for the Spark steps (PySpark 3.5 does not support Python 3.13) and the notebook's own packages. It then prints
the resolved paths and **stops** if `data/` is missing or holds CSVs in sub-folders or duplicate names, or if a path points
at a retired folder (`MyDrive/cosmic`, `cosmic_OLD_do_not_use`, `_OLD_100MB`)."""),
    ("code", """PROJECT_ROOT = "/content/drive/MyDrive/cosmic_roja"
REPO_URL = "https://github.com/lowkey2556/cosmic-events.git"
REPO_BRANCH = "v3-phase2-4-verified"
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
    sh(f"git -C /content/cosmic fetch -q origin {REPO_BRANCH}")
    sh(f"git -C /content/cosmic switch {REPO_BRANCH}")
    sh(f"git -C /content/cosmic merge --ff-only origin/{REPO_BRANCH}")
else:
    sh(f"git clone -q --branch {REPO_BRANCH} {REPO_URL} /content/cosmic")
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

    # ------------------------------------------------------------------ Phase 1
    ("md", """## Phase 1 — ZTF files (already in Drive: verified, never downloaded twice)
`ztf_download` checks `v3_5GB/ztf_raw/MANIFEST.csv`: files already verified in Drive are skipped, so when all 117 files are
there this cell only confirms them. `ztf_select` runs only if no selection was saved yet."""),
    ("code", """import os
V3 = os.environ["V3_ROOT"]
if not os.path.exists(f"{V3}/outputs/ztf_selection.csv"):
    sh("python run.py ztf_select")          # 5-15 min, only on a brand-new Drive folder
else:
    print("ZTF selection already saved: ztf_select skipped")
sh("python run.py ztf_download")            # no download when every file is already verified"""),
    ("code", run("ztf_audit", "reads every downloaded file (5-15 min); writes ztf_audit_*.csv", data=False)),
    ("code", run("candidates", "optional: faint VSX stars for more AAVSO downloads (VizieR, else HEASARC)", data=False)),
    ("md", """## Step 0 — AAVSO dataset audit (plain Python, before the Big Data stack)"""),
    ("code", run("audit")),

    # ------------------------------------------------------------------ stack
    ("md", """## The Big Data stack: install, start, smoke tests
Java 8 from apt; Hadoop, Spark, Hive and Derby tarballs from `PROJECT_ROOT/work/cache/` (downloaded only if missing)."""),
    ("code", 'sh("bash setup/install_stack.sh")'),
    ("code", 'sh("bash setup/start_stack.sh")'),
    ("code", """# Load the stack's environment (JAVA_HOME, HADOOP_HOME, SPARK_HOME, HIVE_HOME, PATH...) into this kernel
import os, re
env_file = open('setup/ENV_LOCATION').read().strip()
for line in open(env_file):
    m = re.match(r'export (\\w+)=(.*)', line.strip())
    if m:
        os.environ[m.group(1)] = os.path.expandvars(m.group(2))
!java -version 2>&1 | head -1; hadoop version | head -1; spark-submit --version 2>&1 | grep -m1 version; hive --version 2>/dev/null | head -1"""),
    ("code", "sh(\"bash -c 'for t in 1 2 3 4 5; do bash setup/smoke_tests.sh $t || exit 1; done'\")   # each must print PASS"),
    ("code", 'sh("command -v dot >/dev/null || apt-get -qq install -y graphviz")\n' + run("architecture")),

    # ------------------------------------------------------------------ Phase 2
    ("md", """## Phase 2a — AAVSO in Spark: ingest, cleaning, observer scores A_k
`ingest` (HDFS + partitioned Hive table) → `preprocess` (Spark SQL cleaning, counted per step) → `score` (A_k,
leave-one-out consensus) → `features` (per-star, for K-Means). One Spark session for the four steps."""),
    ("code", run("ingest+preprocess+score+features")),
    ("md", """## Phase 2b — ZTF in Spark: ingest, clean (≥ 5 GB check), VSX labels
`ztf_ingest`: the 117 files → HDFS `field=/filter=` folders → partitioned Hive table `ztf_lightcurves_raw` (schema read from
the files). `ztf_clean`: invalid / flagged / saturated / repeated epochs and short light curves removed with Spark array
functions → `ztf_lightcurves_clean`, its size checked against 5 GB, a copy kept in `v3_5GB/clean/` (a later run restores
from it instead of recleaning). `ztf_label`: VSX (VizieR, else NASA HEASARC `aavsovsx`) cross-matched within 2″; every ZTF
object within 10″ of an AAVSO star is excluded from training."""),
    ("code", run("ztf_ingest+ztf_clean+ztf_label", "the longest Spark part: 8.5 GB of nested arrays")),
    ("code", """!hive -S -e "SHOW PARTITIONS cosmic.ztf_lightcurves_clean;" 2>/dev/null | head -12
!hdfs dfs -du -s -h /cosmic/ztf/raw /cosmic/ztf/clean"""),
    ("md", """## Phase 2c — Windows (AAVSO and ZTF) and A_k validation against ZTF
`windows`: 100-day windows (stride 50 d, ≥ 20 observations), 13 summary features, RGA-Net token sequences, 10 star-grouped
70/15/15 splits. `ztf_windows`: the same windows for labelled ZTF light curves (A_k = 1), class-balanced, ≤ 50,000.
`akval`: Spearman ρ between A_k and each observer's scatter against ZTF."""),
    ("code", run("windows+ztf_windows+akval")),

    # ------------------------------------------------------------------ Phase 3
    ("md", """## Phase 3a — K-Means (supporting) and the four Spark MLlib models
NB, DT, RF, LR on 10 splits; with and without observer features; data setting (AAVSO only, or AAVSO + ZTF training
windows) and grid chosen on validation AUPRC; validation and test probabilities saved."""),
    ("code", run("cluster+classify")),
    ("md", """## Phase 3b — RGA-Net (PyTorch, GPU): ZTF pre-training, validation grid, 4 variants × 10 seeds
Pre-trains once on ZTF windows, chooses lr × dropout × hybrid head separately within EACH split, then fine-tunes
`full`, `no_pretrain`, `no_gate`, `observer_blind` on every split. Training-only augmentation. **Resumable:** after a
disconnect just run the cell again — finished pre-training, grid points and (variant, seed) runs are reused."""),
    ("code", run("rganet")),
    ("md", """## Phase 3c — Distributed inference (Spark) and the comparison (saved predictions only)
`infer`: Spark broadcasts the trained models and scores every window (`mapInPandas`) → Hive `window_predictions`, checked
against PyTorch. `compare`: MCC (threshold chosen on validation), ROC-AUC, AUPRC, κ, Brier; prior approaches = classic models
WITHOUT observer features (Fix 1); paired Wilcoxon tests; `COMPARISON.md`."""),
    ("code", run("infer") + "\n" + run("compare")),

    # ------------------------------------------------------------------ Phase 4
    ("md", """## Phase 4 — Dashboard tables, the 13 figures with captions, DATA_FLOW.txt
`dashboard`: tidy tables as CSV + Parquet + Hive `dash_*`, and `DATA_DICTIONARY.md`. `report`: the 13 figures (PNG + HTML
+ JSON) with generated "Inference:" captions, and `DATA_FLOW.txt` from the measured records. Old v2 files (bar charts,
histograms, old tables) are moved to `outputs/retired_v2/`."""),
    ("code", run("dashboard") + "\n" + run("report")),
    ("md", """## Optional, slow — scalability (YARN with 1, 2, 4 executors, 3 runs each; about 35-40 min)
Run it last; then run `report` again so DATA_FLOW.txt includes it."""),
    ("code", "# !python run.py scalability --data real && python run.py report --data real"),
]

# ------------------------------------------------------------------ RESULTS (reads saved files only)
RESULTS = [
    ("md", f"""# {RESULTS_TITLE}
Everything below **only reads files** the pipeline saved in `MyDrive/cosmic_roja/v3_5GB/` (no Spark, no training). Run the
next cell first: it mounts Drive if needed. Each figure is followed by its generated **Inference:** caption. The last cell
lists anything missing and ends with **{END_MARKER}**."""),
    ("code", """import os, json
import pandas as pd
from IPython.display import Image, Markdown, display
if not os.path.isdir("/content/drive/MyDrive"):
    from google.colab import drive
    drive.mount("/content/drive")
V3 = os.environ.get("V3_ROOT", "/content/drive/MyDrive/cosmic_roja/v3_5GB")
OUT, FIG, DASH = f"{V3}/outputs", f"{V3}/outputs/figures", f"{V3}/dashboard_data"
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
CAPS = json.load(open(f"{OUT}/figure_captions.json", encoding="utf-8")) if os.path.exists(f"{OUT}/figure_captions.json") else {}
SHOWN, MISSING = [], []

def table(name, title, query=None, cols=None):
    path = f"{OUT}/{name}"
    display(Markdown(f"**{title}** — `{name}`"))
    if not os.path.exists(path):
        MISSING.append(name); display(Markdown(f"*not produced yet: {name}*")); return None
    d = pd.read_csv(path)
    if query: d = d.query(query)
    if cols: d = d[[c for c in cols if c in d.columns]]
    display(d); SHOWN.append(name); return d

def figure(name, interactive=True):
    # interactive Plotly figure from its JSON (PNG fallback), then its generated caption
    j, p = f"{FIG}/{name}.json", f"{FIG}/{name}.png"
    if not (os.path.exists(j) or os.path.exists(p)):
        MISSING.append(name); display(Markdown(f"*figure not produced yet: {name}*")); return
    try:
        import plotly.io as pio
        if not interactive: raise RuntimeError
        pio.read_json(j).show()
    except Exception:
        display(Image(p, width=1100))
    cap = CAPS.get(name, {}).get("caption")
    display(Markdown(f"> {cap}" if cap else f"*no caption saved for {name}*"))
    if not cap: MISSING.append(name + " (caption)")
    SHOWN.append(name)

def text(name, path=None, markdown=False):
    path = path or f"{OUT}/{name}"
    if not os.path.exists(path):
        MISSING.append(name); print(f"not produced yet: {name}"); return
    t = open(path, encoding="utf-8").read()
    display(Markdown(t)) if markdown else print(t)
    SHOWN.append(name)
print("results folder:", OUT)"""),
    ("md", "## 1. Run record: measured step timings and the data flow"),
    ("code", """table("step_timings.csv", "Wall time of every pipeline step (measured on the latest run of each step)")
text("DATA_FLOW.txt")"""),
    ("md", "## 2. Data: audit, cleaning, composition"),
    ("code", """table("audit_summary.csv", "AAVSO audit (50 MB check)")
table("ztf_audit_summary.csv", "ZTF audit (files, GB, light curves, epochs, our stars)")
table("ztf_clean_size_check.csv", "Cleaned ZTF size vs the 5 GB target")
for n in FIGURES_01: figure(n)
figure("02_data_quality"); figure("03_cleaning_sankey"); figure("07_composition_sunburst")
table("ztf_label_counts.csv", "ZTF light curves labelled from VSX (2 arcsec) and excluded near AAVSO stars")
table("ztf_window_counts.csv", "ZTF pre-training windows (class-balanced, capped)")
table("window_summary.csv", "AAVSO windows"); table("window_counts_by_class.csv", "AAVSO windows per class")
table("star_attrition.csv", "Stars not used, and why", query="status != 'used'", cols=["auid", "name", "vsx_type", "status", "reason"])"""),
    ("md", "## 3. Observer reliability A_k (and its check against ZTF)"),
    ("code", """figure("04_lightcurve_ak"); figure("05_ak_violin")
table("ak_summary_by_type.csv", "A_k by observation type (scored observers)")
figure("06_ak_validation")
table("ak_validation_summary.csv", "A_k vs scatter against ZTF (Spearman)")
table("spearman_score_vs_uncertainty.csv", "A_k vs self-reported uncertainty (Spearman)")"""),
    ("md", "## 4. Main result: proposed RGA-Net vs the prior approaches (10 star-grouped splits)"),
    ("code", """figure("08_comparison_table")
table("comparison_table.csv", "All rows, window and star level (mean ± std over seeds)",
      cols=["display_name", "role", "level", "seeds", "mcc", "roc_auc", "pr_auc", "kappa", "brier"])
figure("09_prior_vs_proposed")
table("prior_vs_proposed_paired.csv", "Paired per-seed differences RGA-Net − prior (MCC, AUPRC), Wilcoxon p",
      query="metric in ['mcc', 'pr_auc']")
figure("10_radar"); figure("13_confusion_grid")"""),
    ("md", "## 5. Fix 1: observer features on classic models (our framework, not prior work)"),
    ("code", """figure("11_observer_feature_dumbbell")
table("observer_feature_gain.csv", "Without -> with observer features (MCC)", query="metric == 'mcc'")"""),
    ("md", "## 6. RGA-Net: pre-training, chosen setting, ablation, learned λ"),
    ("code", """table("rganet_pretraining.csv", "ZTF pre-training (hold-out by light curve)")
table("rganet_grid.csv", "Validation grid (separate selection within each split)")
figure("12_ablation")
table("ablation_table.csv", "Ablation (mean ± std)", cols=["display_name", "level", "seeds", "mcc", "roc_auc", "pr_auc", "kappa", "brier"])
table("ablation_paired.csv", "Ablation, paired over seeds (variant − full)", query="metric == 'mcc'")
table("rganet_lambda.csv", "Learned gate strengths λ")
d = table("ml_selected_params.csv", "ML models: chosen data setting and settings per seed")
if d is not None: display(d.groupby(["variant", "model", "data_setting"]).size().rename("seeds").reset_index())"""),
    ("md", "## 7. Distributed inference, scalability, K-Means (supporting)"),
    ("code", """table("distributed_inference_check.csv", "Spark vs PyTorch probabilities")
table("window_predictions_summary.csv", "Hive window_predictions per seed and set")
table("scalability.csv", "Preprocessing + windowing through YARN (table only)")
table("kmeans_silhouette.csv", "K-Means silhouette"); table("kmeans_crosstab_label.csv", "Cluster vs label")"""),
    ("md", "## 8. COMPARISON.md, the appendix metrics, and the dashboard data"),
    ("code", """text("COMPARISON.md", markdown=True)
table("metrics_appendix.csv", "Appendix: precision, recall, accuracy, F1, balanced accuracy",
      cols=["display_name", "level", "precision", "recall", "accuracy", "f1", "balanced_accuracy"])
if os.path.isdir(DASH):
    print("dashboard_data:", sorted(f for f in os.listdir(DASH) if f.endswith((".csv", ".parquet", ".md", ".yaml"))))
text("DATA_DICTIONARY.md", f"{DASH}/DATA_DICTIONARY.md", markdown=True)"""),
    ("code", f"""# End of the results. Lists anything that was expected but not found.
print(f"{{len(SHOWN)}} result items shown from {{OUT}}")
if MISSING:
    print("NOT PRODUCED YET (run the step that makes them):", ", ".join(sorted(set(MISSING))))
    print("RESULTS INCOMPLETE")
else:
    print("{END_MARKER}")"""),
]


def _figures_01_line():
    return "FIGURES_01 = " + repr(list(FIGURES[0]))


def cells():
    out = list(CELLS)
    res = list(RESULTS)
    kind, src = res[1]
    res[1] = (kind, src + "\n" + _figures_01_line())
    return out + res


def build(path=None):
    nb_cells = []
    for kind, src in cells():
        lines = src.split("\n")
        source = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
        if kind == "md":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            nb_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source})
    nb = {"cells": nb_cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                          "language_info": {"name": "python"}, "accelerator": "GPU",
                                          "colab": {"provenance": [], "gpuType": "T4"}},
          "nbformat": 4, "nbformat_minor": 0}
    path = path or os.path.join(ROOT, "notebooks", "cosmic_colab.ipynb")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
    print(f"wrote {path} ({len(nb_cells)} cells)")
    return nb


if __name__ == "__main__":
    build()
