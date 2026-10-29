# Cosmic Event Crowdsourcing Platform

Detects anomalous (eruptive) variable stars in crowdsourced **AAVSO** observations. A Big Data
pipeline (HDFS, YARN, Spark, Spark SQL, Hive, MLlib) ingests, cleans and windows the light curves
and measures every observer's reliability; the proposed deep learning model, **RGA-Net**
(Reliability-Gated Attention Network, PyTorch), gates its attention by that reliability. RGA-Net
is compared with four Spark MLlib models (Naive Bayes, Decision Tree, Random Forest, Logistic
Regression) on the same windows and the same 10 star-grouped splits.

**Tools:** Hadoop HDFS · YARN · Apache Spark (PySpark) · Spark SQL · Hive · Spark MLlib · PyTorch
(the deep learning model only) · **Plotly** for every graph (inline in the notebook, saved as PNG via
`kaleido` and as interactive HTML) · Graphviz for the architecture diagram. Matplotlib and Seaborn
are not used anywhere. Everything runs on a single-node pseudo-distributed cluster **inside Google Colab**.

| Phase / step | What | Tool |
|---|---|---|
| 1 `ztf_select`, `ztf_download`, `ztf_audit` | 117 ZTF DR24 light-curve files (8.53 GB, 4.1 M light curves, 841 M epochs), MD5-verified, never downloaded twice | Python |
| 0 `audit` | AAVSO dataset audit (50 MB check) | Python |
| 2 `ingest`, `preprocess`, `score`, `features` | AAVSO → HDFS → partitioned Hive; Spark SQL cleaning; observer reliability A_k | HDFS, Spark, Hive |
| 2 `ztf_ingest`, `ztf_clean`, `ztf_label` | ZTF → HDFS → Hive (field × filter); array cleaning, ≥ 5 GB check, Drive copy; VSX labels (2″), AAVSO overlaps excluded | HDFS, Spark SQL, Hive, astroquery |
| 2 `windows`, `ztf_windows`, `akval` | 100-day windows + token sequences (AAVSO, and ≤ 50,000 class-balanced ZTF); 10 star-grouped splits; A_k vs ZTF (Spearman) | Spark |
| 3 `cluster`, `classify` | K-Means (supporting); NB/DT/RF/LR with/without observer features, AAVSO vs AAVSO + ZTF chosen on validation | Spark MLlib |
| 3 `rganet` | ZTF pre-training, validation grid, 4 variants × 10 seeds, training-only augmentation, resumable | PyTorch (GPU) |
| 3 `infer`, `compare` | Spark scores every window; MCC/ROC-AUC/AUPRC/κ/Brier, Fix 1, paired Wilcoxon, `COMPARISON.md` | Spark, pandas, SciPy |
| 4 `dashboard`, `report` | dashboard tables + DATA_DICTIONARY; 13 figures with generated captions; DATA_FLOW.txt | Spark, Plotly |
| `scalability` | YARN with 1, 2, 4 executors | Spark on YARN |

Docs: [PROPOSED_MODEL](docs/PROPOSED_MODEL.md) · [CLASSIFICATION](docs/CLASSIFICATION.md) ·
[PREPROCESSING](docs/PREPROCESSING.md) · [REVIEW2_CHECKLIST](docs/REVIEW2_CHECKLIST.md) ·
[DATA_DOWNLOAD](docs/DATA_DOWNLOAD.md) · [INSTALL](docs/INSTALL.md) ·
[DATA_FLOW_EXPLAINED](docs/DATA_FLOW_EXPLAINED.md) ·
**[COLAB_GUIDE](docs/COLAB_GUIDE.md) (start here)**.

## Open the notebook

**Step-by-step guide: [docs/COLAB_GUIDE.md](docs/COLAB_GUIDE.md).**

1. Every path comes from one setting, `PROJECT_ROOT` = `MyDrive/cosmic_roja`. The AAVSO CSVs go
   **directly in `cosmic_roja/data/`**; everything the pipeline writes goes under `cosmic_roja/v3_5GB/`
   (`ztf_raw/`, `clean/`, `work/`, `outputs/`, `dashboard_data/`); labels and the stack's tarball cache stay in
   `work/` (see [docs/DATA_DOWNLOAD.md](docs/DATA_DOWNLOAD.md)). The old `MyDrive/cosmic`
   folder is never used: the notebook stops if a path points there.
2. Open <https://colab.research.google.com/github/lowkey2556/cosmic-events/blob/v3-phase2-4-verified/notebooks/cosmic_colab.ipynb>
   and choose a **T4 GPU** runtime.
3. Run the cells in order (order and measured timings: [docs/COLAB_GUIDE.md](docs/COLAB_GUIDE.md)). The last section,
   **▶️ RESULTS — start here**, only reads saved files and ends with `ALL RESULTS SHOWN`.
   Outputs are saved to `MyDrive/cosmic_roja/v3_5GB/outputs/`.
4. To share: **File → Save** (keep the outputs), then **Share → Anyone with the link → Viewer**.

## Restart the stack after a Colab reset

A reset wipes the VM, but Drive keeps `cosmic_roja/work/` (tarball cache, labels, window tensors,
models, predictions) and the outputs. Just **Run all** again. If only the kernel restarted (the VM is still
alive), run `bash setup/start_stack.sh` (it starts only what is not running) and the
"load the stack's environment" cell. `bash setup/stop_stack.sh` stops everything.
RGA-Net and the comparison need no stack: after a reset they can be re-run straight from `work/`.

## Web UIs from Colab

Run the "Optional: open the web UIs" cell: **YARN ResourceManager (8088)** and the **HDFS NameNode
(9870)**; the **Spark UI** is on port 4040 while a job runs. Command-line equivalents:
`yarn application -list -appStates ALL`, `hdfs dfsadmin -report`, `jps`.

## Running steps yourself

```bash
python run.py audit --data real                       # any single step
python run.py preprocess+windows --data real          # several steps in one Spark session
python run.py all --data real                         # audit .. report (not Phase 1 downloads, not scalability)
bash scripts/local_e2e.sh                             # whole chain on synthetic data, Spark local mode (Java 8/11/17)
python run.py windows --set windows.length_days=80    # any setting can be overridden
python -m pytest -q tests                             # unit tests (local-mode Spark, synthetic data only)
```

All parameters live in [config/settings.yaml](config/settings.yaml). Synthetic data
(`python scripts/make_synthetic.py`) is for building and testing only. It goes to a separate HDFS
folder (`/cosmic_synth`), a separate Hive database (`cosmic_synth`) and a separate output folder
(`synthetic_run/`), and is never written to the real `v3_5GB/` folder.

## Layout

```
config/       settings.yaml, label_map.yaml, stars.yaml
setup/        install_stack.sh, start_stack.sh, stop_stack.sh, smoke_tests.sh, smoke_spark_hive.py
src/cosmic/   audit, rawfiles, ingest, labels (VSX), preprocess, observer_score, features, windowing,
              windows, splits, cluster, classify, rganet, infer, evaluation, compare, scalability,
              plots (Plotly), architecture (Graphviz), spark_session, config
scripts/      make_synthetic.py (test data), build_notebook.py (generates the Colab notebook)
notebooks/    cosmic_colab.ipynb
tests/        pytest unit tests
docs/         COLAB_GUIDE, PROPOSED_MODEL, CLASSIFICATION, PREPROCESSING, REVIEW2_CHECKLIST,
              DATA_DOWNLOAD, INSTALL, figures/architecture.{png,svg,dot}
```
