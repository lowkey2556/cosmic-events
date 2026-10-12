# Cosmic Event Crowdsourcing Platform

Detects anomalous (eruptive) variable stars in crowdsourced **AAVSO** observations. A Big Data
pipeline (HDFS, YARN, Spark, Spark SQL, Hive, MLlib) ingests, cleans and windows the light curves
and measures every observer's reliability; the proposed deep learning model, **RGA-Net**
(Reliability-Gated Attention Network, PyTorch), gates its attention by that reliability. RGA-Net
is compared with four Spark MLlib models (Naive Bayes, Decision Tree, Random Forest, Logistic
Regression) on the same windows and the same 5 star-grouped splits.

**Tools:** Hadoop HDFS · YARN · Apache Spark (PySpark) · Spark SQL · Hive · Spark MLlib · PyTorch
(the deep learning model only) · **Plotly** for every graph (inline in the notebook, saved as PNG via
`kaleido` and as interactive HTML) · Graphviz for the architecture diagram. Matplotlib and Seaborn
are not used anywhere. Everything runs on a single-node pseudo-distributed cluster **inside Google Colab**.

| Step | What | Tool |
|---|---|---|
| 0 | Dataset audit: sizes, rows, stars, observers, coverage, class balance, 50 MB check, skipped files | Python |
| 1 | Ingestion and storage: HDFS, Parquet, Hive external table, partition pruning, VSX labels | HDFS, Spark, Hive |
| 2 | Preprocessing: dedup, null keys, comment codes, upper limits, thinning, band normalisation | Spark SQL |
| 3 | Observer agreement score A_k (leave-one-out consensus) | Spark SQL |
| 4 | Per-star features (for K-Means) | Spark SQL |
| 5 | Windowing: per-window ML features, RGA-Net token sequences, 5 star-grouped splits | Spark, Hive |
| 6 | K-Means (supporting analysis) | Spark MLlib |
| 7 | The 4 ML models | Spark MLlib |
| 8 | RGA-Net + ablations | PyTorch (GPU) |
| 9 | Comparison: tables, Plotly figures, `COMPARISON.md` | Python, Plotly |
| 10 | Scalability: YARN with 1, 2, 4 executors | Spark on YARN |

Docs: [PROPOSED_MODEL](docs/PROPOSED_MODEL.md) · [CLASSIFICATION](docs/CLASSIFICATION.md) ·
[PREPROCESSING](docs/PREPROCESSING.md) · [REVIEW2_CHECKLIST](docs/REVIEW2_CHECKLIST.md) ·
[DATA_DOWNLOAD](docs/DATA_DOWNLOAD.md) · [INSTALL](docs/INSTALL.md) ·
**[COLAB_GUIDE](docs/COLAB_GUIDE.md) (start here)**.

## Open the notebook

**Step-by-step guide: [docs/COLAB_GUIDE.md](docs/COLAB_GUIDE.md).**

1. The AAVSO CSVs live in `MyDrive/cosmic_roja/` (sub-folders `raw/`, `existing/`, read recursively);
   see [docs/DATA_DOWNLOAD.md](docs/DATA_DOWNLOAD.md).
2. Open <https://colab.research.google.com/github/lowkey2556/cosmic-events/blob/main/notebooks/cosmic_colab.ipynb>
   and choose a **T4 GPU** runtime.
3. **Runtime → Run all.** The notebook mounts Drive, runs the audit, installs and starts the stack,
   runs the five smoke tests, then Steps 1–10, and shows every Review 2 table and figure inline.
   Outputs are saved to `MyDrive/cosmic_roja/outputs/review2/`.
4. To share: **File → Save** (keep the outputs), then **Share → Anyone with the link → Viewer**.

## Restart the stack after a Colab reset

A reset wipes the VM, but Drive keeps the tarball cache, `cosmic_roja/work/` (window tensors,
predictions) and the outputs. Just **Run all** again. If only the kernel restarted (the VM is still
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
python run.py all --data real                         # audit .. compare (not scalability)
python run.py windows --set windows.length_days=80    # any setting can be overridden
python -m pytest -q tests                             # unit tests (local-mode Spark, synthetic data only)
```

All parameters live in [config/settings.yaml](config/settings.yaml). Synthetic data
(`python scripts/make_synthetic.py`) is for building and testing only. It goes to a separate HDFS
folder (`/cosmic_synth`), a separate Hive database (`cosmic_synth`) and a separate output folder
(`synthetic_run/`), and is **never** written to `review2/`.

## Layout

```
config/       settings.yaml, label_map.yaml, stars.yaml
setup/        install_stack.sh, start_stack.sh, stop_stack.sh, smoke_tests.sh, smoke_spark_hive.py
src/cosmic/   audit, rawfiles, ingest, labels (VSX), preprocess, observer_score, features, windowing,
              windows, splits, cluster, classify, rganet, evaluation, compare, scalability,
              plots (Plotly), architecture (Graphviz), spark_session, config
scripts/      make_synthetic.py (test data), build_notebook.py (generates the Colab notebook)
notebooks/    cosmic_colab.ipynb
tests/        pytest unit tests
docs/         COLAB_GUIDE, PROPOSED_MODEL, CLASSIFICATION, PREPROCESSING, REVIEW2_CHECKLIST,
              DATA_DOWNLOAD, INSTALL, figures/architecture.{png,svg,dot}
```
