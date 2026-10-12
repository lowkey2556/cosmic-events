# Cosmic Event Crowdsourcing Platform

A Spark pipeline that detects anomalous celestial events in crowdsourced **AAVSO** variable-star
observations, using each observer's computed reliability as an explicit classifier feature.

**Tools:** Hadoop HDFS · YARN · Apache Spark (PySpark) · Spark SQL · Hive · Spark MLlib (Decision
Tree, Naive Bayes, K-Means) · Matplotlib · Tableau. Everything runs on a single-node
pseudo-distributed cluster **inside Google Colab**.

| Step | What |
|---|---|
| 0 | Dataset audit (sizes, rows, stars, observers, coverage, class balance, skipped files) |
| 1 | Ingestion and storage (HDFS, Parquet, Hive external table, partition pruning) |
| 2 | Preprocessing (Spark SQL) |
| 3 | Observer agreement score |
| 4 | Features |
| 5 | Classification |
| 6 | K-Means |

## Open the notebook

1. Put the AAVSO CSVs in sub-folders of one Drive data folder (see [docs/DATA_DOWNLOAD.md](docs/DATA_DOWNLOAD.md))
   and set `DATA_ROOT` in the notebook's first configuration cell.
2. Open `notebooks/cosmic_colab.ipynb` in Colab (**File → Open notebook → GitHub**, paste the repo URL).
3. In the second code cell, set `REPO_URL` to this repository.
4. **Runtime → Run all.** The notebook mounts Drive, installs and starts the stack, runs the five
   smoke tests and then Steps 1–6, and shows every Review 2 table and figure inline. Outputs are also saved to
   `MyDrive/cosmic/outputs/review2/`.
5. To share: **File → Save** (keep the outputs), then **Share → Anyone with the link → Viewer**.

## Restart the stack after a Colab reset

A reset wipes the VM, but Drive keeps the tarball cache and the outputs. Just **Run all** again.
The install then only extracts cached tarballs (target under 10 minutes including the stack start), and Step 1
re-ingests from Drive. If only the kernel restarted (the VM is still alive), run:

```bash
bash setup/start_stack.sh        # idempotent: starts only what is not already running
```

and then re-run the "load the stack's environment" cell. `bash setup/stop_stack.sh` stops everything.

## Web UIs from Colab

Run the "Optional: open the web UIs" cell. It calls `google.colab.output.serve_kernel_port_as_window`
for **YARN ResourceManager (8088)** and the **HDFS NameNode (9870)**. While a Spark job is running, the
**Spark UI** is on port 4040 (`serve_kernel_port_as_window(4040)`), and is also linked from the
YARN application page. Colab's port proxy may refuse some UI sub-pages; the command-line equivalents
are `yarn application -list -appStates ALL`, `hdfs dfsadmin -report` and `jps`.

## Running steps yourself

```bash
python run.py ingest|preprocess|score|features|classify|cluster|architecture [--data real|synthetic]
python run.py milestone1 --data real          # everything in Milestone 1
python run.py score --set observer_score.window_days=0.5   # any setting can be overridden
python -m pytest -q tests                      # unit tests (local-mode Spark, synthetic data)
```

All parameters live in [config/settings.yaml](config/settings.yaml). Synthetic data
(`python scripts/make_synthetic.py`) is for building and testing only. It goes to a separate HDFS
folder (`/cosmic_synth`), a separate Hive database (`cosmic_synth`) and a separate output folder
(`synthetic_run/`), and is **never** written to `review2/`.

## Layout

```
config/       settings.yaml, label_map.yaml, stars.yaml (44-star download list)
setup/        install_stack.sh, start_stack.sh, stop_stack.sh, smoke_tests.sh, smoke_spark_hive.py
src/cosmic/   one module per step: ingest, preprocess, observer_score, features, classify, cluster
              + labels (VSX), plots (Matplotlib), architecture (diagram), spark_session, config
scripts/      make_synthetic.py (test data), build_notebook.py (generates the Colab notebook)
notebooks/    cosmic_colab.ipynb
tests/        pytest unit tests
docs/         PROPOSED_MODEL, PREPROCESSING, CLASSIFICATION, INSTALL, DATA_DOWNLOAD, REVIEW2_CHECKLIST
```
