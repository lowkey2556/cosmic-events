# How to run the project in Google Colab

> **v3 (in progress) — read this first.** The notebook now starts with **one SETUP cell**: run it first in every
> session. It mounts Drive, sets every path, gets the code and installs Python (including a Python 3.12 environment
> for Spark, because PySpark 3.5 does not run on Colab's Python 3.13). Everything new is written under
> `cosmic_roja/v3_5GB/` (outputs in `v3_5GB/outputs/`); `data/` and `work/cache/` are only read; `_OLD_100MB/` is never
> touched. **Phase 1** (ZTF download + audit + candidate stars) is the next section after SETUP. The rest of this
> guide still describes the v2 run and will be rewritten when v3 is finished.

A step-by-step guide to running the whole Review 2 pipeline: the dataset audit, the Big Data
stack (HDFS, YARN, Spark, Hive), preprocessing, windowing, the four ML models, RGA-Net, the
comparison and the scalability test. Times below are rough estimates for about 100 MB of data on
a standard Colab GPU machine; your runs may be faster or slower.

---

## 1. First-time setup

### 1.1 Check your Google Drive layout

```
MyDrive/
└── cosmic_roja/                 <- PROJECT_ROOT: every path below is derived from it
    ├── data/                    <- ALL the AAVSO CSVs, directly in this folder (no sub-folders)
    ├── outputs/review2/         <- created automatically: every table and figure for Review 2
    └── work/                    <- created automatically: models, window tensors, splits, predictions,
        ├── labels/              <-   vsx_labels.csv (star types from VizieR, fetched when missing)
        └── cache/               <-   Hadoop/Spark/Hive downloads (saves time after a reset)
```

The old `MyDrive/cosmic` folder (renamed `cosmic_OLD_do_not_use`) is never read or written:
the notebook stops if any path points there.

- File names don't matter (but no two may share a name). Stars are identified by their `auid`.
- Only `.csv` files **directly in `data/`** are read. If any CSV is still in a sub-folder of
  `data/` (e.g. the old `raw/` or `existing/`), the notebook stops and lists it: move it into
  `data/` itself. Outputs and work files are never read as data.
- On the first run with this layout, the VSX labels are fetched again into `work/labels/` (1 s per star) and the
  stack tarballs are downloaded again into `work/cache/` (about 1.5 GB, once).

### 1.2 Open the notebook

Click: **https://colab.research.google.com/github/lowkey2556/cosmic-events/blob/main/notebooks/cosmic_colab.ipynb**

(Or in Colab: **File → Open notebook → GitHub**, type `lowkey2556/cosmic-events`, choose
`notebooks/cosmic_colab.ipynb`.)

To keep your own copy with outputs: **File → Save a copy in Drive**. Work in that copy from now on.

### 1.3 Switch to a GPU runtime

**Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save.** RGA-Net trains on the GPU.
Everything else (the Big Data stack) runs on the CPU of the same machine. If "High-RAM" is offered,
turn it on.

### 1.4 Mount Google Drive

Run the first code cell (`drive.mount(...)`). A pop-up asks for permission: choose your Google
account and allow access. You should see `Mounted at /content/drive`.

---

## 2. Run order

The first time, run the cells **one by one from the top** and check each result. Later you can use
**Runtime → Run all**.

| # | Notebook section | Time (estimate) | What you should see when it works |
|---|---|---|---|
| 1 | **0. Mount Google Drive and get the code** | < 1 min | `Mounted at /content/drive`, then the latest commits |
| 2 | **Project folder and path check** | seconds | `Resolved paths:` (PROJECT_ROOT, data, outputs, work, labels file, tarball cache — all under `cosmic_roja`), then `N CSV files directly in …/data`. An error here stops the run → see §5 |
| 3 | **GPU check** | seconds | `GPU: Tesla T4` (or similar). `NONE` → see §5 |
| 4 | **Step 0: Dataset audit** | 1–3 min | Files, Per star, Class balance, Skipped files, Unmatched VSX types, then **SUMMARY** with PASS/FAIL |
| 5 | **1. Install the stack** | 6–8 min first time; 2–4 min after (cache) | Ends without `ERROR`; `%%time` shows the time |
| 6 | **2. Start HDFS, YARN, Derby, Hive** | 1–2 min | `NameNode is up (port 9000)`, `ResourceManager is up`, `metastore is up (port 9083)` |
| 7 | **Load the stack's environment** | seconds | Java 1.8, Hadoop 3.3.6, Spark 3.5.9, Hive 3.1.3 |
| 8 | **3. Smoke tests** (5 cells) | ~3 min | Each prints `SMOKE TEST n: PASS - …` (see §2.2) |
| 9 | Optional: web UIs | seconds | Links to YARN (8088) and HDFS (9870) |
| 10 | **4. Architecture diagram** | < 1 min | `saved …/architecture.png`, `.svg`, `.dot` |
| 11 | **5. Step 1: Ingestion** | 5–10 min | `hdfs dfs -put` lines; VSX look-ups (only stars not yet in `labels/`); partitions; pruning table; dataset summary |
| 12 | **6. Step 2: Preprocessing** | 3–8 min | The row-count table after each step (thinning is step 5) |
| 13 | **7. Step 3: Observer score** | 3–10 min | Score summary, top/bottom 10 observers, Spearman ρ |
| 14 | **8. Step 4: Per-star features** | 2–5 min | Feature log |
| 15 | **9. Step 5: Windowing** | 5–15 min | Windows per class and per split, the list of stars not used (with reasons), `saved …/windows_seq.npz` |
| 16 | **10. Step 6: K-Means** | 1–3 min | Silhouette per k |
| 17 | **11. Step 7: ML models** | 10–30 min | One line per seed / feature set / model with its chosen setting and validation PR-AUC |
| 18 | **12. Step 8: RGA-Net** | 10–40 min on a T4 GPU (hours on CPU) | `RGA-Net on GPU: …`, then one line per variant and seed (best epoch, validation PR-AUC, λ), then the learned λ table |
| 18b | **12b. Step 8b: Distributed inference** | 2–10 min | `Broadcasting 5 trained models…`, `Scored N (window, model) pairs…`, then the Hive query output and the Spark-vs-PyTorch check (max difference near 0, e.g. below 0.0001) |
| 19 | **13. Step 9: Comparison** | < 1 min | The comparison, star-level and ablation tables; `saved …/COMPARISON.md` |
| 20 | **14. Step 10: Scalability** | 30–90 min (9 runs) | One line per run (`… s (Spark steps), executors granted by YARN: n`), then the summary table |
| 21 | **Review 2 results** | < 1 min | Every table and interactive figure, and COMPARISON.md |

### 2.1 Reading the audit result (Step 0)

The **SUMMARY** table at the end:

- `target 50 MB: total size` and **`target 50 MB: unique-observation size` → PASS or FAIL.** The
  second one is the size after removing observations that appear in more than one file. **Quote
  this one on the slide.**
- `repeated rows`: how many rows were repeats from overlapping downloads.
- **Skipped files** should say `(none)`. **Unmatched VSX types** lists stars that get no label.

### 2.2 What a passing smoke test looks like

```
SMOKE TEST 1: PASS - ...
SMOKE TEST 2: PASS - ...
SMOKE TEST 3: PASS - ...
SMOKE TEST 4: PASS - ...
SMOKE TEST 5: PASS - ...
```

If any says `FAIL`, stop and see §5.

### 2.3 Checking which stars are used (Step 5)

After windowing, the cell prints how many stars are in the raw data, how many are used, and a
table of every star that is **not** used with the reason, for example:

- `unlabelled (no VSX match)` or `unlabelled (VSX type GCAS not in label map)`: no class to learn;
- `all rows removed at: …`: a preprocessing step removed every row of that star;
- `no 100-day window with >= 20 observations (N clean detections in total)`: too little data.

The full list is `star_attrition.csv`.

---

## 3. What each output means

All files are in `MyDrive/cosmic_roja/outputs/review2/`. Every graph is a **Plotly** figure saved
as `fig_*.png` (slides), `fig_*.html` (interactive: open it in a browser) and `fig_*.json` (used by
the notebook). `figure_export_log.csv` lists any figure whose PNG export failed; the HTML is
always kept.

### Dataset slide

| File | What it shows | How to read it |
|---|---|---|
| `audit_summary.csv` | Total size, PASS/FAIL against 50 MB, rows, stars, observers, dates | Quote the unique-observation MB and its PASS/FAIL. |
| `dataset_summary.csv` | Size in HDFS (raw CSV and Parquet), rows, stars, observers, bands, observation types, date range | The dataset as loaded into Hive. |
| `audit_class_balance.csv` | Anomalous / normal / unlabelled stars, rows, windows | Shows how balanced the two classes are. |
| `fig_obs_per_star` | Observations per star, coloured by class | Taller bar = more observations (log scale). |

### Architecture slide

`architecture.png` (+ `.svg`, `.dot`): follow the arrows from Drive to Plotly; the colour of each
box is the tool family (legend bottom right); orange dashed arrows = tables registered in Hive.

### Preprocessing results slide

| File | What it shows | How to read it |
|---|---|---|
| `preprocess_row_counts.csv`, `fig_preprocessing_funnel` | Rows left after each step, with the % each step removed | The orange bar is thinning: it removes most rows, all from observers who took more than 10 points of one star in one night. |
| `partition_pruning.csv`, `fig_partition_pruning` | Files and MB scanned by the same query, partitioned vs unpartitioned | Fewer MB scanned on the partitioned table = pruning works. |
| `fig_lightcurve_before_after` | One star, raw vs band-normalised, coloured by observer | Up = brighter. After normalisation the bands line up on one curve. |
| `fig_observer_score_hist`, `observer_score_summary.csv` | Distribution of the observer agreement score A_k | Right = agrees more with other observers. |
| `star_attrition.csv` | Every star: rows after each step, windows, used or not, why | Explains exactly which stars are in the models. |

### Proposed model slide (RGA-Net)

| File | What it shows | How to read it |
|---|---|---|
| `fig_rganet_training_curves` | Training (solid) and validation (dashed) loss per epoch, and validation PR-AUC; one colour per split | The star marks the epoch whose weights were kept. Validation loss rising while training loss falls = overfitting, which early stopping cuts off. |
| `rganet_learned_lambda.csv` | The learned gate strengths λ per seed | λ near 0 = the network chose not to use the gate; larger λ = more attention to reliable observers. |
| `ablation_table.csv`, `fig_ablation` | Full vs no-gate vs observer-blind | If "full" beats "no gate", the gate itself helps; if "no gate" beats "observer-blind", observer information helps at all. Check the ± spread before concluding. |

### Distributed inference (Big Data slide)

| File | What it shows | How to read it |
|---|---|---|
| Hive table `window_predictions`, `window_predictions_summary.csv` | The trained RGA-Net applied by Spark to every window (one row per window and split seed) | Shows the trained deep model running inside the Big Data stack, in parallel, with results in Hive. |
| `distributed_inference_check.csv` | Largest difference between Spark's and PyTorch's probabilities on the validation/test windows | Should be near 0 (floating-point noise): Spark reproduces the trained model exactly. |

### Comparison slide (main result)

| File | What it shows | How to read it |
|---|---|---|
| `comparison_table.csv` | RGA-Net vs NB, DT, RF, LR on test windows: 6 metrics, mean ± std over 5 splits, training time | Higher is better for every metric. Compare the gap with the ± values. |
| `comparison_table_star_level.csv` | The same per star (mean window probability) | The stricter view; fewer test items, so bigger ± values. |
| `fig_comparison_metrics` | Grouped bars with error bars | Each group is a model; error bars = spread over the 5 splits. |
| `fig_roc_pr_curves` | ROC and precision-recall curves of all 5 models on one split | Curve closer to the top-left (ROC) or top-right (PR) = better ranking of anomalous windows. |
| `fig_confusion_matrices` | One 2×2 table per model on one split | Diagonal = correct; bottom-left = missed anomalous windows; top-right = false alarms. |
| `COMPARISON.md` | Written summary from the numbers | Says which model is best on which metric, by how much, and whether the gap is bigger than the seed-to-seed spread. Use its wording; don't overstate. |
| `ml_observer_features_table.csv` | ML models with vs without the observer features (backup slide) | Shows whether observer information helps the classical models. |

### Scalability slide

`scalability.csv`, `fig_scalability`: mean ± std time of preprocessing + windowing with 1, 2 and 4
YARN executors, the speedup, how many executors YARN actually granted, and the CPU core count. On a
2-core machine, expect little gain beyond 2 executors: that is a property of the machine, not a bug.

---

## 4. When Colab disconnects or resets

**What is lost:** everything on the Colab machine: the installed stack, HDFS contents, Hive tables,
and the cloned code.

**What survives (in Drive):** everything under `cosmic_roja/`: your data (`data/`), `work/` (window
tensors `windows_seq.npz`, `splits.csv`, models, predictions, `labels/vsx_labels.csv`, the download
cache `cache/`) and `outputs/review2/` (every table and figure).

**Quickest way back:**

| What you want | What to run |
|---|---|
| Just **look at** the results | Mount Drive, then the **Review 2 results** cells. No stack needed. |
| Re-run **RGA-Net** or the **comparison** | Mount Drive, clone, the path-check cell, then `!python run.py rganet --data real` and/or `!python run.py compare --data real`. No stack needed: they read `work/`. (Step 8b, distributed inference, needs the stack and the Hive tables from Steps 1–5.) |
| Re-run the **ML models** or anything in Spark | The Hive tables are gone, so run: install → start → environment → Steps 1, 2, 3, 5 → then Step 7. (Step 4 and 6 only if you need K-Means.) |
| **Kernel restarted** only (machine still alive) | Mount, clone, the path-check cell, `!bash setup/start_stack.sh`, the environment cell; HDFS and Hive are still there, so continue where you stopped. |

The audit (Step 0) needs no stack at all. You can re-run it any time after mounting Drive.

---

## 5. Common errors and fixes

| Problem | What to do |
|---|---|
| `FileNotFoundError: The data folder does not exist: …/cosmic_roja/data` | Drive isn't mounted, or the folder name differs. Re-run the mount cell; in the Files panel (left) open `drive/MyDrive/cosmic_roja` and check that `data` exists (capital letters matter). Fix `PROJECT_ROOT` in the path-check cell if needed. |
| `ValueError: … CSV file(s) are in sub-folders of …/data` | Move the listed files directly into `data/` and delete the empty sub-folders (e.g. `raw/`, `existing/`). |
| `ValueError: Duplicate file names in the data folder` | Two files differ only in upper/lower case; rename or delete one. |
| `ValueError: These paths point at the retired folder` | `PROJECT_ROOT` was set to the old folder. Set it to `/content/drive/MyDrive/cosmic_roja`. |
| GPU check says `NONE` | **Runtime → Change runtime type → T4 GPU**, then reconnect and run from the top. If Colab says no GPU is available (usage limits), you can still run everything; RGA-Net will train on the CPU and take much longer. |
| HDFS or YARN not starting; a smoke test says `FAIL` | Run `!bash setup/start_stack.sh` again (it only starts what's missing), then `!jps`: you should see NameNode, DataNode, ResourceManager, NodeManager and RunJar (the metastore). If it still fails: **Runtime → Disconnect and delete runtime**, then run from the top. |
| Hive metastore errors (`Could not connect to meta store`, `thrift://localhost:9083`) | Run `!bash setup/start_stack.sh`, then smoke test 4. |
| `Table or view not found: cosmic.window_features` (or another table) | That table comes from an earlier step that was not run in this session (e.g. after a reset). Run the earlier steps first (see §4). |
| Out of memory: `Java heap space`, `Container killed … beyond memory limits`, or the session crashes | Turn on **High-RAM** if offered and re-run the failed step. You can lower Spark's parallelism: `!python run.py <step> --data real --set spark.shuffle_partitions=24`. For RGA-Net on a small GPU: `--set rganet.batch_size=32`. |
| `CUDA out of memory` | `!python run.py rganet --data real --set rganet.batch_size=32` |
| Files listed under **Skipped files** | Their first line isn't the AAVSO header; they were **not used**. Re-download them as CSV with a header row. Don't edit files by hand. |
| `ModuleNotFoundError` (e.g. `yaml`, `plotly`) | Re-run the install cell (it installs `requirements.txt`). For the audit alone, its own cell installs what it needs. |
| `PNG export failed` warnings | The HTML files are still saved. Tell me what `figure_export_log.csv` says. |
| `RuntimeError: Graphviz dot not found` | `!apt-get -qq install -y graphviz`, then the architecture cell again. |
| VizieR look-ups fail (network) | Re-run Step 1 later. Stars already looked up are kept in `cosmic_roja/work/labels/vsx_labels.csv`. |
| Scalability: "YARN granted fewer executors than requested" | The machine ran out of memory for more executors; the table shows what was granted. Report it as measured. |

---

## 6. Adding new data

1. Put the new CSVs **directly into `cosmic_roja/data/`** (no sub-folders; each file name must be
   new). Don't delete or rename old files.
2. Re-run, in order: **Project folder and path check**, **Step 0 audit**, then (with the stack running) **Steps 1 → 9**,
   and Step 10 if you want new scalability numbers. Step 1 clears `/cosmic/raw` in HDFS and
   re-uploads everything, so removed files never linger. New stars are looked up in VSX automatically.
3. Check the audit's **Skipped files**, **Unmatched VSX types** and **repeated rows**, and the
   windowing step's list of stars not used.
4. Every table and figure is regenerated from the new data. Nothing old is reused.

---

## 7. Before the review

1. Make sure the last full run used the final data (check `audit_summary.csv`).
2. **File → Save** (this keeps the outputs in the notebook).
3. **Share → General access: Anyone with the link → Viewer → Copy link.** That is your view-only
   Colab link.
4. **Figures for slides:** in Drive, open `cosmic_roja/outputs/review2/`, select the `fig_*.png`
   files and `architecture.png`, right-click → **Download** (Drive zips them). Open `COMPARISON.md`
   for the wording of the results.
5. Final checklist:
   - [ ] Audit: unique-observation size **PASS** (≥ 50 MB); no unexpected skipped files
   - [ ] All 5 smoke tests PASS
   - [ ] `star_attrition.csv` reviewed: you can explain every star that is not used
   - [ ] Steps 1–10 ran without errors on the final data
   - [ ] `comparison_table.csv`, `ablation_table.csv`, `rganet_learned_lambda.csv`, `COMPARISON.md` present
   - [ ] `distributed_inference_check.csv`: Spark vs PyTorch difference near 0
   - [ ] `figure_export_log.csv` shows `png = True` for every figure
   - [ ] Slides quote `COMPARISON.md` honestly (including where RGA-Net is not best)
   - [ ] Notebook saved with outputs and shared view-only

---

## 8. Live demo tips

If the faculty asks to see the stack working, run these after the stack is started (each takes
seconds):

1. **HDFS:** `!hdfs dfs -ls -h "/cosmic/raw" | head -20` — the raw CSVs stored in HDFS.
2. **Hive:** `!hive -S -e "USE cosmic; SHOW TABLES;"` and
   `!hive -S -e "SELECT label, COUNT(*) FROM cosmic.window_features GROUP BY label;"` — the
   pipeline's tables, and the windows per class, read by Hive itself.
3. **YARN:** `!yarn application -list -appStates ALL | tail -10` — every Spark job that ran on YARN
   (or open the YARN web UI on port 8088 with the optional cell).
4. **Spark on YARN, live:** `!bash setup/smoke_tests.sh 3` — submits SparkPi to YARN and prints PASS.
5. **The deep model inside Spark:** `!hive -S -e "SELECT window_id, seed, ROUND(prob, 3) FROM cosmic.window_predictions LIMIT 10;"`
   — RGA-Net's predictions, computed in parallel by Spark executors, read back by Hive.

To show RGA-Net itself, open `src/cosmic/rganet.py` at `GatedSelfAttention.forward`: the line
`s = s + self.lam() * log_a[...]` is the reliability gate, `score(i, j) = q·k/√d + λ·log(A_j + ε)`.
