# How to run the project in Google Colab

This guide covers **what is built so far**: the dataset audit (Step 0), the Big Data stack, and
pipeline Steps 1–6. Sections marked *(coming later)* will be filled in as windowing, RGA-Net,
the model comparison and the scalability test are added. The final version of this guide is
written when everything else is finished.

---

## 1. First-time setup

### 1.1 Check your Google Drive layout

Your data must be laid out like this before you start:

```
MyDrive/
├── cosmic_roja/            <- your data (DATA_ROOT)
│   ├── raw/                <- the 2024 CSVs (observations_20261012_*.csv)
│   └── existing/           <- the January 2025 CSVs
└── cosmic/                 <- created automatically on the first run
    ├── cache/              <- Hadoop/Spark/Hive downloads (saves time after a reset)
    ├── labels/             <- vsx_labels.csv (star types fetched from VizieR)
    └── outputs/review2/    <- every table and figure for Review 2
```

- File names don't matter. Stars are identified by their `auid`.
- Only `.csv` files are read. ZIPs and other files are ignored, and so is any folder named `DOWNLOAD_LOG`.
- You can add new sub-folders inside `cosmic_roja/` later; they are found automatically.

### 1.2 Open the notebook

Click this link:

**https://colab.research.google.com/github/lowkey2556/cosmic-events/blob/main/notebooks/cosmic_colab.ipynb**

(Or in Colab: **File → Open notebook → GitHub**, paste `lowkey2556/cosmic-events`, pick
`notebooks/cosmic_colab.ipynb`.)

To keep your own copy with outputs: **File → Save a copy in Drive**. Work in that copy from now on.

### 1.3 Choose a runtime

**Runtime → Change runtime type**:
- For now (Steps 0–6), **CPU** is fine. Choose **High-RAM** if it's offered.
- *(coming later)* RGA-Net training will need **T4 GPU**. The Big Data stack works on a GPU runtime too.

### 1.4 Mount Google Drive

Run the first code cell (`drive.mount(...)`). A pop-up asks for permission. Choose your Google
account and allow access. You should see `Mounted at /content/drive`.

---

## 2. Run order

Run the cells **from top to bottom**, one at a time the first time, so you can check each result.
(Later you can use **Runtime → Run all**.)

| # | Notebook section | Time (approx.) | What you should see when it works |
|---|---|---|---|
| 1 | **0. Mount Google Drive and get the code** | < 1 min | `Mounted at /content/drive`, then a list of recent commits (`git log`) |
| 2 | **Data folder** | seconds | A list of folders: `.../cosmic_roja`, `.../cosmic_roja/existing`, `.../cosmic_roja/raw`. If you see `AssertionError: Drive folder not found`, see §5. |
| 3 | **Step 0: Dataset audit** | 1–3 min | Tables printed: Files, Per star, Class balance, Skipped files, Unmatched VSX types, then a **SUMMARY**. See §2.1. |
| 4 | **1. Install the stack** | 6–8 min first time; ~2–4 min after (cache) | Ends without `ERROR`. The `%%time` line at the bottom shows how long it took. |
| 5 | **2. Start HDFS, YARN, Derby, Hive metastore** | 1–2 min | Lines like `NameNode is up (port 9000)`, `ResourceManager is up`, `metastore is up (port 9083)` |
| 6 | **Load the stack's environment** | seconds | Version lines: Java 1.8, Hadoop 3.3.6, Spark 3.5.9, Hive 3.1.3 |
| 7 | **3. Smoke tests** (5 cells) | ~3 min total | Each cell prints `SMOKE TEST n: PASS - ...`. **All five must say PASS** before you continue. |
| 8 | Optional: web UIs | seconds | Links that open the YARN (8088) and HDFS (9870) pages |
| 9 | **4. Raw data in Drive** | seconds | The list of CSV files that will be ingested |
| 10 | **5. Architecture diagram** | < 1 min | `saved .../architecture.png`, `.svg`, `.dot` |
| 11 | **Step 1: Ingestion and storage** | 3–8 min | `hdfs dfs -put` lines, VSX lookups (one per star, ~1 s each, first time only), partition list, pruning table, dataset summary |
| 12 | **Step 2: Preprocessing** | 2–5 min | The row-count table after each cleaning step |
| 13 | **Step 3: Observer agreement score** | 3–10 min | Observer score summary, top/bottom 10 observers, Spearman ρ |
| 14 | **Step 4: Features** | 2–5 min | Feature table saved |
| 15 | **Step 5: Classification** (one seed) | 2–5 min | Metrics table for NB, DT, LR, RF |
| 16 | **Step 6: K-Means** | 1–3 min | Silhouette per k, chosen k |
| 17 | **Review 2 deliverables** | < 1 min | Every table and interactive Plotly figure shown inline |

Times depend on the data size and on Colab. With ~100 MB of data, expect the higher end.

### 2.1 Reading the audit result (Step 0)

The **SUMMARY** table at the end is what matters most:

- `total CSV size, MB` and **`target 50 MB: total size` → PASS or FAIL.**
- `unique-observation size, MB` and **`target 50 MB: unique-observation size` → PASS or FAIL.**
  This is the size after removing observations that appear in more than one file
  (overlapping downloads). **Use this number on the slide.** It is the honest one.
- `repeated rows`: how many rows were repeats. A large number means some downloads overlap.
- `stars`, `observers`, `bands`, `first date`, `last date`.
- `windows (raw detections, before cleaning)`: a first count of the 100-day windows the
  models will use. The final count after cleaning can be lower.

Also check:
- **Skipped files**: should say `(none)`. Any file listed there had a header that is not the
  AAVSO export header, so it was **not used**.
- **Unmatched VSX types**: stars that get no label (no VSX match, or a type not in
  `config/label_map.yaml`). They are excluded from training. Send me this list.
- **Class balance**: how many stars, rows and windows are anomalous / normal / unlabelled.

1 MB here = 1,048,576 bytes, the same as Windows and Google Drive show.

### 2.2 What a passing smoke test looks like

```
SMOKE TEST 1: PASS - ...
SMOKE TEST 2: PASS - ...
SMOKE TEST 3: PASS - ...
SMOKE TEST 4: PASS - ...
SMOKE TEST 5: PASS - ...
```

If any says `FAIL`, stop and see §5.

---

## 3. What each output means

All files are in `MyDrive/cosmic/outputs/review2/`. Every graph is a **Plotly** figure, saved
three ways:

- `fig_*.png`: for the slides;
- `fig_*.html`: interactive (zoom, hover); double-click it in Drive or download and open in a browser;
- `fig_*.json`: used by the notebook to show the figure inline.

`figure_export_log.csv` lists any figure whose PNG export failed. The HTML is always kept, and
the notebook prints the failures. Tell me if there are any.

| File | What it shows | Slide | How to read it |
|---|---|---|---|
| `audit_summary.csv` | Total size, PASS/FAIL against 50 MB, rows, stars, observers, dates | Dataset | Quote the unique-observation MB and its PASS/FAIL. |
| `audit_files.csv` | Size, rows, stars, observers, dates per file | Dataset | `repeated_rows` > 0 means the file overlaps an earlier one. |
| `audit_stars.csv` | Per star: rows, observers, dates, coverage, windows | Dataset | Stars with few windows add little to training. |
| `audit_class_balance.csv` | Anomalous / normal / unlabelled by stars, rows, windows | Dataset | Shows whether the two classes are balanced. |
| `audit_skipped_files.csv`, `audit_unmatched_vsx_types.csv` | What was left out, and why | Dataset (backup) | Should be empty or short. |
| `dataset_summary.csv`, `rows_per_band.csv`, `rows_per_type.csv`, `label_counts.csv` | The ingested dataset in numbers | Dataset | The totals after loading into Hive. |
| `fig_obs_per_star.png/.html` | Observations per star, coloured by class | Dataset | Taller bar = more observations (log scale). |
| `architecture.png` (+ `.svg`, `.dot`) | The pipeline, with the tool at each stage | Architecture | Follow the arrows; the colours show the tool family. |
| `preprocess_row_counts.csv`, `fig_preprocessing_funnel.png/.html` | Rows left after each cleaning step | Preprocessing results | Each bar is one step; the drop between bars is what that step removed. |
| `partition_pruning.csv`, `fig_partition_pruning.png/.html` | Files and MB scanned by the same query, partitioned vs unpartitioned | Preprocessing results | Fewer MB scanned on the partitioned table = pruning works. |
| `fig_lightcurve_before_after.png/.html`, `lightcurve_star.csv` | One star's light curve, raw vs band-normalised, coloured by observer | Preprocessing results | Up = brighter. After normalisation, bands line up on one curve. |
| `fig_observer_score_hist.png/.html`, `observer_score_summary.csv`, `observer_scores.csv` | Distribution of observer agreement scores A_k | Preprocessing results | Right = agrees more with other observers. |
| `observer_top_bottom10.csv` | Most and least consistent observers | Preprocessing results | Bottom 10 have the largest scatter against peers. |
| `spearman_score_vs_uncertainty.csv` | Does A_k agree with observers' own error bars? | Preprocessing results | Negative ρ = observers with bigger stated errors score lower. |
| `model_metrics.csv`, `fig_model_comparison.png/.html`, `fig_confusion_*.png/.html` | Milestone 1 classifiers, **one seed** | (preliminary) | Will be replaced by the 5-seed comparison. |
| `fig_kmeans_silhouette.png/.html`, `kmeans_*.csv` | K-Means cluster quality | (supporting) | Highest silhouette = chosen k. |
| *(coming later)* comparison table, ablation table, ROC/PR curves, confusion-matrix grid, training curves, scalability table | The main Review 2 results | Comparison | Explained here when built. |

---

## 4. When Colab disconnects or resets

**What is lost:** everything on the Colab machine. That means the installed stack, HDFS
contents, Hive tables, and the cloned code.

**What survives (in Drive):** your data (`cosmic_roja/`), the download cache (`cosmic/cache/`),
the VSX labels (`cosmic/labels/`), and every output (`cosmic/outputs/review2/`).

**Quickest way back:**
- **Only the kernel restarted** (**Runtime → Restart session**; the machine is still alive):
  run the Drive, code and Data folder cells, then `!bash setup/start_stack.sh` and the
  "Load the stack's environment" cell. HDFS and Hive data are still there; skip the install.
- **Full reset / disconnected** (the machine is gone): **Runtime → Run all**. The install uses
  the cache, so it's faster. HDFS starts empty, so Step 1 must re-ingest, and every later step
  must re-run, because each reads the previous step's Hive tables.
- **Only want to look at results?** Run the Drive cell, then the cells under
  **Review 2 deliverables**. They read straight from Drive and need no stack.

The audit (Step 0) needs no stack at all. You can re-run it any time after mounting Drive.

---

## 5. Common errors and fixes

| Problem | What to do |
|---|---|
| `AssertionError: Drive folder not found: /content/drive/MyDrive/cosmic_roja` | Drive isn't mounted, or the folder name differs. Re-run the mount cell. In the Files panel (left), open `drive/MyDrive` and check the exact name (capital letters matter). Fix `DATA_ROOT` in the Data folder cell if needed. |
| Files listed under **Skipped files** | Their first line isn't the AAVSO header. Open one in Drive and compare with `docs/DATA_DOWNLOAD.md`. Re-download it as CSV with a header row. Don't edit files by hand. |
| A smoke test says `FAIL`, or HDFS/YARN didn't start | Run `!bash setup/start_stack.sh` again (it only starts what's missing). Then run `!jps`. You should see NameNode, DataNode, ResourceManager, NodeManager and the metastore (RunJar). Logs are in the folder printed by the start script. If it still fails: **Runtime → Disconnect and delete runtime**, then **Run all**. |
| Hive metastore errors (`Could not connect to meta store`, `thrift://localhost:9083`) | The metastore didn't start or stopped. Run `!bash setup/start_stack.sh`, then smoke test 4. |
| `Java heap space`, `Container killed ... beyond memory limits`, or the session crashes | Use a **High-RAM** runtime if available. Re-run the failed step. You can also lower parallelism: `!python run.py <step> --data real --set spark.shuffle_partitions=24`. |
| `ModuleNotFoundError` (e.g. `yaml`, `plotly`) | Re-run the install cell (it installs `requirements.txt`). For the audit alone: `!pip -q install pyyaml astropy astroquery`. |
| `RuntimeError: Graphviz dot not found` | Run `!apt-get -qq install -y graphviz`, then the architecture cell again. |
| `PNG export failed` warnings | The HTML files are still saved. Tell me what `figure_export_log.csv` says. |
| VizieR lookups fail (network) | Re-run Step 1 later. Stars already looked up are kept in `cosmic/labels/vsx_labels.csv` and are not queried again. |
| *(coming later)* GPU not available | Covered when RGA-Net is added. |

---

## 6. Adding new data

1. Put the new CSVs in a **new sub-folder** of `cosmic_roja/` (or into `raw/`). Don't delete or rename old files.
2. Re-run, in order: **Data folder**, **Step 0 audit**, then (with the stack running) **Steps 1 → 6**.
   Step 1 clears `/cosmic/raw` in HDFS and re-uploads everything, so removed files never linger.
3. New stars are looked up in VSX automatically (only the new ones).
4. Check the audit's **Skipped files**, **Unmatched VSX types** and **repeated rows**.

---

## 7. Before the review

1. **Run all** once more on the final data, and check that every smoke test says PASS.
2. **File → Save** (this keeps the outputs in the notebook).
3. **Share → General access: Anyone with the link → Viewer → Copy link.** That is your
   view-only Colab link.
4. **Download figures for slides:** in Drive, open `cosmic/outputs/review2/`, select the
   `fig_*.png` files and `architecture.png`, then right-click → **Download** (Drive zips them).
5. Checklist:
   - [ ] Audit: unique-observation size PASS (≥ 50 MB), no unexpected skipped files
   - [ ] All 5 smoke tests PASS
   - [ ] Steps 1–6 ran without errors on the final data
   - [ ] `figure_export_log.csv` shows PNG = True for every figure
   - [ ] Notebook saved with outputs and shared view-only
   - [ ] *(coming later)* comparison, ablation and scalability results present

---

## 8. Live demo tips

If the faculty asks to see the stack working, run these cells live (after the stack is started):

1. **HDFS:** `!hdfs dfs -ls -R -h "/cosmic/raw" | head -20` shows the raw files stored in HDFS.
2. **Hive:** `!hive -S -e "USE cosmic; SHOW TABLES;"` and
   `!hive -S -e "SHOW PARTITIONS cosmic.observations_raw;"` show the tables and partitions as Hive sees them.
3. **YARN:** `!yarn application -list -appStates ALL | tail -10` lists the Spark jobs that ran on YARN.
   (Or open the YARN web UI on port 8088 with the optional cell.)
4. **A smoke test:** `!bash setup/smoke_tests.sh 3` submits SparkPi to YARN live and prints PASS.

Each of these takes a few seconds, so they're safe to run in front of the panel.
