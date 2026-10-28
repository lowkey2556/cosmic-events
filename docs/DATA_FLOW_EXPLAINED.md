# The data flow, explained

`v3_5GB/outputs/DATA_FLOW.txt` is written by the `report` step from **measurements of the latest run**. Every step that
moves or changes data saves what it measured (location, format, size in bytes, rows, files/partitions, tool, what
changed, seconds) to `v3_5GB/work/dataflow/NN_<stage>.json` (`src/cosmic/dataflow.py`). Nothing in DATA_FLOW.txt is typed
by hand. A stage that has not run is printed as `not run yet`. The file ends with the wall time of every `run.py` step
(`step_timings.csv`). This page explains how to read it. The numbers quoted here are the ones measured on Colab on
28 Oct 2026; stages marked *(v3 only)* have not run on Colab yet, so their numbers will appear only after the first v3
run.

## The journey in one picture

```
AAVSO website --CSV--> Drive data/ --hdfs dfs -put--> HDFS /cosmic/raw --Spark--> Hive observations_raw (Parquet, partitioned)
IRSA ZTF ---Parquet--> Colab disk --> Drive v3_5GB/ztf_raw --Hadoop FS API--> HDFS /cosmic/ztf/raw --> Hive ztf_lightcurves_raw
                                         |
               Spark SQL cleaning --> Hive obs_clean                + Hive ztf_lightcurves_clean (copy in Drive v3_5GB/clean)
                                         |
               A_k (Spark SQL self-join) --> Hive observer_scores     VSX cross-match --> Hive ztf_labels
                                         |                                     |
               Spark windowing --> Hive window_* (+ ztf_window_*) --> .npz files in Drive v3_5GB/work
                    |                                      |
           Spark MLlib NB/DT/RF/LR              PyTorch RGA-Net: pre-train on ZTF, fine-tune on AAVSO (GPU)
                    |                                      |  checkpoints in Drive v3_5GB/work/models
           Hive ml_predictions                  Spark mapInPandas --> Hive window_predictions
                    \______________________________________/
                 compare (pandas/SciPy) --> report (Plotly) + dashboard (Spark) --> Drive v3_5GB/outputs, dashboard_data
```

## Where the data lives, and why there

| Place | What it is | Survives a Colab reset? |
|---|---|---|
| **Google Drive** (`/content/drive/MyDrive/cosmic_roja`) | the permanent copy: inputs, ZTF files, checkpoints, results | yes |
| **Colab local disk** (`/content`, `/tmp`) | fast scratch space: the ZTF download lands here first, then is MD5-checked and copied to Drive; the clone of the code; the Python envs | no |
| **HDFS** (`hdfs://localhost:9000`, its blocks on the Colab disk) | where Spark reads and writes the Big Data tables | no |
| **Hive metastore** (Derby, `thrift://localhost:9083`) | names, schemas, partitions and HDFS locations of every table, so Spark SQL and the Hive CLI see the same tables | no |
| **GPU memory** | RGA-Net batches during training only | no |

Because HDFS and Hive disappear with the VM, the steps that matter keep a copy in Drive (the cleaned ZTF table,
windows as `.npz`, splits, models, predictions). On a fresh VM a Spark step rebuilds any missing table first
(`src/cosmic/pipeline.py`), and the cleaned ZTF table is restored from Drive instead of re-cleaned.

## The 16 stages

| # | Stage (DATA_FLOW.txt) | Made by | Measured on Colab |
|---|---|---|---|
| 01 | AAVSO observations (website → Drive) | `ingest` | 35 CSV files, 101.07 MB, 637,966 rows |
| 02 | ZTF light curves (IRSA → Colab disk → Drive) | `ztf_ingest` | 117 files, 8.53 GB (`ztf_audit`: 4,101,784 light curves, 840,991,347 epochs, 46 fields, 2018-03-20 to 2025-10-21) |
| 03 | AAVSO raw CSVs in HDFS | `ingest` | 101.1 MB in `/cosmic/raw` |
| 04 | AAVSO raw table in Hive | `ingest` | 18.64 MB of Snappy Parquet, 20 partitions (obs_year × sky_region) |
| 05 | ZTF raw Parquet in HDFS + Hive | `ztf_ingest` | *(v3 only)* |
| 06 | AAVSO cleaned observations | `preprocess` | 188,662 rows (29.6 % of raw; thinning removed 410,469) |
| 07 | ZTF cleaned light curves | `ztf_clean` | *(v3 only)*; with the 5 GB check |
| 08 | Observer agreement scores A_k | `score` | 808 observers, 412 scored with enough overlap |
| 09 | ZTF objects labelled from VSX | `ztf_label` | *(v3 only)* |
| 10 | A_k checked against ZTF | `akval` | *(v3 only)* |
| 11 | AAVSO windows, features, token sequences | `windows` | 795 windows from 50 stars (v2 run) |
| 12 | ZTF windows (pre-training pool) | `ztf_windows` | *(v3 only)* |
| 13 | Spark MLlib models and predictions | `classify` | *(v3 only: 10 seeds, 2 data settings)* |
| 14 | RGA-Net checkpoints | `rganet` | *(v3 only: pre-training + 40 fine-tuned models)* |
| 15 | RGA-Net predictions in Hive (Spark) | `infer` | 3,975 (window, model) pairs scored in 46 s; Spark vs PyTorch max difference 2.7e-7 (v2 run) |
| 16 | Results, charts, captions, dashboard data | `report` | *(v3 only)* |

## Reading one block

```
[07] ZTF cleaned light curves
     What happens : Spark SQL cleans the epoch arrays in place with higher-order functions ...
     Location     : Hive cosmic.ztf_lightcurves_clean at hdfs://localhost:9000/cosmic/ztf/clean; copy in .../v3_5GB/clean/...
     Format       : Parquet (Snappy), partitioned by field/filter
     Size         : <bytes on disk, from the Hadoop FileSystem API>   (compression: raw X GB -> clean Y GB (PASS/FAIL vs 5 GB))
     Rows         : <epochs kept>   Files/partitions: <data files>
     Tool         : Spark SQL on YARN
     Changed      : <epochs removed>; <light curves dropped>; catflags column dropped (all 0 after cleaning)
     Time taken   : <seconds>
     counting pass: <seconds of the first, counting scan>
```

* **Size** is bytes on disk as measured (`os.path.getsize` for Drive/Colab files, `getContentSummary` for HDFS),
  printed with 1 MB = 1,048,576 bytes.
* **Rows** means AAVSO observations, ZTF *epochs* (photometric points; one ZTF row holds a whole light curve), windows,
  or predictions, depending on the stage. The block says which.
* **Changed** compares with the previous stage: rows removed, columns added or dropped, format changes.
* The **TOTALS** table sums the measured stage times. **STEP TIMINGS** lists the wall time of each `run.py` step,
  including Spark start-up and writing outputs, so the two do not add up to the same number.

## Format changes along the way, and why

| From → to | Why |
|---|---|
| AAVSO CSV → Parquet (Snappy), partitioned by `obs_year`/`sky_region` | columnar and compressed (101 MB of CSV became 18.6 MB); queries on one year read only that year's folders (`partition_pruning.csv`) |
| ZTF Parquet (nested arrays) → HDFS `field=/filter=` folders, no rewrite | the files are already columnar; the folders make field and filter Hive partitions, so per-field queries skip the rest |
| ZTF arrays cleaned **in place** (`filter`, `arrays_zip`, `array_distinct`) | 841 million epochs are never exploded into rows for cleaning; exploded only when windows are cut, for the sampled light curves |
| Hive window tables → `.npz` in Drive | PyTorch on the GPU reads NumPy arrays; the file survives resets, so training needs no stack |
| PyTorch `.pt` checkpoints → broadcast to Spark executors | Spark scores every window in parallel (`mapInPandas`); the predictions go back to Hive |
| CSV/Parquet results → Plotly PNG + HTML + JSON, dashboard CSV + Parquet + Hive `dash_*` | slides, interactive notebook display, and a dashboard-ready star schema |

## How to regenerate

`!python run.py report --data real` rewrites DATA_FLOW.txt from the latest records at any time (no stack needed). To
refresh one stage's numbers, re-run the step that makes it.
