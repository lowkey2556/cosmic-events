# Big Data processing steps, in plain English

*(Review 2, slide 4.)* Each step names the tool it uses and why. Row counts after every step
are logged (Hive table `preprocess_log`, file `preprocess_row_counts.csv`).

## Step 1: Ingestion and storage

| What | Tool | Why |
|---|---|---|
| Copy the raw AAVSO CSV exports from Google Drive into HDFS `/cosmic/raw/` with `hdfs dfs -put` | **Hadoop HDFS** | Distributed, fault-tolerant storage. Every later step reads from the cluster, not from a laptop folder. |
| Fetch each star's VSX type and coordinates (only the stars in our files) | astroquery → VizieR `B/vsx` (main server, then mirrors); NASA HEASARC's copy `aavsovsx` when no VizieR server answers; saved as Hive table `star_labels` | Gives the ground-truth class and the sky position |
| Read the CSVs with an **explicit schema** (every column named; read as text, then cast) | **Spark** | Blank uncertainties and `NA` comparison stars become nulls instead of crashing the job |
| Add `obs_year` (from the Julian Date) and `sky_region` (declination band from VSX; `UNKNOWN` if missing) | **Spark SQL** | These become the partition columns |
| Write **Parquet + Snappy**, partitioned by `obs_year` / `sky_region` | **Spark → HDFS** | Columnar and compressed, so a query reads only the columns and folders it needs |
| Register the **external** Hive table `observations_raw` on that HDFS folder (`MSCK REPAIR TABLE` discovers the partitions) | **Hive metastore** | Spark and the Hive CLI see one named table. "External" means dropping the table never deletes the data. |
| Run the same year-restricted query on the partitioned table and on an unpartitioned copy (`observations_flat`), reading the **files and bytes actually scanned** from Spark's query metrics | **Spark SQL** | Evidence that partition pruning skips whole folders. **Bytes scanned** is the meaningful number. *Files* scanned can be higher for the partitioned table, because each partition folder holds its own small file, while the unpartitioned copy has a few large ones. We report both as measured. |

## Step 2: Preprocessing, all in Spark SQL

1. **Deduplicate** on `(auid, jd, band, observer, mag)`. The same report submitted twice would
   otherwise count double. We use `ROW_NUMBER()` over that key and keep the first.
2. **Drop rows with null keys** (`auid`, `jd`, `band`, `observer`, `mag`). Without these, an
   observation can't be placed or compared.
3. **Drop unreliable comment codes.** If the AAVSO comment code contains any of
   `B U W L Z I S` (bright sky, clouds, poor seeing, low altitude, questionable accuracy,
   uncertain identification, comparison-sequence problem), the row is dropped. The list is set
   in `settings.yaml`.
4. **Split out "fainter-than" rows** (`fainterthan = TRUE`). The star was *not* seen; the
   magnitude is only an upper limit, so these rows are excluded from all magnitude statistics.
   They go to `obs_upper_limits` and still count toward observing cadence (gaps, time span) in
   Step 4.
5. **Thin dense time series** (its own step in the results table and chart, drawn in a second
   colour). Keep at most **N = 10 points per (star, band, observer, night)**, evenly spaced in
   time; the cap is `preprocess.max_obs_per_observer_night` in `settings.yaml` (`null` or `0`
   switches thinning off).

   *Why.* A few CCD observers record time series: hundreds of points of one star in one night,
   one every minute or two. On the real data this step removed about two thirds of the rows
   (68 % in the first full run); by construction, every removed row comes from an observer-night
   with more than 10 points. Without the cap:
   - those few observers would supply most of the rows, so every statistic (amplitudes,
     cadence, the windows the models learn from) would describe *their* nights, not the star;
   - in the observer score (Step 3), their points would dominate everyone else's consensus,
     and the self-join's cost grows with the square of the points per night;
   - a window with one dense night would look "well observed" while covering one night.

   *What is kept.* The observer's night is not removed: up to 10 points spread evenly across it
   remain (first point kept, then every ⌈n/10⌉-th, ordered by time with fixed tie-breakers, so
   the result is reproducible). Observers who report a few points per night — almost all
   visual observers — are not affected at all.
6. **Band normalisation.** `mag_norm = mag − median(mag)` within each `(auid, band)`. V, B and
   visual magnitudes of the same star sit at different levels. Subtracting each band's median
   puts all bands on a common zero point, so we can compare observers and measure amplitude.
   `(auid, band)` groups with fewer than 20 rows are dropped because their median is unstable.

Output: Hive table `obs_clean` (one row per usable detection, with `mag_norm` and a unique
`obs_id`).

Rows are counted after every step **in total** (`preprocess_row_counts.csv`, with the rows and
the % removed by each step; drawn as the Sankey chart `figures/03_cleaning_sankey`) and **per star**
(`preprocess_rows_per_star.csv`, Hive table `preprocess_star_log`). The per-star counts feed
`star_attrition.csv` (Step 5), so any star that disappears can be traced to the exact step that
removed it.

Also saved for the report: `data_quality_aavso.csv` (size, rows, nulls, duplicates, flagged rows, bands, stars,
observers before vs after) and a 10-row raw/cleaned sample of one star (`sample_aavso_raw.csv`,
`sample_aavso_clean.csv`).

## Step 2b: ZTF light curves at Big Data scale (Phase 2, Spark)

The 117 ZTF DR24 files (8.53 GB; 4.1 million light curves; 841 million epochs) hold **one row per light curve** (one
object in one filter) with the epochs as arrays (`hmjd`, `mag`, `magerr`, `clrcoeff`, `catflags`). Spark cleans the
arrays in place with higher-order functions (`arrays_zip`, `filter`, `array_distinct`, `transform`), so the 841 million
epochs are never exploded into rows for cleaning.

| Step | What | Tool |
|---|---|---|
| `ztf_ingest` | Copy every file from Drive into HDFS `/cosmic/ztf/raw/field=<id>/filter=<zg/zr/zi>/` (Hadoop FileSystem API); register the external Hive table `ztf_lightcurves_raw`, partitioned by field and filter, with the column types **read from the files themselves** | HDFS, Hive |
| `ztf_clean` 1 | Keep valid filters (zg, zr, zi) and positions | Spark SQL |
| 2 | Remove epochs with a missing/NaN magnitude or error, or error ≤ 0 | Spark SQL `filter` |
| 3 | Remove epochs with any quality flag (`catflags != 0`) | Spark SQL |
| 4 | Remove saturated epochs (`mag ≤ 12.5`, `ztf.saturation_mag`) | Spark SQL |
| 5 | Remove exact repeated epochs | `array_distinct` |
| 6 | Drop light curves with fewer than 20 good epochs | Spark SQL |
| | Write `ztf_lightcurves_clean` (Snappy Parquet, partitioned by field/filter); check its size against **5 GB** (PASS/FAIL, and an estimate of the extra raw data needed when short); keep a copy in `v3_5GB/clean/` that later runs restore instead of re-cleaning (only when it was made from the same raw files) | Spark, HDFS, Hive |
| `ztf_label` | VSX variables of our label-map types around every field centre (VizieR, else HEASARC), cross-matched to ZTF objects within **2″** (declination-bucket equi-join + exact angular distance); **every ZTF object within 10″ of an AAVSO star is flagged and excluded from training** → Hive `ztf_labels` | Spark SQL + astroquery |
| `ztf_windows` | The same 100-day windows (A_k = 1, offset 0), class-balanced, ≤ 50,000 → `ztf_window_features`, `ztf_window_sequences`, `work/ztf_windows_seq.npz` | Spark |
| `akval` | AAVSO observations matched to the nearest unsaturated ZTF point of the same star within 1 day; per (star, AAVSO band, ZTF filter) median offset removed; each observer's robust scatter against ZTF compared with A_k (Spearman ρ, with numbers of observers and points; "too small to conclude" below 10 observers) | Spark SQL + SciPy |

Counts after every ZTF step: `ztf_clean_steps.csv` (the ZTF half of the Sankey chart); before/after table:
`data_quality_ztf.csv`; size check: `ztf_clean_size_check.csv`; labels: `ztf_label_counts.csv`; windows:
`ztf_window_counts.csv`; A_k check: `ak_validation_summary.csv`, `ak_vs_ztf_observers.csv`. Where the data is at every
stage, with measured sizes and times: `outputs/DATA_FLOW.txt` (see [DATA_FLOW_EXPLAINED.md](DATA_FLOW_EXPLAINED.md)).

## Step 3: Observer agreement score

See [PROPOSED_MODEL.md](PROPOSED_MODEL.md). This is a **Spark SQL self-join** of `obs_clean`
with itself on `(auid, band)`, restricted to `|Δt| ≤ 1 day` and different observers. To keep
the join small, each row gets a day bucket `floor(jd / Δt)` and is joined only with the three
neighbouring buckets (a fast equi-join). The exact `|Δt|` test is then applied. Medians and
MADs are computed with Spark SQL `percentile`. Outputs: Hive tables `observer_residuals` and
`observer_scores`.

## Step 4: Per-star features (for K-Means)

One row per star, built with Spark SQL aggregates. The Lomb–Scargle periodogram (astropy) runs
**per star inside Spark** with `groupBy("auid").applyInPandas(...)`, so it is parallel across
stars and the full dataset is never collected to the driver. Output: Hive table `star_features`.

## Step 5: Windowing (Spark)

Each star's light curve is cut into 100-day windows (stride 50 days, ≥ 20 observations): Spark
SQL assigns observations to windows, computes the per-window ML features (with Lomb–Scargle per
window in `applyInPandas`), and builds the RGA-Net token sequences (`applyInPandas`). Outputs:
Hive tables `windows`, `window_features`, `window_sequences`, `splits` (Parquet on HDFS),
`star_attrition.csv`, and `work/windows_seq.npz` for PyTorch. Details:
[CLASSIFICATION.md](CLASSIFICATION.md).

## Step 6: K-Means

All stars, including unlabelled ones, are clustered with **Spark MLlib K-Means** on the
standardised per-star features. k = 2..8 is chosen by silhouette score. The clusters are
cross-tabulated against the labels. This is a supporting analysis, not part of the comparison.

## Steps 7–9: Models and comparison

The four Spark MLlib models (Step 7), RGA-Net in PyTorch (Step 8) and the comparison (Step 9):
see [CLASSIFICATION.md](CLASSIFICATION.md) and [PROPOSED_MODEL.md](PROPOSED_MODEL.md).

## Step 10: Scalability

The Spark part that does the heavy lifting — preprocessing plus windowing — is re-run through
YARN with 1, 2 and 4 executors (1 core, 1 GB each), 3 times each. `scalability.csv` reports the
mean ± std time, the speedup over 1 executor, how many executors YARN actually granted, and the
machine's CPU core count (a table; no chart). On a 2-core Colab VM, at most 2 executors can run
truly in parallel, so a flat speedup beyond 2 is expected there.

## Why "never group on `target`"

The same star appears as "SS Cyg" and "SS CYG", for example. Every grouping and join uses
`auid`, the AAVSO's unique star identifier.
