"""Step 2 — Preprocessing. Everything here is Spark SQL.

What it does, in order (row counts are logged after every step into the Hive table
`preprocess_log`):
  1. Deduplicate: identical (auid, jd, band, observer, mag) rows are the same report
     submitted twice; keep one.
  2. Drop rows whose key fields (auid, jd, band, observer, mag) are missing.
  3. Drop rows carrying an unreliable AAVSO comment code (list in settings.yaml).
  4. Split out "fainter-than" rows (the star was NOT seen; mag is only an upper limit)
     into `obs_upper_limits`. They are excluded from magnitude statistics but still
     count toward observing cadence in Step 4.
  5. Thin dense time series: at most N points per (star, band, observer, night), evenly
     spaced in time (N = preprocess.max_obs_per_observer_night; null or 0 switches it off).
     A few CCD observers take hundreds of points of one star in one night; without the cap
     they would supply most of the rows and dominate every statistic, every window and the
     observer scores. Thinning keeps their night but removes the repetition.
  6. Band-normalise: mag_norm = mag − median(mag) for each (auid, band). (auid, band) groups
     with fewer than preprocess.min_group_size rows (20) are dropped, EXCEPT a star's main bands
     (V, Visual, CV, B; preprocess.main_bands), which are kept with at least
     preprocess.min_main_band_rows rows (5), so a star is not lost because its minor bands are small.
Rows are also counted PER STAR after every step (Hive table `preprocess_star_log`), so any
star that disappears can be traced to the exact step that removed it.
Rows AND stars are counted after every step (preprocess_row_counts.csv: the input of the Sankey chart).
Outputs: Hive tables `obs_clean`, `obs_upper_limits`, `preprocess_log`, `preprocess_star_log`;
data_quality_aavso.csv (size, rows, nulls, duplicates, flagged rows, bands, stars, observers, before
and after cleaning); a Parquet copy of obs_clean in V3_ROOT/clean/; the data-flow records.

Why: crowdsourced data contain duplicates, gaps and flagged-bad reports. Different
filters (V, B, visual) sit at different magnitude levels, so comparing observers or
computing amplitudes requires putting every band on a common zero point first.
"""
import os
import time

import pandas as pd

from . import dataflow
from .spark_session import path_bytes, save_csv, save_hive_table, table_bytes, table_files


def _count(spark, view):
    return spark.sql(f"SELECT COUNT(*) FROM {view}").first()[0]


def _stars(spark, view):
    return spark.sql(f"SELECT COUNT(DISTINCT auid) FROM {view} WHERE auid IS NOT NULL").first()[0]


def _step(spark, name, view):
    return (name, _count(spark, view), _stars(spark, view))


def transform(spark, source_view, settings):
    """Pure Spark SQL pipeline on a view of typed observations. Returns (clean_df, upper_df, log_pdf)."""
    p = settings["preprocess"]
    codes = "".join(c.upper() for c in p["unreliable_comment_codes"])
    log = [_step(spark, "0 raw rows", source_view)]

    spark.sql(f"""
        SELECT * FROM (
          SELECT *, ROW_NUMBER() OVER (PARTITION BY auid, jd, band, observer, mag
                                       ORDER BY source_file, row_num) AS dup_rank
          FROM {source_view}) WHERE dup_rank = 1""").drop("dup_rank").createOrReplaceTempView("pp_dedup")
    log.append(_step(spark, "1 after deduplication", "pp_dedup"))

    spark.sql("""SELECT * FROM pp_dedup
                 WHERE auid IS NOT NULL AND jd IS NOT NULL AND band IS NOT NULL AND band <> ''
                   AND observer IS NOT NULL AND observer <> '' AND mag IS NOT NULL""") \
        .createOrReplaceTempView("pp_keys")
    log.append(_step(spark, "2 after dropping null keys", "pp_keys"))

    spark.sql(f"""SELECT * FROM pp_keys
                  WHERE commentcode IS NULL OR NOT (commentcode RLIKE '[{codes}]')""") \
        .createOrReplaceTempView("pp_codes")
    log.append(_step(spark, f"3 after dropping comment codes {codes}", "pp_codes"))

    upper = spark.sql("SELECT * FROM pp_codes WHERE fainterthan")
    spark.sql("SELECT * FROM pp_codes WHERE NOT fainterthan").createOrReplaceTempView("pp_det")
    log.append(_step(spark, "4 detections (fainter-than split out)", "pp_det"))
    n_upper = upper.count()

    m = int(p.get("max_obs_per_observer_night") or 0)
    if m > 0:  # rows ordered by time (ties broken by mag, file and row) so the kept points are reproducible
        spark.sql(f"""
            SELECT * FROM (
              SELECT *, ROW_NUMBER() OVER (PARTITION BY auid, band, observer, FLOOR(jd - 0.5)
                                           ORDER BY jd, mag, source_file, row_num) AS rn,
                        COUNT(*)     OVER (PARTITION BY auid, band, observer, FLOOR(jd - 0.5)) AS n_night
              FROM pp_det)
            WHERE n_night <= {m} OR (rn - 1) % CAST(CEIL(n_night / {m}) AS INT) = 0""") \
            .drop("rn", "n_night").createOrReplaceTempView("pp_thin")
        log.append(_step(spark, f"5 after thinning dense time series (<= {m} per observer per night)", "pp_thin"))
    else:
        spark.table("pp_det").createOrReplaceTempView("pp_thin")
        log.append(_step(spark, "5 thinning switched off", "pp_thin"))

    min_n = int(p["min_group_size"])
    main = ", ".join("'" + b.replace("'", "''") + "'" for b in p.get("main_bands") or []) or "''"
    min_main = int(p.get("min_main_band_rows", min_n))
    clean = spark.sql(f"""
        SELECT monotonically_increasing_id() AS obs_id, auid, target, jd, mag, uncertainty, band, obs_type,
               observer, commentcode, obs_year, sky_region,
               grp_median AS band_median, mag - grp_median AS mag_norm
        FROM (SELECT *, percentile(mag, 0.5) OVER (PARTITION BY auid, band) AS grp_median,
                        COUNT(*) OVER (PARTITION BY auid, band) AS grp_n
              FROM pp_thin)
        WHERE grp_n >= {min_n} OR (band IN ({main}) AND grp_n >= {min_main})""")
    return clean, upper, log, n_upper


STAR_STEPS = [("rows_after_dedup", "pp_dedup"), ("rows_after_null_keys", "pp_keys"),
              ("rows_after_comment_codes", "pp_codes"), ("rows_detections", "pp_det"),
              ("rows_after_thinning", "pp_thin")]


def star_counts(spark, clean_view, source_view="pp_source"):
    """Rows per star after every step (0 where a step removed the star completely)."""
    out = None
    for col, view in [("rows_raw", source_view)] + STAR_STEPS + [("rows_after_band_groups", clean_view)]:
        c = spark.sql(f"SELECT auid, COUNT(*) AS {col} FROM {view} WHERE auid IS NOT NULL GROUP BY auid").toPandas()
        out = c if out is None else out.merge(c, on="auid", how="left")
    cols = ["rows_raw"] + [c for c, _ in STAR_STEPS] + ["rows_after_band_groups"]
    out[cols] = out[cols].fillna(0).astype("int64")
    return out


def quality(spark, view, key_cols, codes):
    """Rows, nulls in key fields, exact duplicates, flagged rows, bands, stars and observers of a view."""
    nul = " OR ".join(f"{c} IS NULL" for c in key_cols)
    r = spark.sql(f"""SELECT COUNT(*) AS rows, SUM(CASE WHEN {nul} THEN 1 ELSE 0 END) AS nulls,
                             SUM(CASE WHEN commentcode RLIKE '[{codes}]' THEN 1 ELSE 0 END) AS flagged,
                             COUNT(DISTINCT band) AS bands, COUNT(DISTINCT auid) AS stars,
                             COUNT(DISTINCT observer) AS observers FROM {view}""").first().asDict()
    distinct = spark.sql(f"SELECT COUNT(*) FROM (SELECT DISTINCT {', '.join(key_cols)} FROM {view})").first()[0]
    r["duplicates"] = r["rows"] - distinct
    return r


def run(spark, cfg):
    t0 = time.time()
    p = cfg.section("preprocess")
    codes = "".join(c.upper() for c in p["unreliable_comment_codes"])
    spark.table(cfg.table("observations_raw")).createOrReplaceTempView("pp_source")
    clean, upper, log, n_upper = transform(spark, "pp_source", cfg.settings)
    save_hive_table(clean, cfg, "obs_clean")
    save_hive_table(upper, cfg, "obs_upper_limits")
    t_clean = cfg.table("obs_clean")
    log.append(_step(spark, f"6 after band normalisation (groups >= {p['min_group_size']}; main bands >= "
                            f"{p.get('min_main_band_rows', p['min_group_size'])})", t_clean))

    stars = star_counts(spark, t_clean)
    save_hive_table(spark.createDataFrame(stars), cfg, "preprocess_star_log")
    save_csv(stars, cfg, "preprocess_rows_per_star.csv")

    pdf = pd.DataFrame(log, columns=["step", "rows", "stars"])
    pdf["removed"] = (pdf["rows"].shift(1) - pdf["rows"]).fillna(0).astype(int)
    pdf["stars_removed"] = (pdf["stars"].shift(1) - pdf["stars"]).fillna(0).astype(int)
    pdf["pct_removed_by_step"] = (100 * pdf["removed"] / pdf["rows"].shift(1)).fillna(0).round(2)
    pdf["pct_of_raw"] = (100 * pdf["rows"] / pdf["rows"].iloc[0]).round(2)
    print(pdf.to_string(index=False))
    print(f"Stars before cleaning: {int(pdf['stars'].iloc[0])}; after: {int(pdf['stars'].iloc[-1])}. "
          f"Fainter-than upper limits kept separately for cadence: {n_upper}")
    save_hive_table(spark.createDataFrame(pdf), cfg, "preprocess_log")
    save_csv(pdf, cfg, "preprocess_row_counts.csv")

    # data quality before vs after (key fields as in deduplication)
    keys = ["auid", "jd", "band", "observer", "mag"]
    raw_bytes = path_bytes(spark, cfg.hdfs_uri(f"{cfg.hdfs_base}/raw"))
    before = quality(spark, "pp_source", keys, codes)
    after = quality(spark, t_clean, keys, codes)
    clean_bytes = table_bytes(spark, t_clean)
    dq = pd.DataFrame([dict(dataset="AAVSO", stage="before cleaning (raw CSV in HDFS)", size_bytes=raw_bytes, **before),
                       dict(dataset="AAVSO", stage="after cleaning (obs_clean, Parquet)", size_bytes=clean_bytes, **after)])
    dq["size_mb"] = (dq["size_bytes"] / 1024 ** 2).round(2)
    dq = dq[["dataset", "stage", "size_mb", "size_bytes", "rows", "nulls", "duplicates", "flagged", "bands", "stars",
             "observers"]]
    print(dq.to_string(index=False))
    save_csv(dq, cfg, "data_quality_aavso.csv")

    # a Parquet copy of the cleaned table in Drive (V3_ROOT/clean) for resuming and for study
    copy = os.path.join(cfg.clean_dir, "aavso_obs_clean.parquet")
    spark.table(t_clean).toPandas().to_parquet(copy, index=False)
    print(f"saved {copy}")
    dataflow.record(cfg, "aavso_clean", "Spark SQL removes repeated reports, rows with missing keys, flagged comment "
                    "codes and fainter-than limits, thins dense CCD nights and normalises every band to its median.",
                    f"Hive table {t_clean} (HDFS) + Drive {copy}", "Parquet (Snappy)", "Spark SQL on YARN",
                    size_bytes=clean_bytes, rows=after["rows"], files=table_files(spark, t_clean),
                    changes=f"{before['rows'] - after['rows']:,} rows removed ({before['stars']} -> {after['stars']} "
                            f"stars); columns added: obs_id, band_median, mag_norm", seconds=time.time() - t0)
    return pdf
