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
  5. Thin dense CCD time series: at most N points per (star, band, observer, night),
     evenly spaced, so one observer's 500-point night cannot dominate.
  6. Band-normalise: mag_norm = mag − median(mag) for each (auid, band), dropping
     (auid, band) groups that are too small for a stable median.
Outputs: Hive tables `obs_clean`, `obs_upper_limits`, `preprocess_log`, the row-count
funnel chart and one star's light curve before/after normalisation.

Why: crowdsourced data contain duplicates, gaps and flagged-bad reports. Different
filters (V, B, visual) sit at different magnitude levels, so comparing observers or
computing amplitudes requires putting every band on a common zero point first.
"""
import pandas as pd

from . import plots
from .spark_session import save_csv, save_hive_table


def _count(spark, view):
    return spark.sql(f"SELECT COUNT(*) FROM {view}").first()[0]


def transform(spark, source_view, settings):
    """Pure Spark SQL pipeline on a view of typed observations. Returns (clean_df, upper_df, log_pdf)."""
    p = settings["preprocess"]
    codes = "".join(c.upper() for c in p["unreliable_comment_codes"])
    log = [("0 raw rows", _count(spark, source_view))]

    spark.sql(f"""
        SELECT * FROM (
          SELECT *, ROW_NUMBER() OVER (PARTITION BY auid, jd, band, observer, mag
                                       ORDER BY source_file, row_num) AS dup_rank
          FROM {source_view}) WHERE dup_rank = 1""").drop("dup_rank").createOrReplaceTempView("pp_dedup")
    log.append(("1 after deduplication", _count(spark, "pp_dedup")))

    spark.sql("""SELECT * FROM pp_dedup
                 WHERE auid IS NOT NULL AND jd IS NOT NULL AND band IS NOT NULL AND band <> ''
                   AND observer IS NOT NULL AND observer <> '' AND mag IS NOT NULL""") \
        .createOrReplaceTempView("pp_keys")
    log.append(("2 after dropping null keys", _count(spark, "pp_keys")))

    spark.sql(f"""SELECT * FROM pp_keys
                  WHERE commentcode IS NULL OR NOT (commentcode RLIKE '[{codes}]')""") \
        .createOrReplaceTempView("pp_codes")
    log.append((f"3 after dropping comment codes {codes}", _count(spark, "pp_codes")))

    upper = spark.sql("SELECT * FROM pp_codes WHERE fainterthan")
    spark.sql("SELECT * FROM pp_codes WHERE NOT fainterthan").createOrReplaceTempView("pp_det")
    log.append(("4 detections (fainter-than split out)", _count(spark, "pp_det")))
    n_upper = upper.count()

    m = int(p["max_obs_per_observer_night"])
    spark.sql(f"""
        SELECT * FROM (
          SELECT *, ROW_NUMBER() OVER (PARTITION BY auid, band, observer, FLOOR(jd - 0.5) ORDER BY jd) AS rn,
                    COUNT(*)     OVER (PARTITION BY auid, band, observer, FLOOR(jd - 0.5)) AS n_night
          FROM pp_det)
        WHERE n_night <= {m} OR (rn - 1) % CAST(CEIL(n_night / {m}) AS INT) = 0""") \
        .drop("rn", "n_night").createOrReplaceTempView("pp_thin")
    log.append((f"5 after thinning time series (<= {m}/observer/night)", _count(spark, "pp_thin")))

    min_n = int(p["min_group_size"])
    clean = spark.sql(f"""
        SELECT monotonically_increasing_id() AS obs_id, auid, target, jd, mag, uncertainty, band, obs_type,
               observer, commentcode, obs_year, sky_region,
               grp_median AS band_median, mag - grp_median AS mag_norm
        FROM (SELECT *, percentile(mag, 0.5) OVER (PARTITION BY auid, band) AS grp_median,
                        COUNT(*) OVER (PARTITION BY auid, band) AS grp_n
              FROM pp_thin)
        WHERE grp_n >= {min_n}""")
    return clean, upper, log, n_upper


def run(spark, cfg):
    spark.table(cfg.table("observations_raw")).createOrReplaceTempView("pp_source")
    clean, upper, log, n_upper = transform(spark, "pp_source", cfg.settings)
    save_hive_table(clean, cfg, "obs_clean")
    save_hive_table(upper, cfg, "obs_upper_limits")
    n_clean = spark.table(cfg.table("obs_clean")).count()
    log.append((f"6 after band normalisation (groups >= {cfg.section('preprocess')['min_group_size']})", n_clean))

    pdf = pd.DataFrame(log, columns=["step", "rows"])
    pdf["removed"] = (pdf["rows"].shift(1) - pdf["rows"]).fillna(0).astype(int)
    pdf["pct_of_raw"] = (100 * pdf["rows"] / pdf["rows"].iloc[0]).round(2)
    pdf.loc[len(pdf)] = ["(side table) fainter-than upper limits kept for cadence", n_upper, 0,
                         round(100 * n_upper / pdf["rows"].iloc[0], 2)]
    print(pdf.to_string(index=False))
    save_hive_table(spark.createDataFrame(pdf), cfg, "preprocess_log")
    save_csv(pdf, cfg, "preprocess_row_counts.csv")
    plots.funnel(pdf.iloc[:-1], cfg.output_dir)
    lightcurve_figure(spark, cfg)
    return pdf


def lightcurve_figure(spark, cfg):
    p = cfg.section("plots")
    t = cfg.table("obs_clean")
    auid = p["lightcurve_star"]
    if auid is None:  # star with the most distinct observers, preferring multi-band stars
        auid = spark.sql(f"""SELECT auid FROM {t} GROUP BY auid
                             ORDER BY COUNT(DISTINCT band) > 1 DESC, COUNT(DISTINCT observer) DESC, auid
                             LIMIT 1""").first()[0]
    name = spark.sql(f"SELECT COALESCE(name, data_name) FROM {cfg.table('star_labels')} WHERE auid = '{auid}'").first()
    name = name[0] if name else auid
    one = spark.sql(f"SELECT jd, mag, mag_norm, band, observer FROM {t} WHERE auid = '{auid}'")
    n = one.count()
    if n > 50000:  # plotting only: keep the figure readable
        one = one.sample(fraction=50000 / n, seed=1)
    pdf = one.toPandas()
    top = list(pdf["observer"].value_counts().index[: p["lightcurve_top_observers"]])
    save_csv(pd.DataFrame({"auid": [auid], "name": [name], "rows_plotted": [len(pdf)], "rows_total": [n]}),
             cfg, "lightcurve_star.csv")
    plots.lightcurve_before_after(pdf, f"{name} ({auid})", top, cfg.output_dir)
