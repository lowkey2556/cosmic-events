"""Step 4 — Feature engineering (owner: Rojashree). One row per star.

What it does:
  Baseline (observer-agnostic) features, from the band-normalised detections:
    amp_robust        p95 − p5 of mag_norm (robust amplitude; outbursts give large values)
    std, skewness, kurtosis of mag_norm (outbursting stars are strongly skewed toward bright)
    log_ls_period, ls_power   Lomb–Scargle best period (log10 days) and its peak power
                              (astropy, run per star inside Spark with applyInPandas)
  Cadence features, from detections AND fainter-than rows (both are "someone looked"):
    gap_median, gap_max (days between consecutive observations), n_obs, time_span
  Observer features (the proposed addition), from Step 3's scores:
    w_s               Σ A_k·n_k / Σ n_k — the observation-weighted agreement score of the
                      observers who watched this star
    n_observers       distinct observers
    mean_abs_offset   mean |median residual| of this star's (adequately scored) observers
  Stars with too few detections for Lomb–Scargle get NaN there; the count is logged and
  the value is imputed later (median fitted on the training stars only).
Output: Hive table `star_features` (+ labels, coordinates) and `feature_log.csv`.

Why: the classifier works per star, so each star's light curve is summarised into
numbers that separate eruptive (sudden, one-sided brightening) from periodic stars.
"""
import numpy as np
import pandas as pd

from .spark_session import save_csv, save_hive_table

BASELINE = ["amp_robust", "std", "skewness", "kurtosis", "log_ls_period", "ls_power",
            "gap_median", "gap_max", "n_obs", "time_span"]
OBSERVER = ["w_s", "n_observers", "mean_abs_offset"]
LS_SCHEMA = "auid STRING, ls_period DOUBLE, ls_power DOUBLE, n_ls_points LONG, ls_ok BOOLEAN"


def lomb_scargle_star(pdf, p):
    """Per-star Lomb–Scargle best period (runs on executors inside applyInPandas)."""
    from astropy.timeseries import LombScargle

    auid = pdf["auid"].iloc[0]
    n = len(pdf)
    out = dict(auid=auid, ls_period=np.nan, ls_power=np.nan, n_ls_points=n, ls_ok=False)
    if n >= p["ls_min_points"]:
        if n > p["ls_max_points"]:
            pdf = pdf.sample(n=p["ls_max_points"], random_state=p["ls_seed"])
        t = pdf["jd"].to_numpy() - pdf["jd"].min()
        y = pdf["mag_norm"].to_numpy()
        span = float(t.max())
        if span > 0 and np.std(y) > 0:
            fmin = 1.0 / min(p["ls_max_period_days"], span)
            fmax = 1.0 / p["ls_min_period_days"]
            freq, power = LombScargle(t, y).autopower(minimum_frequency=fmin, maximum_frequency=fmax,
                                                      samples_per_peak=p["ls_samples_per_peak"])
            i = int(np.nanargmax(power))
            out.update(ls_period=float(1.0 / freq[i]), ls_power=float(power[i]), ls_ok=True)
    return pd.DataFrame([out])


def build(spark, cfg):
    p = cfg.section("features")
    clean, upper = cfg.table("obs_clean"), cfg.table("obs_upper_limits")
    mags = spark.sql(f"""
        SELECT auid,
               percentile(mag_norm, 0.95) - percentile(mag_norm, 0.05) AS amp_robust,
               stddev(mag_norm) AS std, skewness(mag_norm) AS skewness, kurtosis(mag_norm) AS kurtosis,
               COUNT(*) AS n_detections
        FROM {clean} GROUP BY auid""")
    cadence = spark.sql(f"""
        WITH t AS (SELECT auid, jd FROM {clean}
                   UNION ALL
                   SELECT u.auid, u.jd FROM {upper} u WHERE u.auid IN (SELECT DISTINCT auid FROM {clean})),
        g AS (SELECT auid, jd, jd - LAG(jd) OVER (PARTITION BY auid ORDER BY jd) AS gap FROM t)
        SELECT auid, percentile(gap, 0.5) AS gap_median, MAX(gap) AS gap_max,
               COUNT(*) AS n_obs, MAX(jd) - MIN(jd) AS time_span
        FROM g GROUP BY auid""")
    ls = spark.table(clean).select("auid", "jd", "mag_norm").groupBy("auid") \
        .applyInPandas(lambda pdf: lomb_scargle_star(pdf, p), schema=LS_SCHEMA)
    obs = spark.sql(f"""
        WITH so AS (SELECT auid, observer, COUNT(*) AS n_k FROM {clean} GROUP BY auid, observer)
        SELECT so.auid,
               SUM(s.score * so.n_k) / SUM(so.n_k) AS w_s,
               COUNT(*) AS n_observers,
               AVG(CASE WHEN NOT s.low_n THEN ABS(s.median_residual) END) AS mean_abs_offset
        FROM so JOIN {cfg.table('observer_scores')} s ON so.observer = s.observer
        GROUP BY so.auid""")
    for name, df in [("f_mags", mags), ("f_cad", cadence), ("f_ls", ls), ("f_obs", obs)]:
        df.createOrReplaceTempView(name)
    return spark.sql(f"""
        SELECT m.auid, COALESCE(l.name, l.data_name) AS name, l.vsx_type, l.primary_type, l.label,
               l.ra, l.dec, l.sky_region,
               m.amp_robust, m.std, m.skewness, m.kurtosis,
               CASE WHEN ls.ls_ok THEN LOG10(ls.ls_period) END AS log_ls_period,
               CASE WHEN ls.ls_ok THEN ls.ls_power END AS ls_power,
               c.gap_median, c.gap_max, CAST(c.n_obs AS DOUBLE) AS n_obs, c.time_span,
               o.w_s, CAST(o.n_observers AS DOUBLE) AS n_observers, o.mean_abs_offset,
               m.n_detections, ls.ls_period, COALESCE(ls.ls_ok, false) AS ls_ok
        FROM f_mags m
        LEFT JOIN f_cad c ON m.auid = c.auid
        LEFT JOIN f_ls ls ON m.auid = ls.auid
        LEFT JOIN f_obs o ON m.auid = o.auid
        LEFT JOIN {cfg.table('star_labels')} l ON m.auid = l.auid""")


def run(spark, cfg):
    feats = build(spark, cfg)
    save_hive_table(feats, cfg, "star_features")
    f = spark.table(cfg.table("star_features"))
    log = spark.sql(f"""SELECT COUNT(*) AS stars,
                               SUM(CASE WHEN ls_ok THEN 0 ELSE 1 END) AS stars_too_sparse_for_lomb_scargle,
                               SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS anomalous,
                               SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) AS normal,
                               SUM(CASE WHEN label IS NULL THEN 1 ELSE 0 END) AS unlabelled
                        FROM {cfg.table('star_features')}""").toPandas()
    log["ls_min_points"] = cfg.section("features")["ls_min_points"]
    print(log.T.to_string(header=False))
    save_csv(log, cfg, "feature_log.csv")
    pdf = f.toPandas()  # one row per star: small
    save_csv(pdf, cfg, "star_features.csv")
    return pdf
