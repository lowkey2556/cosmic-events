"""Step 3 — Observer agreement score (owner: Sanjhana). The novel part of the project.

What it does, for every observation of star s, band b, time t by observer k:
  1. Leave-one-out consensus = median mag_norm of the SAME star and band within
     |t' − t| <= Δt (default 1 day), using only OTHER observers' observations.
     This is a Spark self-join on (auid, band) with the time-window condition and
     observer != observer — no Python loops. To keep the join small, each row is
     put in a day bucket floor(jd/Δt) and only the three neighbouring buckets are
     joined (an equi-join), then the exact |Δt| condition is applied.
  2. Residual r = mag_norm − consensus.
  3. Robust scatter per observer: σ_k = 1.4826 × median(|r − median(r)|)  (the MAD,
     scaled to equal the standard deviation for Gaussian noise).
  4. Agreement score A_k = exp(−σ_k / σ_ref), where σ_ref = median of all σ_k.
     A_k = 1 means perfect agreement with peers; a typical observer scores e^−1 ≈ 0.37.
  5. Observers with fewer than 20 residuals get the population median score and are
     flagged low_n (too little overlap to judge them).
  6. Each observer's median residual is stored as their systematic offset — so a
     consistent-but-offset observer is NOT penalised in A_k (the MAD is taken around
     their own median), but the offset is still available as a feature.
Outputs: Hive tables `observer_residuals` and `observer_scores`; score histogram by
observation type; top/bottom-10 table; Spearman correlation of score vs self-reported
uncertainty (computed with Spark MLlib).

Why: amateurs disagree on the same star on the same night. Instead of averaging the
disagreement away, we measure it from the archive itself and give it to the classifier.
"""
import math

import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

from . import plots
from .spark_session import save_csv, save_hive_table


def residuals(spark, clean_view, window_days):
    w = float(window_days)
    return spark.sql(f"""
        WITH base AS (
          SELECT obs_id, auid, band, observer, obs_type, jd, mag_norm,
                 CAST(FLOOR(jd / {w}) AS BIGINT) AS day
          FROM {clean_view}),
        a AS (SELECT *, explode(array(day - 1, day, day + 1)) AS nb_day FROM base)
        SELECT a.obs_id, a.auid, a.band, a.observer, a.obs_type, a.jd, a.mag_norm,
               percentile(b.mag_norm, 0.5)   AS consensus,
               COUNT(*)                      AS n_peer_obs,
               COUNT(DISTINCT b.observer)    AS n_peer_observers,
               a.mag_norm - percentile(b.mag_norm, 0.5) AS residual
        FROM a JOIN base b
          ON a.auid = b.auid AND a.band = b.band AND b.day = a.nb_day
         AND ABS(a.jd - b.jd) <= {w} AND a.observer <> b.observer
        GROUP BY a.obs_id, a.auid, a.band, a.observer, a.obs_type, a.jd, a.mag_norm""")


def scores(spark, clean_view, resid_view, settings):
    p = settings["observer_score"]
    min_n, scale = int(p["min_residuals"]), float(p["mad_scale"])
    spark.sql(f"""
        WITH med AS (
          SELECT observer, COUNT(*) AS n_residuals, percentile(residual, 0.5) AS median_residual
          FROM {resid_view} GROUP BY observer)
        SELECT m.observer, m.n_residuals, m.median_residual,
               {scale} * percentile(ABS(r.residual - m.median_residual), 0.5) AS sigma
        FROM {resid_view} r JOIN med m ON r.observer = m.observer
        GROUP BY m.observer, m.n_residuals, m.median_residual""").createOrReplaceTempView("os_sigma")
    sigma_ref = spark.sql(f"SELECT percentile(sigma, 0.5) FROM os_sigma WHERE n_residuals >= {min_n}").first()[0]
    if sigma_ref is None:
        raise RuntimeError(f"No observer has >= {min_n} residuals: not enough overlap to score observers")
    sigma_ref = max(float(sigma_ref), 1e-6)
    median_score = spark.sql(f"""SELECT percentile(EXP(-sigma / {sigma_ref}), 0.5) FROM os_sigma
                                 WHERE n_residuals >= {min_n}""").first()[0]
    out = spark.sql(f"""
        WITH obs AS (
          SELECT observer, COUNT(*) AS n_obs, COUNT(DISTINCT auid) AS n_stars,
                 percentile(uncertainty, 0.5) AS median_uncertainty,
                 COUNT(uncertainty) AS n_uncertainty
          FROM {clean_view} GROUP BY observer),
        typ AS (
          SELECT observer, obs_type FROM (
            SELECT observer, obs_type, ROW_NUMBER() OVER (PARTITION BY observer ORDER BY COUNT(*) DESC, obs_type) rk
            FROM {clean_view} GROUP BY observer, obs_type) WHERE rk = 1)
        SELECT o.observer, t.obs_type, o.n_obs, o.n_stars,
               COALESCE(s.n_residuals, 0) AS n_residuals, s.median_residual, s.sigma,
               CASE WHEN COALESCE(s.n_residuals, 0) >= {min_n} THEN EXP(-s.sigma / {sigma_ref})
                    ELSE {median_score} END AS score,
               COALESCE(s.n_residuals, 0) < {min_n} AS low_n,
               o.median_uncertainty, o.n_uncertainty,
               CAST({sigma_ref} AS DOUBLE) AS sigma_ref
        FROM obs o JOIN typ t ON o.observer = t.observer
        LEFT JOIN os_sigma s ON o.observer = s.observer""")
    return out, sigma_ref, median_score


def spearman(df):
    """Spearman rank correlation of score vs median self-reported uncertainty (Spark MLlib)."""
    d = df.where("NOT low_n AND median_uncertainty IS NOT NULL").select("score", "median_uncertainty")
    n = d.count()
    if n < 3:
        return float("nan"), n
    vec = VectorAssembler(inputCols=["score", "median_uncertainty"], outputCol="v").transform(d)
    rho = Correlation.corr(vec, "v", "spearman").first()[0].toArray()[0, 1]
    return float(rho), n


def run(spark, cfg):
    spark.table(cfg.table("obs_clean")).createOrReplaceTempView("os_clean")
    res = residuals(spark, "os_clean", cfg.section("observer_score")["window_days"])
    save_hive_table(res, cfg, "observer_residuals")
    spark.table(cfg.table("observer_residuals")).createOrReplaceTempView("os_resid")
    sc, sigma_ref, median_score = scores(spark, "os_clean", "os_resid", cfg.settings)
    save_hive_table(sc, cfg, "observer_scores")
    sc = spark.table(cfg.table("observer_scores"))

    n_res = spark.table(cfg.table("observer_residuals")).count()
    pdf = sc.toPandas()  # one row per observer: small
    min_n = cfg.section("observer_score")["min_residuals"]
    summary = pd.DataFrame([dict(observations_with_consensus=n_res, observers=len(pdf),
                                 observers_scored=int((~pdf["low_n"]).sum()),
                                 observers_low_n=int(pdf["low_n"].sum()), sigma_ref=sigma_ref,
                                 median_score=median_score, min_residuals=min_n)])
    print(summary.T.to_string(header=False))
    save_csv(summary, cfg, "observer_score_summary.csv")
    save_csv(pdf.sort_values("score", ascending=False), cfg, "observer_scores.csv")

    q = pdf[~pdf["low_n"]].sort_values("score", ascending=False)
    cols = ["observer", "obs_type", "score", "sigma", "median_residual", "n_residuals", "n_obs", "n_stars"]
    k = min(10, len(q) // 2)  # never list the same observer in both groups
    top = q.head(k)[cols].assign(group=f"top {k}")
    bottom = q.tail(k)[cols].iloc[::-1].assign(group=f"bottom {k}")
    tb = pd.concat([top, bottom])
    print(tb.to_string(index=False))
    save_csv(tb, cfg, "observer_top_bottom10.csv")

    rho, n = spearman(sc)
    sp = pd.DataFrame([dict(spearman_rho=rho, n_observers=n,
                            note="score vs observer's median self-reported uncertainty; "
                                 "observers with >= min_residuals and at least one reported uncertainty")])
    print(sp.to_string(index=False))
    save_csv(sp, cfg, "spearman_score_vs_uncertainty.csv")
    plots.score_histogram(pdf, cfg.output_dir)
    if cfg.dataset == "synthetic":
        print("SANITY CHECK:", sanity_check(pdf))
    return pdf


def sanity_check(pdf, noisy="NOISY", offset="OFFST"):
    """On synthetic data: the noisy observer must score clearly lower; the offset one keeps a good score."""
    pdf = pdf.set_index("observer")
    med = pdf.loc[~pdf["low_n"], "score"].median()
    return dict(noisy_score=float(pdf.loc[noisy, "score"]), median_score=float(med),
                noisy_clearly_lower=bool(pdf.loc[noisy, "score"] < 0.5 * med),
                offset_median_residual=float(pdf.loc[offset, "median_residual"]),
                offset_score=float(pdf.loc[offset, "score"]),
                offset_detected=bool(abs(pdf.loc[offset, "median_residual"] - 0.6) < 0.2),
                noisy_low_n=bool(pdf.loc[noisy, "low_n"]) or math.isnan(pdf.loc[noisy, "sigma"]))
