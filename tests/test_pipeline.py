"""Unit tests of the pipeline logic (local-mode Spark, synthetic data only)."""
import math
import os

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------- config & labels
def test_synthetic_outputs_never_go_to_review2():
    from cosmic.config import load_config
    cfg = load_config("synthetic", ["datasets.synthetic.output_subdir=review2"])
    with pytest.raises(ValueError):
        _ = cfg.output_dir


def test_primary_type_and_labels():
    from cosmic.labels import label_for, primary_type
    lm = {"anomalous": ["UGSS", "ZAND", "UG"], "normal": ["M", "SRB", "EA"]}
    assert primary_type("UGSS+E") == "UGSS"
    assert primary_type("SRB:") == "SRB"
    assert primary_type("ZAND|SR") == "ZAND"
    assert primary_type(None) is None
    assert label_for("UGSS+E", lm) == 1
    assert label_for("M", lm) == 0
    assert label_for("NR+ELL", lm) is None


def test_sky_region():
    from cosmic.labels import sky_region
    edges = [-90, -30, 0, 30, 60, 90]
    assert sky_region(45.0, edges) == "DEC_P30_P60"
    assert sky_region(-45.0, edges) == "DEC_M90_M30"
    assert sky_region(90.0, edges) == "DEC_P60_P90"
    assert sky_region(float("nan"), edges) == "UNKNOWN"


# ---------------------------------------------------------------- ingestion
def test_header_and_blank_values(spark, typed_view):
    from cosmic.ingest import RAW_COLUMNS
    import make_synthetic
    assert make_synthetic.COLUMNS == RAW_COLUMNS
    r = spark.sql(f"""SELECT COUNT(*) n, COUNT(uncertainty) n_unc, COUNT(cmag) n_cmag,
                             SUM(CASE WHEN obs_type='Visual' AND comp1_c IS NOT NULL THEN 1 ELSE 0 END) visual_comp,
                             COUNT(DISTINCT auid) stars, COUNT(DISTINCT target) spellings
                      FROM {typed_view}""").first()
    assert r.n > 3000
    assert r.n_unc < r.n             # visual blanks -> null, no crash
    assert r.visual_comp == 0        # 'NA' comparison stars -> null
    assert r.spellings > r.stars     # same star, several spellings: we must group on auid


# ---------------------------------------------------------------- preprocessing
def test_preprocessing_counts_and_normalisation(spark, clean_views, typed_view):
    log = clean_views["log"]
    rows = [n for _, n in log]
    assert all(a >= b for a, b in zip(rows, rows[1:])), log
    assert rows[1] < rows[0]                       # duplicates were removed
    assert clean_views["n_upper"] > 0              # fainter-than rows split out
    assert spark.sql("SELECT COUNT(*) FROM t_upper WHERE NOT fainterthan").first()[0] == 0
    bad = spark.sql("SELECT COUNT(*) FROM t_clean WHERE commentcode RLIKE '[BUWLZIS]'").first()[0]
    assert bad == 0
    med = spark.sql("""SELECT MAX(ABS(m)) FROM (SELECT percentile(mag_norm, 0.5) m FROM t_clean
                       GROUP BY auid, band)""").first()[0]
    assert med < 1e-9                               # every (auid, band) median is zero after normalising
    dup = spark.sql("""SELECT COUNT(*) FROM (SELECT auid, jd, band, observer, mag, COUNT(*) c FROM t_clean
                       GROUP BY auid, jd, band, observer, mag HAVING c > 1)""").first()[0]
    assert dup == 0


# ---------------------------------------------------------------- observer score
@pytest.fixture(scope="module")
def scored(spark, clean_views, settings):
    from cosmic.observer_score import residuals, scores
    residuals(spark, "t_clean", settings["observer_score"]["window_days"]).cache() \
        .createOrReplaceTempView("t_resid")
    sc, sigma_ref, med = scores(spark, "t_clean", "t_resid", settings)
    return sc.toPandas(), sigma_ref, med


def test_leave_one_out_consensus_matches_brute_force(spark, scored, settings):
    w = settings["observer_score"]["window_days"]
    clean = spark.sql("SELECT obs_id, auid, band, observer, jd, mag_norm FROM t_clean WHERE auid = '900-SYN-001'").toPandas()
    res = spark.sql("SELECT obs_id, consensus FROM t_resid WHERE auid = '900-SYN-001'").toPandas().set_index("obs_id")
    checked = 0
    for _, r in clean.sample(60, random_state=0).iterrows():
        peers = clean[(clean.auid == r.auid) & (clean.band == r.band) & (clean.observer != r.observer)
                      & ((clean.jd - r.jd).abs() <= w)]
        if len(peers) == 0:
            assert r.obs_id not in res.index
        else:
            assert res.loc[r.obs_id, "consensus"] == pytest.approx(np.median(peers.mag_norm), abs=1e-9)
            checked += 1
    assert checked > 10


def test_sanity_noisy_observer_scores_clearly_lower(scored):
    from cosmic.observer_score import sanity_check
    pdf, sigma_ref, med = scored
    s = sanity_check(pdf)
    print(s)
    assert s["noisy_clearly_lower"], s
    assert not s["noisy_low_n"]
    assert s["offset_detected"], s                  # +0.6 mag systematic offset recovered
    assert s["offset_score"] > s["median_score"] * 0.8  # consistent-but-offset observer is not penalised
    assert ((pdf["score"] > 0) & (pdf["score"] <= 1)).all()
    low = pdf[pdf["low_n"]]
    assert (low["score"] == pytest.approx(med)).all() if len(low) else True


# ---------------------------------------------------------------- features
def test_lomb_scargle_recovers_period():
    from cosmic.features import lomb_scargle_star
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 700, 400))
    y = 2.0 * np.sin(2 * np.pi * t / 123.0) + rng.normal(0, 0.1, t.size)
    p = dict(ls_min_points=20, ls_max_points=20000, ls_seed=1, ls_min_period_days=0.1,
             ls_max_period_days=1000.0, ls_samples_per_peak=5)
    out = lomb_scargle_star(pd.DataFrame({"auid": "X", "jd": t + 2459000, "mag_norm": y}), p)
    assert out["ls_ok"].iloc[0]
    assert out["ls_period"].iloc[0] == pytest.approx(123.0, rel=0.03)
    sparse = lomb_scargle_star(pd.DataFrame({"auid": "Y", "jd": t[:5], "mag_norm": y[:5]}), p)
    assert not sparse["ls_ok"].iloc[0] and math.isnan(sparse["ls_period"].iloc[0])


def test_lomb_scargle_inside_spark(spark):
    from cosmic.features import LS_SCHEMA, lomb_scargle_star
    t = np.linspace(0, 300, 200)
    pdf = pd.concat([pd.DataFrame({"auid": a, "jd": t, "mag_norm": np.sin(2 * np.pi * t / P)})
                     for a, P in [("A", 10.0), ("B", 50.0)]])
    p = dict(ls_min_points=20, ls_max_points=20000, ls_seed=1, ls_min_period_days=0.1,
             ls_max_period_days=1000.0, ls_samples_per_peak=5)
    out = spark.createDataFrame(pdf).groupBy("auid").applyInPandas(lambda d: lomb_scargle_star(d, p), LS_SCHEMA) \
        .toPandas().set_index("auid")
    assert out.loc["A", "ls_period"] == pytest.approx(10.0, rel=0.03)
    assert out.loc["B", "ls_period"] == pytest.approx(50.0, rel=0.03)


# ---------------------------------------------------------------- classification
def test_grouped_split_by_star(spark):
    from cosmic.classify import grouped_split
    pdf = pd.DataFrame({"auid": [f"S{i:02d}" for i in range(40)], "label": [i % 2 for i in range(40)],
                        "x": np.arange(40.0)})
    tr, te, w = grouped_split(spark.createDataFrame(pdf), 0.7, 42)
    a, b = set(tr.toPandas().auid), set(te.toPandas().auid)
    assert not a & b and len(a | b) == 40
    assert tr.where("label = 1").count() == 14 and tr.where("label = 0").count() == 14
    tr2, _, _ = grouped_split(spark.createDataFrame(pdf), 0.7, 42)
    assert set(tr2.toPandas().auid) == a            # reproducible for a fixed seed
    assert w[0] == pytest.approx(1.0) and w[1] == pytest.approx(1.0)


def test_metrics(spark):
    from cosmic.classify import metrics
    p = spark.createDataFrame(pd.DataFrame({"label": [1, 1, 1, 0, 0, 0, 0], "prediction": [1.0, 1, 0, 0, 0, 0, 1]}))
    m = metrics(p)
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (2, 1, 3, 1)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 3 / 4) / 2)
