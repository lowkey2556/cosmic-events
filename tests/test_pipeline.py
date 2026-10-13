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
    assert primary_type("RRAB/BL") == "RRAB"
    assert primary_type("NL/VY") == "NL"                  # VY Scl
    assert primary_type("RRAB/BL", [":", "+", "|"]) == "RRAB/BL"


def test_label_map_file():
    from cosmic.config import load_config
    from cosmic.labels import label_for
    lm = load_config("synthetic").load_yaml("label_map")
    assert label_for("NR", lm) == 1                       # T CrB
    assert label_for("UGWZ/IBWD", lm) == 1                # WZ Sge
    assert label_for("NL/VY+E", lm) == 1                  # VY Scl
    assert label_for("M", lm) == 0


def test_unmatched_types_are_listed(capsys):
    from cosmic.labels import assign_labels
    stars = pd.DataFrame({"auid": ["A", "B", "C"], "data_name": ["X1", "X2", "X3"]})
    vsx = pd.DataFrame({"auid": ["A", "B"], "name": ["X1", "X2"], "ra": [1.0, 2.0], "dec": [10.0, -40.0],
                        "type": ["UGSS", "GCAS"]})
    out, un = assign_labels(stars, vsx, {"anomalous": ["UGSS"], "normal": ["M"]}, [-90, -30, 0, 30, 60, 90])
    assert list(out["label"].astype(object)) == [1, pd.NA, pd.NA]
    assert dict(zip(un["auid"], un["reason"])) == {"B": "primary type not in label map", "C": "no VSX match"}
    assert "GCAS" in capsys.readouterr().out


# ---------------------------------------------------------------- raw file discovery
def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write(text)





def _good():
    from cosmic.rawfiles import RAW_COLUMNS
    return ",".join(RAW_COLUMNS) + "\n1,SS CYG,000-BCP-220,2460676.6,12.5,0.05,False,V,CCD,ABC,,,,,,,,,\n"


def test_flat_data_folder_and_header_check(tmp_path):
    from cosmic.rawfiles import check_headers, list_csv_files
    d = str(tmp_path / "data")
    _write(os.path.join(d, "a.csv"), _good())
    _write(os.path.join(d, "b.CSV"), "﻿" + _good())                 # BOM, upper-case extension
    _write(os.path.join(d, "bad.csv"), "star,date,mag\nx,1,2\n")
    _write(os.path.join(d, "empty.csv"), "")
    _write(os.path.join(d, "x.zip"), "zip")                              # not a CSV: ignored
    files = list_csv_files(d)
    assert sorted(os.path.basename(f) for f in files) == ["a.csv", "b.CSV", "bad.csv", "empty.csv"]
    ok, skipped = check_headers(files, d)
    assert len(ok) == 2
    assert sorted(s["file"] for s in skipped) == ["bad.csv", "empty.csv"]


def test_data_folder_stops_on_subfolders_and_duplicate_names(tmp_path):
    from cosmic.rawfiles import list_csv_files
    d = tmp_path / "data"
    _write(str(d / "a.csv"), _good())
    _write(str(d / "raw" / "b.csv"), _good())                            # old layout: CSVs in a sub-folder
    with pytest.raises(ValueError, match="sub-folders"):
        list_csv_files(str(d))
    os.remove(d / "raw" / "b.csv")
    os.rmdir(d / "raw")
    assert len(list_csv_files(str(d))) == 1
    _write(str(d / "A.CSV"), _good())                                    # same name as a.csv apart from case
    names = os.listdir(d)
    if len({n.lower() for n in names}) < len(names):                     # both exist (case-sensitive file system, as on Colab)
        with pytest.raises(ValueError, match="Duplicate file names"):
            list_csv_files(str(d))
    with pytest.raises(FileNotFoundError):
        list_csv_files(str(tmp_path / "missing"))


def test_every_path_derives_from_project_root(monkeypatch, tmp_path):
    from cosmic.config import check_paths, load_config
    monkeypatch.setenv("COSMIC_PROJECT_ROOT", str(tmp_path))
    cfg = load_config("real")
    p = {k: v.replace(os.sep, "/") for k, v in cfg.paths().items()}
    root = str(tmp_path).replace(os.sep, "/")
    assert p == {"PROJECT_ROOT": root, "data (CSVs)": root + "/data", "outputs": root + "/outputs/review2",
                 "work": root + "/work", "VSX labels file": root + "/work/labels/vsx_labels.csv",
                 "tarball cache": root + "/work/cache"}
    with pytest.raises(FileNotFoundError, match="data folder does not exist"):
        check_paths(cfg)                                                  # data/ missing
    (tmp_path / "data").mkdir()
    check_paths(cfg)
    assert load_config("synthetic").data_dir.replace(os.sep, "/").endswith("data/synthetic/raw")  # env is real-only


@pytest.mark.parametrize("root", ["/content/drive/MyDrive/cosmic", "/content/drive/MyDrive/cosmic/",
                                  "/content/drive/MyDrive/cosmic_OLD_do_not_use"])
def test_retired_folders_are_refused(monkeypatch, root):
    from cosmic.config import check_paths, is_retired, load_config
    monkeypatch.setenv("COSMIC_PROJECT_ROOT", root)
    with pytest.raises(ValueError, match="retired folder"):
        check_paths(load_config("real"), need_data=False)
    assert not is_retired("/content/drive/MyDrive/cosmic_roja/data")    # the new project folder is fine
    assert is_retired("/content/drive/MyDrive/cosmic/cache/x.tgz")


def test_vsx_name_candidates():
    from cosmic.labels import vsx_name_candidates
    assert vsx_name_candidates("SS CYG")[0] == "SS Cyg"
    assert vsx_name_candidates("T CRB")[0] == "T CrB"
    assert vsx_name_candidates("Z UMA")[0] == "Z UMa"
    assert vsx_name_candidates("KHI CYG")[0] == "khi Cyg"
    assert vsx_name_candidates("OMI CET")[0] == "omi Cet"
    assert vsx_name_candidates("V603 AQL")[0] == "V0603 Aql"
    assert vsx_name_candidates("V 603 Aql")[0] == "V0603 Aql"


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


def test_comment_with_line_break_stays_one_record(spark, tmp_path):
    from cosmic.ingest import RAW_COLUMNS, read_raw
    f = tmp_path / "x.csv"
    f.write_text(",".join(RAW_COLUMNS) + "\n"
                 + '1,T CrB,000-BBW-825,2460702.946285,10.1,0.01,False,V,CCD,SCMA,,,,,,,,,"line one\r\nline two"\n'
                 + "2,T CrB,000-BBW-825,2460702.95,10.2,0.01,False,V,CCD,SCMA,,,,,,,,,\n", newline="")
    rows = read_raw(spark, f.as_uri()).collect()
    assert len(rows) == 2 and rows[0]["comments"].startswith("line one") and rows[1]["jd"] == "2460702.95"


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
    t = np.sort(np.random.default_rng(3).uniform(0, 300, 200))  # irregular, like real AAVSO sampling
    pdf = pd.concat([pd.DataFrame({"auid": a, "jd": t, "mag_norm": np.sin(2 * np.pi * t / P)})
                     for a, P in [("A", 10.0), ("B", 50.0)]])
    p = dict(ls_min_points=20, ls_max_points=20000, ls_seed=1, ls_min_period_days=0.1,
             ls_max_period_days=1000.0, ls_samples_per_peak=5)
    out = spark.createDataFrame(pdf).groupBy("auid").applyInPandas(lambda d: lomb_scargle_star(d, p), LS_SCHEMA) \
        .toPandas().set_index("auid")
    assert out.loc["A", "ls_period"] == pytest.approx(10.0, rel=0.03)
    assert out.loc["B", "ls_period"] == pytest.approx(50.0, rel=0.03)


# ---------------------------------------------------------------- splits and metrics (shared by all models)
def test_star_split_disjoint_stratified_reproducible():
    from cosmic.splits import check_disjoint, make_splits
    stars = pd.DataFrame({"auid": [f"S{i:02d}" for i in range(57)], "label": [int(i < 31) for i in range(57)]})
    sp = make_splits(stars, [1, 2, 3, 4, 5])
    check_disjoint(sp)
    for seed, d in sp.groupby("seed"):
        assert len(d) == 57 and d["auid"].is_unique                   # every star in exactly one set
        c = d.groupby(["set", "label"]).size()
        assert c[("train", 1)] == 22 and c[("val", 1)] == 5 and c[("test", 1)] == 4
        assert c[("train", 0)] == 18 and c[("val", 0)] == 4 and c[("test", 0)] == 4
    assert sp[sp.seed == 1].equals(make_splits(stars, [1]))            # reproducible
    tests = {s: set(d.loc[d["set"] == "test", "auid"]) for s, d in sp.groupby("seed")}
    assert len({frozenset(v) for v in tests.values()}) == 5             # 5 different splits


def test_metrics_by_hand():
    from cosmic.evaluation import at_threshold, average_precision, choose_threshold, roc_auc
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.3, 0.4, 0.2, 0.1, 0.85])
    # positives beat 4, 3 and 2 of the 4 negatives -> AUC = 9 / 12
    assert roc_auc(y, p) == pytest.approx(9 / 12)
    # sorted: .9(+) .85(-) .8(+) .4(-) .3(+) ... -> AP = 1/3*1 + 1/3*(2/3) + 1/3*(3/5)
    assert average_precision(y, p) == pytest.approx((1 + 2 / 3 + 3 / 5) / 3)
    m = at_threshold(y, p, 0.5)
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (2, 1, 3, 1)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 3 / 4) / 2)
    assert roc_auc([1, 0], [0.7, 0.7]) == pytest.approx(0.5) and math.isnan(roc_auc([1, 1], [0.1, 0.2]))
    assert choose_threshold([1, 1, 0, 0], [0.9, 0.6, 0.5, 0.1]) == pytest.approx(0.6)
    try:  # cross-check with scikit-learn when it is installed (it is on Colab)
        from sklearn.metrics import average_precision_score, roc_auc_score
        r = np.random.default_rng(0)
        yy, pp = r.integers(0, 2, 300), np.round(r.random(300), 2)   # rounding creates ties
        assert roc_auc(yy, pp) == pytest.approx(roc_auc_score(yy, pp))
        assert average_precision(yy, pp) == pytest.approx(average_precision_score(yy, pp))
    except ImportError:
        pass


def test_threshold_comes_from_validation_only():
    from cosmic.evaluation import evaluate
    r = np.random.default_rng(1)
    val = pd.DataFrame({"set": "val", "auid": [f"V{i % 6}" for i in range(60)], "label": [i % 6 % 2 for i in range(60)]})
    val["prob"] = np.clip(0.3 * val["label"] + r.random(60) * 0.7, 0, 1)
    test = pd.DataFrame({"set": "test", "auid": [f"T{i % 6}" for i in range(60)], "label": [i % 6 % 2 for i in range(60)],
                         "prob": r.random(60)})
    a = evaluate(pd.concat([val, test]))
    flipped = test.assign(label=1 - test["label"])
    b = evaluate(pd.concat([val, flipped]))
    assert a["window"]["threshold"] == b["window"]["threshold"]           # test labels never move the threshold
    assert a["star"]["threshold"] == b["star"]["threshold"]
    assert a["window"]["n_test"] == 60 and a["star"]["n_test"] == 6


# ---------------------------------------------------------------- windowing
def test_window_membership_formula_matches_window_counts():
    from cosmic.windowing import window_counts, window_ks
    r = np.random.default_rng(2)
    dt = np.sort(r.uniform(0, 731, 3000)); dt[0] = 0.0
    dt = np.r_[dt, [50.0, 100.0, 150.0, 99.999999]]   # exact window edges
    dt.sort()
    starts, counts = window_counts(dt, 100, 50)
    from collections import Counter
    c = Counter(k for x in dt for k in window_ks(x, 100, 50))
    assert [c.get(k, 0) for k in range(len(starts))] == list(counts)
    with pytest.raises(ValueError):
        window_ks(10.0, 50, 100)                          # stride > length is refused


def test_build_sequence_subsample_pad_and_gaps():
    from cosmic.windows import build_sequence, vocab
    n = 7
    pdf = pd.DataFrame({"window_id": "A#0000", "t_days": [5.0, 0.0, 1.0, 3.0, 3.0, 10.0, 20.0],
                        "mag_norm": np.arange(n, dtype=float), "uncertainty": [0.1, np.nan, 0.2, 0.3, np.nan, 0.1, 0.1],
                        "a_k": 0.5, "obs_offset": 0.0, "band_idx": 2, "type_idx": 3})
    pad = build_sequence(pdf, 10).iloc[0]
    assert pad["seq_len"] == 7 and pad["n_obs"] == 7 and len(pad["mag_norm"]) == 10
    assert pad["t_days"][:7] == sorted(pdf["t_days"]) and pad["t_days"][7:] == [0.0, 0.0, 0.0]
    assert pad["logdt"][0] == 0.0 and pad["logdt"][1] == pytest.approx(np.log1p(1.0))
    assert sum(pad["unc_missing"]) == 2
    sub = build_sequence(pdf, 4).iloc[0]
    assert sub["seq_len"] == 4 and sub["n_obs"] == 7
    assert sub["t_days"][0] == 0.0 and sub["t_days"][-1] == 20.0     # first and last kept
    assert vocab(pd.DataFrame({"value": ["V", "Vis.", "B"], "n": [5, 9, 1]}), 2) == {"Vis.": 2, "V": 3}


class _ViewCfg:
    """Minimal stand-in for Config in local-mode Spark tests: tables are temp views named t_<table>."""

    def __init__(self, settings, overrides=None):
        import copy
        self.settings = copy.deepcopy(settings)
        for (sec, key), v in (overrides or {}).items():
            self.settings[sec][key] = v

    def table(self, name):
        return f"t_{name}"

    def section(self, name):
        return self.settings[name]


@pytest.fixture(scope="module")
def windowed(spark, clean_views, scored, settings, synth_dir):
    from cosmic.labels import assign_labels
    from cosmic.windows import assign_windows, window_features, window_sequences
    pdf, _, _ = scored
    spark.table("t_clean").createOrReplaceTempView("t_obs_clean")
    spark.createDataFrame(pdf[["observer", "score", "low_n", "median_residual"]]).createOrReplaceTempView("t_observer_scores")
    vsx = pd.read_csv(os.path.join(synth_dir, "labels", "vsx_labels.csv"))
    stars = vsx[["auid"]].assign(data_name=vsx["name"])
    lab, _ = assign_labels(stars, vsx, {"anomalous": ["UGSS", "UGSU", "UGZ", "ZAND"], "normal": ["M", "SRB", "SRA"]},
                           settings["ingest"]["dec_band_edges"])
    spark.createDataFrame(lab.astype(object).where(lab.notna(), None),
                          "auid STRING, data_name STRING, name STRING, ra DOUBLE, dec DOUBLE, vsx_type STRING, "
                          "primary_type STRING, label INT, sky_region STRING").createOrReplaceTempView("t_star_labels")
    cfg = _ViewCfg(settings, {("windows", "seq_len"): 32})
    vb, vt = assign_windows(spark, cfg)
    feats = window_features(spark, cfg).toPandas()
    seqs = window_sequences(spark, cfg).toPandas()
    w_obs = spark.table("w_obs").select("auid", "jd", "window_id", "k", "t_days").toPandas()
    clean = spark.table("t_clean").select("auid", "jd").toPandas()
    return dict(feats=feats, seqs=seqs, w_obs=w_obs, clean=clean, cfg=cfg, vb=vb)


def test_spark_windows_match_the_window_rule(windowed):
    from cosmic.windowing import count_windows
    w = windowed["cfg"].section("windows")
    feats, clean = windowed["feats"], windowed["clean"]
    got = feats.groupby("auid").size()
    for auid, d in clean.groupby("auid"):
        want = count_windows(d["jd"], w["length_days"], w["stride_days"], w["min_obs"])
        assert got.get(auid, 0) == want, auid
    assert (feats["n_obs"] >= w["min_obs"]).all()
    wo = windowed["w_obs"]
    assert ((wo["t_days"] >= 0) & (wo["t_days"] < w["length_days"])).all()
    assert feats["window_id"].is_unique and feats["label"].notna().any()
    assert ((feats["w_a"] > 0) & (feats["w_a"] <= 1)).all()


def test_spark_sequences(windowed):
    seqs, feats = windowed["seqs"].set_index("window_id"), windowed["feats"].set_index("window_id")
    assert set(seqs.index) == set(feats.index)
    assert (seqs["seq_len"] == np.minimum(seqs["n_obs"], 32)).all()
    assert (seqs["n_obs"] == feats.loc[seqs.index, "n_obs"].astype(int)).all()
    assert seqs["mag_norm"].map(len).eq(32).all()
    t = seqs.iloc[0]["t_days"][:seqs.iloc[0]["seq_len"]]
    assert list(t) == sorted(t)


def test_classify_fit_predict_on_windows(spark, windowed):
    from cosmic.classify import fit_predict
    from cosmic.splits import make_splits
    feats = windowed["feats"]
    lab = feats.dropna(subset=["label"]).drop_duplicates("auid")[["auid", "label"]].astype({"label": int})
    sp = make_splits(lab, [1])
    c = dict(dt_max_depth=[3], dt_min_instances=[1], rf_num_trees=[10], rf_max_depth=[3], lr_reg_param=[0.01],
             lr_elastic_net=[0.0], lr_max_iter=50)
    pred, chosen = fit_predict(spark.createDataFrame(feats.astype(object).where(feats.notna(), None)),
                               spark.createDataFrame(sp), c, [1])
    assert set(pred["model"]) == {"NB", "DT", "RF", "LR"} and set(pred["set"]) == {"val", "test"}
    assert set(pred["variant"]) == {"with_observer", "without_observer"}
    assert pred["prob"].between(0, 1).all()
    test_stars = set(sp.loc[sp["set"] == "test", "auid"])
    assert set(pred.loc[pred["set"] == "test", "auid"]) <= test_stars     # only test stars in the test predictions
    assert len(chosen) == 8


# ---------------------------------------------------------------- RGA-Net (PyTorch, synthetic tensors only)
def _toy_windows(tmp_path, n_stars=24, per_star=6, L=24, seed=0):
    """Synthetic token tensors for tests: anomalous windows have one bright excursion."""
    import json
    r = np.random.default_rng(seed)
    N = n_stars * per_star
    label = np.repeat(np.arange(n_stars) % 2, per_star)
    t = np.sort(r.uniform(0, 100, (N, L)), axis=1)
    mag = r.normal(0, 0.1, (N, L)) - 1.5 * label[:, None] * (np.abs(t - 50) < 8)
    a = r.uniform(0.2, 1.0, (N, L))
    num = np.stack([mag, t, np.log1p(np.diff(t, prepend=t[:, :1], axis=1)), np.full((N, L), 0.1),
                    np.zeros((N, L)), a, np.zeros((N, L))], -1).astype(np.float32)
    mask = np.ones((N, L), bool); mask[:, -4:] = False
    np.savez_compressed(tmp_path / "windows_seq.npz", num=num, mask=mask, band=np.full((N, L), 2), otype=np.full((N, L), 2),
                        label=label, window_id=np.array([f"S{i // per_star:02d}#{i % per_star:04d}" for i in range(N)]),
                        auid=np.array([f"S{i // per_star:02d}" for i in range(N)]),
                        feature_names=np.array(["mag_norm", "t_days", "logdt", "unc", "unc_missing", "a_k", "obs_offset"]))
    (tmp_path / "vocab.json").write_text(json.dumps(dict(n_band=4, n_type=4)))
    from cosmic.splits import make_splits
    stars = pd.DataFrame({"auid": [f"S{i:02d}" for i in range(n_stars)], "label": np.arange(n_stars) % 2})
    make_splits(stars, [1, 2]).to_csv(tmp_path / "splits.csv", index=False)


def test_reliability_gate_shifts_attention():
    import torch
    from cosmic.rganet import GatedSelfAttention
    torch.manual_seed(0)
    att = GatedSelfAttention(8, 2, 0.0, gate=True, lambda_init=5.0)
    x = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, True, True, False]])
    log_a = torch.log(torch.tensor([[0.9, 0.01, 0.5, 0.9]]) + 1e-3)
    B, L, d = x.shape
    q, k, v = att.qkv(x).view(B, L, 3, 2, 4).permute(2, 0, 3, 1, 4)
    s = q @ k.transpose(-1, -2) / 2.0 + att.lam() * log_a[:, None, None, :]
    w = torch.softmax(s.masked_fill(~mask[:, None, None, :], float("-inf")), -1)
    assert torch.all(w[..., 3] == 0)                                    # padding gets no attention
    assert torch.all(w[..., 0] > w[..., 1])                             # reliable observer outweighs unreliable one
    assert float(att.lam().detach()) > 0
    att.forward(x, mask, log_a)                                         # runs


def test_rganet_trains_and_ablations_run(tmp_path):
    import torch
    from cosmic.config import load_config
    from cosmic.rganet import load_windows, train_one
    _toy_windows(tmp_path)
    data = load_windows(str(tmp_path))
    split = pd.read_csv(tmp_path / "splits.csv").query("seed == 1")
    p = dict(load_config("synthetic").section("rganet"), d_model=16, heads=2, layers=1, max_epochs=6, patience=3,
             batch_size=16)
    res = {}
    for v in ("full", "no_gate", "observer_blind"):
        pred, hist, info = train_one(data, split, v, 1, p, torch.device("cpu"), log=lambda *_: None)
        res[v] = (pred, hist, info)
        assert set(pred["set"]) == {"val", "test"} and pred["prob"].between(0, 1).all()
        assert 1 <= info["best_epoch"] <= len(hist) <= 6
        test_stars = set(split.loc[split["set"] == "test", "auid"])
        assert set(pred.loc[pred["set"] == "test", "auid"]) == test_stars
    assert res["full"][2]["lambda_layer1"] > 0 and res["no_gate"][2]["lambda_layer1"] == 0
    assert res["observer_blind"][2]["parameters"] < res["no_gate"][2]["parameters"]   # 2 fewer token inputs
    pred2, _, _ = train_one(data, split, "full", 1, p, torch.device("cpu"), log=lambda *_: None)
    assert np.allclose(pred2["prob"], res["full"][0]["prob"], atol=1e-5)               # fixed seed: reproducible


def test_compare_end_to_end_on_synthetic_predictions(tmp_path):
    """Synthetic predictions in a temporary folder only: checks every table, figure and COMPARISON.md is produced."""
    from cosmic import compare
    from cosmic.config import load_config
    r = np.random.default_rng(3)
    rows = []
    for seed in (1, 2, 3):
        for s, n in (("val", 40), ("test", 40)):
            auid = np.array([f"{s}{seed}_{i % 8}" for i in range(n)]); y = (np.arange(n) % 8 < 4).astype(int)
            for model, variant in [("RGA-Net", "full"), ("RGA-Net", "no_gate"), ("RGA-Net", "observer_blind"),
                                   *[(m, v) for m in ("NB", "DT", "RF", "LR") for v in ("with_observer", "without_observer")]]:
                rows.append(pd.DataFrame(dict(model=model, variant=variant, seed=seed, set=s, auid=auid,
                                              window_id=[f"{a}#{i}" for i, a in enumerate(auid)], label=y,
                                              prob=np.clip(0.4 * y + r.random(n) * 0.6, 0, 1))))
    pred = pd.concat(rows)
    cfg = load_config("synthetic", [f"datasets.synthetic.project_root={tmp_path}",
                                    f"datasets.synthetic.output_subdir={tmp_path / 'out'}"])
    work = cfg.work_dir
    pred[pred.model == "RGA-Net"].to_csv(os.path.join(work, "predictions_rganet.csv"), index=False)
    pred[pred.model != "RGA-Net"].to_csv(os.path.join(work, "predictions_ml.csv"), index=False)
    pd.DataFrame([dict(variant=v, seed=s, train_seconds=10.0 + s, lambda_layer1=0.5, lambda_pool=0.7)
                  for v in ("full", "no_gate", "observer_blind") for s in (1, 2, 3)]).to_csv(
        os.path.join(work, "rganet_runs.csv"), index=False)
    pd.DataFrame([dict(epoch=e, train_loss=1 / e, val_loss=1.2 / e, val_pr_auc=0.5 + e / 100, seed=s, variant="full")
                  for s in (1, 2, 3) for e in range(1, 6)]).to_csv(os.path.join(work, "rganet_history.csv"), index=False)
    pd.DataFrame([dict(model=m, variant=v, seed=s, fit_seconds=2.0) for m in ("NB", "DT", "RF", "LR")
                  for v in ("with_observer", "without_observer") for s in (1, 2, 3)]).to_csv(
        os.path.join(cfg.output_dir, "ml_selected_params.csv"), index=False)
    cfg.settings["split"]["curves_seed"] = 1
    comp = compare.run(cfg)
    assert list(comp["model"]) == ["RGA-Net", "NB", "DT", "RF", "LR"]
    out = cfg.output_dir
    for f in ("comparison_table.csv", "comparison_table_star_level.csv", "ablation_table.csv",
              "ml_observer_features_table.csv", "metrics_per_seed.csv", "COMPARISON.md",
              "fig_comparison_metrics.html", "fig_roc_pr_curves.html", "fig_confusion_matrices.html",
              "fig_rganet_training_curves.html", "fig_ablation.html"):
        assert os.path.exists(os.path.join(out, f)), f
    md = open(os.path.join(out, "COMPARISON.md"), encoding="utf-8").read()
    assert "seed-to-seed spread" in md and "RGA-Net" in md
    assert "review2" not in out.replace("\\", "/").split("/")


# ---------------------------------------------------------------- dataset audit (synthetic, no Spark)
def test_window_rule():
    from cosmic.windowing import count_windows, longest_coverage, window_counts
    jd = np.concatenate([np.arange(0, 100, 1.0), np.arange(300, 330, 1.0)])
    starts, counts = window_counts(jd, 100, 50)
    assert list(starts) == [0, 50, 100, 150, 200, 250, 300]
    assert list(counts) == [100, 50, 0, 0, 0, 30, 30]
    assert count_windows(jd, 100, 50, 20) == 4
    assert count_windows(jd, 100, 50, 31) == 2
    assert longest_coverage(jd, 30) == 99.0               # the 200-day gap breaks coverage
    assert longest_coverage(jd, 250) == 329.0
    assert count_windows([], 100, 50, 20) == 0


def test_audit_on_synthetic(synth_dir, tmp_path):
    import shutil
    from cosmic import audit
    from cosmic.config import load_config
    from cosmic.rawfiles import RAW_COLUMNS
    data = tmp_path / "data"
    shutil.copytree(os.path.join(synth_dir, "raw"), data)
    n_synth = len(os.listdir(data))
    first = sorted(os.listdir(data))[0]
    copy = "zz_copy_" + first                                          # read after the original
    shutil.copy(data / first, data / copy)                             # an overlapping download
    (data / "bad.csv").write_text("star,date\nx,1\n")
    # one record whose quoted comment contains a line break
    rec = '9,T CrB,000-BBW-825,2460702.9,10.1,0.01,False,V,CCD,SCMA,,,,,,,,,"line one\r\nline two"\n'
    (data / "multi.csv").write_bytes((",".join(RAW_COLUMNS) + "\n" + rec).encode())
    cfg = load_config("synthetic", [f"datasets.synthetic.project_root={tmp_path}", f"datasets.synthetic.data_subdir={data}",
                                    f"datasets.synthetic.labels_file={os.path.join(synth_dir, 'labels', 'vsx_labels.csv')}",
                                    f"datasets.synthetic.output_subdir={tmp_path / 'out'}"])
    r = audit.run(cfg, fetch_vsx=False)
    s = dict(zip(r["summary"]["item"], r["summary"]["value"]))
    files = r["files"].set_index("file")
    dup_rows = files.loc[copy, "rows"]
    assert files.loc[copy, "repeated_rows"] == dup_rows
    assert files.loc["multi.csv", "rows"] == 1
    # the copied file repeats all its rows; the synthetic files also contain ~1% exact re-submissions
    assert s["repeated rows (same auid, jd, band, observer, mag)"] == files["repeated_rows"].sum() > dup_rows
    assert list(r["skipped"]["file"]) == ["bad.csv"]
    assert s["csv files accepted"] == n_synth + 2
    assert s["unique-observation size, MB (repeats across files removed)"] < s["total CSV size, MB (1 MB = 1,048,576 bytes)"]
    assert s["target 50 MB: total size"] == "FAIL"                       # synthetic data is tiny
    # every row is counted once: rows in files = sum of per-star unique rows + repeats
    assert r["stars"]["rows"].sum() + files["repeated_rows"].sum() == s["rows (all files)"]
    # exact bytes: the multi-line record is counted with all its bytes
    assert audit.iter_records(str(data / "multi.csv")).__next__()[0] == len(rec.encode())
    bal = r["balance"].set_index("class")
    assert bal["stars"].sum() == r["stars"]["auid"].nunique()
    assert bal.loc["unlabelled", "stars"] >= 2                          # the synthetic file has 2 unmatched types
    assert len(r["unmatched"]) == bal.loc["unlabelled", "stars"]


# ---------------------------------------------------------------- figures (Plotly only)
def test_no_matplotlib_or_seaborn_anywhere():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hits = []
    for d in ("src", "scripts", "setup", "tests"):
        for dirpath, _, files in os.walk(os.path.join(root, d)):
            for f in files:
                if f.endswith(".py") and f != "test_pipeline.py":
                    text = open(os.path.join(dirpath, f), encoding="utf-8").read()
                    if "import matplotlib" in text or "import seaborn" in text or "from matplotlib" in text:
                        hits.append(f)
    assert hits == []
    reqs = [l.split("#")[0].strip().lower() for l in open(os.path.join(root, "requirements.txt"))]
    assert not any(r.startswith(("matplotlib", "seaborn")) for r in reqs)


def test_plotly_figure_saved_as_html_json_and_png_or_logged(tmp_path):
    from cosmic import plots
    pdf = pd.DataFrame({"jd": [2460000.0, 2460001.0, 2460002.0], "mag": [10.0, 11.0, 10.5],
                        "mag_norm": [-0.5, 0.5, 0.0], "band": ["V"] * 3, "observer": ["A", "B", "A"]})
    plots.lightcurve_before_after(pdf, "X", ["A", "B"], str(tmp_path))
    base = tmp_path / "fig_lightcurve_before_after"
    assert (tmp_path / (base.name + ".html")).exists() and (tmp_path / (base.name + ".json")).exists()
    log = pd.read_csv(tmp_path / "figure_export_log.csv")
    assert list(log["figure"]) == [base.name]
    assert (tmp_path / (base.name + ".png")).exists() == bool(log["png"].iloc[0])   # PNG or a logged failure
    import plotly.io as pio
    fig = pio.read_json(str(tmp_path / (base.name + ".json")))
    assert fig.layout.yaxis.autorange == "reversed" and fig.layout.yaxis2.autorange == "reversed"


def test_architecture_dot_has_windowing_and_rganet():
    from cosmic.architecture import dot_source
    src = dot_source()
    assert src.startswith("digraph") and "windowing" in src and "RGA-Net" in src and "PyTorch" in src


def test_star_attrition_explains_every_star(spark, windowed, typed_view):
    from cosmic.preprocess import star_counts
    from cosmic.splits import make_splits
    from cosmic.windows import star_attrition
    spark.createDataFrame(star_counts(spark, "t_clean", source_view=typed_view)).createOrReplaceTempView("t_preprocess_star_log")
    spark.table(typed_view).createOrReplaceTempView("t_observations_raw")
    feats = windowed["feats"]
    per_star = feats.groupby("auid").size().rename("windows").reset_index()
    lab = feats.dropna(subset=["label"]).drop_duplicates("auid")[["auid", "label"]].astype({"label": int})
    att = star_attrition(spark, windowed["cfg"], per_star, make_splits(lab, [1]))
    raw_stars = {r[0] for r in spark.sql(f"SELECT DISTINCT auid FROM {typed_view}").collect()}
    assert set(att["auid"]) == raw_stars                                    # every raw star is listed
    used = att[att["status"] == "used"]
    assert set(used["auid"]) == set(lab["auid"]) and used["set_first_seed"].notna().all()
    assert (att.loc[att["status"] != "used", "reason"] != "").all()          # every unused star has a reason
    assert att["reason"].str.startswith("unlabelled").sum() >= 2             # the synthetic unlisted types
    cols = [c for c in att.columns if c.startswith("rows_")]
    assert (att[cols].diff(axis=1).iloc[:, 1:] <= 0).all().all()             # counts never grow step to step


def test_only_the_data_folder_is_read(tmp_path, monkeypatch):
    from cosmic.config import load_config
    from cosmic.rawfiles import discover
    monkeypatch.setenv("COSMIC_PROJECT_ROOT", str(tmp_path))
    for rel in ("data/a.csv", "data/b.csv", "outputs/review2/audit_files.csv", "work/labels/vsx_labels.csv",
                "work/splits.csv", "stray.csv"):
        _write(str(tmp_path / rel), _good())
    _, files, skipped = discover(load_config("real"))
    rel = sorted(os.path.relpath(f, tmp_path).replace(os.sep, "/") for f in files)
    assert rel == ["data/a.csv", "data/b.csv"] and skipped == []


def test_underscore_and_padded_names():
    from cosmic.labels import norm_name, vsx_name_candidates
    assert vsx_name_candidates("DY_PEG")[0] == "DY Peg"
    assert vsx_name_candidates("V603_AQL")[0] == "V0603 Aql"
    assert norm_name("DY_PEG") == norm_name("DY Peg")


# ---------------------------------------------------------------- distributed inference (Spark + saved RGA-Net)
def _toy_sequence_frame(data, idx):
    """The toy windows in the shape of the Hive table window_sequences (one array column per token feature)."""
    names = list(data["feature_names"])
    rows = dict(window_id=data["window_id"][idx], seq_len=data["mask"][idx].sum(1).astype(int),
                band_idx=[list(map(int, r)) for r in data["band"][idx]],
                type_idx=[list(map(int, r)) for r in data["otype"][idx]])
    for j, n in enumerate(names):
        rows[n] = [list(map(float, r)) for r in data["num"][idx][..., j]]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def saved_models(tmp_path_factory):
    import torch
    from cosmic.config import load_config
    from cosmic.rganet import load_windows, train_one
    d = tmp_path_factory.mktemp("rga")
    _toy_windows(d)
    data = load_windows(str(d))
    split = pd.read_csv(d / "splits.csv").query("seed == 1")
    p = dict(load_config("synthetic").section("rganet"), d_model=16, heads=2, layers=1, max_epochs=3, patience=2,
             batch_size=16)
    out = {}
    for v in ("full", "observer_blind"):
        pred, _, _ = train_one(data, split, v, 1, p, torch.device("cpu"), log=lambda *_: None,
                               save_path=str(d / f"rganet_{v}_seed1.pt"))
        out[v] = pred
    return d, data, out


def test_saved_model_reproduces_training_predictions(saved_models):
    from cosmic.rganet import load_model, predict_frame
    d, data, preds = saved_models
    for v, pred in preds.items():
        model, meta = load_model(str(d / f"rganet_{v}_seed1.pt"))
        assert meta["variant"] == v and meta["seed"] == 1
        idx = np.flatnonzero(np.isin(data["window_id"], pred["window_id"]))
        got = predict_frame(_toy_sequence_frame(data, idx), [(model, meta)]).set_index("window_id")["prob"]
        want = pred.set_index("window_id")["prob"]
        assert np.allclose(got.loc[want.index], want, atol=1e-5)          # same preprocessing, same numbers


def test_spark_distributed_inference_matches_pytorch(spark, saved_models):
    from cosmic.infer import OUT_SCHEMA, load_blobs, scorer
    d, data, preds = saved_models
    blobs = load_blobs(str(d), ["full"])
    assert list(blobs) == ["rganet_full_seed1.pt"]
    sdf = spark.createDataFrame(_toy_sequence_frame(data, np.arange(len(data["window_id"])))).repartition(3)
    out = sdf.mapInPandas(scorer(spark.sparkContext.broadcast(blobs)), schema=OUT_SCHEMA).toPandas()
    assert len(out) == len(data["window_id"]) and set(out["variant"]) == {"full"}   # every window scored once
    ref = preds["full"].set_index("window_id")["prob"]
    got = out.set_index("window_id")["prob"]
    assert np.allclose(got.loc[ref.index], ref, atol=1e-5)


def test_duplicate_names_detected():
    from cosmic.rawfiles import duplicate_names
    assert duplicate_names(["a.csv", "b.csv", "A.CSV", "c.csv"]) == [["a.csv", "A.CSV"]]
    assert duplicate_names(["observations_1.csv", "observations_2.csv"]) == []
