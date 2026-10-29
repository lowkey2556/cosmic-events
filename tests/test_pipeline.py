"""Unit tests of the pipeline logic (local-mode Spark, synthetic data only)."""
import json
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
    monkeypatch.delenv("V3_ROOT", raising=False)
    p = {k: v.replace(os.sep, "/") for k, v in cfg.paths().items()}
    root = str(tmp_path).replace(os.sep, "/")
    v3 = root + "/v3_5GB"
    assert p == {"PROJECT_ROOT": root, "data (AAVSO CSVs, read only)": root + "/data",
                 "tarball cache (read only)": root + "/work/cache", "VSX labels file": root + "/work/labels/vsx_labels.csv",
                 "V3_ROOT (all new writes)": v3, "outputs": v3 + "/outputs", "work": v3 + "/work",
                 "ZTF raw": v3 + "/ztf_raw", "clean": v3 + "/clean", "dashboard data": v3 + "/dashboard_data"}
    with pytest.raises(FileNotFoundError, match="data folder does not exist"):
        check_paths(cfg)                                                  # data/ missing
    (tmp_path / "data").mkdir()
    check_paths(cfg)
    assert load_config("synthetic").data_dir.replace(os.sep, "/").endswith("data/synthetic/raw")  # env is real-only


def test_no_write_folder_outside_v3_root(monkeypatch, tmp_path):
    from cosmic.config import check_paths, load_config
    monkeypatch.setenv("COSMIC_PROJECT_ROOT", str(tmp_path))
    (tmp_path / "data").mkdir()
    with pytest.raises(ValueError, match="outside V3_ROOT"):
        check_paths(load_config("real", [f"datasets.real.output_subdir={tmp_path / 'outputs'}"]))


@pytest.mark.parametrize("root", ["/content/drive/MyDrive/cosmic", "/content/drive/MyDrive/cosmic/",
                                  "/content/drive/MyDrive/cosmic_OLD_do_not_use",
                                  "/content/drive/MyDrive/cosmic_roja/_OLD_100MB"])
def test_retired_folders_are_refused(monkeypatch, root):
    from cosmic.config import check_paths, is_retired, load_config
    monkeypatch.setenv("COSMIC_PROJECT_ROOT", root)
    with pytest.raises(ValueError, match="retired folder"):
        check_paths(load_config("real"), need_data=False)
    assert not is_retired("/content/drive/MyDrive/cosmic_roja/data")    # the new project folder is fine
    assert is_retired("/content/drive/MyDrive/cosmic/cache/x.tgz")
    assert is_retired("/content/drive/MyDrive/cosmic_roja/_OLD_100MB/outputs/x.csv")


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
    rows = [n for _, n, _ in log]
    stars = [k for _, _, k in log]
    assert all(a >= b for a, b in zip(stars, stars[1:])) and stars[0] > 0
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


def _synthetic_predictions(seeds=(1, 2, 3), with_observer_boost=0.0, rng_seed=3):
    """Saved-prediction files in the shape every model writes (validation and test windows, several seeds)."""
    r = np.random.default_rng(rng_seed)
    rows = []
    for seed in seeds:
        for s, n in (("val", 40), ("test", 40)):
            auid = np.array([f"{s}{seed}_{i % 8}" for i in range(n)]); y = (np.arange(n) % 8 < 4).astype(int)
            for model, variant in [("RGA-Net", "full"), ("RGA-Net", "no_pretrain"), ("RGA-Net", "no_gate"),
                                   ("RGA-Net", "observer_blind"),
                                   *[(m, v) for m in ("NB", "DT", "RF", "LR") for v in ("with_observer", "without_observer")]]:
                sep = 0.4 + (with_observer_boost if variant == "with_observer" else 0.0)
                rows.append(pd.DataFrame(dict(model=model, variant=variant, seed=seed, set=s, auid=auid,
                                              window_id=[f"{a}#{i}" for i, a in enumerate(auid)], label=y,
                                              prob=np.clip(sep * y + r.random(n) * 0.6, 0, 1))))
    return pd.concat(rows, ignore_index=True)


def _compare_cfg(tmp_path, pred, seeds=(1, 2, 3)):
    from cosmic.config import load_config
    cfg = load_config("synthetic", [f"datasets.synthetic.project_root={tmp_path}",
                                    f"datasets.synthetic.output_subdir={tmp_path / 'out'}"])
    work = cfg.work_dir
    pred[pred.model == "RGA-Net"].to_csv(os.path.join(work, "predictions_rganet.csv"), index=False)
    pred[pred.model != "RGA-Net"].assign(data_setting="aavso").to_csv(os.path.join(work, "predictions_ml.csv"),
                                                                      index=False)
    pd.DataFrame([dict(variant=v, seed=s, train_seconds=10.0 + s, lambda_layer1=0.5 + s / 100, lambda_layer2=0.6,
                       lambda_pool=0.7) for v in ("full", "no_pretrain", "no_gate", "observer_blind") for s in seeds]) \
        .to_csv(os.path.join(work, "rganet_runs.csv"), index=False)
    return cfg


COMPARE_FILES = ("per_seed_metrics.csv", "comparison_table.csv", "prior_vs_proposed_table.csv",
                 "prior_vs_proposed_paired.csv", "observer_feature_gain.csv", "ablation_table.csv", "ablation_paired.csv",
                 "rganet_lambda.csv", "metrics_appendix.csv", "COMPARISON.md")


def test_compare_end_to_end_on_synthetic_predictions(tmp_path):
    """Synthetic predictions in a temporary folder only: every Phase 3 table and COMPARISON.md is produced from the saved
    predictions, with MCC/ROC-AUC/AUPRC/kappa/Brier, both levels, and 10-seed style mean ± std."""
    from cosmic import compare
    cfg = _compare_cfg(tmp_path, _synthetic_predictions())
    comp = compare.run(cfg)
    out = cfg.output_dir
    for f in COMPARE_FILES:
        assert os.path.exists(os.path.join(out, f)), f
    w = comp[comp["level"] == "window"]
    assert list(w["model"]) == ["NB", "DT", "RF", "LR", "NB", "DT", "RF", "LR", "RGA-Net"]
    assert set(comp["level"]) == {"window", "star"} and (comp["seeds"] == 3).all()
    for m in ("mcc", "roc_auc", "pr_auc", "kappa", "brier"):
        assert comp[m].str.contains(" ± ").all() and comp[f"{m}_mean"].notna().all()
    ab = pd.read_csv(os.path.join(out, "ablation_table.csv"))
    assert set(ab["variant"]) == {"full", "no_pretrain", "no_gate", "observer_blind"}
    lam = pd.read_csv(os.path.join(out, "rganet_lambda.csv"))
    assert set(lam["variant"]) == {"full", "no_pretrain"}                 # only gated variants have a λ
    md = open(os.path.join(out, "COMPARISON.md"), encoding="utf-8").read()
    for phrase in ("Prior approaches = the four classic ML models without observer features", "Wilcoxon",
                   "larger than the seed spread", "Honest note", "Learned λ"):
        assert phrase in md, phrase
    assert "review2" not in out.replace("\\", "/").split("/")


def test_fix1_prior_approaches_are_classic_models_without_observer_features(tmp_path):
    """Fix 1: the 'prior approach' rows and the RGA-Net-vs-prior statistics use ONLY the classic models WITHOUT observer
    features; the with-observer models are labelled as our framework and never change the prior comparison."""
    from cosmic import compare
    from cosmic.evaluation import evaluate_all
    pred = _synthetic_predictions(with_observer_boost=0.5)                # with-observer models are much better
    cfg = _compare_cfg(tmp_path, pred)
    comp = compare.run(cfg)
    prior = comp[comp["role"] == "prior approach"]
    assert set(prior["variant"]) == {"without_observer"} and set(prior["model"]) == {"NB", "DT", "RF", "LR"}
    fw = comp[comp["role"] == "our framework on a classic model"]
    assert set(fw["variant"]) == {"with_observer"} and not fw["display_name"].str.contains("without").any()
    assert set(comp.loc[comp["role"] == "proposed", "variant"]) == {"full"}
    pvp = pd.read_csv(os.path.join(cfg.output_dir, "prior_vs_proposed_paired.csv"))
    assert set(pvp["prior_model"]) == {"NB", "DT", "RF", "LR"}
    # changing the with-observer predictions does not move the prior-vs-proposed statistics at all
    per_seed = evaluate_all(pred)
    pred2 = pred.copy()
    wo = pred2["variant"] == "with_observer"
    pred2.loc[wo, "prob"] = 1 - pred2.loc[wo, "prob"]
    a = compare.prior_vs_proposed(per_seed, "window")
    b = compare.prior_vs_proposed(evaluate_all(pred2), "window")
    pd.testing.assert_frame_equal(a, b)
    gain = pd.read_csv(os.path.join(cfg.output_dir, "observer_feature_gain.csv"))
    g = gain[(gain["level"] == "window") & (gain["metric"] == "mcc")]
    assert (g["with_mean"] > g["without_mean"]).all()                      # the boost shows up only here
    md = open(os.path.join(cfg.output_dir, "COMPARISON.md"), encoding="utf-8").read()
    assert "not prior approaches" in md


def test_paired_wilcoxon_matches_scipy():
    from scipy.stats import wilcoxon
    from cosmic.compare import paired, wilcoxon_p
    d = np.array([0.10, 0.05, -0.02, 0.08, 0.12, 0.03, 0.07, -0.01, 0.09, 0.04])
    assert wilcoxon_p(d) == pytest.approx(wilcoxon(d).pvalue)
    assert math.isnan(wilcoxon_p(np.zeros(10))) and math.isnan(wilcoxon_p([0.1]))
    ps = pd.DataFrame([dict(model=m, variant="v", level="window", seed=s, mcc=b + (d[s - 1] if m == "A" else 0),
                            roc_auc=0.8, pr_auc=0.7, kappa=0.5, brier=0.2 - (0.01 if m == "A" else 0))
                       for m, b in (("A", 0.5), ("B", 0.5)) for s in range(1, 11)])
    r = paired(ps, ("A", "v"), ("B", "v"), "window").set_index("metric")
    assert r.loc["mcc", "mean_difference"] == pytest.approx(d.mean()) and r.loc["mcc", "seeds_ahead"] == 8
    assert r.loc["brier", "seeds_ahead"] == 10                              # lower Brier counts as ahead
    assert r.loc["mcc", "wilcoxon_p"] == pytest.approx(wilcoxon(d).pvalue)


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
    d = pd.DataFrame({"jd": [2460000.0, 2460001.0, 2460002.0], "mag": [10.0, 11.0, 10.5], "band": ["V"] * 3,
                      "observer": ["A", "B", "A"], "A_k": [0.9, 0.3, 0.9]})
    fig, cap = plots.lightcurve_ak(d, "X")
    plots.save(fig, str(tmp_path), "04_lightcurve_ak")
    base = "04_lightcurve_ak"
    assert (tmp_path / (base + ".html")).exists() and (tmp_path / (base + ".json")).exists()
    log = pd.read_csv(tmp_path / "figure_export_log.csv")
    assert list(log["figure"]) == [base]
    assert (tmp_path / (base + ".png")).exists() == bool(log["png"].iloc[0])   # PNG or a logged failure
    import plotly.io as pio
    f = pio.read_json(str(tmp_path / (base + ".json")))
    assert f.layout.yaxis.autorange == "reversed"                               # brighter = up
    assert cap.startswith("Inference:") and "3 cleaned observations" in cap


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


# ---------------------------------------------------------------- v3 Phase 0: robustness
def test_window_npz_has_no_object_arrays_and_loads(tmp_path):
    from cosmic.rganet import load_windows
    from cosmic.windows import NUM_TOKEN_FEATURES, save_window_npz
    L = 6
    seq = pd.DataFrame({"window_id": ["000-AAA-111#0000", "000-BBB-22#0001"], "auid": ["000-AAA-111", "000-BBB-22"],
                        "label": [1, None], "seq_len": [6, 3], "band_idx": [[2] * L] * 2, "type_idx": [[3] * L] * 2,
                        **{c: [[0.5] * L] * 2 for c in NUM_TOKEN_FEATURES}})
    save_window_npz(str(tmp_path / "windows_seq.npz"), seq, L)
    (tmp_path / "vocab.json").write_text('{"n_band": 4, "n_type": 5}')
    z = np.load(tmp_path / "windows_seq.npz", allow_pickle=False)              # no pickle needed at all
    assert all(z[k].dtype != object for k in z.files)
    assert z["window_id"].dtype.kind == "U" and list(z["auid"]) == ["000-AAA-111", "000-BBB-22"]
    d = load_windows(str(tmp_path))
    assert d["num"].shape == (2, L, len(NUM_TOKEN_FEATURES)) and list(d["label"]) == [1, -1]
    assert d["mask"].sum(1).tolist() == [6, 3]
    # a file written the old way (object string arrays) still loads
    np.savez(tmp_path / "windows_seq.npz", **{**{k: z[k] for k in z.files},
                                             "window_id": np.array(list(z["window_id"]), dtype=object)})
    assert list(load_windows(str(tmp_path))["window_id"]) == list(z["window_id"])


def test_fresh_machine_rebuilds_missing_tables_in_order():
    from cosmic.pipeline import plan_steps
    none = lambda t: False
    assert [s for s, _ in plan_steps(["infer"], none)] == ["ingest", "preprocess", "score", "windows", "infer"]
    assert [s for s, _ in plan_steps(["classify"], none)] == ["ingest", "preprocess", "score", "windows", "classify"]
    every = lambda t: True
    assert plan_steps(["infer"], every) == [("infer", "requested")]          # nothing rebuilt when tables exist
    partial = lambda t: t in {"observations_raw", "star_labels", "observations_flat"}
    assert [s for s, _ in plan_steps(["score"], partial)] == ["preprocess", "score"]
    # scalability: only the missing prerequisites, not the timed steps themselves
    assert [s for s, _ in plan_steps(["preprocess", "windows"], partial, only_missing=True)] == ["preprocess", "score"]
    assert plan_steps(["preprocess", "windows"], every, only_missing=True) == []


def test_install_script_finds_drive_cache_without_env_vars():
    text = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "setup", "install_stack.sh"),
                encoding="utf-8").read()
    assert "DRIVE_CACHE=/content/drive/MyDrive/cosmic_roja/work/cache" in text
    assert 'echo "using cached $f' in text and "_OLD_100MB" in text


# ---------------------------------------------------------------- v3 Phase 1: ZTF and candidates (no network)
def test_ztf_checksum_and_field_parsing():
    from cosmic.ztf import parse_checksums, parse_field_centres
    idx = parse_checksums("6cff324f8bfb274c12e807a7385d87e1  ./0/field000199/ztf_000199_zg_c09_q1_dr24.parquet\n"
                          "2f74  ./1/field001240/ztf_001240_zr_c16_q4_dr24.parquet\nnot a line\n")
    assert idx.to_dict("records") == [
        dict(rel_path="0/field000199/ztf_000199_zg_c09_q1_dr24.parquet", md5="6cff324f8bfb274c12e807a7385d87e1",
             field=199, filter="zg", ccd=9, q=1),
        dict(rel_path="1/field001240/ztf_001240_zr_c16_q4_dr24.parquet", md5="2f74", field=1240, filter="zr", ccd=16, q=4)]
    f = parse_field_centres("% ID RA Dec Ebv GalLong GalLat EclLong EclLat Entry\n"
                            "000600     10.0  24.0   0.1   120.0  -38.5   20.0   15.0   1\n")
    assert f.to_dict("records") == [dict(field=600, ra=10.0, dec=24.0, gal_lat=-38.5)]


def test_ztf_budget_saturation_and_separation():
    from cosmic.ztf import ang_sep_deg, api_int, fill_to_budget, unsaturated_mask
    assert (api_int("0xd"), api_int("0x1"), api_int(731), api_int("731")) == (13, 1, 731, 731)  # API hex ids
    c = pd.DataFrame({"size_bytes": [4, 4, 4, 1]})
    assert fill_to_budget(0, c, 8, 10) == ([0, 1], 8)                         # stops once the minimum is reached
    assert fill_to_budget(5, c, 9, 10) == ([0], 9)
    assert fill_to_budget(5, c, 10, 10) == ([0, 3], 10)                       # skips files that would exceed max
    m = unsaturated_mask([11.0, 13.0, 14.0, np.nan], [0, 0, 32768, 0], 12.5)
    assert m.tolist() == [False, True, False, False]                          # saturated / flagged / missing out
    assert ang_sep_deg(10.0, 20.0, 10.0, 21.0) == pytest.approx(1.0)
    assert ang_sep_deg(0.0, 0.0, 0.0, 2 / 3600) * 3600 == pytest.approx(2.0)


def test_candidates_balanced_across_classes_and_types():
    from cosmic.candidates import GCVS_NAME, balanced_pick
    c = pd.DataFrame({"label": [1] * 6 + [0] * 6, "primary_type": ["UGSS"] * 4 + ["NL"] * 2 + ["M"] * 3 + ["EA"] * 3,
                      "rank": range(12)})
    p = balanced_pick(c, 6)
    assert (p["label"] == 1).sum() == 3 and (p["label"] == 0).sum() == 3
    assert set(p.loc[p["label"] == 1, "primary_type"]) == {"UGSS", "NL"}       # types are spread, not one type only
    assert GCVS_NAME.match("SS Cyg") and GCVS_NAME.match("V1504 Cyg") and GCVS_NAME.match("omi Cet")
    assert not GCVS_NAME.match("CSS 101127:000130+050624") and not GCVS_NAME.match("ASASSN-13cx")


def _ztf_test_project(tmp_path):
    """A tiny project with one AAVSO star and one synthetic ZTF DR-format file (test data only)."""
    import hashlib
    import pyarrow as pa
    import pyarrow.parquet as pq
    from cosmic.config import load_config
    from cosmic.rawfiles import RAW_COLUMNS
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.csv").write_text(",".join(RAW_COLUMNS) + "\n1,SS CYG,000-BCP-220,2460676.6,12.5,0.05,False,V,CCD,ABC,,,,,,,,,\n")
    (tmp_path / "labels.csv").write_text("auid,name,ra,dec,type\n000-BCP-220,SS Cyg,10.0,20.0,UGSS\n")
    src = tmp_path / "src"
    src.mkdir()
    name = "ztf_000600_zr_c01_q1_dr24.parquet"
    t = pa.table({"objectid": [1, 2, 3], "filterid": pa.array([2, 2, 2], pa.int8()), "fieldid": pa.array([600] * 3, pa.int16()),
                  "rcid": pa.array([0] * 3, pa.int8()), "objra": pa.array([10.0, 10.1, 10.0 + 1 / 3600], pa.float32()),
                  "objdec": pa.array([20.0, 20.1, 20.0], pa.float32()), "nepochs": [4, 2, 3],
                  "hmjd": [[58300.0, 58301.0, 58302.0, 58303.0], [58300.5, 58310.0], [59000.0, 59001.0, 59002.0]],
                  "mag": pa.array([[11.0, 13.0, 14.0, 15.0], [18.0, 18.1], [16.0, 16.1, 16.2]], pa.list_(pa.float32())),
                  "magerr": pa.array([[0.01] * 4, [0.1] * 2, [0.05] * 3], pa.list_(pa.float32())),
                  "clrcoeff": pa.array([[0.1] * 4, [0.1] * 2, [0.1] * 3], pa.list_(pa.float32())),
                  "catflags": pa.array([[0, 0, 32768, 0], [0, 0], [0, 0, 0]], pa.list_(pa.int32()))})
    pq.write_table(t, src / name)
    md5 = hashlib.md5((src / name).read_bytes()).hexdigest()
    cfg = load_config("synthetic", [f"datasets.synthetic.project_root={tmp_path}", f"datasets.synthetic.data_subdir={data}",
                                    f"datasets.synthetic.labels_file={tmp_path / 'labels.csv'}",
                                    f"datasets.synthetic.output_subdir={tmp_path / 'out'}",
                                    f"ztf.tmp_dir={tmp_path / 'ztftmp'}"])
    sel = pd.DataFrame([dict(rel_path=f"0/field000600/{name}", md5=md5, field=600, filter="zr", ccd=1, q=1,
                             url=(src / name).as_uri(), size_bytes=(src / name).stat().st_size, priority=1,
                             reason="contains AAVSO star(s): SS Cyg")])
    sel.to_csv(os.path.join(cfg.output_dir, "ztf_selection.csv"), index=False)
    pd.DataFrame([dict(auid="000-BCP-220", name="SS Cyg", label=1, field=600, ccd=1, q=1, filter="zr", epochs=7,
                       unsaturated=5, median_mag=15.0)]).to_csv(os.path.join(cfg.output_dir, "ztf_star_tiles.csv"), index=False)
    return cfg, sel, name


def test_ztf_download_verifies_resumes_and_audit_counts(tmp_path, monkeypatch, capsys):
    from cosmic import ztf
    monkeypatch.setattr(ztf.shutil, "which", lambda _: None)                 # no wget on the test machine: urllib
    cfg, sel, name = _ztf_test_project(tmp_path)
    ztf.run_download(cfg)
    dest = os.path.join(cfg.ztf_raw_dir, "field000600", name)
    assert os.path.exists(dest) and ztf.md5_of(dest) == sel["md5"].iat[0]
    assert "downloaded, checksum OK" in capsys.readouterr().out
    ztf.run_download(cfg)                                                       # second run: nothing to do
    assert "to download: 0" in capsys.readouterr().out
    assert len(pd.read_csv(os.path.join(cfg.ztf_raw_dir, "MANIFEST.csv"))) == 1
    s = ztf.run_audit(cfg).set_index("item")["value"]
    assert s["files"] == 1 and s["light curves (rows: one per object and filter)"] == 3
    assert s["epochs (photometric points)"] == 9 and s["filters"] == "zr"
    assert s["AAVSO stars found in the downloaded ZTF files (2 arcsec)"] == 1
    st = pd.read_csv(os.path.join(cfg.output_dir, "ztf_aavso_stars.csv"))
    # objects 1 and 3 are within 2"; object 1: 11.0 saturated, 14.0 flagged -> 2 unsaturated; object 3: 3
    assert sorted(st["objectid"]) == [1, 3] and st["unsaturated"].sum() == 5


def test_ztf_download_stops_on_checksum_mismatch(tmp_path, monkeypatch):
    from cosmic import ztf
    monkeypatch.setattr(ztf.shutil, "which", lambda _: None)
    cfg, sel, _ = _ztf_test_project(tmp_path)
    sel.assign(md5="0" * 32).to_csv(os.path.join(cfg.output_dir, "ztf_selection.csv"), index=False)
    with pytest.raises(RuntimeError, match="MD5 mismatch"):
        ztf.run_download(cfg)
    assert not os.path.exists(os.path.join(cfg.ztf_raw_dir, "MANIFEST.csv"))   # nothing recorded as verified


def test_vizier_falls_back_to_a_mirror(monkeypatch):
    import astroquery.vizier
    from cosmic import vizier as vz
    calls = []

    class FakeVizier:
        def __init__(self, vizier_server=None, **kw):
            self.srv = vizier_server

        def query_constraints(self, **kw):
            calls.append(self.srv)
            if self.srv == "down.example":
                raise ConnectionError("connect timeout")
            return [[{"Name": "SS Cyg"}]]

    monkeypatch.setattr(astroquery.vizier, "Vizier", FakeVizier)
    vz._working.clear()
    assert vz.pick_server(["down.example", "mirror.example", "never.example"]) == "mirror.example"
    assert calls == ["down.example", "mirror.example"]                       # stops at the first that answers
    assert vz.pick_server(["down.example", "mirror.example", "never.example"]) == "mirror.example"
    assert calls == ["down.example", "mirror.example"]                       # remembered: no new test queries
    vz._working.clear()
    with pytest.raises(RuntimeError, match="No VizieR server answered"):
        vz.pick_server(["down.example"])


def test_field_ranking_without_vizier(monkeypatch):
    from cosmic import vsx
    from cosmic import ztf
    from cosmic.config import load_config

    def unreachable(*a, **k):
        raise RuntimeError("No VSX backend answered (synthetic test)")

    monkeypatch.setattr(vsx, "backend", unreachable)
    centres = pd.DataFrame({"field": [1, 2], "ra": [10.0, 20.0], "dec": [5.0, 6.0]})
    assert ztf.vsx_richness(load_config("synthetic"), centres, 3.5) is None    # caller ranks by distance instead


# ---------------------------------------------------------------- Phase 3 metrics (MCC, kappa, Brier, thresholds)
def test_mcc_kappa_brier_by_hand_and_against_sklearn():
    from cosmic.evaluation import at_threshold, brier, choose_threshold, kappa_from, mcc_from
    # tp=3, fp=1, fn=1, tn=5: MCC = (15 - 1) / sqrt(4 * 4 * 6 * 6) = 14 / 24
    assert mcc_from(3, 1, 1, 5) == pytest.approx(14 / 24)
    # po = 0.8, pe = (4*4 + 6*6) / 100 = 0.52 -> kappa = 0.28 / 0.48
    assert kappa_from(3, 1, 1, 5) == pytest.approx(0.28 / 0.48)
    assert mcc_from(5, 0, 0, 0) == 0.0                                       # undefined -> 0, never NaN
    assert brier([1, 0, 1], [0.8, 0.3, 0.4]) == pytest.approx((0.04 + 0.09 + 0.36) / 3)
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.7, 0.35, 0.6, 0.3, 0.2, 0.1, 0.05])
    t = choose_threshold(y, p)
    # t = 0.35: tp 3, fp 1, fn 0, tn 4 -> MCC 12 / sqrt(240) = 0.775 beats t = 0.7 (10 / sqrt(180) = 0.745)
    assert t == pytest.approx(0.35)
    try:
        from sklearn.metrics import brier_score_loss, cohen_kappa_score, matthews_corrcoef
    except ImportError:
        return
    r = np.random.default_rng(5)
    yy, pp = r.integers(0, 2, 400), r.random(400)
    for thr in (0.3, 0.5, 0.71):
        m = at_threshold(yy, pp, thr)
        pred = (pp >= thr).astype(int)
        assert m["mcc"] == pytest.approx(matthews_corrcoef(yy, pred))
        assert m["kappa"] == pytest.approx(cohen_kappa_score(yy, pred))
    assert brier(yy, pp) == pytest.approx(brier_score_loss(yy, pp))
    # the chosen threshold really maximises MCC over every candidate threshold
    best = max(matthews_corrcoef(yy, (pp >= c).astype(int)) for c in np.unique(pp))
    assert at_threshold(yy, pp, choose_threshold(yy, pp))["mcc"] == pytest.approx(best)


def test_evaluate_reports_five_metrics_at_both_levels():
    from cosmic.evaluation import METRICS, evaluate_all, mean_std
    pred = _synthetic_predictions(seeds=(1, 2))
    per_seed = evaluate_all(pred)
    assert set(per_seed["level"]) == {"window", "star"}
    for m in METRICS + ["threshold", "tp", "fp", "fn", "tn"]:
        assert per_seed[m].notna().all(), m
    star = per_seed[per_seed["level"] == "star"]
    assert (star["n_test"] == 8).all()                                       # 8 test stars per seed
    ms = mean_std(per_seed)
    assert (ms["seeds"] == 2).all() and ms["mcc"].str.contains(" ± ").all()


# ---------------------------------------------------------------- leakage
def test_leakage_rganet_statistics_and_augmentation_use_training_windows_only(tmp_path, monkeypatch):
    import torch
    from cosmic import rganet
    from cosmic.config import load_config
    _toy_windows(tmp_path)
    data = rganet.load_windows(str(tmp_path))
    split = pd.read_csv(tmp_path / "splits.csv").query("seed == 1")
    p = dict(load_config("synthetic").section("rganet"), d_model=16, heads=2, layers=1, max_epochs=2, patience=2,
             batch_size=16)
    calls = []
    real = rganet.augment
    monkeypatch.setattr(rganet, "augment", lambda xb, *a, **k: (calls.append(len(xb["mask"])), real(xb, *a, **k))[1])
    save = tmp_path / "m.pt"
    rganet.train_one(data, split, "no_pretrain", 1, p, torch.device("cpu"), log=lambda *_: None, save_path=str(save))
    _, meta = rganet.load_model(str(save))
    n_train = int(np.isin(data["auid"], split.loc[split["set"] == "train", "auid"]).sum())
    assert calls and sum(calls) == 2 * n_train                               # augmentation saw train windows only
    # token statistics come from the training windows only: change every val/test token -> identical statistics
    other = ~np.isin(data["auid"], split.loc[split["set"] == "train", "auid"])
    data2 = dict(data, num=data["num"].copy())
    data2["num"][other] += 100.0
    rganet.train_one(data2, split, "no_pretrain", 1, p, torch.device("cpu"), log=lambda *_: None, save_path=str(save))
    _, meta2 = rganet.load_model(str(save))
    assert np.allclose(meta["mean"], meta2["mean"]) and np.allclose(meta["std"], meta2["std"])


def test_leakage_ztf_windows_only_ever_train_the_ml_models(spark, windowed):
    from cosmic.classify import fit_predict
    from cosmic.splits import make_splits
    feats = windowed["feats"]
    lab = feats.dropna(subset=["label"]).drop_duplicates("auid")[["auid", "label"]].astype({"label": int})
    sp = make_splits(lab, [1])
    z = feats.dropna(subset=["label"]).head(40).copy()
    z["auid"] = "ZTF" + z["auid"]; z["window_id"] = "ZTF" + z["window_id"]
    c = dict(dt_max_depth=[3], dt_min_instances=[1], rf_num_trees=[5], rf_max_depth=[3], lr_reg_param=[0.01],
             lr_elastic_net=[0.0], lr_max_iter=30, data_settings=["aavso", "aavso_ztf"])
    sdf = lambda d: spark.createDataFrame(d.astype(object).where(d.notna(), None))  # noqa: E731
    pred, chosen = fit_predict(sdf(feats), spark.createDataFrame(sp), c, [1], ztf=sdf(z))
    assert not pred["auid"].str.startswith("ZTF").any()                      # never in validation or test
    assert set(pred["data_setting"]) <= {"aavso", "aavso_ztf"}
    ztf_runs = chosen[chosen["data_setting"] == "aavso_ztf"]
    assert (ztf_runs["n_train_ztf"] == 40).all() and (chosen.loc[chosen["data_setting"] == "aavso", "n_train_ztf"] == 0).all()
    for s in ("val", "test"):
        assert set(pred.loc[pred["set"] == s, "auid"]) <= set(sp.loc[sp["set"] == s, "auid"])


def test_leakage_ztf_objects_near_aavso_stars_are_flagged_and_match_radius(spark):
    from cosmic.ztf_spark import match_labels
    a = 1 / 3600
    spark.createDataFrame(pd.DataFrame({
        "objectid": [1, 2, 3, 4], "field": 1, "filter": "zr",
        "objra": [10.0, 10.0 + 1.5 * a, 50.0 + 3 * a, 80.0], "objdec": [20.0, 20.0, -5.0, 60.0 + 1 * a]})) \
        .createOrReplaceTempView("t_obj")
    spark.createDataFrame(pd.DataFrame({"name": ["V1", "V2", "V3"], "ra": [10.0, 50.0, 80.0], "dec": [20.0, -5.0, 60.0],
                                        "type": ["UGSS", "M", "EA"], "primary_type": ["UGSS", "M", "EA"],
                                        "label": [1, 0, 0]})).createOrReplaceTempView("t_vsx")
    spark.createDataFrame(pd.DataFrame({"auid": ["000-AAA-001"], "ra": [80.0], "dec": [60.0]})) \
        .createOrReplaceTempView("t_aav")
    out = match_labels(spark, "t_obj", "t_vsx", "t_aav", 2.0, 10.0).toPandas().set_index("objectid")
    assert sorted(out.index) == [1, 2, 4]                                    # object 3 is 3" away: no label
    assert out.loc[1, "vsx_name"] == "V1" and out.loc[2, "vsx_name"] == "V1" and out.loc[2, "sep_arcsec"] == pytest.approx(1.5 * np.cos(np.radians(20)), abs=0.01)  # RA offset x cos(dec)
    assert out.loc[4, "aavso_overlap"] and not out.loc[1, "aavso_overlap"]     # 1" from an AAVSO star -> excluded


# ---------------------------------------------------------------- Phase 4: captions, DATA_FLOW, dashboard dictionary
def _caption_inputs():
    raw = pd.DataFrame({"jd": [1.0, 2.0], "mag": [10.0, 11.0], "fainterthan": [False, False]})
    clean = pd.DataFrame({"jd": [1.0, 2.0], "mag": [10.0, 11.0], "mag_norm": [-0.5, 0.5], "A_k": [0.9, 0.4]})
    dq = pd.DataFrame([dict(dataset="AAVSO", stage="before", size_mb=10.0, rows=100, nulls=3, duplicates=2, flagged=1,
                            bands=3, stars=5, observers=4),
                       dict(dataset="AAVSO", stage="after", size_mb=4.0, rows=80, nulls=0, duplicates=0, flagged=0,
                            bands=3, stars=5, observers=4)])
    steps = {"AAVSO": pd.DataFrame({"step": ["0 raw", "1 dedup", "2 thin"], "rows": [100, 95, 80]})}
    scores = pd.DataFrame({"observer": list("ABCDEF"), "obs_type": ["CCD", "CCD", "Visual", "Visual", "DSLR", "CCD"],
                           "score": [0.9, 0.8, 0.4, 0.5, 0.6, 0.7], "low_n": [False] * 5 + [True]})
    per_obs = pd.DataFrame({"observer": list("ABCD"), "A_k": [0.9, 0.7, 0.5, 0.3], "scatter_vs_ztf_mag": [0.05, 0.1, 0.2, 0.3],
                            "n_matched": [20, 30, 40, 50], "low_n": [False] * 4})
    summ = dict(spearman_rho=-1.0, p_value=0.0, observers=4, matched_observations=140, enough_to_conclude=False)
    comp = pd.DataFrame({"source": ["AAVSO", "AAVSO", "ZTF"], "class": ["anomalous", "unlabelled", "normal"],
                         "vsx_type": ["UGSS", None, "M"], "rows": [50, 10, 70]})
    return raw, clean, dq, steps, scores, per_obs, summ, comp


def test_every_chart_has_a_generated_inference_caption():
    from cosmic import plots
    raw, clean, dq, steps, scores, per_obs, summ, comp = _caption_inputs()
    lc = pd.DataFrame({"jd": [1.0, 2.0, 3.0], "mag": [10.0, 10.5, 11.0], "band": ["V", "V", "B"],
                       "observer": ["A", "B", "A"], "A_k": [0.9, 0.3, 0.9]})
    caps = [plots.caption_samples(raw, clean, "AAVSO"), plots.data_quality(dq)[1], plots.cleaning_sankey(steps)[1],
            plots.lightcurve_ak(lc, "X")[1], plots.ak_violin(scores)[1], plots.ak_validation(per_obs, summ)[1],
            plots.composition_sunburst(comp)[1]]
    for c in caps:
        assert c.startswith("Inference: ") and len(c) > 40, c
    assert "No nulls or duplicates remain" in caps[1]                        # data-driven, not a fixed sentence
    assert "too few to conclude" in caps[5]                                  # small overlap is said, not hidden


def test_report_makes_figures_and_captions_from_saved_files_and_retires_v2(tmp_path):
    import json
    from cosmic import compare, report
    cfg = _compare_cfg(tmp_path, _synthetic_predictions())
    compare.run(cfg)
    out = cfg.output_dir
    for f in ("fig_comparison_metrics.png", "fig_obs_per_star.html", "comparison_table_star_level.csv"):  # v2 leftovers
        open(os.path.join(out, f), "w").close()
    report.run(cfg)
    caps = json.load(open(os.path.join(out, "figure_captions.json"), encoding="utf-8"))
    for name in ("08_comparison_table", "09_prior_vs_proposed", "10_radar", "11_observer_feature_dumbbell",
                 "12_ablation", "13_confusion_grid"):
        assert caps[name]["caption"].startswith("Inference:"), name
        assert os.path.exists(os.path.join(cfg.figures_dir, name + ".json"))
    fig09 = open(os.path.join(cfg.figures_dir, "09_prior_vs_proposed.json"), encoding="utf-8").read()
    assert "dtype" not in fig09 and "without observer features" in fig09     # one clean row per prior approach
    md = open(os.path.join(out, "figure_captions.md"), encoding="utf-8").read()
    assert "Not made (input missing)" in md and "02_data_quality" in md       # missing inputs are listed, never faked
    assert not [f for f in os.listdir(out) if f.startswith("fig_")]           # old v2 charts moved away...
    assert sorted(os.listdir(os.path.join(out, "retired_v2"))) == ["comparison_table_star_level.csv",
                                                                   "fig_comparison_metrics.png", "fig_obs_per_star.html"]
    assert os.path.exists(os.path.join(out, "DATA_FLOW.txt"))


def test_data_flow_is_generated_from_recorded_measurements(tmp_path):
    from cosmic import dataflow
    from cosmic.config import load_config
    cfg = load_config("synthetic", [f"datasets.synthetic.project_root={tmp_path}",
                                    f"datasets.synthetic.output_subdir={tmp_path / 'out'}"])
    dataflow.record(cfg, "ztf_clean", "Spark cleans the arrays.", "HDFS /x", "Parquet", "Spark SQL",
                    size_bytes=5 * 2 ** 30 + 1, rows=123456789, files=7, compression="raw 8.5 GB -> clean 5.0 GB",
                    changes="epochs removed", seconds=42.04, extra={"counting pass": "10 s"})
    dataflow.record(cfg, "aavso_source", "CSV files.", "/data", "CSV", "manual", size_bytes=2048, rows=10)
    dataflow.record_timing(cfg, "ztf_clean", 42.04, "Spark (yarn)")
    dataflow.record_timing(cfg, "rganet", 7.0, "plain Python + PyTorch")
    dataflow.record_timing(cfg, "ztf_clean", 40.0, "Spark (yarn)")            # a rerun replaces the old time
    text = open(dataflow.write(cfg), encoding="utf-8").read()
    assert "[07] ZTF cleaned light curves" in text and "Size         : 5.00 GB" in text
    assert "Rows         : 123,456,789" in text and "Time taken   : 42.0 s" in text and "counting pass" in text
    assert "Size         : 2.00 KB" in text
    assert text.count("not run yet") == len(dataflow.STAGES) - 2             # nothing typed by hand
    assert "STEP TIMINGS" in text and "ztf_clean" in text and "47.0 s" in text  # 40.0 + 7.0
    with pytest.raises(KeyError):
        dataflow.record(cfg, "no_such_stage", "", "", "", "")


# ---------------------------------------------------------------- fresh machine, notebook, no bar charts
def test_fresh_machine_plans_the_ztf_steps(monkeypatch, tmp_path):
    from cosmic.config import load_config
    from cosmic.pipeline import plan_steps
    plan = [s for s, _ in plan_steps(["akval"], lambda t: False)]
    assert plan.index("ingest") < plan.index("preprocess") < plan.index("score") < plan.index("akval")
    assert plan.index("ztf_clean") < plan.index("akval")
    plan = [s for s, _ in plan_steps(["ztf_windows"], lambda t: False)]
    assert plan.index("ztf_clean") < plan.index("ztf_label") < plan.index("ztf_windows")
    have = {"observations_raw", "star_labels", "obs_clean", "ztf_lightcurves_clean", "ztf_labels"}
    assert [s for s, _ in plan_steps(["ztf_windows"], lambda t: t in have)] == ["ztf_windows"]
    # on a fresh machine every path comes from PROJECT_ROOT; the non-Spark steps read only V3_ROOT/work files
    monkeypatch.setenv("COSMIC_PROJECT_ROOT", str(tmp_path / "proj"))
    monkeypatch.delenv("V3_ROOT", raising=False)
    cfg = load_config("real")
    for p in (cfg.work_dir, cfg.output_dir, cfg.figures_dir, cfg.dashboard_dir, cfg.clean_dir, cfg.ztf_raw_dir):
        assert p.startswith(str(tmp_path / "proj" / "v3_5GB")), p
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for f in ("rganet.py", "compare.py", "report.py", "dataflow.py", "plots.py", "evaluation.py"):
        assert "/content" not in open(os.path.join(root, "src", "cosmic", f), encoding="utf-8").read(), f


def test_local_hdfs_stand_in(tmp_path, monkeypatch):
    from cosmic import spark_session
    monkeypatch.setenv("COSMIC_LOCAL_FS_ROOT", str(tmp_path / "fs"))
    src = tmp_path / "a.csv"; src.write_text("x\n")
    spark_session.hdfs("-mkdir", "-p", "/c/raw")
    spark_session.hdfs("-put", "-f", str(src), "/c/raw/")
    assert (tmp_path / "fs" / "c" / "raw" / "a.csv").read_text() == "x\n"
    assert "a.csv" in spark_session.hdfs("-ls", "-R", "-h", "/c/raw")
    spark_session.hdfs("-rm", "-r", "-f", "-skipTrash", "/c/raw")
    assert not (tmp_path / "fs" / "c" / "raw").exists()


def _notebook(tmp_path):
    import build_notebook
    nb = build_notebook.build(str(tmp_path / "nb.ipynb"))
    return build_notebook, nb, ["".join(c["source"]) for c in nb["cells"]]


def test_notebook_runs_every_step_in_order_and_ends_with_the_results(tmp_path):
    import json
    import re
    bn, nb, src = _notebook(tmp_path)
    json.loads(open(tmp_path / "nb.ipynb", encoding="utf-8").read())          # valid JSON notebook
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            body = "\n".join(l for l in "".join(c["source"]).split("\n") if not l.lstrip().startswith(("!", "%")))
            compile(body, "cell", "exec")                                     # every Python cell compiles
    steps = [s for cell in src for m in re.findall(r"run\.py ([a-z_+]+)", cell) for s in m.split("+")]
    order = ["ztf_download", "ztf_audit", "audit", "architecture", "ingest", "preprocess", "score", "features",
             "ztf_ingest", "ztf_clean", "ztf_label", "windows", "ztf_windows", "akval", "cluster", "classify", "rganet",
             "infer", "compare", "dashboard", "report"]
    pos = [steps.index(s) for s in order]
    assert pos == sorted(pos), steps
    first_results = next(i for i, s in enumerate(src) if s.startswith(f"# {bn.RESULTS_TITLE}"))
    results = src[first_results:]
    assert not any("run.py" in s for s in results)                           # results only read saved files
    assert results[-1].rstrip().endswith('print("ALL RESULTS SHOWN")')
    for group in bn.FIGURES:
        for name in group:
            assert any(name in s for s in results), name
    assert len(bn.FIGURES) == 13


def test_no_bar_charts_histograms_or_v2_results_anywhere(tmp_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bad = ("go.Bar(", "px.bar(", "go.Histogram(", "px.histogram(", "go.Funnel(", "barmode")
    hits = []
    for d in ("src", "scripts"):
        for dirpath, _, files in os.walk(os.path.join(root, d)):
            for f in files:
                if f.endswith(".py"):
                    text = open(os.path.join(dirpath, f), encoding="utf-8").read()
                    hits += [f"{f}: {b}" for b in bad if b in text]
    assert hits == []
    _, _, src = _notebook(tmp_path)
    for old in ("fig_comparison_metrics", "fig_obs_per_star", "fig_observer_score_hist", "fig_preprocessing_funnel",
                "fig_scalability", "comparison_table_star_level", "ml_observer_features_table", "rganet_learned_lambda"):
        assert not any(old in s for s in src), old
    committed = json.load(open(os.path.join(root, "notebooks", "cosmic_colab.ipynb"), encoding="utf-8"))
    assert ["".join(c["source"]) for c in committed["cells"]] == src           # committed notebook is up to date
