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


def test_recursive_discovery_and_header_check(tmp_path):
    from cosmic.rawfiles import RAW_COLUMNS, check_headers, find_csv_files
    good = ",".join(RAW_COLUMNS) + "\n1,SS CYG,000-BCP-220,2460676.6,12.5,0.05,False,V,CCD,ABC,,,,,,,,,\n"
    root = str(tmp_path)
    _write(os.path.join(root, "AAVSO_DATA", "EXISTING", "a.csv"), good)
    _write(os.path.join(root, "AAVSO_DATA", "NEW_2025", "a.csv"), good)          # same name, other round
    _write(os.path.join(root, "AAVSO_DATA", "NEW_2025", "b.CSV"), "﻿" + good)  # BOM, upper-case extension
    _write(os.path.join(root, "AAVSO_DATA", "NEW_2025", "bad.csv"), "star,date,mag\nx,1,2\n")
    _write(os.path.join(root, "AAVSO_DATA", "NEW_2025", "empty.csv"), "")
    _write(os.path.join(root, "AAVSO_DATA", "DOWNLOAD_LOG", "log.csv"), good)
    _write(os.path.join(root, "AAVSO_DATA", "NEW_2025", "x.zip"), "zip")
    files = find_csv_files(root)
    rel = sorted(os.path.relpath(f, root).replace(os.sep, "/") for f in files)
    assert rel == ["AAVSO_DATA/EXISTING/a.csv", "AAVSO_DATA/NEW_2025/a.csv", "AAVSO_DATA/NEW_2025/b.CSV",
                   "AAVSO_DATA/NEW_2025/bad.csv", "AAVSO_DATA/NEW_2025/empty.csv"]
    ok, skipped = check_headers(files, root)
    assert len(ok) == 3
    assert sorted(os.path.basename(s["file"]) for s in skipped) == ["bad.csv", "empty.csv"]


def test_data_root_override(monkeypatch):
    from cosmic.config import load_config
    monkeypatch.setenv("COSMIC_DATA_ROOT", "/content/drive/MyDrive/X")
    assert load_config("real").raw_dir == "/content/drive/MyDrive/X"
    assert load_config("synthetic").raw_dir.replace(os.sep, "/").endswith("data/synthetic/raw")  # env is real-only


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
    rows = read_raw(spark, "file://" + str(f)).collect()
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
    root = tmp_path / "data"
    raw = root / "raw"
    shutil.copytree(os.path.join(synth_dir, "raw"), raw / "EXISTING")
    shutil.copytree(os.path.join(synth_dir, "labels"), root / "labels")
    first = sorted(os.listdir(raw / "EXISTING"))[0]
    (raw / "NEW_2025").mkdir()
    shutil.copy(raw / "EXISTING" / first, raw / "NEW_2025" / first)       # an overlapping download
    (raw / "NEW_2025" / "bad.csv").write_text("star,date\nx,1\n")
    (raw / "DOWNLOAD_LOG").mkdir()
    shutil.copy(raw / "EXISTING" / first, raw / "DOWNLOAD_LOG" / "x.csv")
    # one record whose quoted comment contains a line break
    rec = '9,T CrB,000-BBW-825,2460702.9,10.1,0.01,False,V,CCD,SCMA,,,,,,,,,"line one\r\nline two"\n'
    (raw / "NEW_2025" / "multi.csv").write_bytes((",".join(RAW_COLUMNS) + "\n" + rec).encode())
    cfg = load_config("synthetic", [f"datasets.synthetic.root={root}", f"datasets.synthetic.output_subdir={tmp_path / 'out'}"])
    r = audit.run(cfg, fetch_vsx=False)
    s = dict(zip(r["summary"]["item"], r["summary"]["value"]))
    files = r["files"].set_index("file")
    dup_rows = files.loc[f"NEW_2025/{first}", "rows"]
    assert files.loc[f"NEW_2025/{first}", "repeated_rows"] == dup_rows
    assert files.loc["NEW_2025/multi.csv", "rows"] == 1
    # the copied file repeats all its rows; the synthetic files also contain ~1% exact re-submissions
    assert s["repeated rows (same auid, jd, band, observer, mag)"] == files["repeated_rows"].sum() > dup_rows
    assert list(r["skipped"]["file"]) == ["NEW_2025/bad.csv"]
    assert s["csv files accepted"] == len(os.listdir(raw / "EXISTING")) + 2
    assert s["unique-observation size, MB (repeats across files removed)"] < s["total CSV size, MB (1 MB = 1,048,576 bytes)"]
    assert s["target 50 MB: total size"] == "FAIL"                       # synthetic data is tiny
    # every row is counted once: rows in files = sum of per-star unique rows + repeats
    assert r["stars"]["rows"].sum() + files["repeated_rows"].sum() == s["rows (all files)"]
    # exact bytes: the multi-line record is counted with all its bytes
    assert audit.iter_records(str(raw / "NEW_2025" / "multi.csv")).__next__()[0] == len(rec.encode())
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
