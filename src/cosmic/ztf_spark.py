"""Phase 2 — ZTF at Big Data scale in Spark: ingest, clean, label, window; and A_k checked against ZTF.

ZTF light-curve files hold ONE ROW PER LIGHT CURVE (an object in one filter) with the epochs as arrays
(hmjd, mag, magerr, clrcoeff, catflags). Spark cleans those arrays with higher-order functions
(filter, arrays_zip, array_distinct), so ~850 million epochs are cleaned without exploding them into rows.

Steps (each is a run.py step):
  ztf_ingest   Copies V3_ROOT/ztf_raw/*.parquet into HDFS /<base>/ztf/raw/field=<id>/filter=<zg|zr|zi>/ (Hadoop
               FileSystem API, one JVM) and registers the external Hive table `ztf_lightcurves_raw`, partitioned by
               field and filter.
  ztf_clean    Quality cuts, counted at every step (ztf_clean_steps.csv, the Sankey input):
                 1. missing or invalid magnitude / error removed   2. catflags != 0 (quality flags) removed
                 3. saturated epochs (mag <= ztf.saturation_mag) removed   4. exact repeated epochs removed
                 5. light curves with fewer than ztf.min_clean_epochs good epochs dropped
               Only the valid filters (zg, zr, zi) are kept. Writes `ztf_lightcurves_clean` (Parquet, partitioned by
               field and filter), reports its size on disk against ztf.clean_target_gb (5 GB) and, if short, how
               many more raw files that would take at the measured clean/raw ratio. Saves raw/clean samples, the
               before/after data-quality table, and a copy in V3_ROOT/clean/ (later runs restore from it).
  ztf_label    Queries VSX (vsx.py) around every field centre, keeps our label-map types, and cross-matches ZTF
               objects to VSX variables within ztf.label_match_arcsec (Spark equi-join on declination buckets, then
               the exact angular distance). EVERY ZTF object within ztf.aavso_exclusion_arcsec of ANY AAVSO star is
               flagged and excluded from the training pool, so no AAVSO star can leak into pre-training.
  ztf_windows  The same windows as AAVSO (100 d, stride 50 d, >= 20 observations, L = 200 tokens). ZTF is one
               professional observer, so every token has A_k = 1 and offset = 0 (the reliability gate is inactive on
               ZTF). Light curves and then windows are sampled class-balanced from all fields (ztf.pretrain_windows,
               default 50,000) and the counts are logged. Writes `ztf_window_features`, `ztf_window_sequences` and
               V3_ROOT/work/ztf_windows_seq.npz for pre-training.
  akval        A_k validation: AAVSO observations of the same star matched to the nearest unsaturated ZTF point within
               akval.window_days; the per-(star, AAVSO band, ZTF filter) median difference is removed; each observer's
               robust scatter against ZTF is compared with A_k (Spearman rho, with numbers of points and observers).
               If too few observers overlap, it says so.
"""
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from . import dataflow
from .spark_session import path_bytes, save_csv, save_hive_table, table_bytes, table_files

RAW_DDL = """objectid BIGINT, filterid TINYINT, fieldid SMALLINT, rcid TINYINT, objra FLOAT, objdec FLOAT,
  nepochs BIGINT, hmjd ARRAY<DOUBLE>, mag ARRAY<FLOAT>, magerr ARRAY<FLOAT>, clrcoeff ARRAY<FLOAT>, catflags ARRAY<INT>"""
CLEAN_DDL = """objectid BIGINT, rcid TINYINT, objra FLOAT, objdec FLOAT, nepochs INT,
  hmjd ARRAY<DOUBLE>, mag ARRAY<FLOAT>, magerr ARRAY<FLOAT>, clrcoeff ARRAY<FLOAT>"""
SEP_ARCSEC = ("DEGREES(2 * ASIN(SQRT(POW(SIN(RADIANS({d2} - {d1}) / 2), 2) + COS(RADIANS({d1})) * COS(RADIANS({d2})) "
              "* POW(SIN(RADIANS({r2} - {r1}) / 2), 2)))) * 3600")


def sep_sql(r1, d1, r2, d2):
    return SEP_ARCSEC.format(r1=r1, d1=d1, r2=r2, d2=d2)


def _fs(spark):
    jvm = spark._jvm
    return jvm, jvm.org.apache.hadoop.fs.FileSystem.get(jvm.java.net.URI("hdfs://localhost:9000/"),
                                                         spark._jsc.hadoopConfiguration())


def _manifest(cfg):
    path = os.path.join(cfg.ztf_raw_dir, "MANIFEST.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("No downloaded ZTF files (V3_ROOT/ztf_raw/MANIFEST.csv): run ztf_download first.")
    from .ztf import FILE_RE
    man = pd.read_csv(path).drop_duplicates("rel_path")
    m = man["rel_path"].str.extract(FILE_RE.pattern)
    man["field"], man["filter"] = m[0].astype(int), m[1]
    man["local"] = [os.path.join(cfg.ztf_raw_dir, f"field{f:06d}", os.path.basename(r))
                    for f, r in zip(man["field"], man["rel_path"])]
    return man


def _manifest_key(man):
    return hashlib.md5("".join(sorted(man["md5"])).encode()).hexdigest()


# ------------------------------------------------------------------ ztf_ingest
def ingest_run(spark, cfg):
    t0 = time.time()
    man = _manifest(cfg)
    base = cfg.hdfs_uri(f"{cfg.hdfs_base}/ztf/raw")
    jvm, fs = _fs(spark)
    fs.delete(jvm.org.apache.hadoop.fs.Path(base), True)
    for r in man.itertuples():
        dst = jvm.org.apache.hadoop.fs.Path(f"{base}/field={r.field}/filter={r.filter}/{os.path.basename(r.local)}")
        fs.copyFromLocalFile(False, True, jvm.org.apache.hadoop.fs.Path("file://" + r.local), dst)
    t_put = time.time()
    print(f"copied {len(man)} files ({man['size_bytes'].sum() / 2**30:.2f} GB) into {base} in {t_put - t0:.0f} s")
    t = cfg.table("ztf_lightcurves_raw")
    spark.sql(f"DROP TABLE IF EXISTS {t}")
    spark.sql(f"""CREATE EXTERNAL TABLE {t} ({RAW_DDL}) PARTITIONED BY (field INT, filter STRING)
                  STORED AS PARQUET LOCATION '{base}'""")
    spark.sql(f"MSCK REPAIR TABLE {t}")
    parts = spark.sql(f"SHOW PARTITIONS {t}").count()
    s = spark.sql(f"SELECT COUNT(*) AS lightcurves, SUM(nepochs) AS epochs, COUNT(DISTINCT field) AS fields FROM {t}").first()
    size = path_bytes(spark, base)
    print(f"Hive {t}: {parts} partitions (field x filter), {s['lightcurves']:,} light curves, {s['epochs']:,} epochs, "
          f"{size / 2**30:.2f} GB")
    dataflow.record(cfg, "ztf_source", "Public ZTF DR light curves (one Parquet file per field/CCD/quadrant/filter) "
                    "downloaded from IRSA to the Colab disk, MD5-checked, then copied to Google Drive.",
                    f"{cfg.ztf_raw_dir} (via /content/ztf_tmp on the Colab disk)", "Parquet (nested arrays)",
                    "wget + md5 (ztf_download)", size_bytes=int(man["size_bytes"].sum()), rows=int(s["epochs"]),
                    files=len(man), changes="(source)", extra={"light curves": f"{s['lightcurves']:,}"})
    dataflow.record(cfg, "ztf_hdfs_hive_raw", "The files are copied into HDFS, one folder per field and filter, and "
                    "registered as an external, partitioned Hive table (no rewrite).",
                    f"HDFS {base}; Hive table {t}", "Parquet (nested arrays), partitioned by field/filter",
                    "Hadoop FileSystem API + Hive metastore", size_bytes=size, rows=int(s["epochs"]),
                    files=f"{len(man)} files in {parts} partitions", changes="none (copied as is)",
                    seconds=time.time() - t0)


# ------------------------------------------------------------------ ztf_clean
def _restore_clean(spark, cfg, man):
    """Copy a previous run's cleaned table from Drive back into HDFS when it was made from the same raw files."""
    z = cfg.section("ztf")
    copy = os.path.join(cfg.clean_dir, "ztf_lightcurves_clean")
    marker = os.path.join(copy, "_COSMIC_SOURCE.json")
    if not (z.get("copy_clean_to_drive", True) and os.path.exists(marker)):
        return False
    with open(marker) as fh:
        if json.load(fh).get("manifest") != _manifest_key(man):
            return False
    jvm, fs = _fs(spark)
    dst = cfg.hdfs_uri(f"{cfg.hdfs_base}/ztf/clean")
    fs.delete(jvm.org.apache.hadoop.fs.Path(dst), True)
    fs.copyFromLocalFile(False, True, jvm.org.apache.hadoop.fs.Path("file://" + copy), jvm.org.apache.hadoop.fs.Path(dst))
    _register_clean(spark, cfg, dst)
    print(f"restored the cleaned ZTF table from {copy} (same raw files as when it was made)")
    return True


def _register_clean(spark, cfg, loc):
    t = cfg.table("ztf_lightcurves_clean")
    spark.sql(f"DROP TABLE IF EXISTS {t}")
    spark.sql(f"""CREATE EXTERNAL TABLE {t} ({CLEAN_DDL}) PARTITIONED BY (field INT, filter STRING)
                  STORED AS PARQUET LOCATION '{loc}'""")
    spark.sql(f"MSCK REPAIR TABLE {t}")


def clean_run(spark, cfg):
    t0 = time.time()
    z = cfg.section("ztf")
    man = _manifest(cfg)
    if _restore_clean(spark, cfg, man):
        return
    if not spark.catalog.tableExists("ztf_lightcurves_raw", cfg.db):
        print("fresh machine: ztf_lightcurves_raw is missing -> running ztf_ingest first")
        ingest_run(spark, cfg)
    raw = cfg.table("ztf_lightcurves_raw")
    sat, min_ep = float(z["saturation_mag"]), int(z["min_clean_epochs"])
    filters = ", ".join(f"'{f}'" for f in z["filters"])
    spark.sql(f"""
        SELECT objectid, field, filter, rcid, objra, objdec, z0, z1, z2, z3, array_distinct(z3) AS z4
        FROM (SELECT *, filter(z2, x -> x.mag > {sat}) AS z3
              FROM (SELECT *, filter(z1, x -> x.catflags = 0) AS z2
                    FROM (SELECT *, filter(z0, x -> x.mag IS NOT NULL AND NOT isnan(x.mag) AND x.magerr IS NOT NULL
                                                   AND NOT isnan(x.magerr) AND x.magerr > 0 AND x.hmjd IS NOT NULL) AS z1
                          FROM (SELECT objectid, field, filter, rcid, objra, objdec,
                                       arrays_zip(hmjd, mag, magerr, clrcoeff, catflags) AS z0
                                FROM {raw} WHERE filter IN ({filters}) AND objra IS NOT NULL AND objdec IS NOT NULL))))
        """).createOrReplaceTempView("zc_steps")
    # pass 1: counts after every step (one scan; nothing is exploded)
    c = spark.sql(f"""SELECT COUNT(*) AS lc0, SUM(size(z0)) AS e0, SUM(size(z1)) AS e1, SUM(size(z2)) AS e2,
                             SUM(size(z3)) AS e3, SUM(size(z4)) AS e4,
                             SUM(CASE WHEN size(z4) >= {min_ep} THEN 1 ELSE 0 END) AS lc5,
                             SUM(CASE WHEN size(z4) >= {min_ep} THEN size(z4) ELSE 0 END) AS e5,
                             COUNT(DISTINCT objectid) AS obj0
                      FROM zc_steps""").first().asDict()
    raw_all = spark.sql(f"SELECT COUNT(*) AS lc, SUM(nepochs) AS e FROM {raw}").first()
    t_count = time.time()
    # pass 2: write the cleaned table (arrays only; catflags dropped because every kept epoch has catflags = 0)
    loc = cfg.hdfs_uri(f"{cfg.hdfs_base}/ztf/clean")
    spark.sql(f"""SELECT objectid, rcid, objra, objdec, size(z4) AS nepochs,
                         transform(z4, x -> x.hmjd) AS hmjd, transform(z4, x -> x.mag) AS mag,
                         transform(z4, x -> x.magerr) AS magerr, transform(z4, x -> x.clrcoeff) AS clrcoeff,
                         field, filter
                  FROM zc_steps WHERE size(z4) >= {min_ep}""") \
        .write.mode("overwrite").partitionBy("field", "filter").option("compression", "snappy").parquet(loc)
    _register_clean(spark, cfg, loc)
    t_clean = cfg.table("ztf_lightcurves_clean")
    size, raw_size = path_bytes(spark, loc), path_bytes(spark, cfg.hdfs_uri(f"{cfg.hdfs_base}/ztf/raw"))
    steps = pd.DataFrame([
        ("0 raw epochs (all filters)", raw_all["e"], raw_all["lc"]),
        (f"1 valid filters ({', '.join(z['filters'])}) and positions", c["e0"], c["lc0"]),
        ("2 missing or invalid magnitude/error removed", c["e1"], c["lc0"]),
        ("3 quality flags (catflags != 0) removed", c["e2"], c["lc0"]),
        (f"4 saturated epochs (mag <= {sat}) removed", c["e3"], c["lc0"]),
        ("5 exact repeated epochs removed", c["e4"], c["lc0"]),
        (f"6 light curves with < {min_ep} good epochs dropped", c["e5"], c["lc5"])],
        columns=["step", "rows", "lightcurves"])
    steps["rows"] = steps["rows"].astype("int64")
    steps["removed"] = (steps["rows"].shift(1) - steps["rows"]).fillna(0).astype("int64")
    steps["pct_removed_by_step"] = (100 * steps["removed"] / steps["rows"].shift(1)).fillna(0).round(2)
    print("\nZTF cleaning (rows = photometric epochs):\n" + steps.to_string(index=False))
    save_csv(steps, cfg, "ztf_clean_steps.csv")
    gb, target = size / 2**30, float(z["clean_target_gb"])
    verdict = "PASS" if gb >= target else "FAIL"
    msg = f"cleaned ZTF Parquet on disk: {gb:.3f} GB -> {verdict} against {target:g} GB"
    if verdict == "FAIL":
        ratio = size / max(raw_size, 1)
        need_raw = (target * 2**30 - size) / max(ratio, 1e-9)
        per_file = float(man["size_bytes"].mean())
        msg += (f". At the measured clean/raw ratio ({ratio:.2f}) about {need_raw / 2**30:.1f} GB more raw data "
                f"(~{int(np.ceil(need_raw / per_file))} more files of the current average size) would be needed "
                "(an estimate from this run's ratio).")
    print(msg)
    dq = pd.DataFrame([
        dict(dataset="ZTF", stage="before cleaning (raw Parquet)", size_bytes=raw_size, rows=int(raw_all["e"]),
             nulls=int(c["e0"] - c["e1"]), duplicates=int(c["e3"] - c["e4"]), flagged=int(c["e1"] - c["e2"]),
             saturated=int(c["e2"] - c["e3"]), bands=spark.sql(f"SELECT COUNT(DISTINCT filter) FROM {raw}").first()[0],
             stars=int(c["obj0"]), observers=1),
        dict(dataset="ZTF", stage="after cleaning (ztf_lightcurves_clean)", size_bytes=size, rows=int(c["e5"]),
             nulls=0, duplicates=0, flagged=0, saturated=0,
             bands=spark.sql(f"SELECT COUNT(DISTINCT filter) FROM {t_clean}").first()[0],
             stars=spark.sql(f"SELECT COUNT(DISTINCT objectid) FROM {t_clean}").first()[0], observers=1)])
    dq["size_mb"] = (dq["size_bytes"] / 2**20).round(2)
    save_csv(dq, cfg, "data_quality_ztf.csv")
    save_csv(pd.DataFrame([dict(clean_gb=round(gb, 3), target_gb=target, result=verdict, raw_gb=round(raw_size / 2**30, 3),
                                message=msg)]), cfg, "ztf_clean_size_check.csv")
    _samples(spark, cfg, raw, t_clean)
    if z.get("copy_clean_to_drive", True):
        copy = os.path.join(cfg.clean_dir, "ztf_lightcurves_clean")
        jvm, fs = _fs(spark)
        import shutil
        shutil.rmtree(copy, ignore_errors=True)
        fs.copyToLocalFile(False, jvm.org.apache.hadoop.fs.Path(loc), jvm.org.apache.hadoop.fs.Path("file://" + copy), True)
        with open(os.path.join(copy, "_COSMIC_SOURCE.json"), "w") as fh:
            json.dump(dict(manifest=_manifest_key(man), made=time.strftime("%Y-%m-%d %H:%M")), fh)
        print(f"copied the cleaned table to {copy}")
    dataflow.record(cfg, "ztf_clean", "Spark SQL cleans the epoch arrays in place with higher-order functions "
                    "(filter / arrays_zip / array_distinct): invalid values, flagged and saturated epochs and exact "
                    "repeats are removed, then light curves that are too short are dropped.",
                    f"Hive {t_clean} at {loc}; copy in {os.path.join(cfg.clean_dir, 'ztf_lightcurves_clean')}",
                    "Parquet (Snappy), partitioned by field/filter", "Spark SQL on YARN", size_bytes=size,
                    rows=int(c["e5"]), files=table_files(spark, t_clean),
                    compression=f"raw {raw_size / 2**30:.2f} GB -> clean {gb:.2f} GB ({verdict} vs {target:g} GB)",
                    changes=f"{int(raw_all['e'] - c['e5']):,} epochs removed; {int(raw_all['lc'] - c['lc5']):,} light "
                            "curves dropped; catflags column dropped (all 0 after cleaning)",
                    seconds=time.time() - t0, extra={"counting pass": f"{t_count - t0:.0f} s"})


def _samples(spark, cfg, raw, clean, n=10):
    """One ZTF light curve: 10 raw epochs with every original column, then 10 cleaned epochs with derived columns."""
    o = spark.sql(f"""SELECT objectid, filter FROM {clean} WHERE nepochs BETWEEN 100 AND 3000
                      ORDER BY objectid LIMIT 1""").first()
    if o is None:
        return
    oid, filt = o["objectid"], o["filter"]
    r = spark.sql(f"""SELECT objectid, field, filter, rcid, objra, objdec, nepochs, e.hmjd, e.mag, e.magerr,
                             e.clrcoeff, e.catflags
                      FROM {raw} LATERAL VIEW inline(arrays_zip(hmjd, mag, magerr, clrcoeff, catflags)) e
                      WHERE objectid = {oid} AND filter = '{filt}' ORDER BY e.hmjd LIMIT {n}""").toPandas()
    c = spark.sql(f"""SELECT objectid, filter, e.hmjd + 2400000.5 AS jd, e.mag, e.magerr AS uncertainty,
                             e.mag - percentile(e.mag, 0.5) OVER () AS mag_norm, 1.0 AS A_k, 0.0 AS offset_mag
                      FROM {clean} LATERAL VIEW inline(arrays_zip(hmjd, mag, magerr)) e
                      WHERE objectid = {oid} AND filter = '{filt}'""").orderBy("jd").limit(n).toPandas()
    save_csv(r, cfg, "sample_ztf_raw.csv")
    save_csv(c, cfg, "sample_ztf_clean.csv")


# ------------------------------------------------------------------ ztf_label
def _field_centres(cfg):
    from .ztf import http_get, parse_field_centres
    path = os.path.join(cfg.work_dir, "ZTF_Fields.txt")
    if not os.path.exists(path):
        with open(path, "wb") as fh:
            fh.write(http_get(cfg.section("ztf")["fields_url"]))
    with open(path, encoding="utf-8") as fh:
        return parse_field_centres(fh.read())


def vsx_for_fields(cfg, fields):
    """Labelled VSX variables around every field centre (cached per field in V3_ROOT/work/ztf_vsx/)."""
    from . import vsx
    from .labels import label_for, primary_type
    z = cfg.section("ztf")
    lm = cfg.load_yaml("label_map")
    types = sorted(set(lm["anomalous"]) | set(lm["normal"]))
    cache = os.path.join(cfg.work_dir, "ztf_vsx")
    os.makedirs(cache, exist_ok=True)
    cen = _field_centres(cfg).set_index("field")
    frames = []
    for f in sorted(fields):
        path = os.path.join(cache, f"field{f:06d}.csv")
        if os.path.exists(path):
            d = pd.read_csv(path)
        else:
            c = cen.loc[f]
            t0 = time.time()
            d = vsx.cone(cfg, c["ra"], c["dec"], float(z["field_query_radius_deg"]), types=types)
            d.to_csv(path, index=False)
            print(f"  field {f}: {len(d)} VSX variables of our types ({time.time() - t0:.0f} s)", flush=True)
        frames.append(d)
    v = pd.concat(frames, ignore_index=True).drop_duplicates("name")
    seps = lm.get("type_separators")
    v["primary_type"] = v["type"].map(lambda t: primary_type(t, seps))
    v["label"] = v["type"].map(lambda t: label_for(t, lm))
    return v.dropna(subset=["label"])


def label_run(spark, cfg):
    t0 = time.time()
    z = cfg.section("ztf")
    clean = cfg.table("ztf_lightcurves_clean")
    fields = [r[0] for r in spark.sql(f"SELECT DISTINCT field FROM {clean}").collect()]
    print(f"querying VSX around {len(fields)} field centres (cached after the first run) ...")
    v = vsx_for_fields(cfg, fields)
    print(f"{len(v)} labelled VSX variables in our fields: {int((v['label'] == 1).sum())} anomalous, "
          f"{int((v['label'] == 0).sum())} normal")
    spark.createDataFrame(v[["name", "ra", "dec", "type", "primary_type", "label"]].astype(
        {"ra": float, "dec": float, "label": int})).createOrReplaceTempView("zl_vsx")
    stars = spark.sql(f"SELECT auid, ra, dec FROM {cfg.table('star_labels')} WHERE ra IS NOT NULL AND dec IS NOT NULL")
    stars.createOrReplaceTempView("zl_aavso")
    m, x = float(z["label_match_arcsec"]), float(z["aavso_exclusion_arcsec"])
    b1, b2 = 0.001, max(0.001, x / 3600 * 1.5)  # declination bucket sizes (deg) > the match radius
    spark.sql(f"SELECT DISTINCT objectid, field, filter, objra, objdec FROM {clean}").createOrReplaceTempView("zl_obj")
    spark.sql(f"""
        WITH vb AS (SELECT v.*, explode(array(CAST(FLOOR(dec / {b1}) AS BIGINT) - 1, CAST(FLOOR(dec / {b1}) AS BIGINT),
                                              CAST(FLOOR(dec / {b1}) AS BIGINT) + 1)) AS b FROM zl_vsx v),
             m AS (SELECT o.objectid, o.field, o.filter, o.objra, o.objdec, vb.name AS vsx_name, vb.type AS vsx_type,
                          vb.primary_type, vb.label, {sep_sql('o.objra', 'o.objdec', 'vb.ra', 'vb.dec')} AS sep_arcsec
                   FROM zl_obj o JOIN vb ON CAST(FLOOR(o.objdec / {b1}) AS BIGINT) = vb.b),
             best AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY objectid ORDER BY sep_arcsec, vsx_name) AS rk
                      FROM m WHERE sep_arcsec <= {m}),
             ab AS (SELECT a.*, explode(array(CAST(FLOOR(dec / {b2}) AS BIGINT) - 1, CAST(FLOOR(dec / {b2}) AS BIGINT),
                                              CAST(FLOOR(dec / {b2}) AS BIGINT) + 1)) AS b FROM zl_aavso a),
             near AS (SELECT DISTINCT o.objectid
                      FROM zl_obj o JOIN ab ON CAST(FLOOR(o.objdec / {b2}) AS BIGINT) = ab.b
                      WHERE {sep_sql('o.objra', 'o.objdec', 'ab.ra', 'ab.dec')} <= {x})
        SELECT b.objectid, b.field, b.filter, b.objra, b.objdec, b.vsx_name, b.vsx_type, b.primary_type, b.label,
               ROUND(b.sep_arcsec, 3) AS sep_arcsec, (n.objectid IS NOT NULL) AS aavso_overlap
        FROM best b LEFT JOIN near n ON b.objectid = n.objectid WHERE b.rk = 1""").createOrReplaceTempView("zl_out")
    save_hive_table(spark.table("zl_out"), cfg, "ztf_labels")
    lab = cfg.table("ztf_labels")
    counts = spark.sql(f"""SELECT CASE label WHEN 1 THEN 'anomalous' ELSE 'normal' END AS class,
                                  SUM(CASE WHEN aavso_overlap THEN 0 ELSE 1 END) AS lightcurves_in_pool,
                                  SUM(CASE WHEN aavso_overlap THEN 1 ELSE 0 END) AS excluded_near_aavso_star,
                                  COUNT(DISTINCT vsx_name) AS vsx_variables
                           FROM {lab} GROUP BY 1 ORDER BY 1""").toPandas()
    print("\nZTF light curves matched to VSX (within %.1f\"):\n" % m + counts.to_string(index=False))
    save_csv(counts, cfg, "ztf_label_counts.csv")
    comp = spark.sql(f"""SELECT 'ZTF' AS source,
                                CASE WHEN l.objectid IS NULL OR l.aavso_overlap THEN 'unlabelled'
                                     WHEN l.label = 1 THEN 'anomalous' ELSE 'normal' END AS class,
                                CASE WHEN l.objectid IS NULL OR l.aavso_overlap THEN 'unlabelled' ELSE l.primary_type END
                                     AS vsx_type,
                                SUM(c.nepochs) AS rows, COUNT(DISTINCT c.objectid) AS stars
                         FROM {clean} c LEFT JOIN {lab} l ON c.objectid = l.objectid
                         GROUP BY 1, 2, 3 ORDER BY rows DESC""").toPandas()
    save_csv(comp, cfg, "composition_ztf.csv")
    dataflow.record(cfg, "ztf_labels", "VSX variables of our types around every field centre (astroquery: VizieR or "
                    "HEASARC) are cross-matched to ZTF objects in Spark (declination-bucket equi-join + exact angular "
                    "distance); objects near any AAVSO star are excluded from training.", f"Hive {lab}",
                    "Parquet (Hive table)", "Spark SQL + astroquery", size_bytes=table_bytes(spark, lab),
                    rows=int(counts["lightcurves_in_pool"].sum() + counts["excluded_near_aavso_star"].sum()),
                    changes=f"{int(counts['lightcurves_in_pool'].sum()):,} labelled light curves in the training pool; "
                            f"{int(counts['excluded_near_aavso_star'].sum()):,} excluded (within {x:g}\" of an AAVSO star)",
                    seconds=time.time() - t0)


# ------------------------------------------------------------------ ztf_windows
def ztf_obs_sql(cfg, sampled_view):
    """Exploded observations of the sampled ZTF light curves, in the same shape as AAVSO's (A_k = 1, offset = 0)."""
    clean = cfg.table("ztf_lightcurves_clean")
    return f"""SELECT CONCAT('ZTF', CAST(c.objectid AS STRING)) AS auid, e.hmjd + 2400000.5 AS jd,
                      e.mag - percentile(e.mag, 0.5) OVER (PARTITION BY c.objectid) AS mag_norm, c.filter AS band,
                      'CCD' AS obs_type, 'ZTF' AS observer, CAST(e.magerr AS DOUBLE) AS uncertainty,
                      1.0 AS a_k, 0.0 AS obs_offset, false AS low_n
               FROM {clean} c LEFT SEMI JOIN {sampled_view} s ON c.objectid = s.objectid
               LATERAL VIEW inline(arrays_zip(hmjd, mag, magerr)) e"""


def windows_run(spark, cfg):
    from .windows import assign_windows_from, export_npz, make_vocab, window_features, window_sequences
    t0 = time.time()
    z = cfg.section("ztf")
    seed, per_class, cap = int(z["sample_seed"]), int(z["max_lightcurves_per_class"]), int(z["pretrain_windows"])
    lab = cfg.table("ztf_labels")
    spark.sql(f"""SELECT objectid, label FROM (
                    SELECT objectid, label, ROW_NUMBER() OVER (PARTITION BY label ORDER BY xxhash64(objectid, {seed})) rk
                    FROM {lab} WHERE NOT aavso_overlap) WHERE rk <= {per_class}""").cache() \
        .createOrReplaceTempView("zw_sampled")
    lc = spark.sql("SELECT label, COUNT(*) AS n FROM zw_sampled GROUP BY label").toPandas()
    print("light curves sampled per class (hash sample over all fields):\n" + lc.to_string(index=False))
    vb, vt = make_vocab(spark, cfg)
    assign_windows_from(spark, cfg, ztf_obs_sql(cfg, "zw_sampled"), "zw_all")
    spark.sql(f"""SELECT CONCAT('ZTF', CAST(objectid AS STRING)) AS auid, label, vsx_name AS name,
                         vsx_name AS data_name, primary_type AS vsx_type FROM {lab}""").createOrReplaceTempView("zw_labels")
    spark.sql(f"""SELECT window_id FROM (
                    SELECT w.window_id, l.label,
                           ROW_NUMBER() OVER (PARTITION BY l.label ORDER BY xxhash64(w.window_id, {seed})) AS rk
                    FROM (SELECT DISTINCT window_id, auid FROM zw_all) w JOIN zw_labels l ON w.auid = l.auid)
                  WHERE rk <= {cap // 2}""").createOrReplaceTempView("zw_keep")
    spark.sql("SELECT a.* FROM zw_all a LEFT SEMI JOIN zw_keep k ON a.window_id = k.window_id").cache() \
        .createOrReplaceTempView("zw_obs")
    save_hive_table(window_features(spark, cfg, "zw_obs", "zw_labels"), cfg, "ztf_window_features")
    save_hive_table(window_sequences(spark, cfg, "zw_obs"), cfg, "ztf_window_sequences")
    feats = cfg.table("ztf_window_features")
    counts = spark.sql(f"""SELECT CASE label WHEN 1 THEN 'anomalous' ELSE 'normal' END AS class, COUNT(*) AS windows,
                                  COUNT(DISTINCT auid) AS lightcurves FROM {feats} GROUP BY 1 ORDER BY 1""").toPandas()
    total_avail = spark.sql("SELECT COUNT(DISTINCT window_id) FROM zw_all").first()[0]
    print(f"\nZTF pre-training windows (cap {cap:,}, class-balanced; {total_avail:,} windows were available):\n"
          + counts.to_string(index=False))
    save_csv(counts.assign(windows_available=total_avail, cap=cap), cfg, "ztf_window_counts.csv")
    path = export_npz(spark, cfg, vb, vt, "ztf_window_sequences", "ztf_window_features", "ztf_windows_seq.npz")
    dataflow.record(cfg, "ztf_windows", "Labelled ZTF light curves (AAVSO-overlapping objects excluded) are exploded "
                    "and cut into the same 100-day windows; A_k = 1 and offset = 0 on every token; class-balanced "
                    "sample for pre-training.", f"Hive {feats}, {cfg.table('ztf_window_sequences')}; {path}",
                    "Parquet (Hive) + NumPy .npz", "Spark SQL / applyInPandas",
                    size_bytes=table_bytes(spark, feats) + table_bytes(spark, cfg.table("ztf_window_sequences")),
                    rows=int(counts["windows"].sum()),
                    changes=f"{int(counts['windows'].sum()):,} windows kept of {total_avail:,} available",
                    seconds=time.time() - t0, extra={"npz file": f"{path} ({os.path.getsize(path) / 2**20:.1f} MB)"})
    for v in ("zw_obs", "zw_all", "zw_sampled"):
        spark.catalog.dropTempView(v)


# ------------------------------------------------------------------ akval
def ak_scatter(matched, min_points):
    """Per observer: robust scatter (1.4826 x MAD) of AAVSO - ZTF after removing each (star, AAVSO band, ZTF filter)
    median offset. matched: auid, band, filter, observer, diff. Returns observer, n_matched, scatter_vs_ztf_mag."""
    d = matched.copy()
    d["corrected"] = d["diff"] - d.groupby(["auid", "band", "filter"])["diff"].transform("median")
    g = d.groupby("observer")["corrected"]
    out = pd.DataFrame({"n_matched": g.size(),
                        "scatter_vs_ztf_mag": g.apply(lambda s: 1.4826 * np.median(np.abs(s - np.median(s))))})
    return out[out["n_matched"] >= min_points].reset_index()


def akval_run(spark, cfg):
    from scipy.stats import spearmanr
    t0 = time.time()
    a, z = cfg.section("akval"), cfg.section("ztf")
    m, w = float(z["match_arcsec"]), float(a["window_days"])
    clean = cfg.table("ztf_lightcurves_clean")
    spark.sql(f"SELECT auid, ra, dec FROM {cfg.table('star_labels')} WHERE ra IS NOT NULL").createOrReplaceTempView("av_s")
    spark.sql(f"""
        WITH s AS (SELECT auid, ra, dec, explode(array(CAST(FLOOR(dec / 0.001) AS BIGINT) - 1,
                   CAST(FLOOR(dec / 0.001) AS BIGINT), CAST(FLOOR(dec / 0.001) AS BIGINT) + 1)) AS b FROM av_s),
             o AS (SELECT c.objectid, c.filter, c.hmjd, c.mag, s.auid,
                          {sep_sql('c.objra', 'c.objdec', 's.ra', 's.dec')} AS sep
                   FROM {clean} c JOIN s ON CAST(FLOOR(c.objdec / 0.001) AS BIGINT) = s.b)
        SELECT o.auid, o.filter, e.hmjd + 2400000.5 AS jd, e.mag AS zmag
        FROM o LATERAL VIEW inline(arrays_zip(hmjd, mag)) e WHERE o.sep <= {m}""").cache().createOrReplaceTempView("av_z")
    n_z = spark.table("av_z").count()
    pairs = spark.sql(f"""
        WITH aa AS (SELECT c.obs_id, c.auid, c.jd, c.mag, c.band, c.observer,
                           explode(array(CAST(FLOOR(c.jd / {w}) AS BIGINT) - 1, CAST(FLOOR(c.jd / {w}) AS BIGINT),
                                         CAST(FLOOR(c.jd / {w}) AS BIGINT) + 1)) AS b
                    FROM {cfg.table('obs_clean')} c LEFT SEMI JOIN (SELECT DISTINCT auid FROM av_z) s ON c.auid = s.auid),
             zz AS (SELECT *, CAST(FLOOR(jd / {w}) AS BIGINT) AS b FROM av_z),
             j AS (SELECT aa.obs_id, aa.auid, aa.band, aa.observer, aa.mag, zz.filter, zz.zmag, ABS(aa.jd - zz.jd) AS dt,
                          ROW_NUMBER() OVER (PARTITION BY aa.obs_id ORDER BY ABS(aa.jd - zz.jd), zz.filter) AS rk
                   FROM aa JOIN zz ON aa.auid = zz.auid AND aa.b = zz.b WHERE ABS(aa.jd - zz.jd) <= {w})
        SELECT auid, band, filter, observer, mag - zmag AS diff, dt FROM j WHERE rk = 1""").toPandas()
    per_obs = ak_scatter(pairs, int(a["min_points_per_observer"])) if len(pairs) else \
        pd.DataFrame(columns=["observer", "n_matched", "scatter_vs_ztf_mag"])
    scores = spark.table(cfg.table("observer_scores")).select("observer", "obs_type", "score", "low_n").toPandas()
    per_obs = per_obs.merge(scores, on="observer", how="left").rename(columns={"score": "A_k"})
    scored = per_obs[per_obs["low_n"] == False]  # noqa: E712 - A_k measured, not the median fill-in
    n_obs = len(scored)
    if n_obs >= 3:
        rho, p = spearmanr(scored["A_k"], scored["scatter_vs_ztf_mag"])
    else:
        rho, p = float("nan"), float("nan")
    enough = n_obs >= int(a["min_observers"])
    verdict = (f"Spearman rho = {rho:.3f} (p = {p:.3g}) between A_k and scatter against ZTF, over {n_obs} observers and "
               f"{len(pairs):,} matched observations." + ("" if enough else
                                                          f" Too few observers overlap with ZTF (< {a['min_observers']}) "
                                                          "to conclude anything."))
    print(verdict)
    save_csv(per_obs, cfg, "ak_vs_ztf_observers.csv")
    save_csv(pd.DataFrame([dict(spearman_rho=rho, p_value=p, observers=n_obs, matched_observations=len(pairs),
                                ztf_points_of_aavso_stars=n_z, stars_matched=pairs["auid"].nunique() if len(pairs) else 0,
                                enough_to_conclude=enough, verdict=verdict)]), cfg, "ak_validation_summary.csv")
    dataflow.record(cfg, "ak_validation", "AAVSO observations are matched (Spark join on star and 1-day buckets) to the "
                    "nearest unsaturated ZTF point of the same star; per-band offsets removed; each observer's scatter "
                    "against ZTF compared with A_k.", "outputs/ak_vs_ztf_observers.csv, ak_validation_summary.csv",
                    "CSV", "Spark SQL + pandas/SciPy", rows=len(pairs),
                    changes=f"{len(pairs):,} matched observations, {n_obs} observers", seconds=time.time() - t0)
