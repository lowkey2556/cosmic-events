"""Star labels from the VSX catalogue (VizieR B/vsx).

What it does:
  1. Finds which stars are actually in our AAVSO files (distinct `auid`, with the
     most common `target` spelling as its name) using Spark on the raw CSVs in HDFS.
  2. Fetches name, RA, Dec and variability type for ONLY those stars from VizieR
     with astroquery (one polite request per star). If a labels CSV already exists
     in Drive (e.g. exported by hand from the VizieR website), it is used instead.
  3. Turns the VSX type into a binary label with config/label_map.yaml, matching on
     the primary type (the text before any ':', '+' or '|'), and assigns each star a
     declination band (`sky_region`) that is later used to partition the data.
  4. Saves the result as the Hive table `star_labels`.

Why: the classifier needs a ground-truth class per star, and the sky map / sky_region
partitioning needs coordinates. Labels are per star, so we group on `auid` — never on
`target`, whose spelling varies ("SS Cyg" vs "SS CYG").
"""
import os
import re
import time

import pandas as pd

from .spark_session import save_hive_table

_SPLIT = re.compile(r"[:+|]")


def norm_name(name):
    return re.sub(r"\s+", " ", str(name)).strip().lower()


def primary_type(vsx_type):
    if vsx_type is None or (isinstance(vsx_type, float) and pd.isna(vsx_type)):
        return None
    t = _SPLIT.split(str(vsx_type).strip())[0].strip().upper()
    return t or None


def label_for(vsx_type, label_map):
    p = primary_type(vsx_type)
    if p in set(label_map["anomalous"]):
        return 1
    if p in set(label_map["normal"]):
        return 0
    return None


def sky_region(dec, edges):
    """Declination band name such as DEC_P30_P60; UNKNOWN when Dec is missing."""
    if dec is None or pd.isna(dec):
        return "UNKNOWN"
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= dec < hi or (hi == edges[-1] and dec == hi):
            f = lambda v: ("P" if v >= 0 else "M") + f"{abs(int(v)):02d}"
            return f"DEC_{f(lo)}_{f(hi)}"
    return "UNKNOWN"


def fetch_vsx(names, pause_s=1.0):
    """Query VizieR B/vsx for each name; returns rows name, ra, dec, type, queried_name."""
    from astroquery.vizier import Vizier

    v = Vizier(columns=["Name", "RAJ2000", "DEJ2000", "Type"], row_limit=5)
    rows = []
    for name in names:
        try:
            res = v.query_constraints(catalog="B/vsx/vsx", Name=f"={name}")
        except Exception as exc:  # network errors are reported, not hidden
            print(f"  VizieR query failed for {name!r}: {exc}")
            res = None
        if res and len(res[0]):
            r = res[0][0]
            rows.append(dict(queried_name=name, name=str(r["Name"]), ra=float(r["RAJ2000"]),
                             dec=float(r["DEJ2000"]), type=str(r["Type"])))
            print(f"  {name:12s} -> {r['Name']} {r['Type']}")
        else:
            print(f"  {name:12s} -> NOT FOUND in VSX")
        time.sleep(pause_s)
    return pd.DataFrame(rows, columns=["queried_name", "name", "ra", "dec", "type"])


def stars_in_data(spark, raw_hdfs_glob):
    """auid -> most frequent target spelling, computed in Spark on the raw files."""
    df = spark.read.option("header", True).csv(raw_hdfs_glob).select("auid", "target")
    df.createOrReplaceTempView("raw_names")
    return spark.sql("""
        SELECT auid, target AS data_name, n FROM (
          SELECT auid, target, COUNT(*) AS n,
                 ROW_NUMBER() OVER (PARTITION BY auid ORDER BY COUNT(*) DESC, target) AS rk
          FROM raw_names WHERE auid IS NOT NULL GROUP BY auid, target) WHERE rk = 1
    """).toPandas()


def build_labels(spark, cfg, raw_hdfs_glob):
    label_map = cfg.load_yaml("label_map")
    edges = cfg.section("ingest")["dec_band_edges"]
    data_stars = stars_in_data(spark, raw_hdfs_glob)
    print(f"{len(data_stars)} distinct auids in the raw data")

    if os.path.exists(cfg.labels_file):
        print(f"Using existing labels file {cfg.labels_file}")
        vsx = pd.read_csv(cfg.labels_file)
        vsx.columns = [c.strip().lower() for c in vsx.columns]
        vsx = vsx.rename(columns={"raj2000": "ra", "dej2000": "dec"})
    else:
        if cfg.dataset != "real":
            raise FileNotFoundError(cfg.labels_file)
        star_list = cfg.load_yaml("star_list")
        listed = [s["name"] for grp in ("eruptive", "periodic") for s in star_list[grp]]
        by_norm = {norm_name(n): n for n in listed}
        queries = pd.DataFrame({"auid": data_stars["auid"],
                                "queried_name": [by_norm.get(norm_name(t), t) for t in data_stars["data_name"]]})
        print(f"Fetching {len(queries)} stars from VizieR B/vsx (1 s between requests)")
        vsx = queries.merge(fetch_vsx(list(queries["queried_name"])), on="queried_name", how="inner")
        os.makedirs(os.path.dirname(cfg.labels_file), exist_ok=True)
        vsx[["auid", "name", "ra", "dec", "type"]].to_csv(cfg.labels_file, index=False)
        print(f"saved {cfg.labels_file}")

    if "auid" not in vsx.columns:  # hand-exported VizieR file: match on name
        m = dict(zip(data_stars["data_name"].map(norm_name), data_stars["auid"]))
        vsx["auid"] = vsx["name"].map(lambda n: m.get(norm_name(n)))

    out = data_stars.merge(vsx[["auid", "name", "ra", "dec", "type"]], on="auid", how="left")
    out["primary_type"] = out["type"].map(primary_type)
    out["label"] = out["type"].map(lambda t: label_for(t, label_map)).astype("Int64")
    out["sky_region"] = out["dec"].map(lambda d: sky_region(d, edges))
    out = out.rename(columns={"type": "vsx_type"})[
        ["auid", "data_name", "name", "ra", "dec", "vsx_type", "primary_type", "label", "sky_region"]]
    missing = out[out["vsx_type"].isna()]
    if len(missing):
        print(f"WARNING: {len(missing)} stars have no VSX match: {list(missing['data_name'])}")
    sdf = spark.createDataFrame(out.astype(object).where(out.notna(), None),
                                "auid STRING, data_name STRING, name STRING, ra DOUBLE, dec DOUBLE, "
                                "vsx_type STRING, primary_type STRING, label INT, sky_region STRING")
    save_hive_table(sdf, cfg, "star_labels")
    print(out.groupby(out["label"].map({1: "anomalous", 0: "normal"}).fillna("unlabelled")).size())
    return out
