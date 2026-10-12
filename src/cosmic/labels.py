"""Star labels from the VSX catalogue (VizieR B/vsx).

What it does:
  1. Finds which stars are actually in our AAVSO files (distinct `auid`, with the
     most common `target` spelling as its name) using Spark on the raw CSVs in HDFS.
  2. Fetches name, RA, Dec and variability type for ONLY those stars from VizieR
     with astroquery (one polite request per star). If a labels CSV already exists
     in Drive (e.g. exported by hand from the VizieR website), it is used instead.
     Stars already in that file are not queried again; only stars that are new in the
     data (e.g. after a new download round) are fetched and appended.
  3. Turns the VSX type into a binary label with config/label_map.yaml, matching on
     the primary type (the text before any ':', '+', '|' or '/'), and assigns each star a
     declination band (`sky_region`) that is later used to partition the data.
  4. Prints every star whose type is not in the label map (or that has no VSX match),
     so nothing is dropped silently; saves the list as unmatched_vsx_types.csv.
  5. Saves the result as the Hive table `star_labels`.

Why: the classifier needs a ground-truth class per star, and the sky map / sky_region
partitioning needs coordinates. Labels are per star, so we group on `auid` — never on
`target`, whose spelling varies ("SS Cyg" vs "SS CYG").
"""
import os
import re
import time

import pandas as pd

DEFAULT_SEPARATORS = [":", "+", "|", "/"]
# IAU abbreviations whose VSX spelling is not simply Capitalised ("CRB" -> "CrB", "UMA" -> "UMa")
_CON = {c.upper(): c for c in ["CMa", "CMi", "CVn", "CrA", "CrB", "LMi", "PsA", "TrA", "UMa", "UMi"]}


def norm_name(name):
    return re.sub(r"\s+", " ", str(name)).strip().lower()


def primary_type(vsx_type, separators=DEFAULT_SEPARATORS):
    """Primary VSX type: the text before the first separator ("UGSS+E" -> "UGSS", "NL/VY" -> "NL")."""
    if vsx_type is None or (isinstance(vsx_type, float) and pd.isna(vsx_type)):
        return None
    t = re.split("|".join(re.escape(c) for c in separators), str(vsx_type).strip())[0].strip().upper()
    return t or None


def label_for(vsx_type, label_map):
    p = primary_type(vsx_type, label_map.get("type_separators", DEFAULT_SEPARATORS))
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


# AAVSO spells Greek letters in capitals ("KHI CYG", "OMI CET"); VizieR's VSX uses "khi Cyg", "omi Cet".
_GREEK = {"ALF": "alf", "ALPHA": "alf", "BET": "bet", "BETA": "bet", "GAM": "gam", "DEL": "del", "EPS": "eps",
          "ZET": "zet", "ETA": "eta", "THE": "the", "IOT": "iot", "KAP": "kap", "LAM": "lam", "MU": "mu.",
          "MU.": "mu.", "NU": "nu.", "NU.": "nu.", "XI": "ksi", "KSI": "ksi", "OMI": "omi", "OMICRON": "omi",
          "PI": "pi.", "PI.": "pi.", "RHO": "rho", "SIG": "sig", "TAU": "tau", "UPS": "ups", "PHI": "phi",
          "KHI": "khi", "CHI": "khi", "PSI": "psi", "OME": "ome"}


def vsx_name_candidates(name):
    """VSX spellings to try for an AAVSO target name, most likely first.

    "SS CYG" -> "SS Cyg"; "T CRB" -> "T CrB"; "KHI CYG" -> "khi Cyg"; "V603 AQL" -> "V0603 Aql"
    (VSX pads V-numbers to four digits).
    """
    parts = re.sub(r"\s+", " ", str(name)).strip().split(" ")
    out = []
    if len(parts) >= 2 and len(parts[-1]) == 3 and parts[-1].isalpha():
        con = _CON.get(parts[-1].upper(), parts[-1].capitalize())
        des = "".join(parts[:-1]).upper()  # "V 603" -> "V603"
        if des in _GREEK:
            out.append(f"{_GREEK[des]} {con}")
        m = re.fullmatch(r"V0*(\d+)", des)
        if m:
            out += [f"V{int(m.group(1)):04d} {con}", f"V{int(m.group(1))} {con}"]
        out.append(f"{des} {con}")
    out.append(" ".join(parts))
    return list(dict.fromkeys(out))  # unique, order kept


def fetch_vsx(names, pause_s=1.0):
    """Query VizieR B/vsx for each name (trying VSX spellings); returns queried_name, name, ra, dec, type."""
    from astroquery.vizier import Vizier

    v = Vizier(columns=["Name", "RAJ2000", "DEJ2000", "Type"], row_limit=5)
    rows = []
    for name in names:
        hit = None
        for cand in vsx_name_candidates(name):
            try:
                res = v.query_constraints(catalog="B/vsx/vsx", Name=f"={cand}")
            except Exception as exc:  # network errors are reported, not hidden
                print(f"  VizieR query failed for {cand!r}: {exc}")
                res = None
            time.sleep(pause_s)
            if res and len(res[0]):
                hit = res[0][0]
                break
        if hit is not None:
            rows.append(dict(queried_name=name, name=str(hit["Name"]), ra=float(hit["RAJ2000"]),
                             dec=float(hit["DEJ2000"]), type=str(hit["Type"])))
            print(f"  {name:12s} -> {hit['Name']:12s} {hit['Type']}")
        else:
            print(f"  {name:12s} -> NOT FOUND in VSX (tried {vsx_name_candidates(name)})")
    return pd.DataFrame(rows, columns=["queried_name", "name", "ra", "dec", "type"])


def stars_in_data(spark, raw_paths):
    """auid -> most frequent target spelling, computed in Spark on the raw files."""
    from .ingest import read_raw  # same reader as Step 1 (multi-line safe)
    df = read_raw(spark, raw_paths).select("auid", "target")
    df.createOrReplaceTempView("raw_names")
    return spark.sql("""
        SELECT auid, target AS data_name, n FROM (
          SELECT UPPER(TRIM(auid)) AS auid, TRIM(target) AS target, COUNT(*) AS n,
                 ROW_NUMBER() OVER (PARTITION BY UPPER(TRIM(auid)) ORDER BY COUNT(*) DESC, TRIM(target)) AS rk
          FROM raw_names WHERE auid IS NOT NULL GROUP BY UPPER(TRIM(auid)), TRIM(target)) WHERE rk = 1
    """).toPandas()


def vsx_rows(cfg, data_stars, fetch=True):
    """VSX rows (auid, name, ra, dec, type) for the stars in `data_stars` (columns auid, data_name).

    Reads the labels CSV in Drive if it exists. Stars not in it are fetched from VizieR (real
    data only) and appended to the file, so each new download round only queries its new stars.
    """
    cols = ["auid", "name", "ra", "dec", "type"]
    vsx = pd.DataFrame(columns=cols)
    if os.path.exists(cfg.labels_file):
        print(f"Using existing labels file {cfg.labels_file}")
        vsx = pd.read_csv(cfg.labels_file)
        vsx.columns = [c.strip().lower() for c in vsx.columns]
        vsx = vsx.rename(columns={"raj2000": "ra", "dej2000": "dec"})
        if "auid" not in vsx.columns:  # hand-exported VizieR file: match on name
            m = dict(zip(data_stars["data_name"].map(norm_name), data_stars["auid"]))
            vsx["auid"] = vsx["name"].map(lambda n: m.get(norm_name(n)))
        vsx = vsx[cols]
    elif cfg.dataset != "real":
        raise FileNotFoundError(cfg.labels_file)

    missing = data_stars[~data_stars["auid"].isin(set(vsx["auid"].dropna()))]
    if len(missing) and fetch and cfg.dataset == "real":
        print(f"Fetching {len(missing)} stars not yet in the labels file from VizieR B/vsx (1 s between requests)")
        try:
            got = fetch_vsx(list(missing["data_name"]))
        except ImportError as exc:
            print(f"WARNING: cannot query VizieR ({exc}); these stars stay without a VSX type")
            got = pd.DataFrame(columns=["queried_name", "name", "ra", "dec", "type"])
        new = missing.rename(columns={"data_name": "queried_name"})[["auid", "queried_name"]]             .merge(got, on="queried_name", how="inner")[cols]
        if len(new):
            vsx = pd.concat([vsx, new], ignore_index=True)
            os.makedirs(os.path.dirname(cfg.labels_file), exist_ok=True)
            vsx.to_csv(cfg.labels_file, index=False)
            print(f"saved {cfg.labels_file} (+{len(new)} stars)")
    return vsx.dropna(subset=["auid"]).drop_duplicates("auid")


def assign_labels(data_stars, vsx, label_map, edges):
    """Join stars to their VSX rows and add primary_type, label (1/0/<NA>) and sky_region.

    Returns (labels, unmatched). `unmatched` lists every star that gets no label, with the reason,
    and each one is printed.
    """
    out = data_stars[["auid", "data_name"]].merge(vsx[["auid", "name", "ra", "dec", "type"]], on="auid", how="left")
    seps = label_map.get("type_separators", DEFAULT_SEPARATORS)
    out["primary_type"] = out["type"].map(lambda t: primary_type(t, seps))
    out["label"] = out["type"].map(lambda t: label_for(t, label_map)).astype("Int64")
    out["sky_region"] = out["dec"].map(lambda d: sky_region(d, edges))
    out = out.rename(columns={"type": "vsx_type"})[
        ["auid", "data_name", "name", "ra", "dec", "vsx_type", "primary_type", "label", "sky_region"]]
    un = out[out["label"].isna()].copy()
    un["reason"] = un["vsx_type"].isna().map({True: "no VSX match", False: "primary type not in label map"})
    un = un[["auid", "data_name", "name", "vsx_type", "primary_type", "reason"]].reset_index(drop=True)
    if len(un):
        print(f"WARNING: {len(un)} stars are unlabelled (excluded from supervised training):")
        for _, r in un.iterrows():
            print(f"  {r['auid']}  {r['data_name']!s:14s} vsx_type={r['vsx_type']!s:14s} "
                  f"primary={r['primary_type']!s:8s} -> {r['reason']}")
    return out, un


def build_labels(spark, cfg, raw_paths):
    from .spark_session import save_csv, save_hive_table
    label_map = cfg.load_yaml("label_map")
    edges = cfg.section("ingest")["dec_band_edges"]
    data_stars = stars_in_data(spark, raw_paths)
    print(f"{len(data_stars)} distinct auids in the raw data")
    out, unmatched = assign_labels(data_stars, vsx_rows(cfg, data_stars), label_map, edges)
    save_csv(unmatched, cfg, "unmatched_vsx_types.csv")
    sdf = spark.createDataFrame(out.astype(object).where(out.notna(), None),
                                "auid STRING, data_name STRING, name STRING, ra DOUBLE, dec DOUBLE, "
                                "vsx_type STRING, primary_type STRING, label INT, sky_region STRING")
    save_hive_table(sdf, cfg, "star_labels")
    print(out.groupby(out["label"].map({1: "anomalous", 0: "normal"}).fillna("unlabelled")).size())
    return out
