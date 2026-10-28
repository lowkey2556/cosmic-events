"""Phase 1 — ZTF public light curves: choose, download and audit (plain Python, runs in Colab).

ZTF data release light curves (IRSA, release set in settings.yaml → ztf.release, DR24 by default)
are stored as one Parquet file per field / CCD / quadrant / filter ("tile"), ~25–60 MB each, with
one row per object and the epochs as arrays (hmjd, mag, magerr, clrcoeff, catflags). Every file
and its MD5 sum is listed in the release's checksum.md5.

What it does:
  ztf_select   1. Finds which tiles contain our AAVSO stars: one cone search (2″) per star with
                  the IRSA ZTF light-curve API, which reports field, CCD and quadrant per epoch.
               2. Priority 1: every filter's file of those tiles.
               3. If that is below the size target (8 GB by default), adds tiles from the nearest (up to
                  ztf.max_neighbour_fields) neighbouring
                  fields (field centres within ztf.neighbour_deg), ranked by how many VSX variables
                  of our label-map types lie within ztf.field_radius_deg of the field centre
                  (counted with astroquery/VizieR B/vsx), until the target is reached without going
                  over the maximum (10 GB). Exact sizes come from HTTP HEAD requests.
               Logs every chosen file and field with the reason (ztf_selection.csv, ztf_fields_chosen.csv).
  ztf_download Downloads the chosen files to the Colab disk (wget, resumable), checks each MD5
               against checksum.md5, copies the file to V3_ROOT/ztf_raw/<field>/ and records it in
               MANIFEST.csv. Files already in Drive with a verified checksum are skipped, so the cell
               can simply be run again after a disconnect. Prints progress and the time remaining.
  ztf_audit    Files, GB, light curves, objects, epochs, filters, date range, and our AAVSO stars:
               matched within 2″ in their tiles, with epochs and UNSATURATED epochs (fainter than
               ztf.saturation_mag, catflags = 0). ZTF saturates for stars brighter than about mag 12.5.

Why: ZTF adds professional, single-observer light curves: data volume for the Big Data part,
labelled examples for pre-training RGA-Net, and an independent reference to validate A_k.
"""
import hashlib
import io
import math
import os
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

UA = {"User-Agent": "cosmic-events (research; contact via IRSA help desk)"}
FILTERS = {1: "zg", 2: "zr", 3: "zi"}
FILE_RE = re.compile(r"ztf_(\d{6})_(z[gri])_c(\d{2})_q(\d)_(dr\d+)\.parquet$")


# ------------------------------------------------------------------ small pure helpers (tested)
def parse_checksums(text):
    """checksum.md5 lines ('<md5>  ./<dir>/fieldNNNNNN/<file>') -> DataFrame of every file."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        md5, rel = parts[0], parts[1].lstrip("./")
        m = FILE_RE.search(rel)
        if m:
            rows.append(dict(rel_path=rel, md5=md5.lower(), field=int(m.group(1)), filter=m.group(2),
                             ccd=int(m.group(3)), q=int(m.group(4))))
    return pd.DataFrame(rows, columns=["rel_path", "md5", "field", "filter", "ccd", "q"])


def parse_field_centres(text):
    """ZTF_Fields.txt -> DataFrame field, ra, dec, gal_lat."""
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("%"):
            continue
        p = line.split()
        rows.append(dict(field=int(p[0]), ra=float(p[1]), dec=float(p[2]), gal_lat=float(p[5])))  # ID RA Dec Ebv GalLong GalLat
    return pd.DataFrame(rows)


def ang_sep_deg(ra1, dec1, ra2, dec2):
    """Great-circle separation in degrees (vectorised)."""
    r1, d1, r2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    a = np.sin((d2 - d1) / 2) ** 2 + np.cos(d1) * np.cos(d2) * np.sin((r2 - r1) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def fill_to_budget(chosen_bytes, candidates, min_bytes, max_bytes):
    """Add candidate rows (in order; column size_bytes) while the total stays <= max_bytes, stopping once it
    reaches min_bytes. Returns (taken candidate index list, final total)."""
    total, taken = chosen_bytes, []
    for i, size in candidates["size_bytes"].items():
        if total >= min_bytes:
            break
        if total + size <= max_bytes:
            taken.append(i)
            total += size
    return taken, total


def unsaturated_mask(mag, catflags, saturation_mag):
    """Epochs that are not saturated (fainter than saturation_mag) and have no quality flag set."""
    mag, catflags = np.asarray(mag, float), np.asarray(catflags)
    return np.isfinite(mag) & (mag > saturation_mag) & (catflags == 0)


def md5_of(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def api_int(v):
    """The light-curve API writes ccdid and qid in hexadecimal ('0xd' = 13) and field in decimal."""
    s = str(v).strip().lower()
    return int(s, 16) if s.startswith("0x") else int(float(s))


def human(nbytes):
    return f"{nbytes / 1024 ** 3:.2f} GB" if nbytes >= 1024 ** 3 else f"{nbytes / 1024 ** 2:.1f} MB"


# ------------------------------------------------------------------ network helpers
def http_get(url, timeout=120, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception as exc:  # noqa: BLE001 - retried, then reported
            if i == tries - 1:
                raise RuntimeError(f"GET {url} failed: {exc}") from exc
            time.sleep(3 * (i + 1))


def http_size(url, timeout=60):
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return int(r.headers["Content-Length"])


def star_epochs(api_url, ra, dec, radius_arcsec):
    """All ZTF epochs within radius of (ra, dec) from the IRSA light-curve API (CSV)."""
    q = urllib.parse.urlencode({"POS": f"CIRCLE {ra} {dec} {radius_arcsec / 3600:.6f}", "FORMAT": "csv"})
    text = http_get(f"{api_url}?{q}").decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(text)) if text.strip() else pd.DataFrame()


# ------------------------------------------------------------------ shared inputs
def aavso_stars(cfg):
    """Our AAVSO stars with VSX name, type, label, RA/Dec (labels file in PROJECT_ROOT/work/labels,
    extended through VizieR for stars not in it yet)."""
    from .labels import assign_labels, vsx_rows
    from .rawfiles import list_csv_files
    seen = {}
    for f in list_csv_files(cfg.data_dir):
        d = pd.read_csv(f, usecols=["auid", "target"], dtype=str)
        d["auid"] = d["auid"].str.strip().str.upper()
        for auid, tgt in d.dropna().groupby("auid")["target"].agg(lambda s: s.mode().iat[0]).items():
            seen.setdefault(auid, tgt.strip())
    stars = pd.DataFrame({"auid": list(seen), "data_name": list(seen.values())})
    lab, _ = assign_labels(stars, vsx_rows(cfg, stars), cfg.load_yaml("label_map"),
                           cfg.section("ingest")["dec_band_edges"])
    return lab


def checksum_index(cfg):
    """Every file of the release (cached in V3_ROOT/work/ztf_checksum_<release>.md5)."""
    z = cfg.section("ztf")
    path = os.path.join(cfg.work_dir, f"ztf_checksum_{z['release']}.md5")
    if not os.path.exists(path):
        print(f"downloading {z['base_url']}/checksum.md5 (about 15 MB) ...")
        data = http_get(f"{z['base_url']}/checksum.md5", timeout=600)
        with open(path, "wb") as fh:
            fh.write(data)
    with open(path, encoding="utf-8") as fh:
        idx = parse_checksums(fh.read())
    idx["url"] = z["base_url"] + "/" + idx["rel_path"]
    return idx


# ------------------------------------------------------------------ ztf_select
def find_star_tiles(stars, z):
    """One API cone search per star (4 at a time) -> per (star, field, ccd, q, filter): epochs, unsaturated epochs."""
    rows = []
    stars = stars.dropna(subset=["ra", "dec"])

    def fetch(s):
        try:
            return star_epochs(z["api_url"], s["ra"], s["dec"], z["match_arcsec"]), None
        except RuntimeError as exc:
            return None, exc

    with ThreadPoolExecutor(4) as ex:
        results = ex.map(fetch, [s for _, s in stars.iterrows()])
        pairs = zip((s for _, s in stars.iterrows()), results)
        for s, (ep, err) in pairs:
            if err is not None:
                print(f"  {s['auid']} {s['data_name']}: API error ({err}); skipped")
                continue
            _summarise_star(s, ep, z, rows)
    return pd.DataFrame(rows, columns=["auid", "name", "label", "field", "ccd", "q", "filter", "epochs", "unsaturated",
                                       "median_mag"])


def _summarise_star(s, ep, z, rows):
    """Append one star's per-tile epoch counts to rows."""
    if ep.empty or "field" not in ep:
        print(f"  {s['auid']} {s['data_name']:14s} not in ZTF (no epochs within {z['match_arcsec']}\")")
        return
    ep["unsat"] = unsaturated_mask(ep["mag"], ep["catflags"], z["saturation_mag"])
    g = ep.groupby(["field", "ccdid", "qid", "filtercode"]).agg(epochs=("mag", "size"), unsaturated=("unsat", "sum"),
                                                                median_mag=("mag", "median")).reset_index()
    for _, r in g.iterrows():
        rows.append(dict(auid=s["auid"], name=s.get("name") or s["data_name"], label=s["label"],
                         field=api_int(r["field"]), ccd=api_int(r["ccdid"]), q=api_int(r["qid"]),
                         filter=r["filtercode"], epochs=int(r["epochs"]), unsaturated=int(r["unsaturated"]),
                         median_mag=round(float(r["median_mag"]), 2)))
    print(f"  {s['auid']} {s['data_name']:14s} {len(ep):6d} epochs, {int(ep['unsat'].sum()):6d} unsaturated, "
          f"fields {sorted(ep['field'].unique().tolist())}", flush=True)


def vsx_richness(cfg, centres, radius_deg):
    """Number of VSX variables of our label-map types within radius of each field centre, or None when no
    VSX backend is reachable (the caller then ranks fields by distance only)."""
    from . import vsx
    from .labels import primary_type
    lm = cfg.load_yaml("label_map")
    wanted = set(lm["anomalous"]) | set(lm["normal"])
    try:
        vsx.backend(cfg)
    except RuntimeError as exc:
        print(f"NOTE: {exc}\nNeighbouring fields are ranked by distance only (no VSX counts).")
        return None
    out = {}
    for _, c in centres.iterrows():
        try:
            res = vsx.cone(cfg, c["ra"], c["dec"], radius_deg, types=sorted(wanted))
            types = [primary_type(t, lm.get("type_separators")) for t in res["type"]]
            out[int(c["field"])] = int(sum(t in wanted for t in types))
        except Exception as exc:  # noqa: BLE001 - reported; field ranked last
            print(f"  VSX count failed for field {int(c['field'])}: {exc}")
            out[int(c["field"])] = 0
    return out


def run_select(cfg):
    z = cfg.section("ztf")
    out = cfg.output_dir
    min_b, max_b = z["target_gb"][0] * 1024 ** 3, z["target_gb"][1] * 1024 ** 3
    stars = aavso_stars(cfg)
    tiles_path, searched_path = os.path.join(out, "ztf_star_tiles.csv"), os.path.join(out, "ztf_searched_stars.csv")
    wanted = set(stars.dropna(subset=["ra", "dec"])["auid"])
    if os.path.exists(tiles_path) and os.path.exists(searched_path) and \
            set(pd.read_csv(searched_path)["auid"]) == wanted:   # same stars as last time: reuse (resume)
        tiles = pd.read_csv(tiles_path)
        print(f"{len(stars)} AAVSO stars; reusing the saved ZTF cone searches ({tiles_path})")
    else:
        print(f"{len(stars)} AAVSO stars; cone-searching ZTF at each ({z['match_arcsec']}\") ...")
        tiles = find_star_tiles(stars, z)
        tiles.to_csv(tiles_path, index=False)
        pd.DataFrame({"auid": sorted(wanted)}).to_csv(searched_path, index=False)
    idx = checksum_index(cfg)
    print(f"{z['release']}: {len(idx)} files in {idx['field'].nunique()} fields")

    # priority 1: every filter of every tile that holds one of our stars
    t1 = tiles.groupby(["field", "ccd", "q"]).agg(stars=("name", lambda s: ", ".join(sorted(set(map(str, s)))))).reset_index()
    p1 = idx.merge(t1, on=["field", "ccd", "q"])
    p1 = p1[p1["filter"].isin(z["filters"])].copy().assign(
        priority=1, reason=lambda d: "contains AAVSO star(s): " + d["stars"])
    print(f"sizing {len(p1)} priority-1 files (tiles holding our stars) ...")
    with ThreadPoolExecutor(8) as ex:
        p1["size_bytes"] = list(ex.map(http_size, p1["url"]))
    chosen = p1.drop(columns="stars")
    total = int(chosen["size_bytes"].sum())
    if total > max_b:  # too much: keep the tiles with the most of our stars' unsaturated epochs first
        score = tiles.groupby(["field", "ccd", "q"])["unsaturated"].sum().rename("score").reset_index()
        chosen = chosen.merge(score, on=["field", "ccd", "q"], how="left").sort_values("score", ascending=False)
        chosen = chosen[chosen["size_bytes"].cumsum() <= max_b].drop(columns="score")
        total = int(chosen["size_bytes"].sum())
    print(f"priority 1: {len(chosen)} files, {human(total)}")

    if total < min_b:  # priority 2: neighbouring fields, richest in labelled VSX variables first
        centres = parse_field_centres(http_get(z["fields_url"]).decode())
        ours = centres[centres["field"].isin(tiles["field"].unique())]
        near = centres[centres["field"].isin(idx["field"].unique())].copy()
        near["dist_deg"] = [ang_sep_deg(ours["ra"], ours["dec"], r, d).min() if len(ours) else np.inf
                            for r, d in zip(near["ra"], near["dec"])]
        near = near[near["dist_deg"] <= z["neighbour_deg"]].sort_values("dist_deg").head(int(z["max_neighbour_fields"]))
        print(f"counting VSX variables in the {len(near)} nearest fields (within {z['neighbour_deg']} deg) ...")
        cache = os.path.join(cfg.work_dir, "ztf_field_vsx_counts.csv")   # VSX counts are reused on a re-run
        known = pd.read_csv(cache).set_index("field")["vsx_variables"].to_dict() if os.path.exists(cache) else {}
        todo = near[~near["field"].isin(known)]
        counted = vsx_richness(cfg, todo, z["field_radius_deg"]) if len(todo) else {}
        rank_by_vsx = counted is not None
        rich = {**known, **(counted or {})}
        if rank_by_vsx:
            pd.DataFrame({"field": list(rich), "vsx_variables": list(rich.values())}).to_csv(cache, index=False)
            near["vsx_variables"] = near["field"].map(rich)
            near = near.sort_values(["vsx_variables", "dist_deg"], ascending=[False, True])
        else:  # VizieR unreachable: nearest fields first, and the log says so
            near["vsx_variables"] = np.nan
            near = near.sort_values("dist_deg")
        rest = idx[idx["filter"].isin(z["filters"]) & ~idx["rel_path"].isin(chosen["rel_path"])]
        for _, f in near.iterrows():
            cand = rest[rest["field"] == f["field"]].sort_values(["ccd", "q", "filter"])
            if cand.empty or total >= min_b:
                continue
            with ThreadPoolExecutor(8) as ex:
                cand = cand.assign(size_bytes=list(ex.map(http_size, cand["url"])))
            take, total = fill_to_budget(total, cand, min_b, max_b)
            if take:
                where = ("other tiles of a field that holds our stars" if f["dist_deg"] < 0.01 else
                         f"neighbouring field, {f['dist_deg']:.1f} deg from our stars' fields")
                why = (f"{where}; {int(f['vsx_variables'])} labelled VSX variables within {z['field_radius_deg']} deg"
                       if np.isfinite(f["vsx_variables"]) else f"{where}; ranked by distance (VizieR was unreachable)")
                chosen = pd.concat([chosen, cand.loc[take].assign(priority=2, reason=why)], ignore_index=True)
    chosen = chosen.sort_values(["priority", "field", "ccd", "q", "filter"])
    chosen.to_csv(os.path.join(out, "ztf_selection.csv"), index=False)
    fl = chosen.groupby(["priority", "field"]).agg(files=("rel_path", "size"), bytes=("size_bytes", "sum"),
                                                   reason=("reason", "first")).reset_index()
    fl.loc[fl["priority"] == 1, "reason"] = "holds AAVSO star(s)"
    fl["gb"] = (fl["bytes"] / 1024 ** 3).round(3)
    fl = fl.drop(columns="bytes")
    fl.to_csv(os.path.join(out, "ztf_fields_chosen.csv"), index=False)
    print(fl.to_string(index=False))
    found = tiles["auid"].nunique()
    print(f"\nAAVSO stars with ZTF epochs: {found} of {len(stars)}; chosen {len(chosen)} files, {human(total)} "
          f"(target {z['target_gb'][0]}-{z['target_gb'][1]} GB)")
    if total < min_b:
        print(f"NOTE: the size target was not reached ({human(total)} < {z['target_gb'][0]} GB): raise "
              "ztf.neighbour_deg to consider more fields.")
    return chosen


# ------------------------------------------------------------------ ztf_download
def _download(url, dest):
    if shutil.which("wget"):
        res = subprocess.run(["wget", "-q", "-c", "--tries=5", "--timeout=120", "-O", dest, url])
        if res.returncode:
            raise RuntimeError(f"wget failed ({res.returncode}) for {url}")
    else:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, open(dest, "wb") as fh:
            shutil.copyfileobj(r, fh, 1 << 22)


def run_download(cfg):
    z = cfg.section("ztf")
    sel_path = os.path.join(cfg.output_dir, "ztf_selection.csv")
    if not os.path.exists(sel_path):
        raise FileNotFoundError("Run the ztf_select step first (it writes ztf_selection.csv).")
    sel = pd.read_csv(sel_path)
    raw = cfg.ztf_raw_dir
    man_path = os.path.join(raw, "MANIFEST.csv")
    man = pd.read_csv(man_path) if os.path.exists(man_path) else pd.DataFrame(columns=["rel_path", "size_bytes", "md5"])
    done = set(man["rel_path"])
    tmp = z.get("tmp_dir") or ("/content/ztf_tmp" if os.path.isdir("/content") else os.path.join(cfg.work_dir, "ztf_tmp"))
    os.makedirs(tmp, exist_ok=True)
    total = int(sel["size_bytes"].sum())
    have = int(sel.loc[sel["rel_path"].isin(done), "size_bytes"].sum())
    todo = sel[~sel["rel_path"].isin(done)]
    print(f"{len(sel)} files, {human(total)} in total; already verified in Drive: {len(sel) - len(todo)} "
          f"({human(have)}); to download: {len(todo)}")
    t0, got = time.time(), 0
    for i, r in enumerate(todo.itertuples(index=False), 1):
        dest = os.path.join(raw, f"field{int(r.field):06d}", os.path.basename(r.rel_path))
        if os.path.exists(dest) and os.path.getsize(dest) == r.size_bytes and md5_of(dest) == r.md5:
            status = "already in Drive, checksum OK"
        else:
            local = os.path.join(tmp, os.path.basename(r.rel_path))
            for attempt in (1, 2):
                _download(r.url, local)
                if md5_of(local) == r.md5:
                    break
                os.remove(local)
                if attempt == 2:
                    raise RuntimeError(f"MD5 mismatch twice for {r.url}")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copyfile(local, dest + ".part")
            os.replace(dest + ".part", dest)
            os.remove(local)
            status = "downloaded, checksum OK"
        pd.DataFrame([dict(rel_path=r.rel_path, size_bytes=r.size_bytes, md5=r.md5)]).to_csv(
            man_path, mode="a", header=not os.path.exists(man_path), index=False)
        got += r.size_bytes
        el = time.time() - t0
        rate = got / el if el > 0 else 0
        eta = (int(todo["size_bytes"].sum()) - got) / rate if rate else float("nan")
        left = f"about {eta / 60:.0f} min left" if math.isfinite(eta) else "estimating time left"
        print(f"[{i}/{len(todo)}] {os.path.basename(r.rel_path)} {human(r.size_bytes)} - {status} | "
              f"{human(have + got)} of {human(total)} | {rate / 1024 ** 2:.1f} MB/s | {left}", flush=True)
    print(f"\nALL {len(sel)} ZTF FILES PRESENT AND VERIFIED in {raw}")


# ------------------------------------------------------------------ ztf_audit
def run_audit(cfg):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    z = cfg.section("ztf")
    out = cfg.output_dir
    raw = cfg.ztf_raw_dir
    man_path = os.path.join(raw, "MANIFEST.csv")
    if not os.path.exists(man_path):
        raise FileNotFoundError("No downloaded ZTF files yet: run ztf_download first.")
    man = pd.read_csv(man_path).drop_duplicates("rel_path")
    rows, t0 = [], time.time()
    for k, rel in enumerate(man["rel_path"], 1):
        m = FILE_RE.search(rel)
        path = os.path.join(raw, f"field{int(m.group(1)):06d}", os.path.basename(rel))
        t = pq.read_table(path, columns=["objectid", "nepochs", "hmjd"])
        flat = pc.list_flatten(t["hmjd"])
        rows.append(dict(file=os.path.basename(rel), field=int(m.group(1)), filter=m.group(2), ccd=int(m.group(3)),
                         q=int(m.group(4)), size_bytes=os.path.getsize(path), lightcurves=t.num_rows,
                         epochs=int(pc.sum(t["nepochs"]).as_py() or 0),
                         mjd_min=pc.min(flat).as_py(), mjd_max=pc.max(flat).as_py()))
        if k % 25 == 0:
            print(f"  audited {k}/{len(man)} files ({time.time() - t0:.0f} s)", flush=True)
    files = pd.DataFrame(rows)
    files.to_csv(os.path.join(out, "ztf_audit_files.csv"), index=False)
    mjd = lambda v: str((pd.Timestamp("1858-11-17") + pd.to_timedelta(v, unit="D")).date())

    # our AAVSO stars, matched within match_arcsec in their tiles
    tiles = pd.read_csv(os.path.join(out, "ztf_star_tiles.csv"))
    stars = aavso_stars(cfg).set_index("auid")
    star_rows = []
    for (auid, fld, ccd, q, filt), _ in tiles.groupby(["auid", "field", "ccd", "q", "filter"]):
        f = files[(files["field"] == fld) & (files["ccd"] == ccd) & (files["q"] == q) & (files["filter"] == filt)]
        if f.empty:
            continue
        path = os.path.join(raw, f"field{int(fld):06d}", f["file"].iat[0])
        t = pq.read_table(path, columns=["objectid", "objra", "objdec", "mag", "catflags"]).to_pandas()
        s = stars.loc[auid]
        sep = ang_sep_deg(t["objra"].to_numpy(), t["objdec"].to_numpy(), s["ra"], s["dec"]) * 3600
        near = t[sep <= z["match_arcsec"]]
        for _, o in near.iterrows():
            unsat = unsaturated_mask(o["mag"], o["catflags"], z["saturation_mag"])
            star_rows.append(dict(auid=auid, name=s.get("name") or s["data_name"], label=s["label"], field=fld,
                                  ccd=ccd, q=q, filter=filt, objectid=int(o["objectid"]), epochs=len(o["mag"]),
                                  unsaturated=int(unsat.sum()), median_mag=round(float(np.nanmedian(o["mag"])), 2)))
    st = pd.DataFrame(star_rows)
    st.to_csv(os.path.join(out, "ztf_aavso_stars.csv"), index=False)
    per_star = st.groupby("auid").agg(epochs=("epochs", "sum"), unsaturated=("unsaturated", "sum")) if len(st) else \
        pd.DataFrame(columns=["epochs", "unsaturated"])
    min_ep = int(z["min_star_epochs"])
    summary = pd.DataFrame([
        ("release", z["release"]), ("files", len(files)), ("fields", files["field"].nunique()),
        ("size, GB (1 GB = 1024^3 bytes)", round(files["size_bytes"].sum() / 1024 ** 3, 3)),
        ("light curves (rows: one per object and filter)", int(files["lightcurves"].sum())),
        ("epochs (photometric points)", int(files["epochs"].sum())),
        ("filters", ", ".join(sorted(files["filter"].unique()))),
        ("first date", mjd(files["mjd_min"].min())), ("last date", mjd(files["mjd_max"].max())),
        ("AAVSO stars searched", len(stars)),
        ("AAVSO stars found in the downloaded ZTF files (2 arcsec)", int(st["auid"].nunique()) if len(st) else 0),
        (f"... of which with >= {min_ep} unsaturated epochs", int((per_star["unsaturated"] >= min_ep).sum())),
        ("saturation limit used (mag)", z["saturation_mag"]),
    ], columns=["item", "value"])
    summary.to_csv(os.path.join(out, "ztf_audit_summary.csv"), index=False)
    print("\n=== ZTF files (first 20) ===\n" + files.drop(columns=["mjd_min", "mjd_max"]).head(20).to_string(index=False))
    print("\n=== Our AAVSO stars in ZTF (epochs summed over filters) ===")
    print(per_star.join(stars[["data_name", "label"]]).sort_values("unsaturated", ascending=False).to_string())
    print("\n=== ZTF SUMMARY ===\n" + summary.to_string(index=False))
    return summary
