"""Synthetic AAVSO-format data generator — FOR BUILDING AND TESTING ONLY.

What it does: writes ~40 fake stars (half outbursting, half periodic, plus two
with types outside the label map) observed by 15 fake observers, in the exact
AAVSO CSV export format, plus a matching fake VSX labels file. One observer
("NOISY") is deliberately noisy and one ("OFFST") has a constant +0.6 mag offset,
so we can check that the observer agreement score behaves as intended.

Why: lets us build and test every pipeline step before real data arrives.
Nothing produced from this data may ever appear in a result, table or figure;
the pipeline writes synthetic runs to synthetic_run/, never to review2/.

Usage: python scripts/make_synthetic.py [--out data/synthetic] [--seed 1]
"""
import argparse
import csv
import os

import numpy as np

COLUMNS = ["#", "target", "auid", "jd", "mag", "uncertainty", "fainterthan", "band", "type",
           "observer", "airmass", "transformed", "comp1_c", "cmag", "comp2_k", "kmag",
           "chart", "commentcode", "comments"]

JD0, JD1 = 2459215.5, 2459945.5  # 2021-01-01 .. 2022-12-31


def make_observers(rng):
    obs = []
    for i in range(13):
        kind = ["Visual", "Visual", "CCD", "Visual", "DSLR"][i % 5]
        sigma = {"Visual": rng.uniform(0.12, 0.22), "CCD": rng.uniform(0.02, 0.05),
                 "DSLR": rng.uniform(0.04, 0.08)}[kind]
        obs.append(dict(code=f"OB{chr(65 + i)}", kind=kind, sigma=sigma, offset=0.0,
                        limit=rng.uniform(13.5, 15.0) if kind == "Visual" else 16.5))
    obs.append(dict(code="NOISY", kind="Visual", sigma=0.9, offset=0.0, limit=14.0))
    obs.append(dict(code="OFFST", kind="CCD", sigma=0.04, offset=0.6, limit=16.5))
    return obs


def light_curve(kind, p, t):
    if kind == "UGSS":  # dwarf nova: quiescence with fast-rise, exponential-decline outbursts
        phase = np.mod(t - p["t0"], p["rec"])
        out = np.where(phase < p["dur"], p["amp"] * np.exp(-phase / (p["dur"] / 2.5)), 0.0)
        return p["q"] - out
    if kind == "RCB":  # bright star with occasional deep fades
        phase = np.mod(t - p["t0"], p["rec"])
        return p["q"] + np.where(phase < p["dur"], p["amp"] * np.sin(np.pi * phase / p["dur"]), 0.0)
    if kind in ("ZAND", "NL"):  # irregular slow wander with flares
        return p["q"] + 0.8 * np.sin(2 * np.pi * t / p["rec"]) + 0.6 * np.sin(2 * np.pi * t / 17.3) ** 9
    if kind in ("M", "SRB"):
        return p["q"] + p["amp"] / 2 * np.sin(2 * np.pi * (t - p["t0"]) / p["P"])
    if kind == "EA":
        phase = np.mod(t - p["t0"], p["P"]) / p["P"]
        return p["q"] + p["amp"] * np.exp(-((np.minimum(phase, 1 - phase)) / 0.04) ** 2)
    if kind in ("RRAB", "DCEP"):
        phase = np.mod(t - p["t0"], p["P"]) / p["P"]
        return p["q"] + p["amp"] * np.where(phase < 0.15, 1 - phase / 0.15, (phase - 0.15) / 0.85)
    if kind == "EW":
        return p["q"] + p["amp"] / 2 * np.abs(np.sin(2 * np.pi * (t - p["t0"]) / p["P"]))
    return p["q"] + 0.3 * np.sin(2 * np.pi * t / 61.0)  # unlisted types: mild variation


def make_stars(rng):
    plan = (["UGSS"] * 12 + ["RCB"] * 4 + ["ZAND"] * 2 + ["NL"] * 2
            + ["M"] * 6 + ["SRB"] * 4 + ["EA"] * 2 + ["RRAB"] * 2 + ["DCEP"] * 2 + ["EW"] * 2
            + ["GCAS", "LB"])
    stars = []
    for i, kind in enumerate(plan, start=1):
        p = dict(t0=rng.uniform(JD0, JD0 + 50))
        if kind == "UGSS":
            p.update(q=rng.uniform(12.5, 14.0), amp=rng.uniform(2.5, 4.0), rec=rng.uniform(30, 80), dur=rng.uniform(8, 15))
        elif kind == "RCB":
            p.update(q=rng.uniform(6, 8), amp=rng.uniform(3, 6), rec=rng.uniform(150, 300), dur=rng.uniform(40, 80))
        elif kind in ("ZAND", "NL"):
            p.update(q=rng.uniform(9, 11.5), rec=rng.uniform(200, 500))
        elif kind == "M":
            p.update(q=rng.uniform(8, 10), amp=rng.uniform(3, 5), P=rng.uniform(150, 400))
        elif kind == "SRB":
            p.update(q=rng.uniform(6, 8), amp=rng.uniform(0.5, 1.5), P=rng.uniform(50, 150))
        elif kind == "EA":
            p.update(q=rng.uniform(7, 9), amp=1.0, P=rng.uniform(1.5, 4))
        elif kind in ("RRAB", "DCEP"):
            p.update(q=rng.uniform(7, 9), amp=0.8, P=rng.uniform(0.5, 1.0) if kind == "RRAB" else rng.uniform(4, 10))
        elif kind == "EW":
            p.update(q=rng.uniform(8, 10), amp=0.6, P=rng.uniform(0.3, 0.5))
        else:
            p.update(q=rng.uniform(5, 7))
        stars.append(dict(n=i, name=f"SYN Star {i:02d}", auid=f"900-SYN-{i:03d}", kind=kind, p=p,
                          ra=rng.uniform(0, 360), dec=rng.uniform(-60, 85)))
    return stars


def generate(out, seed=1):
    rng = np.random.default_rng(seed)
    observers = make_observers(rng)
    stars = make_stars(rng)
    raw = os.path.join(out, "raw")
    os.makedirs(raw, exist_ok=True)
    os.makedirs(os.path.join(out, "labels"), exist_ok=True)
    total = 0
    for s in stars:
        regular = [o for o in observers[:13]]
        pool = list(rng.choice(regular, size=rng.integers(5, 9), replace=False))
        pool += [observers[13], observers[14]]  # NOISY and OFFST watch every star
        nights = np.sort(rng.choice(np.arange(JD0, JD1), size=rng.integers(30, 45), replace=False))
        rows = []
        for night in nights:
            k = rng.integers(2, 5)
            for o in rng.choice(pool, size=min(k, len(pool)), replace=False):
                npts = rng.integers(2, 4) if o["kind"] == "CCD" else 1
                for _ in range(npts):
                    t = night + rng.uniform(0, 0.4)
                    true = float(light_curve(s["kind"], s["p"], np.array([t]))[0])
                    bands = [("V", 0.0), ("B", 0.6)] if o["kind"] == "CCD" else (
                        [("TG", 0.1)] if o["kind"] == "DSLR" else [("Vis.", 0.0)])
                    for band, colour in bands:
                        m = true + colour + o["offset"] + rng.normal(0, o["sigma"])
                        fainter = m > o["limit"] + colour
                        rows.append(dict(t=t, mag=o["limit"] + colour if fainter else m, fainter=fainter,
                                         band=band, o=o))
        with open(os.path.join(raw, f"{s['n']:02d}_{s['name'].replace(' ', '_')}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(COLUMNS)
            for i, r in enumerate(rows, start=1):
                o = r["o"]
                visual = o["kind"] == "Visual"
                code = rng.choice(["B", "U", "Z", "K", "D"]) if rng.random() < 0.03 else ""
                rec = [i, s["name"] if rng.random() < 0.5 else s["name"].upper(), s["auid"], f"{r['t']:.5f}",
                       f"{r['mag']:.3f}", "" if visual else f"{abs(rng.normal(o['sigma'], 0.01)):.3f}",
                       "TRUE" if r["fainter"] else "FALSE", r["band"], o["kind"], o["code"],
                       "" if visual else f"{rng.uniform(1.0, 2.0):.3f}", "0" if visual else "1",
                       "NA" if visual else "000-BBB-111", "NA" if visual else "11.234",
                       "NA" if visual else "000-BBB-222", "NA" if visual else "12.001",
                       f"X{rng.integers(10000, 99999)}", code, ""]
                w.writerow(rec)
                total += 1
                if rng.random() < 0.01:  # exact duplicate submission
                    w.writerow(rec)
                    total += 1
    with open(os.path.join(out, "labels", "vsx_labels.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["auid", "name", "ra", "dec", "type"])
        for s in stars:
            vsx_type = {"UGSS": "UGSS+E", "M": "M", "SRB": "SRB:"}.get(s["kind"], s["kind"])
            w.writerow([s["auid"], s["name"], f"{s['ra']:.5f}", f"{s['dec']:.5f}", vsx_type])
    print(f"wrote {len(stars)} stars, {len(observers)} observers, {total} rows to {out}")
    return total


# ------------------------------------------------------------------ synthetic ZTF (Phase 2 steps; testing only)
ZTF_FIELDS = [(1001, 40.0, 20.0), (1002, 120.0, 45.0), (1003, 250.0, -10.0)]
ZTF_ANOM, ZTF_NORM = ["UGSS", "UGSU", "ZAND"], ["M", "EA", "RRAB"]


def _ztf_row(rng, oid, field, filt, ra, dec, true_mag_fn, n_bad=True):
    """One ZTF light curve (one object in one filter) in the IRSA DR Parquet layout, with some bad epochs."""
    hmjd = np.sort(rng.uniform(JD0, JD1, rng.integers(180, 260))) - 2400000.5
    colour = 0.0 if filt == "zr" else 0.4
    mag = true_mag_fn(hmjd + 2400000.5) + colour + rng.normal(0, 0.03, len(hmjd))
    magerr = rng.uniform(0.015, 0.05, len(hmjd))
    catflags = np.zeros(len(hmjd), np.int32)
    if n_bad:
        catflags[rng.random(len(hmjd)) < 0.05] = 32768                 # flagged epochs
        mag[rng.random(len(hmjd)) < 0.02] = np.nan                     # missing magnitudes
        dup = rng.choice(len(hmjd), 3, replace=False)                  # exact repeated epochs
        hmjd, mag, magerr, catflags = (np.r_[a, a[dup]] for a in (hmjd, mag, magerr, catflags))
    return dict(objectid=int(oid), filterid=1 if filt == "zg" else 2, fieldid=field, rcid=0,
                objra=float(ra), objdec=float(dec), nepochs=len(hmjd), hmjd=hmjd.tolist(),
                mag=mag.astype(np.float32).tolist(), magerr=magerr.astype(np.float32).tolist(),
                clrcoeff=np.full(len(hmjd), 0.1, np.float32).tolist(), catflags=catflags.tolist())


def generate_ztf(v3_root, seed=1, per_class=30, constant=20, aavso_overlap=6):
    """Synthetic ZTF DR light-curve files for the Phase 2 steps (ztf_ingest .. akval), written where the real ones go:
    V3_ROOT/ztf_raw/fieldNNNNNN/ztf_NNNNNN_<filter>_c01_q1_dr24.parquet + MANIFEST.csv. Also writes the offline
    inputs that would otherwise come from the internet: work/ZTF_Fields.txt and the per-field VSX caches
    work/ztf_vsx/fieldNNNNNN.csv. A few ZTF objects sit exactly on synthetic AAVSO stars (same light curves), so the
    AAVSO-exclusion rule and the A_k validation are exercised. FOR TESTING ONLY."""
    import hashlib

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    rng = np.random.default_rng(seed + 100)
    a_rng = np.random.default_rng(seed)
    make_observers(a_rng)
    aavso = make_stars(a_rng)                                           # the same stars generate() wrote
    raw, work = os.path.join(v3_root, "ztf_raw"), os.path.join(v3_root, "work")
    os.makedirs(os.path.join(work, "ztf_vsx"), exist_ok=True)
    with open(os.path.join(work, "ZTF_Fields.txt"), "w") as fh:
        fh.write("% ID RA Dec Ebv Gal_Long Gal_Lat\n")
        for f, ra, dec in ZTF_FIELDS:
            fh.write(f"{f} {ra} {dec} 0.05 100.0 30.0\n")
    overlap = [s for s in aavso if s["kind"] in ("UGSS", "M") and s["p"]["q"] > 12.6][:aavso_overlap]
    if len(overlap) < aavso_overlap:                                    # make the ZTF copy faint enough anyway
        overlap += [s for s in aavso if s["kind"] in ("UGSS", "M") and s not in overlap][:aavso_overlap - len(overlap)]
    man, oid = [], 10 ** 12
    for fi, (field, fra, fdec) in enumerate(ZTF_FIELDS):
        objs, vsx = [], []
        for k in range(per_class * 2 + constant):
            ra, dec = fra + rng.uniform(-1, 1), fdec + rng.uniform(-1, 1)
            if k < per_class * 2:
                kind = (ZTF_ANOM if k < per_class else ZTF_NORM)[k % 3]
                lc_kind = {"UGSU": "UGSS"}.get(kind, kind)
                p = dict(t0=rng.uniform(JD0, JD0 + 50), q=rng.uniform(15, 17), amp=rng.uniform(1.5, 3.0),
                         rec=rng.uniform(30, 80), dur=rng.uniform(8, 15), P=rng.uniform(0.5, 300))
                if lc_kind == "EA":
                    p["P"] = rng.uniform(1.5, 4)
                elif lc_kind == "RRAB":
                    p["P"], p["amp"] = rng.uniform(0.5, 1.0), 0.8
                vsx.append(dict(name=f"ZSYN {field} {k:03d}", ra=ra + 0.3 / 3600, dec=dec, type=kind, var_flag=0))
                fn = (lambda t, kk=lc_kind, pp=p: light_curve(kk, pp, t))
            else:                                                       # constant star, no VSX entry
                q = rng.uniform(14, 18)
                fn = (lambda t, qq=q: np.full(len(t), qq))
            objs.append((ra, dec, fn))
        if fi == 0:                                                     # ZTF copies of AAVSO stars
            for s in overlap:
                p = dict(s["p"], q=max(s["p"]["q"], 13.0))
                objs.append((s["ra"], s["dec"], (lambda t, kk=s["kind"], pp=p: light_curve(kk, pp, t))))
                vsx.append(dict(name=s["name"], ra=s["ra"], dec=s["dec"],
                                type={"UGSS": "UGSS+E"}.get(s["kind"], s["kind"]), var_flag=0))
        vsx.append(dict(name=f"ZSYN {field} far", ra=fra + 3, dec=fdec, type="M", var_flag=0))  # not in ZTF
        pd.DataFrame(vsx).to_csv(os.path.join(work, "ztf_vsx", f"field{field:06d}.csv"), index=False)
        d = os.path.join(raw, f"field{field:06d}")
        os.makedirs(d, exist_ok=True)
        for filt in ("zg", "zr"):
            rows = []
            for ra, dec, fn in objs:
                oid += 1
                rows.append(_ztf_row(rng, oid, field, filt, ra, dec, fn))
            t = pa.Table.from_pylist(rows, schema=pa.schema([
                ("objectid", pa.int64()), ("filterid", pa.int8()), ("fieldid", pa.int16()), ("rcid", pa.int8()),
                ("objra", pa.float32()), ("objdec", pa.float32()), ("nepochs", pa.int64()),
                ("hmjd", pa.list_(pa.float64())), ("mag", pa.list_(pa.float32())), ("magerr", pa.list_(pa.float32())),
                ("clrcoeff", pa.list_(pa.float32())), ("catflags", pa.list_(pa.int32()))]))
            name = f"ztf_{field:06d}_{filt}_c01_q1_dr24.parquet"
            path = os.path.join(d, name)
            pq.write_table(t, path)
            with open(path, "rb") as fh:
                md5 = hashlib.md5(fh.read()).hexdigest()
            man.append(dict(rel_path=f"0/field{field:06d}/{name}", size_bytes=os.path.getsize(path), md5=md5))
    pd.DataFrame(man).to_csv(os.path.join(raw, "MANIFEST.csv"), index=False)
    print(f"wrote {len(man)} synthetic ZTF files to {raw} ({len(overlap)} ZTF objects on AAVSO stars)")
    return raw


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "data", "synthetic"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ztf-v3-root", default=None,
                    help="also write synthetic ZTF files under this V3_ROOT (e.g. synthetic_run/v3_5GB)")
    a = ap.parse_args()
    generate(a.out, a.seed)
    if a.ztf_v3_root:
        generate_ztf(a.ztf_v3_root, a.seed)
