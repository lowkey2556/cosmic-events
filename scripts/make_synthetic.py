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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "data", "synthetic"))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    generate(a.out, a.seed)
