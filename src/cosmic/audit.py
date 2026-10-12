"""Step 0 — Dataset audit of the raw AAVSO CSVs (plain Python, runs before the Big Data stack).

What it does:
  1. Finds every CSV under the data folder exactly as ingestion does (recursive; ZIPs, other
     files and DOWNLOAD_LOG/ ignored) and skips any file whose header is not the AAVSO export
     header (listed, never re-mapped).
  2. Reads every accepted file record by record (a quoted comment containing a line break stays
     one record) and reports, per file and in total: size in MB, rows, stars (by `auid`),
     observers and date range.
  3. Checks the total size against the target (50 MB), twice:
       * total CSV size on disk;
       * size of the unique observations only. Overlapping downloads repeat the same
         observation; Step 2 removes those repeats with the key (auid, jd, band, observer, mag),
         and the audit uses the same key, adding up the exact bytes of each first occurrence
         (header lines not counted).
     Both get PASS/FAIL, so repeated rows can never make the dataset look bigger than it is.
  4. Per star (unique rows only): rows, observers, first and last date, the longest continuous
     coverage in days, and how many time windows the windowing step will cut from its
     detections (same rule as Step "windows"; counted here on raw detections, before cleaning,
     so the final counts after cleaning can be lower).
  5. Class balance (anomalous / normal / unlabelled) by stars and by rows, from VSX types and
     config/label_map.yaml, plus the list of stars whose VSX type is not in the label map.
All tables are printed and saved as audit_*.csv in the output folder.

Why: Review 2 requires at least 50 MB of genuine AAVSO data. This step measures what is
really there before anything else runs, and shows exactly which files and stars are
dropped and why.
"""
import csv
import io
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .labels import assign_labels, vsx_rows
from .rawfiles import RAW_COLUMNS, discover
from .windowing import count_windows, longest_coverage

MIB = 1024 * 1024
_IDX = {c: i for i, c in enumerate(RAW_COLUMNS)}


def jd_to_date(jd):
    if jd is None or not np.isfinite(jd):
        return None
    return str((pd.Timestamp("1970-01-01") + pd.to_timedelta(jd - 2440587.5, unit="D")).date())


def _float(text):
    try:
        v = float(text)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def iter_records(path):
    """Yield (record_bytes, fields) for every data record of one CSV (header skipped).

    A physical line is joined with the next while it has an odd number of quote characters, so
    a quoted comment containing a line break stays in its own record (as Spark's multiLine read
    does). record_bytes is the exact number of bytes the record occupies in the file.
    """
    with open(path, "rb") as fh:
        fh.readline()  # header (already checked)
        buf, quotes = [], 0
        for line in fh:
            buf.append(line)
            quotes += line.count(b'"')
            if quotes % 2:
                continue
            raw = b"".join(buf)
            buf, quotes = [], 0
            text = raw.decode("utf-8", errors="replace")
            if not text.strip():
                continue
            fields = next(csv.reader(io.StringIO(text)), [])
            yield len(raw), fields
        if buf:  # unterminated quote at end of file: still one record
            raw = b"".join(buf)
            yield len(raw), next(csv.reader(io.StringIO(raw.decode("utf-8", errors="replace"))), [])


def _parse(fields):
    """The few typed fields the audit needs, normalised as in ingestion (TRIM/UPPER/CAST)."""
    f = fields + [""] * (len(RAW_COLUMNS) - len(fields))
    get = lambda c: f[_IDX[c]].strip()
    return dict(target=get("target"), auid=get("auid").upper() or None, jd=_float(get("jd")),
                mag=_float(get("mag")), band=get("band") or None, observer=get("observer").upper() or None,
                fainter=get("fainterthan").upper() in ("TRUE", "T", "1", "YES", "Y", "<"))


def scan(root, files):
    """Read all accepted files once. Returns (per-file rows, per-star accumulators, totals).

    repeated_rows of a file = its rows whose (auid, jd, band, observer, mag) was already seen,
    earlier in the same file or in a file read before it.
    """
    seen = set()
    per_file = []
    stars = defaultdict(lambda: dict(rows=0, observers=set(), jd=[], det_jd=[], names=Counter()))
    tot = dict(rows=0, unique_rows=0, unique_bytes=0, bands=set(), observers=set())
    for path in files:
        size = os.path.getsize(path)
        f_rows = f_dupes = 0
        f_stars, f_obs, f_jd = set(), set(), []
        for nbytes, fields in iter_records(path):
            r = _parse(fields)
            f_rows += 1
            if r["auid"]:
                f_stars.add(r["auid"])
            if r["observer"]:
                f_obs.add(r["observer"])
            if r["jd"] is not None:
                f_jd.append(r["jd"])
            key = (r["auid"], r["jd"], r["band"], r["observer"], r["mag"])
            if key in seen:
                f_dupes += 1
                continue
            seen.add(key)
            tot["unique_rows"] += 1
            tot["unique_bytes"] += nbytes
            if r["band"]:
                tot["bands"].add(r["band"])
            if r["observer"]:
                tot["observers"].add(r["observer"])
            if r["auid"]:
                s = stars[r["auid"]]
                s["rows"] += 1
                s["names"][r["target"]] += 1
                if r["observer"]:
                    s["observers"].add(r["observer"])
                if r["jd"] is not None:
                    s["jd"].append(r["jd"])
                    if not r["fainter"] and r["mag"] is not None:
                        s["det_jd"].append(r["jd"])
        tot["rows"] += f_rows
        per_file.append(dict(file=os.path.relpath(path, root).replace(os.sep, "/"), size_bytes=size,
                             size_mb=round(size / MIB, 3), rows=f_rows,
                             repeated_rows=f_dupes, stars=len(f_stars),
                             observers=len(f_obs),
                             first_date=jd_to_date(min(f_jd)) if f_jd else None,
                             last_date=jd_to_date(max(f_jd)) if f_jd else None))
    return pd.DataFrame(per_file), stars, tot


def star_table(stars, w, max_gap):
    rows = []
    for auid, s in stars.items():
        jd = np.array(s["jd"])
        rows.append(dict(auid=auid, data_name=s["names"].most_common(1)[0][0], rows=s["rows"],
                         detections=len(s["det_jd"]), observers=len(s["observers"]),
                         first_date=jd_to_date(jd.min()) if len(jd) else None,
                         last_date=jd_to_date(jd.max()) if len(jd) else None,
                         span_days=round(float(jd.max() - jd.min()), 1) if len(jd) else 0.0,
                         longest_continuous_coverage_days=round(longest_coverage(s["det_jd"], max_gap), 1),
                         windows=count_windows(s["det_jd"], w["length_days"], w["stride_days"], w["min_obs"])))
    return pd.DataFrame(rows).sort_values("rows", ascending=False).reset_index(drop=True)


def _save(pdf, out_dir, name):
    path = os.path.join(out_dir, name)
    pdf.to_csv(path, index=False)
    print(f"saved {path}")


def _show(title, pdf):
    print(f"\n=== {title} ===")
    print(pdf.to_string(index=False) if len(pdf) else "(none)")


def run(cfg, fetch_vsx=True):
    a = cfg.section("audit")
    w = cfg.section("windows")
    target_mb = float(a["target_mb"])
    root, files, skipped = discover(cfg)
    skipped = pd.DataFrame(skipped, columns=["file", "reason", "header_found"])
    files_df, stars, tot = scan(root, files)
    st = star_table(stars, w, float(a["coverage_max_gap_days"])) if stars else pd.DataFrame(
        columns=["auid", "data_name", "rows", "detections", "observers", "first_date", "last_date",
                 "span_days", "longest_continuous_coverage_days", "windows"])

    # labels
    label_map = cfg.load_yaml("label_map")
    data_stars = st[["auid", "data_name"]]
    try:
        vsx = vsx_rows(cfg, data_stars, fetch=fetch_vsx)
    except FileNotFoundError as exc:
        print(f"WARNING: no labels file ({exc}); every star counts as unlabelled")
        vsx = pd.DataFrame(columns=["auid", "name", "ra", "dec", "type"])
    labels, unmatched = assign_labels(data_stars, vsx, label_map, cfg.section("ingest")["dec_band_edges"])
    lab = labels.set_index("auid")
    st.insert(2, "vsx_type", st["auid"].map(lab["vsx_type"]))
    st.insert(3, "class", st["auid"].map(lab["label"].map({1: "anomalous", 0: "normal"})).fillna("unlabelled"))
    bal = st.groupby("class").agg(stars=("auid", "count"), rows=("rows", "sum"), windows=("windows", "sum"))
    bal = bal.reindex(["anomalous", "normal", "unlabelled"], fill_value=0).reset_index()
    for c in ("stars", "rows", "windows"):
        bal[f"{c}_pct"] = (100 * bal[c] / max(bal[c].sum(), 1)).round(1)

    # totals
    total_bytes = int(files_df["size_bytes"].sum()) if len(files_df) else 0
    unique_bytes = tot["unique_bytes"]  # data records only (header lines not counted)
    all_jd = np.concatenate([np.array(s["jd"]) for s in stars.values()]) if stars else np.array([])
    total_mb, unique_mb = total_bytes / MIB, unique_bytes / MIB
    summary = pd.DataFrame([
        ("csv files found", len(files) + len(skipped)), ("csv files accepted", len(files)),
        ("csv files skipped (bad header)", len(skipped)),
        ("total CSV size, MB (1 MB = 1,048,576 bytes)", round(total_mb, 2)),
        ("total CSV size, bytes", total_bytes),
        (f"target {target_mb:g} MB: total size", "PASS" if total_mb >= target_mb else "FAIL"),
        ("unique-observation size, MB (repeats across files removed)", round(unique_mb, 2)),
        (f"target {target_mb:g} MB: unique-observation size", "PASS" if unique_mb >= target_mb else "FAIL"),
        ("rows (all files)", tot["rows"]), ("unique rows", tot["unique_rows"]),
        ("repeated rows (same auid, jd, band, observer, mag)", tot["rows"] - tot["unique_rows"]),
        ("stars (auid)", len(stars)), ("observers", len(tot["observers"])), ("bands", len(tot["bands"])),
        ("first date", jd_to_date(all_jd.min()) if len(all_jd) else None),
        ("last date", jd_to_date(all_jd.max()) if len(all_jd) else None),
        ("windows (raw detections, before cleaning)", int(st["windows"].sum()) if len(st) else 0),
        ("window rule", f"{w['length_days']} d long, {w['stride_days']} d stride, >= {w['min_obs']} obs"),
        ("coverage gap limit, days", a["coverage_max_gap_days"]),
    ], columns=["item", "value"])

    out = cfg.output_dir
    _show("Files", files_df.drop(columns="size_bytes"))
    _show("Per star (unique rows; windows counted on raw detections)", st)
    _show("Class balance", bal)
    _show("Skipped files", skipped)
    _show("Stars with unmatched / missing VSX type", unmatched)
    _show("SUMMARY", summary)
    if total_mb >= target_mb > unique_mb:
        print(f"\nNOTE: the files total {total_mb:.2f} MB, but only {unique_mb:.2f} MB are unique observations. "
              "The 50 MB target is NOT met by genuine unique data.")
    for pdf, name in ((files_df, "audit_files.csv"), (st, "audit_stars.csv"), (bal, "audit_class_balance.csv"),
                      (skipped, "audit_skipped_files.csv"), (unmatched, "audit_unmatched_vsx_types.csv"),
                      (summary, "audit_summary.csv")):
        _save(pdf, out, name)
    return dict(files=files_df, stars=st, balance=bal, skipped=skipped, unmatched=unmatched, summary=summary)
