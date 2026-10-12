"""Finding and checking the raw AAVSO CSV files (plain Python, no Spark needed).

What it does:
  1. Walks the data folder recursively (EXISTING/, NEW_2024/, NEW_2025/ ...) and returns every
     *.csv file. ZIPs and any other non-CSV files are ignored, and so is every folder named in
     `ingest.exclude_dirs` (DOWNLOAD_LOG by default), at any depth.
  2. Reads only the first line of each CSV and compares it with the expected AAVSO export
     header. A file whose header differs is skipped and reported with the header it actually
     has. No column mapping is ever guessed.

Why: data arrives in download rounds, in several sub-folders. Ingestion (Step 1) and the
dataset audit must see exactly the same set of files, and a file in the wrong format must be
visible in the audit, not silently mis-read.
"""
import csv
import io
import os

RAW_COLUMNS = ["#", "target", "auid", "jd", "mag", "uncertainty", "fainterthan", "band", "type",
               "observer", "airmass", "transformed", "comp1_c", "cmag", "comp2_k", "kmag",
               "chart", "commentcode", "comments"]


def find_csv_files(root, exclude_dirs=("DOWNLOAD_LOG",)):
    """Sorted paths of every *.csv under `root`, skipping excluded folder names (case-insensitive)."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Data folder not found: {root}")
    excluded = {d.upper() for d in exclude_dirs or []}
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d.upper() not in excluded)
        out += [os.path.join(dirpath, f) for f in sorted(filenames) if f.lower().endswith(".csv")]
    return out


def read_header(path):
    """Column names in the first line of a CSV (BOM and surrounding spaces removed), or [] if empty."""
    with open(path, "rb") as fh:
        first = fh.readline().decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(io.StringIO(first)))
    return [c.strip() for c in rows[0]] if rows else []


def check_headers(files, root):
    """Split files into (accepted, skipped). Each skipped entry says why; nothing is remapped."""
    accepted, skipped = [], []
    for f in files:
        rel = os.path.relpath(f, root).replace(os.sep, "/")
        found = read_header(f)
        if not found:
            skipped.append(dict(file=rel, reason="empty file", header_found=""))
        elif [c.lower() for c in found] != RAW_COLUMNS:
            skipped.append(dict(file=rel, reason="header does not match the AAVSO export columns",
                                header_found=",".join(found)[:300]))
        else:
            accepted.append(f)
    for s in skipped:
        print(f"SKIPPED {s['file']}: {s['reason']}  found: {s['header_found'][:120]}")
    return accepted, skipped


def discover(cfg):
    """(root, accepted_files, skipped) for the configured dataset."""
    root = cfg.raw_dir
    files = find_csv_files(root, cfg.section("ingest").get("exclude_dirs", ["DOWNLOAD_LOG"]))
    accepted, skipped = check_headers(files, root)
    print(f"{len(files)} CSV files under {root}: {len(accepted)} accepted, {len(skipped)} skipped")
    return root, accepted, skipped
