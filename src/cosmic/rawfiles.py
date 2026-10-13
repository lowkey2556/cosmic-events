"""Finding and checking the raw AAVSO CSV files (plain Python, no Spark needed).

What it does:
  1. Lists the *.csv files DIRECTLY inside the data folder (PROJECT_ROOT/data). Nothing else
     is read: not sub-folders, not the outputs or work folders, not ZIPs or other files.
     If CSVs are found in sub-folders of data/, it stops with an error that lists them (they
     must be moved into data/ itself), so no data is ever skipped silently.
  2. Stops with a clear error if two files would get the same name in HDFS (names that differ
     only in upper/lower case), because one would overwrite the other.
  3. Reads only the first line of each CSV and compares it with the expected AAVSO export
     header. A file whose header differs is skipped and reported with the header it actually
     has. No column mapping is ever guessed.

Why: ingestion (Step 1) and the dataset audit must see exactly the same set of files, and a
file in the wrong place or the wrong format must be visible, not silently mis-read.
"""
import csv
import io
import os

RAW_COLUMNS = ["#", "target", "auid", "jd", "mag", "uncertainty", "fainterthan", "band", "type",
               "observer", "airmass", "transformed", "comp1_c", "cmag", "comp2_k", "kmag",
               "chart", "commentcode", "comments"]


def list_csv_files(data_dir):
    """Sorted paths of the *.csv files directly in data_dir. Raises if data_dir is missing, if CSVs
    sit in sub-folders, or if two file names collide (case-insensitively)."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data folder not found: {data_dir}")
    names = sorted(os.listdir(data_dir))
    files = [os.path.join(data_dir, n) for n in names
             if n.lower().endswith(".csv") and os.path.isfile(os.path.join(data_dir, n))]
    nested = [os.path.relpath(os.path.join(dp, f), data_dir).replace(os.sep, "/")
              for d in names if os.path.isdir(os.path.join(data_dir, d))
              for dp, _, fs in os.walk(os.path.join(data_dir, d)) for f in fs if f.lower().endswith(".csv")]
    if nested:
        raise ValueError(f"{len(nested)} CSV file(s) are in sub-folders of {data_dir}, which are not read. "
                         "Move them directly into data/ (and remove the sub-folders):\n  "
                         + "\n  ".join(nested[:50]) + ("\n  ..." if len(nested) > 50 else ""))
    dups = duplicate_names([os.path.basename(f) for f in files])
    if dups:
        raise ValueError("Duplicate file names in the data folder (they would overwrite each other in HDFS): "
                         + "; ".join(" / ".join(v) for v in dups) + ". Rename or remove the duplicates.")
    return files


def duplicate_names(names):
    """Groups of file names that are equal apart from upper/lower case (and so collide in HDFS
    or on a case-insensitive file system)."""
    seen = {}
    for n in names:
        seen.setdefault(n.lower(), []).append(n)
    return [v for v in seen.values() if len(v) > 1]


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
    """(data_dir, accepted_files, skipped) for the configured dataset."""
    root = cfg.data_dir
    files = list_csv_files(root)
    accepted, skipped = check_headers(files, root)
    print(f"{len(files)} CSV files in {root}: {len(accepted)} accepted, {len(skipped)} skipped")
    return root, accepted, skipped
