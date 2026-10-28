"""DATA_FLOW.txt: where the data is at every stage, written from real measurements.

What it does: every step that moves or transforms data calls `record(...)` with what it measured on
that run — the location (Drive path, Colab disk, HDFS path, Hive table, .npz file, checkpoint), the
format, size in bytes (os.path.getsize / the Hadoop FileSystem API), rows, number of files or
partitions, the tool, what changed compared with the previous stage, and the seconds it took. Each
record is a small JSON file in V3_ROOT/work/dataflow/. `write(cfg)` turns all records into
V3_ROOT/outputs/DATA_FLOW.txt: one plain-English block per stage in pipeline order, an ASCII flow
diagram and a totals table. Nothing in it is typed by hand: a stage that has not run yet is listed
as "not run yet".

Why: so the whole journey of the data — AAVSO website and IRSA ZTF → Drive → Colab disk → HDFS →
Hive → cleaned → A_k → windows → features → models → predictions → results — can be studied with
real numbers.
"""
import json
import os
import time

# order, key, title (stages appear in DATA_FLOW.txt in this order)
STAGES = [
    (1, "aavso_source", "AAVSO observations (website download -> Google Drive)"),
    (2, "ztf_source", "ZTF light curves (IRSA -> Colab disk -> Google Drive)"),
    (3, "aavso_hdfs_raw", "AAVSO raw CSVs in HDFS"),
    (4, "aavso_hive_raw", "AAVSO raw table in Hive (partitioned Parquet)"),
    (5, "ztf_hdfs_hive_raw", "ZTF raw Parquet in HDFS + Hive (partitioned by field and filter)"),
    (6, "aavso_clean", "AAVSO cleaned observations"),
    (7, "ztf_clean", "ZTF cleaned light curves"),
    (8, "ak_scores", "Observer agreement scores A_k"),
    (9, "ztf_labels", "ZTF objects labelled from VSX"),
    (10, "ak_validation", "A_k checked against ZTF"),
    (11, "aavso_windows", "AAVSO windows, features and token sequences"),
    (12, "ztf_windows", "ZTF windows, features and token sequences (pre-training pool)"),
    (13, "ml_models", "Spark MLlib models (NB, DT, RF, LR) and their predictions"),
    (14, "rganet_checkpoints", "RGA-Net: pre-trained and fine-tuned checkpoints"),
    (15, "predictions_hive", "RGA-Net predictions written back to Hive by Spark"),
    (16, "results", "Result tables, charts, captions and dashboard data in Drive"),
]
ORDER = {k: (o, t) for o, k, t in STAGES}


def _dir(cfg):
    d = os.path.join(cfg.work_dir, "dataflow")
    os.makedirs(d, exist_ok=True)
    return d


def record(cfg, stage, what, location, fmt, tool, size_bytes=None, rows=None, files=None, compression=None,
           changes="", seconds=None, extra=None):
    """Save one stage's measurements (overwrites that stage's previous record)."""
    order, title = ORDER[stage]
    rec = dict(order=order, stage=stage, title=title, what=what, location=location, format=fmt, tool=tool,
               size_bytes=None if size_bytes is None else int(size_bytes), rows=None if rows is None else int(rows),
               files=files, compression=compression, changes=changes,
               seconds=None if seconds is None else round(float(seconds), 1), extra=extra or {},
               recorded=time.strftime("%Y-%m-%d %H:%M"))
    with open(os.path.join(_dir(cfg), f"{order:02d}_{stage}.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, default=str)
    return rec


def load(cfg):
    out = {}
    for f in sorted(os.listdir(_dir(cfg))):
        if f.endswith(".json"):
            with open(os.path.join(_dir(cfg), f), encoding="utf-8") as fh:
                r = json.load(fh)
            out[r["stage"]] = r
    return out


def record_timing(cfg, step, seconds, mode):
    """Append one step's measured wall time to outputs/step_timings.csv (the latest run of a step replaces older ones)."""
    import pandas as pd
    path = os.path.join(cfg.output_dir, "step_timings.csv")
    row = pd.DataFrame([dict(step=step, seconds=round(float(seconds), 1), mode=mode,
                             finished=time.strftime("%Y-%m-%d %H:%M:%S"))])
    old = pd.read_csv(path) if os.path.exists(path) else row.iloc[:0]
    pd.concat([old[old["step"] != step], row], ignore_index=True).to_csv(path, index=False)


def human(n):
    if n is None:
        return "-"
    for unit, div in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return f"{n} B"


def fmt_rows(n):
    return "-" if n is None else f"{n:,}"


def write(cfg):
    recs = load(cfg)
    lines = ["DATA FLOW - generated from the measurements of the latest run (do not edit; rerun instead)",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')}.  Sizes: 1 MB = 1,048,576 bytes; 1 GB = 1024 MB.", ""]
    for order, key, title in STAGES:
        r = recs.get(key)
        lines.append(f"[{order:02d}] {title}")
        if r is None:
            lines += ["     not run yet", ""]
            continue
        lines += [f"     What happens : {r['what']}",
                  f"     Location     : {r['location']}",
                  f"     Format       : {r['format']}",
                  f"     Size         : {human(r['size_bytes'])}" + (f"   (compression: {r['compression']})"
                                                                     if r.get("compression") else ""),
                  f"     Rows         : {fmt_rows(r['rows'])}" + (f"   Files/partitions: {r['files']}"
                                                                    if r.get("files") is not None else ""),
                  f"     Tool         : {r['tool']}",
                  f"     Changed      : {r['changes'] or '-'}",
                  f"     Time taken   : {'-' if r['seconds'] is None else str(r['seconds']) + ' s'}"]
        for k, v in (r.get("extra") or {}).items():
            lines.append(f"     {k:<13}: {v}")
        lines.append("")
    lines += ["FLOW DIAGRAM", "",
              "  AAVSO website ---(CSV)---> Drive data/ ---(hdfs dfs -put)---> HDFS /cosmic/raw ---(Spark)---> Hive observations_raw",
              "  IRSA ZTF -------(Parquet)-> Colab disk -> Drive v3_5GB/ztf_raw -> HDFS /cosmic/ztf/raw -> Hive ztf_lightcurves_raw",
              "                                                   |",
              "                     Spark SQL cleaning: Hive obs_clean  +  Hive ztf_lightcurves_clean (copy in Drive clean/)",
              "                                                   |",
              "        A_k (Spark SQL self-join) -> Hive observer_scores      VSX cross-match -> Hive ztf_labels",
              "                                                   |",
              "        Spark windowing -> Hive window_features / window_sequences (+ ztf_*)  -> .npz in Drive work/",
              "                 |                                            |",
              "        Spark MLlib NB/DT/RF/LR                  PyTorch RGA-Net: pre-train on ZTF -> fine-tune on AAVSO",
              "                 |                                            |  (checkpoints in Drive work/models)",
              "        Hive ml_predictions                      Spark mapInPandas -> Hive window_predictions",
              "                 \\______________________________________________/",
              "                         result CSVs, charts, captions, dashboard_data (Drive v3_5GB/outputs)", ""]
    lines += ["TOTALS", "", f"  {'stage':<62}{'size':>12}{'rows':>18}{'time':>10}"]
    for r in sorted(recs.values(), key=lambda r: r["order"]):
        lines.append(f"  {r['title'][:60]:<62}{human(r['size_bytes']):>12}{fmt_rows(r['rows']):>18}"
                     f"{'-' if r['seconds'] is None else str(int(r['seconds'])) + ' s':>10}")
    lines.append(f"  {'stages measured':<62}{len(recs):>12}")
    lines.append(f"  {'total time of the measured stages':<62}{'':>12}{'':>18}"
                 f"{int(sum(r['seconds'] or 0 for r in recs.values())):>8} s")
    tpath = os.path.join(cfg.output_dir, "step_timings.csv")
    if os.path.exists(tpath):
        import pandas as pd
        t = pd.read_csv(tpath)
        lines += ["", "STEP TIMINGS (wall time of each run.py step, measured on the latest run of that step)", ""]
        lines += [f"  {r.step:<16}{r.seconds:>10.1f} s   {r.mode:<28}{r.finished}" for r in t.itertuples()]
        lines.append(f"  {'total':<16}{t['seconds'].sum():>10.1f} s")
    path = os.path.join(cfg.output_dir, "DATA_FLOW.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"saved {path}")
    return path
