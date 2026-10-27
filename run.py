"""Command-line entry point.

    python run.py <step>[+<step>...] [--data real|synthetic] [--set key.sub=value ...]

Steps, in pipeline order:
    paths                   print the resolved paths (all derived from PROJECT_ROOT) and check them
    audit          Step 0   dataset audit of the AAVSO CSVs: sizes, rows, stars, 50 MB check (no Spark)
    ztf_select     Phase 1  choose the ZTF DR light-curve files (tiles) to download; logs every choice
    ztf_download   Phase 1  resumable, MD5-verified download of the chosen files to V3_ROOT/ztf_raw
    ztf_audit      Phase 1  audit of the downloaded ZTF files (size, rows, objects, filters, dates, our stars)
    candidates     Phase 1  faint VSX stars (mag ~13-17) to download more AAVSO data for
    architecture            architecture diagram with Graphviz: PNG + SVG + DOT (no Spark)
    ingest         Step 1   HDFS put, VSX labels, Parquet + Hive external table, partition pruning, dataset summary
    preprocess     Step 2   Spark SQL cleaning, thinning, band normalisation (+ rows per star after every step)
    score          Step 3   observer agreement scores
    features       Step 4   per-star features (used by K-Means)
    windows        Step 5   Spark windowing: per-window ML features, RGA-Net token sequences, star-grouped splits
    cluster        Step 6   K-Means on the per-star features (supporting analysis)
    classify       Step 7   the 4 ML models (Spark MLlib) on the windows
    rganet         Step 8   RGA-Net + ablations (PyTorch, GPU if available; no Spark); saves the models
    infer          Step 8b  distributed inference: Spark applies the trained RGA-Net to ALL windows
    compare        Step 9   comparison tables, Plotly figures, COMPARISON.md (no Spark)
    scalability    Step 10  preprocessing + windowing through YARN at 1, 2, 4 executors, 3 runs each
    all                     audit .. compare in order (not the ZTF steps, not scalability)

Fresh machines: before a Spark step runs, any Hive table it needs that is missing is rebuilt by running
the step that makes it first (src/cosmic/pipeline.py). Spark steps run in the Python 3.12 environment
made by setup/python_env.sh when the current Python is 3.13 or newer (PySpark 3.5 does not support it).

`--data real` (default) reads PROJECT_ROOT/data/*.csv and writes only under V3_ROOT = PROJECT_ROOT/v3_5GB
(PROJECT_ROOT = /content/drive/MyDrive/cosmic_roja, or $COSMIC_PROJECT_ROOT).
`--data synthetic` uses data/synthetic/ and writes to synthetic_run/.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.stdout.reconfigure(line_buffering=True)  # show progress live in Colab "!python" cells

from cosmic.config import check_paths, load_config  # noqa: E402

SPARK_STEPS = ["ingest", "preprocess", "score", "features", "windows", "cluster", "classify", "infer"]
LOCAL_STEPS = ["paths", "audit", "ztf_select", "ztf_download", "ztf_audit", "candidates", "architecture",
               "rganet", "compare", "scalability"]
ALL = ["audit", "architecture", "ingest", "preprocess", "score", "features", "windows", "cluster", "classify",
       "rganet", "infer", "compare"]
SPARK_PY = "/content/cosmic_py/bin/python"


def architecture(cfg):
    from cosmic.architecture import draw
    path = draw(os.path.join(cfg.output_dir, "architecture.png"), dpi=cfg.section("plots")["dpi"])
    for ext in (".png", ".svg", ".dot"):  # keep a copy in the repo for the docs
        shutil.copy(os.path.splitext(path)[0] + ext, cfg.repo_path(os.path.join("docs", "figures", "architecture" + ext)))


def run_local(step, cfg, overrides):
    if step == "paths":
        return
    if step == "audit":
        from cosmic import audit
        print(f"data folder: {cfg.data_dir}")
        audit.run(cfg)
    elif step in ("ztf_select", "ztf_download", "ztf_audit"):
        from cosmic import ztf
        getattr(ztf, step.replace("ztf_", "run_"))(cfg)
    elif step == "candidates":
        from cosmic import candidates
        candidates.run(cfg)
    elif step == "architecture":
        architecture(cfg)
    elif step == "rganet":
        from cosmic import rganet
        rganet.run(cfg)
    elif step == "compare":
        from cosmic import compare
        compare.run(cfg)
    elif step == "scalability":
        from cosmic import scalability
        scalability.run(cfg, overrides)


def spark_python():
    """None if this interpreter can run PySpark 3.5, else the Python 3.12 environment's interpreter."""
    if sys.version_info < (3, 13):
        try:
            import pyspark  # noqa: F401
            return None
        except ImportError:
            pass
    py = os.environ.get("COSMIC_SPARK_PY", SPARK_PY)
    if not os.path.exists(py):
        raise RuntimeError(f"Spark steps need Python <= 3.12 with PySpark, and {py} does not exist. "
                           "Run the notebook's setup cell (it runs setup/python_env.sh) first.")
    return py


def run_spark(steps, cfg, args):
    """Run Spark steps in one session, after rebuilding any missing Hive tables they need."""
    py = spark_python()
    if py and os.path.realpath(py) != os.path.realpath(sys.executable):
        cmd = [py, os.path.abspath(__file__), "+".join(steps), "--data", args.data,
               *sum((["--set", o] for o in args.set), [])]
        if args.timings_json:
            cmd += ["--timings-json", args.timings_json]
        if args.only_missing:
            cmd.append("--only-missing")
        print(f"(Spark steps run with {py}: Python {sys.version.split()[0]} is too new for PySpark 3.5)")
        res = subprocess.run(cmd)
        if res.returncode:
            raise SystemExit(res.returncode)
        return
    from cosmic import classify, cluster, features, infer, ingest, observer_score, preprocess, windows
    from cosmic.pipeline import plan_steps
    from cosmic.spark_session import get_spark
    mods = dict(ingest=ingest, preprocess=preprocess, score=observer_score, features=features, windows=windows,
                cluster=cluster, classify=classify, infer=infer)
    wall0 = time.time()
    spark = get_spark(cfg)
    sc = spark.sparkContext
    print(f"Spark {spark.version} | master={sc.master} | app={sc.applicationId}")
    plan = plan_steps(steps, lambda t: spark.catalog.tableExists(t, cfg.db), only_missing=args.only_missing)
    for s, why in plan:
        if why != "requested":
            print(f"fresh machine: {why} -> running step '{s}' first")
    timings, granted = [], 0
    try:
        for s, _ in plan:
            print(f"\n{'=' * 20} STEP: {s} {'=' * 20}")
            t0 = time.time()
            mods[s].run(spark, cfg)
            timings.append((s, round(time.time() - t0, 1)))
            granted = max(granted, sc._jsc.sc().getExecutorMemoryStatus().size() - 1)  # minus the driver
            print(f"--- {s} finished in {timings[-1][1]} s")
    finally:
        spark.stop()
    print("\nStep timings (s):", timings)
    if args.timings_json:
        with open(args.timings_json, "w") as fh:
            json.dump(dict(steps=timings, steps_seconds=sum(t for _, t in timings), executors_granted=granted,
                           wall_seconds=round(time.time() - wall0, 1)), fh)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", help="one step, several joined with '+', or 'all'")
    ap.add_argument("--data", choices=["real", "synthetic"], default="real")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--only-missing", action="store_true",
                    help="only build the missing Hive tables the given Spark steps need, not the steps themselves")
    ap.add_argument("--timings-json", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    steps = ALL if a.step == "all" else a.step.split("+")
    unknown = [s for s in steps if s not in SPARK_STEPS + LOCAL_STEPS]
    if unknown:
        ap.error(f"unknown step(s) {unknown}; choose from {SPARK_STEPS + LOCAL_STEPS}")
    cfg = load_config(a.data, a.set)
    print(f"dataset={a.data}  hive database={cfg.db}  hdfs={cfg.hdfs_base}  python={sys.version.split()[0]}")
    check_paths(cfg, need_data=bool({"paths", "audit", "ingest", "candidates"} & set(steps)))
    from cosmic import plots
    plots.set_export(**cfg.section("plots"))

    # consecutive Spark steps share one Spark session; other steps run on their own
    group = []
    for s in steps + [None]:
        if s in SPARK_STEPS:
            group.append(s)
            continue
        if group:
            run_spark(group, cfg, a)
            group = []
        if s is not None:
            print(f"\n{'=' * 20} STEP: {s} {'=' * 20}")
            run_local(s, cfg, a.set)


if __name__ == "__main__":
    main()
