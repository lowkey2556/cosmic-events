"""Command-line entry point.

    python run.py <step>[+<step>...] [--data real|synthetic] [--set key.sub=value ...]

Steps, in pipeline order:
    audit          Step 0   dataset audit of the raw CSVs: sizes, rows, stars, 50 MB check (no Spark)
    architecture            architecture diagram with Graphviz: PNG + SVG + DOT (no Spark)
    ingest         Step 1   HDFS put, VSX labels, Parquet + Hive external table, partition pruning, dataset summary
    preprocess     Step 2   Spark SQL cleaning, thinning, band normalisation (+ rows per star after every step)
    score          Step 3   observer agreement scores
    features       Step 4   per-star features (used by K-Means)
    windows        Step 5   Spark windowing: per-window ML features, RGA-Net token sequences, 5 star-grouped
                            splits, star_attrition.csv; exports work/windows_seq.npz for PyTorch
    cluster        Step 6   K-Means on the per-star features (supporting analysis)
    classify       Step 7   the 4 ML models (Spark MLlib) on the windows, 5 splits, with/without observer features
    rganet         Step 8   RGA-Net + ablations (PyTorch, GPU if available; no Spark needed); saves the models
    infer          Step 8b  distributed inference: Spark applies the trained RGA-Net to ALL windows in parallel
                            (mapInPandas) and writes the Hive table window_predictions
    compare        Step 9   comparison tables, Plotly figures, COMPARISON.md (no Spark needed)
    scalability    Step 10  preprocessing + windowing through YARN at 1, 2, 4 executors, 3 runs each
    all                     audit .. compare in order (not scalability)

Steps can be chained with "+": `python run.py preprocess+windows` runs both in one Spark session.
`--data real` (default) uses Drive data and writes to Drive/cosmic_roja/outputs/review2/.
`--data synthetic` uses data/synthetic/ and writes to synthetic_run/ (never review2/).
"""
import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.stdout.reconfigure(line_buffering=True)  # show progress live in Colab "!python" cells

from cosmic.config import load_config  # noqa: E402

SPARK_STEPS = ["ingest", "preprocess", "score", "features", "windows", "cluster", "classify", "infer"]
LOCAL_STEPS = ["audit", "architecture", "rganet", "compare", "scalability"]
ALL = ["audit", "architecture", "ingest", "preprocess", "score", "features", "windows", "cluster", "classify",
       "rganet", "infer", "compare"]


def architecture(cfg):
    from cosmic.architecture import draw
    path = draw(os.path.join(cfg.output_dir, "architecture.png"), dpi=cfg.section("plots")["dpi"])
    for ext in (".png", ".svg", ".dot"):  # keep a copy in the repo for the docs
        shutil.copy(os.path.splitext(path)[0] + ext, cfg.repo_path(os.path.join("docs", "figures", "architecture" + ext)))


def run_local(step, cfg, overrides):
    if step == "audit":
        from cosmic import audit
        print(f"data folder: {cfg.raw_dir}")
        audit.run(cfg)
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


def run_spark(steps, cfg, timings_json=None):
    from cosmic import classify, cluster, features, infer, ingest, observer_score, preprocess, windows
    from cosmic.spark_session import get_spark
    mods = dict(ingest=ingest, preprocess=preprocess, score=observer_score, features=features, windows=windows,
                cluster=cluster, classify=classify, infer=infer)
    wall0 = time.time()
    spark = get_spark(cfg)
    sc = spark.sparkContext
    print(f"Spark {spark.version} | master={sc.master} | app={sc.applicationId}")
    timings, granted = [], 0
    try:
        for s in steps:
            print(f"\n{'=' * 20} STEP: {s} {'=' * 20}")
            t0 = time.time()
            mods[s].run(spark, cfg)
            timings.append((s, round(time.time() - t0, 1)))
            granted = max(granted, sc._jsc.sc().getExecutorMemoryStatus().size() - 1)  # minus the driver
            print(f"--- {s} finished in {timings[-1][1]} s")
    finally:
        spark.stop()
    print("\nStep timings (s):", timings)
    if timings_json:
        with open(timings_json, "w") as fh:
            json.dump(dict(steps=timings, steps_seconds=sum(t for _, t in timings), executors_granted=granted,
                           wall_seconds=round(time.time() - wall0, 1)), fh)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", help="one step, several joined with '+', or 'all'")
    ap.add_argument("--data", choices=["real", "synthetic"], default="real")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--timings-json", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    steps = ALL if a.step == "all" else a.step.split("+")
    unknown = [s for s in steps if s not in SPARK_STEPS + LOCAL_STEPS]
    if unknown:
        ap.error(f"unknown step(s) {unknown}; choose from {SPARK_STEPS + LOCAL_STEPS}")
    cfg = load_config(a.data, a.set)
    from cosmic import plots
    plots.set_export(**cfg.section("plots"))
    print(f"dataset={a.data}  hive database={cfg.db}  hdfs={cfg.hdfs_base}  outputs={cfg.output_dir}")

    # consecutive Spark steps share one Spark session; other steps run on their own
    group = []
    for s in steps + [None]:
        if s in SPARK_STEPS:
            group.append(s)
            continue
        if group:
            run_spark(group, cfg, a.timings_json)
            group = []
        if s is not None:
            print(f"\n{'=' * 20} STEP: {s} {'=' * 20}")
            run_local(s, cfg, a.set)


if __name__ == "__main__":
    main()
