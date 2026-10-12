"""Command-line entry point.

    python run.py <step> [--data real|synthetic] [--set key.sub=value ...]

Steps:
    audit          Step 0  dataset audit of the raw CSVs: sizes, rows, stars, 50 MB check (no Spark needed)
    architecture   draw architecture.png (no Spark needed)
    ingest         Step 1  HDFS put, VSX labels, Parquet + Hive external table, partition pruning, dataset summary
    preprocess     Step 2  Spark SQL cleaning + band normalisation
    score          Step 3  observer agreement scores
    features       Step 4  per-star features
    classify       Step 5  MLlib classifiers (one fixed seed)
    cluster        Step 6  K-Means
    milestone1     all of the above, in order

`--data real` (default) uses Drive data and writes to Drive/cosmic/outputs/review2/.
`--data synthetic` uses data/synthetic/ and writes to synthetic_run/ (never review2/).
"""
import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.stdout.reconfigure(line_buffering=True)  # show progress live in Colab "!python" cells

from cosmic.config import load_config  # noqa: E402

SPARK_STEPS = ["ingest", "preprocess", "score", "features", "classify", "cluster"]


def architecture(cfg):
    from cosmic.architecture import draw
    path = draw(cfg.repo_path(os.path.join("docs", "figures", "architecture.png")),
                dpi=cfg.section("plots")["dpi"])
    shutil.copy(path, os.path.join(cfg.output_dir, "architecture.png"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["audit", "architecture", *SPARK_STEPS, "milestone1"])
    ap.add_argument("--data", choices=["real", "synthetic"], default="real")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args(argv)
    cfg = load_config(a.data, a.set)
    print(f"dataset={a.data}  hive database={cfg.db}  hdfs={cfg.hdfs_base}  outputs={cfg.output_dir}")

    if a.step == "audit":
        from cosmic import audit
        print(f"data folder: {cfg.raw_dir}")
        audit.run(cfg)
        return
    steps = ["architecture", *SPARK_STEPS] if a.step == "milestone1" else [a.step]
    if "architecture" in steps:
        architecture(cfg)
        steps.remove("architecture")
    if not steps:
        return
    from cosmic import classify, cluster, features, ingest, observer_score, preprocess
    from cosmic.spark_session import get_spark
    mods = dict(ingest=ingest, preprocess=preprocess, score=observer_score, features=features,
                classify=classify, cluster=cluster)
    spark = get_spark(cfg)
    print(f"Spark {spark.version} | master={spark.sparkContext.master} | app={spark.sparkContext.applicationId}")
    timings = []
    try:
        for s in steps:
            print(f"\n{'=' * 20} STEP: {s} {'=' * 20}")
            t0 = time.time()
            mods[s].run(spark, cfg)
            timings.append((s, round(time.time() - t0, 1)))
            print(f"--- {s} finished in {timings[-1][1]} s")
    finally:
        spark.stop()
    print("\nStep timings (s):", timings)


if __name__ == "__main__":
    main()
