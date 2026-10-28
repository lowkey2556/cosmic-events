"""Step 8b — Distributed inference: the trained RGA-Net applied to ALL windows, in parallel, by Spark.

What it does:
  1. Loads the trained RGA-Net models that Step 8 saved in work/models/ (by default the full
     model of every split seed; settings.yaml → infer.variants) and broadcasts their weights
     to the Spark executors.
  2. Reads every window's token sequence from the Hive table `window_sequences` and runs the
     models inside Spark with `mapInPandas` (a pandas UDF over batches of windows): each
     executor rebuilds the models once on its CPU and scores its share of the windows. The
     inputs are prepared with exactly the same function used in training (rganet.make_inputs).
  3. Writes one row per (window, model) — window_id, auid, label, seed, variant, prob, the
     window's set in that seed's split (train/val/test, or "none" for unlabelled stars) — to
     the Hive table `window_predictions` (Parquet on HDFS).
  4. Consistency check: for the validation and test windows, the Spark probabilities are
     compared with the probabilities PyTorch produced right after training
     (predictions_rganet.csv); the largest difference is reported in
     distributed_inference_check.csv. They should agree to floating-point precision.
Outputs: Hive `window_predictions`, window_predictions_summary.csv, distributed_inference_check.csv.

Why: training needs a GPU and runs once, but applying the model is embarrassingly parallel;
Spark scores every window of every star (including unlabelled stars and windows outside the
test set) across the cluster, and the result sits in Hive next to the other tables.
"""
import glob
import os
import re
import time

import pandas as pd

from .spark_session import save_csv, save_hive_table

OUT_SCHEMA = "window_id STRING, seed INT, variant STRING, prob DOUBLE"


def load_blobs(model_dir, variants):
    """{file name: bytes} of the saved models to use."""
    out = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "rganet_*_seed*.pt"))):
        m = re.match(r"rganet_(.+)_seed(\d+)\.pt$", os.path.basename(path))
        if m and m.group(1) in variants:
            with open(path, "rb") as fh:
                out[os.path.basename(path)] = fh.read()
    return out


def scorer(blobs_bc):
    """mapInPandas function: rebuild the models once per task, then score every batch of windows."""

    def score(batches):
        import torch

        from .rganet import load_model, predict_frame
        torch.set_num_threads(1)  # one core per executor slot; Spark provides the parallelism
        models = [load_model(b) for _, b in sorted(blobs_bc.value.items())]
        for pdf in batches:
            yield predict_frame(pdf, models)

    return score


def run(spark, cfg):
    variants = list(cfg.section("infer")["variants"])
    blobs = load_blobs(os.path.join(cfg.work_dir, "models"), variants)
    if not blobs:
        raise FileNotFoundError(f"No saved RGA-Net models for {variants} in {cfg.work_dir}/models: run the rganet step first")
    print(f"Broadcasting {len(blobs)} trained models to the executors: {sorted(blobs)}")
    bc = spark.sparkContext.broadcast(blobs)
    t0 = time.time()
    from .windows import SUMMARY  # the hybrid head also needs each window's summary features
    # window_sequences.n_obs (an INT count) would clash with the summary feature n_obs (a DOUBLE): keep the feature
    seq = spark.table(cfg.table("window_sequences")).drop("n_obs").join(
        spark.table(cfg.table("window_features")).select("window_id", *SUMMARY), "window_id")
    probs = seq.mapInPandas(scorer(bc), schema=OUT_SCHEMA)
    idx = spark.table(cfg.table("window_features")).select("window_id", "auid", "label")
    splits = spark.table(cfg.table("splits")).select("seed", "auid", "set")
    out = probs.join(idx, "window_id").join(splits, ["seed", "auid"], "left") \
        .selectExpr("window_id", "auid", "CAST(label AS INT) AS label", "seed", "variant", "prob",
                    "COALESCE(`set`, 'none') AS `set`")
    save_hive_table(out, cfg, "window_predictions")
    secs = time.time() - t0
    pred = spark.table(cfg.table("window_predictions"))
    summary = pred.groupBy("variant", "seed", "set").agg({"window_id": "count", "prob": "avg"}) \
        .withColumnRenamed("count(window_id)", "windows").withColumnRenamed("avg(prob)", "mean_prob") \
        .orderBy("variant", "seed", "set").toPandas()
    n = pred.count()
    print(f"Scored {n} (window, model) pairs on {spark.sparkContext.defaultParallelism} parallel task slots "
          f"in {secs:.1f} s -> Hive table {cfg.table('window_predictions')}")
    print(summary.to_string(index=False))
    save_csv(summary, cfg, "window_predictions_summary.csv")
    from . import dataflow
    from .spark_session import table_bytes
    dataflow.record(cfg, "predictions_hive", "Spark broadcasts the trained models and scores every window with "
                    "mapInPandas; the probabilities are written back to Hive.", f"Hive {cfg.table('window_predictions')}",
                    "Parquet (Hive table)", "Spark mapInPandas + PyTorch (CPU)",
                    size_bytes=table_bytes(spark, cfg.table("window_predictions")), rows=n, seconds=secs)

    check_path = os.path.join(cfg.work_dir, "predictions_rganet.csv")
    if os.path.exists(check_path):
        ref = pd.read_csv(check_path)
        ref = ref[ref["variant"].isin(variants)][["variant", "seed", "window_id", "prob"]]
        sp = pred.where("`set` IN ('val', 'test')").select("variant", "seed", "window_id", "prob").toPandas()
        m = ref.merge(sp, on=["variant", "seed", "window_id"], suffixes=("_pytorch", "_spark"))
        diff = (m["prob_pytorch"] - m["prob_spark"]).abs()
        check = pd.DataFrame([dict(windows_compared=len(m), max_abs_difference=float(diff.max()) if len(m) else None,
                                   mean_abs_difference=float(diff.mean()) if len(m) else None,
                                   seconds=round(secs, 1), pairs_scored=n)])
        print("\nSpark vs PyTorch probabilities on the validation/test windows:\n" + check.to_string(index=False))
        save_csv(check, cfg, "distributed_inference_check.csv")
    return summary
