"""Step 7 — The four ML models (Spark MLlib) on the per-window summary features.

What it does, for each of the 5 star-grouped splits (Hive table `splits`, the same splits
RGA-Net uses) and on the same windows (Hive table `window_features`):
  * Models: Gaussian Naive Bayes (modelType="gaussian"), Decision Tree, Random Forest and
    Logistic Regression.
  * Class weights through weightCol: w = n_train / (2 · n_class_train), counted on the
    TRAINING windows, so the smaller class is not ignored.
  * Median imputation and standardisation (StandardScaler, mean and std) are fitted on the
    training windows only, then applied to validation and test.
  * Small hyperparameter grids (settings.yaml → classify) are tuned on the VALIDATION windows:
    the setting with the best validation PR-AUC is kept. Gaussian Naive Bayes has no
    hyperparameters to tune. The test windows are never used to choose anything.
  * Run twice: with the observer features (the main comparison) and without them
    (supplementary table: does the observer information help the ML models?).
  * Saves every model's predicted probability for the validation and test windows to
    work/predictions_ml.csv. The threshold choice and all metrics are then computed by
    the comparison step with the same code as for RGA-Net (cosmic/evaluation.py).
Outputs: predictions_ml.csv, ml_selected_params.csv (chosen setting and training time per
model, seed and feature set), Hive table `ml_predictions`.

Why: these are the existing machine-learning baselines RGA-Net is compared against, trained
on exactly the same samples and splits.
"""
import itertools
import os
import time

import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import (DecisionTreeClassifier, LogisticRegression, NaiveBayes,
                                       RandomForestClassifier)
from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from .evaluation import average_precision
from .spark_session import save_csv, save_hive_table
from .windows import BASE_FEATURES, OBSERVER_FEATURES

MODELS = ["NB", "DT", "RF", "LR"]
FEATURE_SETS = {"with_observer": BASE_FEATURES + OBSERVER_FEATURES, "without_observer": BASE_FEATURES}


def grid(name, c):
    """Candidate settings for one model (list of dicts)."""
    if name == "NB":
        return [{}]
    if name == "DT":
        return [dict(maxDepth=int(d), minInstancesPerNode=int(m))
                for d, m in itertools.product(c["dt_max_depth"], c["dt_min_instances"])]
    if name == "RF":
        return [dict(numTrees=int(n), maxDepth=int(d)) for n, d in itertools.product(c["rf_num_trees"], c["rf_max_depth"])]
    return [dict(regParam=float(r), elasticNetParam=float(e))
            for r, e in itertools.product(c["lr_reg_param"], c["lr_elastic_net"])]


def make(name, params, seed, c):
    base = dict(labelCol="label", featuresCol="features", weightCol="w")
    if name == "NB":
        return NaiveBayes(modelType="gaussian", **base)
    if name == "DT":
        return DecisionTreeClassifier(seed=seed, **base, **params)
    if name == "RF":
        return RandomForestClassifier(seed=seed, **base, **params)
    return LogisticRegression(maxIter=int(c["lr_max_iter"]), **base, **params)


def class_weights(train):
    counts = {int(r["label"]): r["count"] for r in train.groupBy("label").count().collect()}
    n = sum(counts.values())
    return {k: n / (2.0 * v) for k, v in counts.items()}


def probs(model, df):
    out = model.transform(df).select("window_id", "auid", "label", vector_to_array("probability")[1].alias("prob"))
    return out.toPandas()


def run(spark, cfg):
    seeds = [int(s) for s in cfg.section("split")["seeds"]]
    pred, chosen = fit_predict(spark.table(cfg.table("window_features")), spark.table(cfg.table("splits")),
                               cfg.section("classify"), seeds)
    pred.to_csv(os.path.join(cfg.work_dir, "predictions_ml.csv"), index=False)
    save_csv(pred, cfg, "predictions_ml.csv")
    save_hive_table(spark.createDataFrame(pred), cfg, "ml_predictions")
    save_csv(chosen, cfg, "ml_selected_params.csv")
    return pred


def fit_predict(window_features, splits, c, seeds):
    """Fit, tune on validation and predict, for every seed, feature set and model. Returns (predictions, chosen)."""
    feats = window_features.where("label IS NOT NULL").withColumn("label", F.col("label").cast("double"))
    preds, chosen = [], []
    for seed in seeds:
        d = feats.join(splits.where(F.col("seed") == seed).select("auid", "set"), "auid")
        w = class_weights(d.where("set = 'train'"))
        d = d.withColumn("w", F.when(F.col("label") == 1.0, w.get(1, 1.0)).otherwise(w.get(0, 1.0))).cache()
        for fs_name, cols in FEATURE_SETS.items():
            imp = [f"{x}_imp" for x in cols]
            prep = Pipeline(stages=[Imputer(inputCols=cols, outputCols=imp, strategy="median"),
                                    VectorAssembler(inputCols=imp, outputCol="raw_features"),
                                    StandardScaler(inputCol="raw_features", outputCol="features",
                                                   withMean=True, withStd=True)]).fit(d.where("set = 'train'"))
            X = prep.transform(d).select("window_id", "auid", "label", "w", "set", "features").cache()
            tr, va, te = (X.where(f"set = '{s}'") for s in ("train", "val", "test"))
            for name in MODELS:
                best, tune_t = None, 0.0
                for params in grid(name, c):
                    t0 = time.time()
                    m = make(name, params, seed, c).fit(tr)
                    fit_t = time.time() - t0
                    tune_t += fit_t
                    pv = probs(m, va)
                    ap = average_precision(pv["label"], pv["prob"])
                    if best is None or ap > best[0]:
                        best = (ap, params, m, fit_t, pv)
                ap, params, m, fit_t, pv = best
                pt = probs(m, te)
                for s, p in (("val", pv), ("test", pt)):
                    preds.append(p.assign(model=name, variant=fs_name, seed=seed, set=s))
                chosen.append(dict(model=name, variant=fs_name, seed=seed, params=str(params) if params else "(none)",
                                   val_pr_auc=ap, fit_seconds=round(fit_t, 2), tuning_seconds=round(tune_t, 2),
                                   n_train=tr.count(), weight_anomalous=w.get(1), weight_normal=w.get(0)))
                print(f"seed {seed} {fs_name:16s} {name}: {chosen[-1]['params']}  val PR-AUC {ap:.3f}  "
                      f"fit {fit_t:.1f} s")
            X.unpersist()
        d.unpersist()
    pred = pd.concat(preds, ignore_index=True)[["model", "variant", "seed", "set", "window_id", "auid", "label", "prob"]]
    pred["label"] = pred["label"].astype(int)
    return pred, pd.DataFrame(chosen)
