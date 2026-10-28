"""Step 7 — The four ML models (Spark MLlib) on the per-window summary features.

What it does, for each of the 10 star-grouped splits (Hive table `splits`, the same splits RGA-Net uses):
  * Models: Gaussian Naive Bayes, Decision Tree, Random Forest and Logistic Regression.
  * Two feature sets: WITHOUT observer features (the 10 light-curve features only = the PRIOR APPROACHES the
    proposed model is compared with) and WITH the 3 observer features (w_a, n_observers, mean_abs_offset =
    "our reliability framework applied to classic models"; these are never called prior approaches).
  * Two data settings, chosen per model on VALIDATION: `aavso` (AAVSO training windows only) and `aavso_ztf`
    (AAVSO training windows + a class-balanced sample of labelled ZTF windows, classify.ml_ztf_windows, added to
    the TRAINING set only; ZTF windows never enter validation or test, and no ZTF object near an AAVSO star is
    used).
  * Class weights w = n_train / (2 * n_class_train); median imputation and standardisation fitted on the
    training windows only.
  * Small hyperparameter grids (settings.yaml -> classify): the (data setting, hyperparameters) with the best
    validation AUPRC is kept. The test windows are never used to choose anything.
  * Saves the validation and test probabilities to work/predictions_ml.csv; thresholds and metrics are computed
    later by the comparison step with the same code as for RGA-Net (cosmic/evaluation.py).
Outputs: predictions_ml.csv, ml_selected_params.csv, Hive table `ml_predictions`.
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


def ztf_train_windows(spark, cfg, n):
    """Class-balanced sample (n in total) of labelled ZTF windows for the aavso_ztf data setting, or None."""
    if n <= 0 or not spark.catalog.tableExists("ztf_window_features", cfg.db):
        return None
    seed = int(cfg.section("ztf")["sample_seed"])
    z = spark.sql(f"""SELECT * FROM (SELECT f.*, ROW_NUMBER() OVER (PARTITION BY label
                                                                    ORDER BY xxhash64(window_id, {seed})) AS rk
                                     FROM {cfg.table('ztf_window_features')} f WHERE label IS NOT NULL)
                      WHERE rk <= {n // 2}""").drop("rk")
    return z


def run(spark, cfg):
    seeds = [int(s) for s in cfg.section("split")["seeds"]]
    c = cfg.section("classify")
    ztf = ztf_train_windows(spark, cfg, int(cfg.section("ztf").get("ml_ztf_windows", 0)))
    if ztf is None and "aavso_ztf" in c["data_settings"]:
        print("no ZTF windows (run ztf_windows first): only the AAVSO-only data setting is used")
    pred, chosen = fit_predict(spark.table(cfg.table("window_features")), spark.table(cfg.table("splits")), c, seeds,
                               ztf)
    pred.to_csv(os.path.join(cfg.work_dir, "predictions_ml.csv"), index=False)
    save_hive_table(spark.createDataFrame(pred), cfg, "ml_predictions")
    save_csv(chosen, cfg, "ml_selected_params.csv")
    from . import dataflow
    from .spark_session import table_bytes
    dataflow.record(cfg, "ml_models", "Spark MLlib trains NB, DT, RF and LR on every split (two feature sets, two data "
                    "settings, small grids chosen on validation); validation and test probabilities are saved.",
                    f"Hive {cfg.table('ml_predictions')}; {os.path.join(cfg.work_dir, 'predictions_ml.csv')}",
                    "Parquet (Hive) + CSV", "Spark MLlib on YARN", size_bytes=table_bytes(spark, cfg.table("ml_predictions")),
                    rows=len(pred), changes=f"{len(chosen)} fitted model selections over {len(seeds)} seeds",
                    seconds=float(chosen["tuning_seconds"].sum()))
    print("\nchosen data setting per model and feature set (count over seeds):\n"
          + chosen.groupby(["variant", "model", "data_setting"]).size().to_string())
    return pred


def fit_predict(window_features, splits, c, seeds, ztf=None):
    """Fit, tune on validation and predict, for every seed, feature set and model. Returns (predictions, chosen)."""
    cols = BASE_FEATURES + OBSERVER_FEATURES
    feats = window_features.where("label IS NOT NULL").select("window_id", "auid", "label", *cols)         .withColumn("label", F.col("label").cast("double"))
    settings = [s for s in c.get("data_settings", ["aavso"]) if s == "aavso" or ztf is not None]
    if ztf is not None:
        ztf = ztf.select("window_id", "auid", "label", *cols).withColumn("label", F.col("label").cast("double"))             .withColumn("set", F.lit("train")).withColumn("src", F.lit("ztf"))
    preds, chosen = [], []
    for seed in seeds:
        aav = feats.join(splits.where(F.col("seed") == seed).select("auid", "set"), "auid").withColumn("src", F.lit("aavso"))
        both = aav.unionByName(ztf) if ztf is not None else aav
        for fs_name, fcols in FEATURE_SETS.items():
            cands = []  # (val AUPRC, setting, params, model, fit seconds, val preds, X)
            for setting in settings:
                d = both if setting == "aavso_ztf" else aav
                w = class_weights(d.where("set = 'train'"))
                d = d.withColumn("w", F.when(F.col("label") == 1.0, w.get(1, 1.0)).otherwise(w.get(0, 1.0)))
                imp = [f"{x}_imp" for x in fcols]
                prep = Pipeline(stages=[Imputer(inputCols=fcols, outputCols=imp, strategy="median"),
                                        VectorAssembler(inputCols=imp, outputCol="raw_features"),
                                        StandardScaler(inputCol="raw_features", outputCol="features",
                                                       withMean=True, withStd=True)]).fit(d.where("set = 'train'"))
                X = prep.transform(d).select("window_id", "auid", "label", "w", "set", "src", "features").cache()
                cands.append((setting, X, w))
            for name in MODELS:
                best, tune_t = None, 0.0
                for setting, X, w in cands:
                    tr, va = X.where("set = 'train'"), X.where("set = 'val'")
                    for params in grid(name, c):
                        t0 = time.time()
                        m = make(name, params, seed, c).fit(tr)
                        fit_t = time.time() - t0
                        tune_t += fit_t
                        pv = probs(m, va)
                        ap = average_precision(pv["label"], pv["prob"])
                        if best is None or ap > best[0]:
                            best = (ap, setting, params, m, fit_t, pv, X, w)
                ap, setting, params, m, fit_t, pv, X, w = best
                pt = probs(m, X.where("set = 'test'"))
                for s, p in (("val", pv), ("test", pt)):
                    preds.append(p.assign(model=name, variant=fs_name, seed=seed, set=s, data_setting=setting))
                tr = X.where("set = 'train'")
                chosen.append(dict(model=name, variant=fs_name, seed=seed, data_setting=setting,
                                   params=str(params) if params else "(none)", val_pr_auc=ap,
                                   fit_seconds=round(fit_t, 2), tuning_seconds=round(tune_t, 2),
                                   n_train=tr.count(), n_train_ztf=tr.where("src = 'ztf'").count(),
                                   weight_anomalous=w.get(1), weight_normal=w.get(0)))
                print(f"seed {seed} {fs_name:16s} {name}: {setting:9s} {chosen[-1]['params']}  val AUPRC {ap:.3f}  "
                      f"fit {fit_t:.1f} s", flush=True)
            for _, X, _ in cands:
                X.unpersist()
    pred = pd.concat(preds, ignore_index=True)[["model", "variant", "seed", "set", "window_id", "auid", "label", "prob",
                                                "data_setting"]]
    pred["label"] = pred["label"].astype(int)
    return pred, pd.DataFrame(chosen)
