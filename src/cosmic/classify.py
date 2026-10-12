"""Step 5 — Classification. Spark MLlib only.

What it does:
  * Uses only stars with a label (anomalous = 1, normal = 0).
  * Grouped split BY STAR (auid): 70 % train / 30 % test, stratified by label. The order
    within each class is a hash of (auid, seed), so the split is reproducible, and the
    code asserts that no auid is in both sets. (There is one feature row per star, so a
    row split here is a star split — the assertion makes this explicit.)
  * Class weights via weightCol: w = n_train / (2 · n_class_train), so the smaller class
    is not ignored.
  * Every model is a Pipeline: median Imputer → VectorAssembler → StandardScaler
    (withMean, withStd) → classifier, FITTED ON THE TRAINING STARS ONLY.
  * Existing observer-agnostic classifiers on BASELINE features: Gaussian Naive Bayes,
    Decision Tree, Logistic Regression, Random Forest.
  * Proposed model: Decision Tree on BASELINE + OBSERVER features. Naive Bayes is also
    trained with observer features, so both model families are compared symmetrically.
  * Decision Tree depth from the fixed grid maxDepth ∈ {3, 5}, chosen by cross-validation
    inside the training stars only.
  * Metrics on the test stars: precision, recall and F1 for the anomalous class, balanced
    accuracy ((TPR + TNR) / 2) and the confusion matrix. Plain accuracy is not headlined
    because the classes can be unbalanced.
Outputs: `model_metrics.csv`, Hive tables `model_metrics` and `test_predictions`, the
proposed model's confusion matrix, DT-baseline vs proposed side-by-side, and a grouped
bar chart of all models.

Why: the research claim is tested by comparing the proposed observer-aware model with
existing observer-agnostic classifiers under identical conditions (same split, same
preprocessing, same metrics).
"""
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import (DecisionTreeClassifier, LogisticRegression, NaiveBayes,
                                       RandomForestClassifier)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as F

from . import plots
from .features import BASELINE, OBSERVER
from .spark_session import save_csv, save_hive_table


def grouped_split(df, train_fraction, seed):
    df.createOrReplaceTempView("cls_labelled")
    s = df.sparkSession.sql(f"""
        SELECT *, rk <= GREATEST(1, ROUND({train_fraction} * n_class)) AS is_train FROM (
          SELECT *, ROW_NUMBER() OVER (PARTITION BY label ORDER BY xxhash64(auid, {int(seed)})) AS rk,
                    COUNT(*) OVER (PARTITION BY label) AS n_class
          FROM cls_labelled)""")
    train, test = s.where("is_train"), s.where("NOT is_train")
    overlap = train.select("auid").intersect(test.select("auid")).count()
    assert overlap == 0, f"{overlap} stars are in both train and test"
    counts = train.groupBy("label").count().collect()
    n_train = sum(r["count"] for r in counts)
    weights = {r["label"]: n_train / (2.0 * r["count"]) for r in counts}
    wexpr = F.when(F.col("label") == 1, weights.get(1, 1.0)).otherwise(weights.get(0, 1.0))
    return train.withColumn("w", wexpr), test.withColumn("w", wexpr), weights


def _pipeline(features, clf):
    imp = [f"{c}_imp" for c in features]
    return Pipeline(stages=[
        Imputer(inputCols=features, outputCols=imp, strategy="median"),
        VectorAssembler(inputCols=imp, outputCol="raw_features"),
        StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True),
        clf])


def make_models(c):
    seed = int(c["seed"])
    base = dict(labelCol="label", featuresCol="features", weightCol="w")
    nb = lambda: NaiveBayes(modelType="gaussian", **base)
    dt = lambda: DecisionTreeClassifier(seed=seed, **base)
    return [
        ("NB (baseline)", BASELINE, nb(), False, "existing"),
        ("DT (baseline)", BASELINE, dt(), True, "existing"),
        ("LR (baseline)", BASELINE, LogisticRegression(regParam=float(c["lr_reg_param"]), maxIter=100, **base), False, "existing"),
        ("RF (baseline)", BASELINE, RandomForestClassifier(numTrees=int(c["rf_num_trees"]), seed=seed, **base), False, "existing"),
        ("NB + observer", BASELINE + OBSERVER, nb(), False, "observer-aware"),
        ("DT + observer (proposed)", BASELINE + OBSERVER, dt(), True, "proposed"),
    ]


def metrics(pred):
    r = pred.select(
        F.sum(((F.col("label") == 1) & (F.col("prediction") == 1)).cast("int")).alias("tp"),
        F.sum(((F.col("label") == 0) & (F.col("prediction") == 1)).cast("int")).alias("fp"),
        F.sum(((F.col("label") == 1) & (F.col("prediction") == 0)).cast("int")).alias("fn"),
        F.sum(((F.col("label") == 0) & (F.col("prediction") == 0)).cast("int")).alias("tn")).first()
    tp, fp, fn, tn = (int(r[k] or 0) for k in ("tp", "fp", "fn", "tn"))
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp and (prec + rec) else 0.0
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    return dict(precision=prec, recall=rec, f1=f1, balanced_accuracy=(rec + tnr) / 2,
                tp=tp, fp=fp, fn=fn, tn=tn)


def fit_and_score(train, test, models, c):
    rows, preds, fitted = [], {}, {}
    for name, feats, clf, tune, family in models:
        pipe = _pipeline(feats, clf)
        chosen = ""
        if tune:
            grid = ParamGridBuilder().addGrid(clf.maxDepth, [int(d) for d in c["dt_max_depths"]]).build()
            ev = MulticlassClassificationEvaluator(labelCol="label", metricName="fMeasureByLabel", metricLabel=1.0)
            cv = CrossValidator(estimator=pipe, estimatorParamMaps=grid, evaluator=ev,
                                numFolds=int(c["cv_folds"]), seed=int(c["seed"]))
            cvm = cv.fit(train)
            model = cvm.bestModel
            chosen = f"maxDepth={model.stages[-1].getOrDefault('maxDepth')}"
        else:
            model = pipe.fit(train)
        p = model.transform(test)
        m = metrics(p)
        rows.append(dict(model=name, family=family, features="baseline + observer" if feats != BASELINE else "baseline",
                         n_features=len(feats), params=chosen, **m))
        preds[name] = p.select("auid", "name", "vsx_type", "label", "prediction").withColumn("model", F.lit(name))
        fitted[name] = model
    return pd.DataFrame(rows), preds, fitted


def run(spark, cfg, seed=None):
    c = dict(cfg.section("classify"))
    if seed is not None:
        c["seed"] = seed
    labelled = spark.table(cfg.table("star_features")).where("label IS NOT NULL")
    train, test, weights = grouped_split(labelled, float(c["train_fraction"]), int(c["seed"]))
    train.cache(); test.cache()
    split = pd.DataFrame([dict(seed=int(c["seed"]), train_stars=train.count(), test_stars=test.count(),
                               train_anomalous=train.where("label = 1").count(),
                               test_anomalous=test.where("label = 1").count(),
                               weight_anomalous=weights.get(1), weight_normal=weights.get(0),
                               auid_overlap=0)])
    print(split.to_string(index=False))
    save_csv(split, cfg, "split_summary.csv")
    save_csv(pd.concat([train.select("auid", "name", "label").toPandas().assign(set="train"),
                        test.select("auid", "name", "label").toPandas().assign(set="test")]),
             cfg, "split_stars.csv")

    res, preds, _ = fit_and_score(train, test, make_models(c), c)
    res.insert(0, "seed", int(c["seed"]))
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))
    save_csv(res, cfg, "model_metrics.csv")
    save_hive_table(spark.createDataFrame(res), cfg, "model_metrics")
    allp = None
    for p in preds.values():
        allp = p if allp is None else allp.unionByName(p)
    save_hive_table(allp, cfg, "test_predictions")

    prop = res[res["family"] == "proposed"].iloc[0]
    save_csv(res[res["family"] == "proposed"], cfg, "proposed_model_metrics.csv")
    cm = lambda r: [[r["tn"], r["fp"]], [r["fn"], r["tp"]]]
    dtb = res[res["model"] == "DT (baseline)"].iloc[0]
    plots.confusion(cm(prop), f"Proposed: DT + observer features\n(test stars, seed {c['seed']})",
                    cfg.output_dir, "fig_confusion_proposed.png")
    plots.confusion_side_by_side(cm(dtb), "DT, baseline features", cm(prop), "DT + observer features (proposed)",
                                 cfg.output_dir, "fig_confusion_dt_vs_proposed.png")
    plots.grouped_metrics(res, cfg.output_dir)
    return res
