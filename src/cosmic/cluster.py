"""Step 6 — K-Means clustering. Spark MLlib.

What it does: clusters ALL stars — including those whose VSX type is not in the label
map — on the standardised BASELINE features (median-imputed, then scaled; both fitted
on all stars because clustering uses no labels). It tries k = 2..8, scores each with the
silhouette coefficient (Spark ClusteringEvaluator), keeps the best k, and cross-tabulates
the clusters against the labels (anomalous / normal / unlabelled) and the VSX types.
Outputs: `kmeans_silhouette.csv`, `kmeans_crosstab_label.csv`, `kmeans_crosstab_type.csv`,
the silhouette plot, and the Hive table `star_clusters`.

Why: an unsupervised check of whether the light-curve features separate the star
families on their own, without being told the answer.
"""
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler

from .features import BASELINE
from .spark_session import save_csv, save_hive_table


def run(spark, cfg):
    k_cfg = cfg.section("kmeans")
    stars = spark.table(cfg.table("star_features"))
    n = stars.count()
    imp = [f"{c}_imp" for c in BASELINE]
    prep = Pipeline(stages=[Imputer(inputCols=BASELINE, outputCols=imp, strategy="median"),
                            VectorAssembler(inputCols=imp, outputCol="raw_features"),
                            StandardScaler(inputCol="raw_features", outputCol="features",
                                           withMean=True, withStd=True)]).fit(stars)
    X = prep.transform(stars).cache()
    ev = ClusteringEvaluator(featuresCol="features", predictionCol="cluster", metricName="silhouette")
    rows, models = [], {}
    for k in range(int(k_cfg["k_min"]), min(int(k_cfg["k_max"]), n - 1) + 1):
        m = KMeans(k=k, seed=int(k_cfg["seed"]), featuresCol="features", predictionCol="cluster").fit(X)
        rows.append(dict(k=k, silhouette=ev.evaluate(m.transform(X))))
        models[k] = m
    sil = pd.DataFrame(rows)
    best_k = int(sil.loc[sil["silhouette"].idxmax(), "k"])
    sil["chosen"] = sil["k"] == best_k
    print(sil.to_string(index=False))
    save_csv(sil, cfg, "kmeans_silhouette.csv")

    out = models[best_k].transform(X).selectExpr(
        "auid", "name", "vsx_type", "primary_type", "label", "cluster",
        "CASE label WHEN 1 THEN 'anomalous' WHEN 0 THEN 'normal' ELSE 'unlabelled' END AS label_name")
    save_hive_table(out, cfg, "star_clusters")
    out = spark.table(cfg.table("star_clusters"))
    ct = out.crosstab("cluster", "label_name").toPandas().sort_values("cluster_label_name")
    ct = ct.rename(columns={"cluster_label_name": "cluster"})
    print(ct.to_string(index=False))
    save_csv(ct, cfg, "kmeans_crosstab_label.csv")
    ct2 = out.fillna({"primary_type": "NONE"}).crosstab("cluster", "primary_type").toPandas() \
        .sort_values("cluster_primary_type").rename(columns={"cluster_primary_type": "cluster"})
    save_csv(ct2, cfg, "kmeans_crosstab_type.csv")
    save_csv(out.toPandas(), cfg, "star_clusters.csv")
    return sil, ct
