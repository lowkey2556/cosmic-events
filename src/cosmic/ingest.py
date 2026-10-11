"""Step 1 — Ingestion & storage (owner: Idhazhya).

What it does:
  1. `hdfs dfs -put` the raw AAVSO CSV files from Drive into HDFS /<base>/raw/.
  2. Builds the star labels table (VSX types, coordinates, sky_region) — see labels.py.
  3. Reads the CSVs in Spark with an explicit schema (every column named; text read
     first and then cast, so blanks and "NA" become nulls instead of crashing).
  4. Adds obs_year (from the Julian Date) and sky_region (declination band from VSX).
  5. Writes Parquet + Snappy to HDFS, partitioned by obs_year and sky_region, and
     registers the Hive EXTERNAL table `observations_raw` on that location.
  6. Writes an unpartitioned copy (`observations_flat`) and runs the same
     year-restricted query on both, recording files and bytes actually scanned
     (partition pruning evidence).
  7. Produces the Review 2 "Dataset" slide tables and the observations-per-star chart.

Why: HDFS gives distributed, fault-tolerant storage; Parquet is columnar and compressed,
so later steps read only the columns they need; Hive gives every team member (and the
Hive CLI) the same named tables; partitioning lets queries skip whole folders.
"""
import glob
import os

import pandas as pd
from pyspark.sql import types as T

from . import plots
from .labels import build_labels
from .spark_session import hdfs, save_csv

RAW_COLUMNS = ["#", "target", "auid", "jd", "mag", "uncertainty", "fainterthan", "band", "type",
               "observer", "airmass", "transformed", "comp1_c", "cmag", "comp2_k", "kmag",
               "chart", "commentcode", "comments"]
# Read every column as text with an explicit schema, then cast in Spark SQL.
RAW_SCHEMA = T.StructType([T.StructField(c, T.StringType(), True) for c in RAW_COLUMNS])

TYPED_SELECT = """
SELECT CAST(`#` AS BIGINT)            AS row_num,
       TRIM(target)                    AS target,
       UPPER(TRIM(auid))               AS auid,
       CAST(jd AS DOUBLE)              AS jd,
       CAST(mag AS DOUBLE)             AS mag,
       CAST(uncertainty AS DOUBLE)     AS uncertainty,
       UPPER(TRIM(COALESCE(fainterthan, ''))) IN ('TRUE','T','1','YES','Y','<') AS fainterthan,
       TRIM(band)                      AS band,
       TRIM(type)                      AS obs_type,
       UPPER(TRIM(observer))           AS observer,
       CAST(airmass AS DOUBLE)         AS airmass,
       TRIM(transformed)               AS transformed,
       NULLIF(TRIM(comp1_c), 'NA')     AS comp1_c,
       CAST(cmag AS DOUBLE)            AS cmag,
       NULLIF(TRIM(comp2_k), 'NA')     AS comp2_k,
       CAST(kmag AS DOUBLE)            AS kmag,
       TRIM(chart)                     AS chart,
       UPPER(TRIM(commentcode))        AS commentcode,
       comments,
       regexp_extract(input_file_name(), '[^/]+$', 0) AS source_file
FROM raw_csv
"""

OBS_DDL_COLUMNS = """row_num BIGINT, target STRING, auid STRING, jd DOUBLE, mag DOUBLE, uncertainty DOUBLE,
  fainterthan BOOLEAN, band STRING, obs_type STRING, observer STRING, airmass DOUBLE, transformed STRING,
  comp1_c STRING, cmag DOUBLE, comp2_k STRING, kmag DOUBLE, chart STRING, commentcode STRING,
  comments STRING, source_file STRING"""


def put_raw_files(cfg):
    files = sorted(glob.glob(os.path.join(cfg.raw_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files in {cfg.raw_dir}")
    dest = f"{cfg.hdfs_base}/raw"
    hdfs("-mkdir", "-p", dest)
    hdfs("-put", "-f", *files, dest + "/")
    hdfs("-ls", "-h", dest)
    return cfg.hdfs_uri(dest + "/*.csv")


def check_header(spark, raw_glob):
    found = [c.strip() for c in spark.read.option("header", True).csv(raw_glob).columns]
    if [c.lower() for c in found] != RAW_COLUMNS:
        raise ValueError("Raw CSV header does not match the expected AAVSO export.\n"
                         f"expected: {RAW_COLUMNS}\nfound:    {found}")


def scan_metrics(df):
    """Run df and return files/bytes read by its file scans (from Spark's SQL metrics)."""
    df.collect()
    found = []

    def walk(node):
        if "Scan" in node.nodeName():
            m = node.metrics()
            vals = {}
            for key in ("numFiles", "filesSize", "numPartitions"):
                opt = m.get(key)
                if opt.isDefined():
                    vals[key] = int(opt.get().value())
            found.append(vals)
        ch = node.children()
        for i in range(ch.size()):
            walk(ch.apply(i))

    walk(df._jdf.queryExecution().executedPlan())
    return dict(files_read=sum(v.get("numFiles", 0) for v in found),
                bytes_read=sum(v.get("filesSize", 0) for v in found),
                partitions_read=sum(v.get("numPartitions", 0) for v in found))


def partition_pruning_test(spark, cfg):
    year = cfg.section("ingest")["pruning_test_year"]
    if year is None:
        year = spark.sql(f"SELECT obs_year FROM {cfg.table('observations_raw')} GROUP BY obs_year "
                         "ORDER BY COUNT(*) DESC LIMIT 1").first()[0]
    rows = []
    spark.conf.set("spark.sql.adaptive.enabled", "false")  # keep the physical plan simple to read metrics
    for table in ("observations_raw", "observations_flat"):
        q = (f"SELECT COUNT(*) AS n, AVG(mag) AS mean_mag FROM {cfg.table(table)} "
             f"WHERE obs_year = {int(year)}")
        m = scan_metrics(spark.sql(q))
        n = spark.sql(q).first()["n"]
        total_files = spark.read.parquet(cfg.hdfs_uri(f"{cfg.hdfs_base}/warehouse/{table}")).inputFiles()
        rows.append(dict(table=table, partitioned=table == "observations_raw", query_year=int(year), rows_matched=n,
                         files_read=m["files_read"], bytes_read=m["bytes_read"], total_files=len(total_files)))
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    pdf = pd.DataFrame(rows)
    pdf["bytes_vs_flat"] = pdf["bytes_read"] / pdf.loc[pdf["table"] == "observations_flat", "bytes_read"].iloc[0]
    print(pdf.to_string(index=False))
    save_csv(pdf, cfg, "partition_pruning.csv")
    plots.pruning(pdf.assign(table=pdf["table"].map({"observations_raw": "partitioned",
                                                      "observations_flat": "unpartitioned"})), cfg.output_dir)
    return pdf


def dataset_summary(spark, cfg):
    t = cfg.table("observations_raw")
    s = spark.sql(f"""SELECT COUNT(*) AS total_rows, COUNT(DISTINCT auid) AS stars,
                             COUNT(DISTINCT observer) AS observers, MIN(jd) AS jd_min, MAX(jd) AS jd_max,
                             SUM(CASE WHEN fainterthan THEN 1 ELSE 0 END) AS fainter_than_rows
                      FROM {t}""").toPandas()
    to_date = lambda jd: str(pd.Timestamp("1970-01-01") + pd.to_timedelta(jd - 2440587.5, unit="D"))[:10]
    s["date_min"] = s["jd_min"].map(to_date)
    s["date_max"] = s["jd_max"].map(to_date)
    labels = spark.sql(f"""SELECT CASE label WHEN 1 THEN 'anomalous' WHEN 0 THEN 'normal' ELSE 'unlabelled' END
                                  AS label_name, COUNT(*) AS stars
                           FROM {cfg.table('star_labels')} GROUP BY 1 ORDER BY 1""").toPandas()
    for _, r in labels.iterrows():
        s[f"stars_{r['label_name']}"] = r["stars"]
    print(s.T.to_string(header=False))
    save_csv(s, cfg, "dataset_summary.csv")
    by_band = spark.sql(f"SELECT band, COUNT(*) AS rows FROM {t} GROUP BY band ORDER BY rows DESC").toPandas()
    by_type = spark.sql(f"SELECT obs_type, COUNT(*) AS rows FROM {t} GROUP BY obs_type ORDER BY rows DESC").toPandas()
    save_csv(by_band, cfg, "rows_per_band.csv")
    save_csv(by_type, cfg, "rows_per_type.csv")
    save_csv(labels, cfg, "label_counts.csv")
    per_star = spark.sql(f"""
        SELECT o.auid, COALESCE(l.name, l.data_name) AS name, l.vsx_type,
               CASE l.label WHEN 1 THEN 'anomalous' WHEN 0 THEN 'normal' ELSE 'unlabelled' END AS label_name,
               COUNT(*) AS n
        FROM {t} o LEFT JOIN {cfg.table('star_labels')} l ON o.auid = l.auid
        GROUP BY o.auid, COALESCE(l.name, l.data_name), l.vsx_type, l.label ORDER BY n DESC""").toPandas()
    save_csv(per_star, cfg, "obs_per_star.csv")
    plots.bar_obs_per_star(per_star, cfg.output_dir)
    return s


def run(spark, cfg):
    base = cfg.hdfs_base
    raw_glob = put_raw_files(cfg)
    check_header(spark, raw_glob)
    build_labels(spark, cfg, raw_glob)

    spark.read.csv(raw_glob, schema=RAW_SCHEMA, header=True, quote='"', escape='"', mode="PERMISSIVE") \
        .createOrReplaceTempView("raw_csv")
    edges = cfg.section("ingest")["dec_band_edges"]
    typed = spark.sql(TYPED_SELECT)
    typed.createOrReplaceTempView("typed")
    obs = spark.sql(f"""
        SELECT t.*,
               CAST(year(timestamp_seconds(CAST((t.jd - 2440587.5) * 86400 AS BIGINT))) AS INT) AS obs_year,
               COALESCE(l.sky_region, 'UNKNOWN') AS sky_region
        FROM typed t LEFT JOIN {cfg.table('star_labels')} l ON t.auid = l.auid""")

    part_path = cfg.hdfs_uri(f"{base}/warehouse/observations_raw")
    flat_path = cfg.hdfs_uri(f"{base}/warehouse/observations_flat")
    obs.repartition("obs_year", "sky_region").write.mode("overwrite").partitionBy("obs_year", "sky_region") \
        .option("compression", "snappy").parquet(part_path)
    spark.read.parquet(part_path).write.mode("overwrite").option("compression", "snappy").parquet(flat_path)

    for name in ("observations_raw", "observations_flat"):
        spark.sql(f"DROP TABLE IF EXISTS {cfg.table(name)}")
    spark.sql(f"""CREATE EXTERNAL TABLE {cfg.table('observations_raw')} ({OBS_DDL_COLUMNS})
                  PARTITIONED BY (obs_year INT, sky_region STRING)
                  STORED AS PARQUET LOCATION '{part_path}'""")
    spark.sql(f"MSCK REPAIR TABLE {cfg.table('observations_raw')}")
    spark.sql(f"""CREATE EXTERNAL TABLE {cfg.table('observations_flat')} ({OBS_DDL_COLUMNS},
                  obs_year INT, sky_region STRING) STORED AS PARQUET LOCATION '{flat_path}'""")

    print("\nHDFS layout of the partitioned Parquet table:")
    hdfs("-ls", "-R", "-h", f"{base}/warehouse/observations_raw")
    print("\nSHOW PARTITIONS (Spark SQL via Hive metastore):")
    parts = spark.sql(f"SHOW PARTITIONS {cfg.table('observations_raw')}").toPandas()
    print(parts.to_string(index=False))
    save_csv(parts, cfg, "hive_partitions.csv")

    partition_pruning_test(spark, cfg)
    return dataset_summary(spark, cfg)
