"""Smoke test 5: PySpark on YARN writes a Hive table through the Hive metastore.

What it does: starts a SparkSession with the project's normal settings (YARN,
remote Hive metastore, Hive 3.1.3 client), writes a tiny Hive-format Parquet
table smoke.spark_written, and prints the YARN application id. The shell
script then reads the table back with the Hive CLI.

Why: proves Spark and Hive share one real metastore and one HDFS warehouse.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from cosmic.config import load_config  # noqa: E402
from cosmic.spark_session import get_spark  # noqa: E402

cfg = load_config("real", ["spark.executor_instances=1", "spark.executor_memory=512m", "spark.driver_memory=512m",
                           "spark.executor_memory_overhead=384m"])
spark = get_spark(cfg)
print("YARN application:", spark.sparkContext.applicationId, "| master:", spark.sparkContext.master)
spark.sql("CREATE DATABASE IF NOT EXISTS smoke")
df = spark.createDataFrame([(1, "written-by-pyspark"), (2, "on-yarn")], "id INT, note STRING")
spark.sql("DROP TABLE IF EXISTS smoke.spark_written")
df.write.format("hive").option("fileFormat", "parquet").saveAsTable("smoke.spark_written")
print(spark.sql("DESCRIBE FORMATTED smoke.spark_written").filter("col_name in ('Location','Provider','Type')").toPandas())
spark.stop()
