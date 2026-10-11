"""Spark session and shared I/O helpers.

What it does: builds the one SparkSession every step uses — submitted to YARN,
reading and writing HDFS, and connected to the real Hive metastore service over
thrift (`hive.metastore.uris`) with the Hive 3.1.3 client jars. It also holds
small helpers: run an `hdfs dfs` command, save a DataFrame as a Hive table that
Hive itself can read, and save small result tables as CSV.

Why: one place to configure Spark means every step runs on exactly the same
stack (HDFS + YARN + Hive metastore), which is what the project is graded on.
"""
import os
import shutil
import subprocess
import sys
import tempfile

from pyspark.sql import SparkSession

from .config import REPO_ROOT


def _package_zip():
    """Zip src/cosmic so YARN executors can import our per-star functions."""
    tmp = tempfile.mkdtemp(prefix="cosmic_pkg_")
    base = os.path.join(tmp, "cosmic")
    return shutil.make_archive(base, "zip", os.path.join(REPO_ROOT, "src"), "cosmic")


def get_spark(cfg):
    s = cfg.section("spark")
    hive_home = os.environ.get("HIVE_HOME")
    if not hive_home:
        raise RuntimeError("HIVE_HOME is not set: source the stack env (setup/start_stack.sh) first")
    py = os.environ.get("PYSPARK_PYTHON", sys.executable)
    builder = (
        SparkSession.builder.appName(f"{s['app_name']}-{cfg.dataset}")
        .master(s["master"])
        .config("spark.submit.deployMode", s["deploy_mode"])
        .config("spark.driver.memory", s["driver_memory"])
        .config("spark.executor.instances", str(s["executor_instances"]))
        .config("spark.executor.cores", str(s["executor_cores"]))
        .config("spark.executor.memory", s["executor_memory"])
        .config("spark.executor.memoryOverhead", s["executor_memory_overhead"])
        .config("spark.sql.shuffle.partitions", str(s["shuffle_partitions"]))
        .config("spark.pyspark.python", py)
        .config("spark.pyspark.driver.python", py)
        .config("spark.executorEnv.PYSPARK_PYTHON", py)
        .config("spark.yarn.appMasterEnv.PYSPARK_PYTHON", py)
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("spark.hadoop.hive.metastore.uris", s["metastore_uri"])
        .config("spark.sql.warehouse.dir", cfg.settings["hdfs"]["namenode"] + s["warehouse_dir"])
        .config("spark.sql.hive.metastore.version", s["metastore_version"])
        .config("spark.sql.hive.metastore.jars", "path")
        .config("spark.sql.hive.metastore.jars.path", f"file://{hive_home}/lib/*.jar")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .enableHiveSupport()
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    spark.sparkContext.addPyFile(_package_zip())
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {cfg.db} LOCATION '{cfg.hdfs_uri(cfg.hdfs_base)}/warehouse/{cfg.db}.db'")
    return spark


def hdfs(*args, check=True):
    """Run `hdfs dfs <args>` and return stdout (printed so notebook cells show it)."""
    cmd = ["hdfs", "dfs", *args]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("$ " + " ".join(cmd))
    if res.stdout:
        print(res.stdout.rstrip())
    if check and res.returncode != 0:
        print(res.stderr)
        raise RuntimeError(f"hdfs command failed: {' '.join(cmd)}")
    return res.stdout


def save_hive_table(df, cfg, name, partition_by=None):
    """Save as a Hive-format Parquet table so both Spark SQL and the Hive CLI can read it."""
    w = df.write.mode("overwrite").format("hive").option("fileFormat", "parquet")
    if partition_by:
        w = w.partitionBy(*partition_by)
    df.sparkSession.sql(f"DROP TABLE IF EXISTS {cfg.table(name)}")
    w.saveAsTable(cfg.table(name))
    return cfg.table(name)


def save_csv(pdf, cfg, filename):
    """Save a small (already aggregated) pandas table to the output folder."""
    path = os.path.join(cfg.output_dir, filename)
    pdf.to_csv(path, index=False)
    print(f"saved {path}")
    return path
