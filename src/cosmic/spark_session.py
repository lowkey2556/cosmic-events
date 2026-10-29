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


def is_local(cfg):
    """Local smoke-test mode: spark.master=local[...] and hdfs.namenode=file://... (no YARN, HDFS or Hive service).
    Used only to run the whole step chain on a laptop/CI with synthetic data; Colab always uses YARN + HDFS + Hive."""
    return str(cfg.section("spark")["master"]).startswith("local")


def _local_spark(cfg):
    s = cfg.section("spark")
    base = cfg.settings["hdfs"]["namenode"]
    if not base.startswith("file://"):
        raise ValueError("local mode needs hdfs.namenode=file:///some/folder (no HDFS service is started)")
    meta = os.path.join(base[len("file://"):], "_metastore")
    os.makedirs(meta, exist_ok=True)
    py = os.environ.get("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_PYTHON", py)
    os.environ["PYTHONPATH"] = os.pathsep.join(p for p in (os.path.join(REPO_ROOT, "src"), os.environ.get("PYTHONPATH"))
                                               if p)
    spark = (SparkSession.builder.appName(f"{s['app_name']}-{cfg.dataset}-local").master(s["master"])
             .config("spark.driver.memory", s["driver_memory"])
             .config("spark.sql.shuffle.partitions", str(s["shuffle_partitions"]))
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.sql.parquet.compression.codec", "snappy")
             .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
             .config("spark.sql.warehouse.dir", base + s["warehouse_dir"])
             .config("spark.hadoop.javax.jdo.option.ConnectionURL",
                     f"jdbc:derby:;databaseName={os.path.join(meta, 'metastore_db')};create=true")
             .config("spark.driver.extraJavaOptions", f"-Dderby.system.home={meta}")
             .config("spark.sql.execution.arrow.pyspark.enabled", "true")
             .config("spark.ui.enabled", "false")
             .enableHiveSupport().getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {cfg.db} LOCATION '{cfg.hdfs_uri(cfg.hdfs_base)}/warehouse/{cfg.db}.db'")
    return spark


def get_spark(cfg):
    s = cfg.section("spark")
    if is_local(cfg):
        return _local_spark(cfg)
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


LOCAL_FS_ROOT = os.environ.get("COSMIC_LOCAL_FS_ROOT")  # set by run.py in local smoke-test mode only


def _local_hdfs(args):
    """The few `hdfs dfs` commands the pipeline uses (-rm -r, -mkdir -p, -put -f, -ls), on a local folder that stands
    in for HDFS in local smoke-test mode."""
    import glob
    root = LOCAL_FS_ROOT or os.environ["COSMIC_LOCAL_FS_ROOT"]
    p = lambda x: os.path.join(root, x.lstrip("/"))  # noqa: E731
    op, rest = args[0], [a for a in args[1:] if not a.startswith("-")]
    if op == "-rm":
        for x in rest:
            shutil.rmtree(p(x), ignore_errors=True)
    elif op == "-mkdir":
        for x in rest:
            os.makedirs(p(x), exist_ok=True)
    elif op == "-put":
        *srcs, dst = rest
        os.makedirs(p(dst), exist_ok=True)
        for f in srcs:
            shutil.copy(f, p(dst))
    elif op == "-ls":
        return "\n".join(sorted(glob.glob(os.path.join(p(rest[0]), "**"), recursive=True)))
    return ""


def hdfs(*args, check=True):
    """Run `hdfs dfs <args>` and return stdout (printed so notebook cells show it)."""
    if LOCAL_FS_ROOT or os.environ.get("COSMIC_LOCAL_FS_ROOT"):
        print("$ (local stand-in for HDFS) hdfs dfs " + " ".join(args))
        return _local_hdfs(args)
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


def path_bytes(spark, uri):
    """Total size in bytes of a file or folder on any Hadoop file system (hdfs://, file://)."""
    jvm = spark._jvm
    path = jvm.org.apache.hadoop.fs.Path(uri)
    fs = path.getFileSystem(spark._jsc.hadoopConfiguration())
    return int(fs.getContentSummary(path).getLength()) if fs.exists(path) else 0


def table_location(spark, table):
    rows = spark.sql(f"DESCRIBE FORMATTED {table}").collect()
    for r in rows:
        if (r["col_name"] or "").strip() == "Location":
            return r["data_type"].strip()
    return None


def table_bytes(spark, table):
    """On-disk size of a Hive table's data (its HDFS location)."""
    loc = table_location(spark, table)
    return path_bytes(spark, loc) if loc else 0


def table_files(spark, table):
    """Number of data files of a Hive table (partitions included)."""
    return len(spark.table(table).inputFiles())
