"""Shared test fixtures: a local-mode SparkSession and a synthetic dataset.

Unit tests check the pipeline LOGIC with local-mode Spark on synthetic data (no HDFS,
YARN or Hive needed). The full stack is exercised separately by setup/smoke_tests.sh
and `python run.py milestone1 --data synthetic`.
"""
import os
import pathlib
import sys

import pytest

try:  # load PyTorch before PySpark/PyArrow: on Windows, torch's DLLs fail to load if they come second
    import torch  # noqa: F401
except ImportError:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    # use pip PySpark's local mode and the local filesystem, not the cluster's configuration
    for var in ("SPARK_HOME", "HADOOP_CONF_DIR", "YARN_CONF_DIR"):
        os.environ.pop(var, None)
    # local-mode Python workers inherit this, so they can import the cosmic package (get_spark ships a zip
    # with addPyFile instead; that needs winutils.exe on Windows, so tests use the environment)
    paths = [os.path.join(ROOT, "src")]
    if os.name == "nt":
        # Windows only: torch's DLLs fail to load if PyArrow's are loaded first, so workers load torch just before
        # their first PyArrow import. Not at worker start-up: importing torch can take longer than the ~10 s Spark
        # gives a new Python worker to connect back.
        boot = tmp_path_factory.mktemp("worker_boot")
        (boot / "sitecustomize.py").write_text(
            "import importlib.abc, sys\n"
            "class _TorchFirst(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, name, path, target=None):\n"
            "        if name == 'pyarrow' and 'torch' not in sys.modules:\n"
            "            sys.meta_path.remove(self)\n"
            "            try:\n"
            "                import torch  # noqa: F401\n"
            "            except ImportError:\n"
            "                pass\n"
            "        return None\n"
            "sys.meta_path.insert(0, _TorchFirst())\n")
        paths.insert(0, str(boot))
    os.environ["PYTHONPATH"] = os.pathsep.join(p for p in (*paths, os.environ.get("PYTHONPATH")) if p)
    from pyspark.sql import SparkSession
    s = (SparkSession.builder.master("local[2]").appName("cosmic-tests")
         .config("spark.sql.shuffle.partitions", "4")
         .config("spark.sql.session.timeZone", "UTC")
         .config("spark.ui.enabled", "false")
         .getOrCreate())
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
    import make_synthetic
    out = tmp_path_factory.mktemp("synthetic")
    make_synthetic.generate(str(out), seed=1)
    return str(out)


@pytest.fixture(scope="session")
def settings():
    from cosmic.config import load_config
    return load_config("synthetic").settings


@pytest.fixture(scope="session")
def typed_view(spark, synth_dir):
    """Synthetic CSVs read exactly as Step 1 reads them, as temp view 'typed_obs'."""
    from cosmic.ingest import TYPED_SELECT, read_raw
    files = sorted(p.as_uri() for p in pathlib.Path(synth_dir, "raw").glob("*.csv"))  # a file list, as Step 1 passes
    read_raw(spark, files).createOrReplaceTempView("raw_csv")
    spark.sql(TYPED_SELECT).createOrReplaceTempView("typed_nopart")
    spark.sql("""SELECT *, CAST(year(timestamp_seconds(CAST((jd - 2440587.5) * 86400 AS BIGINT))) AS INT) AS obs_year,
                        'UNKNOWN' AS sky_region FROM typed_nopart""").createOrReplaceTempView("typed_obs")
    return "typed_obs"


@pytest.fixture(scope="session")
def clean_views(spark, typed_view, settings):
    from cosmic.preprocess import transform
    clean, upper, log, n_upper = transform(spark, typed_view, settings)
    clean.cache().createOrReplaceTempView("t_clean")
    upper.createOrReplaceTempView("t_upper")
    return dict(log=log, n_upper=n_upper)
