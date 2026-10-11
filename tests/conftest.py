"""Shared test fixtures: a local-mode SparkSession and a synthetic dataset.

Unit tests check the pipeline LOGIC with local-mode Spark on synthetic data (no HDFS,
YARN or Hive needed). The full stack is exercised separately by setup/smoke_tests.sh
and `python run.py milestone1 --data synthetic`.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


@pytest.fixture(scope="session")
def spark():
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
    from cosmic.ingest import RAW_SCHEMA, TYPED_SELECT
    spark.read.csv(os.path.join(synth_dir, "raw", "*.csv"), schema=RAW_SCHEMA, header=True,
                   quote='"', escape='"').createOrReplaceTempView("raw_csv")
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
