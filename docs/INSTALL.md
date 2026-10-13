# Installation record

Every setup command used by this project, in order. The notebook runs exactly these.

## Versions (checked against the official archives on 2026-10-11)

| Component | Version | Source |
|---|---|---|
| Java | OpenJDK 8 (`openjdk-8-jdk-headless`) | Ubuntu apt (Colab) |
| Hadoop | 3.3.6 | https://archive.apache.org/dist/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz |
| Spark | 3.5.9, pre-built for Hadoop 3 (Scala 2.12) | https://archive.apache.org/dist/spark/spark-3.5.9/spark-3.5.9-bin-hadoop3.tgz |
| PySpark | 3.5.9 (pip, exactly matching) | PyPI |
| Hive | 3.1.3 | https://archive.apache.org/dist/hive/hive-3.1.3/apache-hive-3.1.3-bin.tar.gz |
| Derby (metastore DB, network server) | 10.14.2.0 | https://archive.apache.org/dist/db/derby/db-derby-10.14.2.0/db-derby-10.14.2.0-bin.tar.gz |

Compatibility:
- Spark 3.5 runs on Java 8, 11 and 17, and Hadoop 3.3 and Hive 3.1 run on Java 8. So the whole
  stack uses Java 8.
- Spark 3.5's `spark.sql.hive.metastore.version` accepts `0.12.0`–`2.3.9` and `3.0.0`–`3.1.3`.
  We set `3.1.3` with `spark.sql.hive.metastore.jars=path` pointing at Hive 3.1.3's own `lib/`.
- PySpark 3.5.9 requires Python ≥ 3.8; Colab's Python 3.12 is fine.
- Downloads are SHA-512 verified against the `.sha512` files on archive.apache.org.

## Commands (Colab)

```bash
# 0. In the notebook: mount Drive, then clone the repository
git clone https://github.com/<you>/<repo>.git /content/cosmic && cd /content/cosmic

# 1. Install (about 6-8 min the first time; tarballs are then cached in MyDrive/cosmic/cache)
bash setup/install_stack.sh
#   apt-get install -y openjdk-8-jdk-headless
#   wget + sha512 check: hadoop-3.3.6, spark-3.5.9-bin-hadoop3, apache-hive-3.1.3-bin, db-derby-10.14.2.0-bin
#   tar -xzf ... -C /opt/cosmic_stack/{hadoop,spark,hive,derby}
#   rm hive/lib/guava-19.0.jar; cp hadoop/share/hadoop/common/lib/guava-27.0-jre.jar hive/lib/   (Guava clash)
#   cp derby/lib/derbyclient.jar hive/lib/                                                     (network-server driver)
#   write core-site, hdfs-site, yarn-site (memory/vcores sized from /proc/meminfo and nproc), mapred-site,
#         hadoop-env (HDFS_*_USER / YARN_*_USER = root), hive-site (metastore on jdbc:derby://localhost:1527),
#         spark conf (hive-site with hive.metastore.uris, spark-env, spark-defaults)
#   pip install -r requirements.txt     (pyspark==3.5.9, pyyaml, pytest, astropy, astroquery, ...)
#   writes /opt/cosmic_stack/cosmic_env.sh

# 2. Start (about 1-2 min)
bash setup/start_stack.sh
#   hdfs namenode -format              (first start only)
#   hdfs --daemon start namenode ; hdfs --daemon start datanode
#   hdfs dfsadmin -safemode wait ; hdfs dfs -mkdir -p /tmp /user/hive/warehouse /spark
#   yarn --daemon start resourcemanager ; yarn --daemon start nodemanager
#   zip $SPARK_HOME/jars -> hdfs:///spark/spark-jars.zip   (spark.yarn.archive)
#   java ... org.apache.derby.drda.NetworkServerControl start -h 127.0.0.1 -p 1527
#   schematool -dbType derby -initSchema   (first start only)
#   hive --service metastore              (thrift://localhost:9083)

# 3. Smoke tests
bash setup/smoke_tests.sh 1   # hdfs dfs put + list
bash setup/smoke_tests.sh 2   # YARN RM up, NodeManager RUNNING
bash setup/smoke_tests.sh 3   # SparkPi on YARN
bash setup/smoke_tests.sh 4   # Hive metastore: CREATE TABLE + LOAD + SELECT through the Hive CLI
bash setup/smoke_tests.sh 5   # PySpark on YARN writes smoke.spark_written; Hive CLI reads it

# 4. Pipeline
python run.py all --data real
```

## Problems found while testing, and how the scripts handle them

| Symptom | Cause | Fix in the scripts |
|---|---|---|
| HDFS and YARN daemons die as soon as the cell or shell that started them ends (log: `RECEIVED SIGNAL 1: SIGHUP`) | Hadoop daemons install their own SIGHUP handler, which overrides `nohup` | `start_stack.sh` launches every daemon with `setsid` (its own session, no controlling terminal) |
| `start_stack.sh` never returns when its output is piped (a Colab cell would hang) | `(cd X && java ... &)` backgrounds the whole list, leaving a shell holding stdout | `(cd X; setsid java ... </dev/null >log 2>&1 &)` |
| `toPandas()` fails with `No module named 'distutils'` | PySpark 3.5 imports `distutils`, which Python 3.12 removed | `setuptools` in requirements.txt provides the shim |
| PySpark/pandas errors with pandas 3.x | PySpark 3.5 predates pandas 3 | `pandas>=2.0,<3` (Colab ships 2.2) |
| Local tests on Windows: "Python worker exited unexpectedly (crashed)" for any Python UDF | a Windows worker bug in pip PySpark 3.5.4 | use PySpark 3.5.9 (the version pinned in requirements.txt) |
| Local tests on Windows: torch fails to load `c10.dll` (WinError 1114) | torch's DLLs must load before PyArrow's | `tests/conftest.py` imports torch first; in the pipeline RGA-Net runs in its own process anyway |
| Hive `SELECT ... ORDER BY` hangs on a small machine | Hive launches a MapReduce job; its 1.5 GB ApplicationMaster left no room for a map container | MapReduce AM/map/reduce containers set to 512 MB in `mapred-site.xml`; smoke test 5 uses a plain `SELECT` (a fetch task, no MapReduce) |

## Verified

On 2026-10-11, on the WSL test machine (non-root, 3.7 GB RAM, YARN 2 GB), these were verified:
- `install_stack.sh` and `start_stack.sh` completed; the start took about 1 minute.
- Smoke tests 1–5 all passed.
- `pytest` passed 11 of 11 tests.
- `python run.py milestone1 --data synthetic` (the six steps that existed then) ran on YARN, with every
  table readable from the Hive CLI.

On 2026-10-13, on a Windows 11 laptop (local-mode Spark 3.5.9 from pip, Java 8, Python 3.12, PyTorch CPU;
no HDFS, YARN or Hive), `pytest` passed 34 of 34 tests, including the Spark windowing, the Spark MLlib
models on windows, the star-attrition table, RGA-Net training with its ablations, and the comparison step.
The full pipeline on YARN with the new steps (windowing, ML models, RGA-Net, comparison, scalability) had
not yet been run at that point; its first run is the Colab run on the real data.

None of this used real data, and none of it produced Review 2 outputs.

## After a Colab reset

The VM is wiped, but Drive keeps the tarball cache and all outputs. Re-run the notebook from
the top. The install then only extracts from the cache, and HDFS and the metastore start empty,
so Step 1 re-ingests from Drive.

## Memory sizing (from the machine at install time)

- YARN NodeManager memory = 55% of RAM (rounded to 256 MB). Free Colab (~12.7 GB) gives about 6.9 GB.
- YARN vcores = 2 × nproc (minimum 4). Vcores are a scheduling unit; the CapacityScheduler's
  default resource calculator schedules on memory. This lets the Milestone 2 run with 4
  executors be scheduled on Colab's 2 CPUs. Real speedup is still limited to about 2 CPUs.
- Executors: 2 × (1536 MB + 512 MB overhead); driver 2 GB (outside YARN, client mode).
- The `vmem`/`pmem` checks are off: PySpark workers live inside executor containers on one VM,
  and YARN's allocations already bound memory.

## Testing outside Colab (how these scripts were tested)

The same `install_stack.sh` and `start_stack.sh` were run unchanged on a non-root WSL (Kali)
machine with 3.7 GB RAM. There, without root or apt, the script takes Java 8 from a JDK tarball in
the cache (`jdk8.tar.gz`, Eclipse Temurin 8) and installs to `~/cosmic_stack`. The
`PYTHON=...` variable pointed at a Python 3.12 virtualenv. This fallback is **only** used when
the script is not root. On Colab (root), Java always comes from apt.

```bash
PYTHON=~/cosmic_venv/bin/python CACHE_DIR=~/cosmic_cache bash setup/install_stack.sh
bash setup/start_stack.sh && bash setup/smoke_tests.sh all
python run.py all --data synthetic --set spark.executor_instances=1 --set spark.executor_memory=512m \
    --set spark.executor_memory_overhead=384m --set spark.driver_memory=768m
python -m pytest -q tests
```
