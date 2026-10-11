#!/usr/bin/env bash
# Smoke tests for the stack. Each prints PASS or FAIL and exits non-zero on failure.
#   bash setup/smoke_tests.sh 1    hdfs dfs put + list
#   bash setup/smoke_tests.sh 2    YARN ResourceManager up (and a NodeManager registered)
#   bash setup/smoke_tests.sh 3    SparkPi submitted to YARN
#   bash setup/smoke_tests.sh 4    Hive metastore up: table created and queried through the Hive CLI
#   bash setup/smoke_tests.sh 5    PySpark-on-YARN writes a Hive table that the Hive CLI reads back
#   bash setup/smoke_tests.sh all
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${COSMIC_ENV:-$(cat "$REPO_DIR/setup/ENV_LOCATION" 2>/dev/null || echo /opt/cosmic_stack/cosmic_env.sh)}"
# shellcheck disable=SC1090
source "$ENV_FILE"
TMP="$(mktemp -d)"
pass() { echo "SMOKE TEST $1: PASS - $2"; }
fail() { echo "SMOKE TEST $1: FAIL - $2"; exit 1; }

t1() {
  echo "hello hdfs $(date -u +%FT%TZ)" > "$TMP/hello.txt"
  hdfs dfs -mkdir -p /smoke && hdfs dfs -put -f "$TMP/hello.txt" /smoke/hello.txt || fail 1 "put failed"
  hdfs dfs -ls /smoke
  hdfs dfs -cat /smoke/hello.txt | grep -q "hello hdfs" && pass 1 "file written to and read back from HDFS" || fail 1 "cat failed"
}

t2() {
  curl -s http://localhost:8088/ws/v1/cluster/info | grep -q '"state":"STARTED"' || fail 2 "ResourceManager REST API not STARTED"
  yarn node -list 2>/dev/null | tee "$TMP/nodes.txt"
  grep -q RUNNING "$TMP/nodes.txt" && pass 2 "ResourceManager STARTED, NodeManager RUNNING" || fail 2 "no RUNNING NodeManager"
}

t3() {
  spark-submit --master yarn --deploy-mode client --num-executors 1 --executor-memory 512m --driver-memory 512m \
    --class org.apache.spark.examples.SparkPi "$SPARK_HOME"/examples/jars/spark-examples_*.jar 20 \
    > "$TMP/pi.txt" 2> "$TMP/pi.err"
  grep -h "Pi is roughly" "$TMP/pi.txt" || { tail -30 "$TMP/pi.err"; fail 3 "SparkPi produced no result"; }
  grep -ho "application_[0-9_]*" "$TMP/pi.err" | head -1 | sed 's/^/YARN application: /'
  pass 3 "SparkPi ran on YARN"
}

t4() {
  printf '1\talpha\n2\tbeta\n3\tgamma\n' > "$TMP/rows.tsv"
  hive -S -e "
    CREATE DATABASE IF NOT EXISTS smoke;
    DROP TABLE IF EXISTS smoke.hive_t;
    CREATE TABLE smoke.hive_t (id INT, name STRING) ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t';
    LOAD DATA LOCAL INPATH '$TMP/rows.tsv' INTO TABLE smoke.hive_t;
    SELECT * FROM smoke.hive_t;" 2>"$TMP/hive4.err" | tee "$TMP/hive4.out"
  grep -q gamma "$TMP/hive4.out" && pass 4 "Hive metastore (thrift://localhost:9083) created and queried smoke.hive_t" \
    || { tail -30 "$TMP/hive4.err"; fail 4 "Hive query failed"; }
}

t5() {
  "$PYSPARK_PYTHON" "$REPO_DIR/setup/smoke_spark_hive.py" || fail 5 "PySpark could not write the Hive table"
  hive -S -e "SELECT * FROM smoke.spark_written;" 2>"$TMP/hive5.err" | tee "$TMP/hive5.out"
  grep -q "written-by-pyspark" "$TMP/hive5.out" && pass 5 "table written by PySpark on YARN read back through the Hive CLI" \
    || { tail -30 "$TMP/hive5.err"; fail 5 "Hive could not read the PySpark table"; }
}

case "${1:-all}" in
  1) t1 ;; 2) t2 ;; 3) t3 ;; 4) t4 ;; 5) t5 ;;
  all) t1; t2; t3; t4; t5 ;;
  *) echo "usage: $0 {1|2|3|4|5|all}"; exit 2 ;;
esac
