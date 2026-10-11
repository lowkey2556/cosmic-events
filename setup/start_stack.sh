#!/usr/bin/env bash
# Start the whole stack (safe to re-run; already-running services are left alone):
#   HDFS NameNode + DataNode -> YARN ResourceManager + NodeManager
#   -> Derby network server -> Hive metastore schema (first run only) -> Hive metastore service.
# Daemons are started individually with --daemon start (no SSH needed, works as root on Colab).
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${COSMIC_ENV:-$(cat "$REPO_DIR/setup/ENV_LOCATION" 2>/dev/null || echo /opt/cosmic_stack/cosmic_env.sh)}"
# shellcheck disable=SC1090
source "$ENV_FILE"
LOG="$COSMIC_DATA_DIR/logs"; mkdir -p "$LOG"

port_open() { (echo > "/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1; }
wait_port() {  # wait_port <port> <name> <seconds>
  for _ in $(seq 1 "$3"); do port_open "$1" && { echo "   $2 is up (port $1)"; return 0; }; sleep 1; done
  echo "ERROR: $2 did not open port $1 in $3 s. Logs: $LOG"; return 1
}
running() { jps 2>/dev/null | awk '{print $2}' | grep -qx "$1"; }
# Hadoop daemons install their own SIGHUP handler (overriding nohup), so they would die when the
# launching shell's terminal closes, e.g. at the end of a Colab "!" cell. setsid gives each daemon
# its own session with no controlling terminal, so no SIGHUP ever reaches it.
detach() { setsid "$@" </dev/null; }

echo "== HDFS"
if [ ! -d "$COSMIC_DATA_DIR/namenode/current" ]; then
  hdfs namenode -format -nonInteractive -force >"$LOG/format.log" 2>&1 && echo "   NameNode formatted"
fi
running NameNode || detach hdfs --daemon start namenode
running DataNode || detach hdfs --daemon start datanode
wait_port 9000 NameNode 60
hdfs dfsadmin -safemode wait >/dev/null
hdfs dfs -mkdir -p /tmp /tmp/hive /user/hive/warehouse /spark "/user/$(whoami)"
hdfs dfs -chmod -R 1777 /tmp
hdfs dfs -chmod -R 777 /user/hive/warehouse

echo "== YARN"
running ResourceManager || detach yarn --daemon start resourcemanager
running NodeManager || detach yarn --daemon start nodemanager
wait_port 8088 ResourceManager 60

echo "== Spark jars archive on HDFS (avoids re-uploading ~250 MB on every submit)"
if ! hdfs dfs -test -e /spark/spark-jars.zip; then
  "$PYSPARK_PYTHON" -m zipfile -c "$COSMIC_DATA_DIR/spark-jars.zip" "$SPARK_HOME"/jars/*
  hdfs dfs -put -f "$COSMIC_DATA_DIR/spark-jars.zip" /spark/spark-jars.zip
fi

echo "== Derby network server (Hive metastore database)"
if ! port_open 1527; then
  # "cd ...;" (not "&&") so only the java process is backgrounded and no shell keeps our stdout open
  (cd "$COSMIC_DATA_DIR/derby"; setsid nohup "$JAVA_HOME/bin/java" -Xmx256m -Dderby.system.home="$COSMIC_DATA_DIR/derby" \
     -cp "$DERBY_HOME/lib/derby.jar:$DERBY_HOME/lib/derbynet.jar:$DERBY_HOME/lib/derbytools.jar" \
     org.apache.derby.drda.NetworkServerControl start -h 127.0.0.1 -p 1527 -noSecurityManager \
     </dev/null >"$LOG/derby.log" 2>&1 &)
fi
wait_port 1527 Derby 60

echo "== Hive metastore schema"
if [ ! -f "$COSMIC_DATA_DIR/derby/.schema_initialised" ]; then
  if schematool -dbType derby -info >"$LOG/schematool_info.log" 2>&1; then
    echo "   schema already present"
  else
    schematool -dbType derby -initSchema >"$LOG/schematool.log" 2>&1 || { tail -30 "$LOG/schematool.log"; exit 1; }
    echo "   schema initialised (Hive $(grep -o 'Initialization script hive-schema-[0-9.]*' "$LOG/schematool.log" | head -1))"
  fi
  touch "$COSMIC_DATA_DIR/derby/.schema_initialised"
fi

echo "== Hive metastore service"
if ! port_open 9083; then
  (cd "$LOG"; setsid nohup hive --service metastore </dev/null >"$LOG/metastore.log" 2>&1 &)
fi
wait_port 9083 "Hive metastore" 180

echo "== Running JVMs"
jps | grep -v Jps | sort -k2
echo "== Stack is up."
