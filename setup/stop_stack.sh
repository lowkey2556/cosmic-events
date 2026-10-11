#!/usr/bin/env bash
# Stop every service started by start_stack.sh (data in HDFS and Derby is kept).
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${COSMIC_ENV:-$(cat "$REPO_DIR/setup/ENV_LOCATION" 2>/dev/null || echo /opt/cosmic_stack/cosmic_env.sh)}"
# shellcheck disable=SC1090
source "$ENV_FILE"
pkill -f 'org.apache.hadoop.hive.metastore.HiveMetaStore' || true
"$JAVA_HOME/bin/java" -cp "$DERBY_HOME/lib/derby.jar:$DERBY_HOME/lib/derbynet.jar:$DERBY_HOME/lib/derbytools.jar" \
  org.apache.derby.drda.NetworkServerControl shutdown -h 127.0.0.1 -p 1527 >/dev/null 2>&1 || true
yarn --daemon stop nodemanager || true
yarn --daemon stop resourcemanager || true
hdfs --daemon stop datanode || true
hdfs --daemon stop namenode || true
jps
