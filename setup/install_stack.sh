#!/usr/bin/env bash
# Install the whole Big Data stack on one Linux VM (written for Google Colab's Ubuntu 22.04).
#
#   Java 8 (OpenJDK, apt) | Hadoop 3.3.6 (HDFS + YARN) | Spark 3.5.9 (bin-hadoop3)
#   Hive 3.1.3 (metastore service) | Derby 10.14.2.0 network server (metastore database)
#
# Tarballs are cached in CACHE_DIR (Google Drive on Colab), so a reinstall after a Colab reset
# only extracts them. Re-running this script is safe.
#
# Environment overrides (all optional):
#   STACK_HOME  install prefix         (default /opt/cosmic_stack as root, else ~/cosmic_stack)
#   CACHE_DIR   tarball cache          (default /content/drive/MyDrive/cosmic/cache if Drive is mounted)
#   PYTHON      python for PySpark     (default python3)
#   JAVA_HOME   use an existing Java 8 instead of apt (only for non-root test machines)
set -euo pipefail

HADOOP_VERSION=3.3.6
SPARK_VERSION=3.5.9
HIVE_VERSION=3.1.3
DERBY_VERSION=10.14.2.0
APACHE=https://archive.apache.org/dist

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "$(id -u)" = "0" ]; then DEFAULT_HOME=/opt/cosmic_stack; else DEFAULT_HOME="$HOME/cosmic_stack"; fi
STACK_HOME="${STACK_HOME:-$DEFAULT_HOME}"
if [ -d /content/drive/MyDrive ]; then DEFAULT_CACHE=/content/drive/MyDrive/cosmic/cache; else DEFAULT_CACHE="$HOME/cosmic_cache"; fi
CACHE_DIR="${CACHE_DIR:-$DEFAULT_CACHE}"
PYTHON="${PYTHON:-python3}"
mkdir -p "$STACK_HOME" "$CACHE_DIR"
echo "== STACK_HOME=$STACK_HOME  CACHE_DIR=$CACHE_DIR  PYTHON=$PYTHON"

# ---------------------------------------------------------------- Java 8
if [ -n "${JAVA_HOME:-}" ] && "$JAVA_HOME/bin/java" -version 2>&1 | grep -q '"1\.8'; then
  echo "== Using existing Java 8 at $JAVA_HOME"
elif [ "$(id -u)" = "0" ] && command -v apt-get >/dev/null; then
  echo "== Installing OpenJDK 8 from apt"
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openjdk-8-jdk-headless >/dev/null
  JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
elif [ -f "$CACHE_DIR/jdk8.tar.gz" ]; then
  # Non-root test machines only (no apt): a Java 8 JDK tarball placed in the cache by hand.
  echo "== Extracting Java 8 from $CACHE_DIR/jdk8.tar.gz (non-root test install)"
  rm -rf "$STACK_HOME/jdk8" && mkdir -p "$STACK_HOME/jdk8"
  tar -xzf "$CACHE_DIR/jdk8.tar.gz" -C "$STACK_HOME/jdk8" --strip-components=1
  JAVA_HOME="$STACK_HOME/jdk8"
else
  echo "ERROR: no Java 8 available (not root, no JAVA_HOME, no cached jdk8.tar.gz)"; exit 1
fi
"$JAVA_HOME/bin/java" -version

# ---------------------------------------------------------------- download (cached) + verify
fetch() {  # fetch <url>  -> cached file path on stdout
  local url="$1" f; f="$CACHE_DIR/$(basename "$url")"
  if [ ! -s "$f" ]; then
    echo "== Downloading $url" >&2
    wget -q --tries=3 "$url" -O "$f.part" && mv "$f.part" "$f"
    local sum; sum=$(wget -q -O - "$url.sha512" | tr -d ' \r\n' | grep -oiE '[0-9a-f]{128}' | head -1 || true)
    if [ -n "$sum" ]; then
      echo "$sum  $f" | sha512sum -c - >&2 || { rm -f "$f"; echo "ERROR: checksum mismatch for $f" >&2; exit 1; }
    fi
  else
    echo "== Using cached $f" >&2
  fi
  echo "$f"
}
HADOOP_TGZ=$(fetch "$APACHE/hadoop/common/hadoop-$HADOOP_VERSION/hadoop-$HADOOP_VERSION.tar.gz")
SPARK_TGZ=$(fetch "$APACHE/spark/spark-$SPARK_VERSION/spark-$SPARK_VERSION-bin-hadoop3.tgz")
HIVE_TGZ=$(fetch "$APACHE/hive/hive-$HIVE_VERSION/apache-hive-$HIVE_VERSION-bin.tar.gz")
DERBY_TGZ=$(fetch "$APACHE/db/derby/db-derby-$DERBY_VERSION/db-derby-$DERBY_VERSION-bin.tar.gz")

# ---------------------------------------------------------------- extract
extract() {  # extract <tgz> <dest>
  if [ ! -d "$2" ]; then
    echo "== Extracting $(basename "$1")"
    mkdir -p "$2"
    tar -xzf "$1" -C "$2" --strip-components=1 --exclude='*/share/doc' --exclude='*/docs' --exclude='*/javadoc'
  fi
}
HADOOP_HOME="$STACK_HOME/hadoop"; SPARK_HOME="$STACK_HOME/spark"
HIVE_HOME="$STACK_HOME/hive";     DERBY_HOME="$STACK_HOME/derby"
extract "$HADOOP_TGZ" "$HADOOP_HOME"
extract "$SPARK_TGZ"  "$SPARK_HOME"
extract "$HIVE_TGZ"   "$HIVE_HOME"
extract "$DERBY_TGZ"  "$DERBY_HOME"

# Hive 3.1.x ships Guava 19, which clashes with Hadoop 3.3's Guava 27: use Hadoop's.
rm -f "$HIVE_HOME"/lib/guava-*.jar
cp "$HADOOP_HOME"/share/hadoop/common/lib/guava-*.jar "$HIVE_HOME/lib/"
# Hive talks to the Derby *network* server, so it needs the Derby client driver.
cp "$DERBY_HOME/lib/derbyclient.jar" "$HIVE_HOME/lib/"
echo "== Hive guava now: $(ls "$HIVE_HOME"/lib/guava-*.jar | xargs -n1 basename)"

# ---------------------------------------------------------------- size YARN to this machine
MEM_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
NPROC=$(nproc)
YARN_MB=$(( (MEM_MB * 55 / 100) / 256 * 256 ))
[ "$YARN_MB" -lt 1536 ] && YARN_MB=1536
# vcores are a scheduling unit, not physical cores; allow 4+ so the 1/2/4-executor runs can be scheduled.
YARN_VCORES=$(( NPROC * 2 )); [ "$YARN_VCORES" -lt 4 ] && YARN_VCORES=4
if [ "$MEM_MB" -lt 6000 ]; then DAEMON_HEAP=256m; else DAEMON_HEAP=512m; fi
echo "== Machine: ${MEM_MB} MB RAM, ${NPROC} CPUs -> YARN ${YARN_MB} MB, ${YARN_VCORES} vcores, daemon heap ${DAEMON_HEAP}"

DATA_DIR="$STACK_HOME/data"
mkdir -p "$DATA_DIR/namenode" "$DATA_DIR/datanode" "$DATA_DIR/tmp" "$DATA_DIR/derby" "$DATA_DIR/logs"
RUN_USER="$(whoami)"

# ---------------------------------------------------------------- Hadoop config
HC="$HADOOP_HOME/etc/hadoop"
cat > "$HC/core-site.xml" <<EOF
<?xml version="1.0"?>
<configuration>
  <property><name>fs.defaultFS</name><value>hdfs://localhost:9000</value></property>
  <property><name>hadoop.tmp.dir</name><value>$DATA_DIR/tmp</value></property>
  <property><name>hadoop.proxyuser.$RUN_USER.hosts</name><value>*</value></property>
  <property><name>hadoop.proxyuser.$RUN_USER.groups</name><value>*</value></property>
</configuration>
EOF
cat > "$HC/hdfs-site.xml" <<EOF
<?xml version="1.0"?>
<configuration>
  <property><name>dfs.replication</name><value>1</value></property>
  <property><name>dfs.namenode.name.dir</name><value>file://$DATA_DIR/namenode</value></property>
  <property><name>dfs.datanode.data.dir</name><value>file://$DATA_DIR/datanode</value></property>
  <property><name>dfs.permissions.enabled</name><value>false</value></property>
</configuration>
EOF
cat > "$HC/yarn-site.xml" <<EOF
<?xml version="1.0"?>
<configuration>
  <property><name>yarn.resourcemanager.hostname</name><value>localhost</value></property>
  <property><name>yarn.nodemanager.aux-services</name><value>mapreduce_shuffle</value></property>
  <property><name>yarn.nodemanager.resource.memory-mb</name><value>$YARN_MB</value></property>
  <property><name>yarn.nodemanager.resource.cpu-vcores</name><value>$YARN_VCORES</value></property>
  <property><name>yarn.scheduler.minimum-allocation-mb</name><value>256</value></property>
  <property><name>yarn.scheduler.maximum-allocation-mb</name><value>$YARN_MB</value></property>
  <property><name>yarn.scheduler.maximum-allocation-vcores</name><value>$YARN_VCORES</value></property>
  <property><name>yarn.scheduler.capacity.maximum-am-resource-percent</name><value>0.5</value></property>
  <!-- single VM: Python workers live inside executor containers; YARN sizing (not the checker) bounds memory -->
  <property><name>yarn.nodemanager.vmem-check-enabled</name><value>false</value></property>
  <property><name>yarn.nodemanager.pmem-check-enabled</name><value>false</value></property>
  <property><name>yarn.nodemanager.disk-health-checker.max-disk-utilization-per-disk-percentage</name><value>99</value></property>
  <property><name>yarn.nodemanager.env-whitelist</name><value>JAVA_HOME,HADOOP_COMMON_HOME,HADOOP_HDFS_HOME,HADOOP_CONF_DIR,CLASSPATH_PREPEND_DISTCACHE,HADOOP_YARN_HOME,HADOOP_HOME,PATH,LANG,TZ,HADOOP_MAPRED_HOME</value></property>
</configuration>
EOF
cat > "$HC/mapred-site.xml" <<EOF
<?xml version="1.0"?>
<configuration>
  <property><name>mapreduce.framework.name</name><value>yarn</value></property>
  <property><name>yarn.app.mapreduce.am.env</name><value>HADOOP_MAPRED_HOME=$HADOOP_HOME</value></property>
  <property><name>mapreduce.map.env</name><value>HADOOP_MAPRED_HOME=$HADOOP_HOME</value></property>
  <property><name>mapreduce.reduce.env</name><value>HADOOP_MAPRED_HOME=$HADOOP_HOME</value></property>
</configuration>
EOF
# Colab runs as root: Hadoop 3 refuses to start daemons as root unless these are set.
sed -i '/# cosmic-begin/,/# cosmic-end/d' "$HC/hadoop-env.sh"
cat >> "$HC/hadoop-env.sh" <<EOF
# cosmic-begin
export JAVA_HOME=$JAVA_HOME
export HADOOP_HEAPSIZE_MAX=$DAEMON_HEAP
export HADOOP_LOG_DIR=$DATA_DIR/logs
export HDFS_NAMENODE_USER=$RUN_USER
export HDFS_DATANODE_USER=$RUN_USER
export HDFS_SECONDARYNAMENODE_USER=$RUN_USER
export YARN_RESOURCEMANAGER_USER=$RUN_USER
export YARN_NODEMANAGER_USER=$RUN_USER
# cosmic-end
EOF

# ---------------------------------------------------------------- Hive config (metastore on Derby network server)
cat > "$HIVE_HOME/conf/hive-site.xml" <<EOF
<?xml version="1.0"?>
<configuration>
  <property><name>hive.metastore.uris</name><value>thrift://localhost:9083</value></property>
  <property><name>javax.jdo.option.ConnectionURL</name><value>jdbc:derby://localhost:1527/metastore_db;create=true</value></property>
  <property><name>javax.jdo.option.ConnectionDriverName</name><value>org.apache.derby.jdbc.ClientDriver</value></property>
  <property><name>javax.jdo.option.ConnectionUserName</name><value>APP</value></property>
  <property><name>javax.jdo.option.ConnectionPassword</name><value>mine</value></property>
  <property><name>hive.metastore.warehouse.dir</name><value>/user/hive/warehouse</value></property>
  <property><name>hive.metastore.schema.verification</name><value>false</value></property>
  <property><name>datanucleus.schema.autoCreateAll</name><value>false</value></property>
  <property><name>hive.metastore.event.db.notification.api.auth</name><value>false</value></property>
  <property><name>hive.execution.engine</name><value>mr</value></property>
  <property><name>hive.exec.scratchdir</name><value>/tmp/hive</value></property>
  <property><name>hive.cli.print.header</name><value>true</value></property>
  <property><name>hive.server2.enable.doAs</name><value>false</value></property>
</configuration>
EOF
cat > "$HIVE_HOME/conf/hive-env.sh" <<EOF
export HADOOP_HOME=$HADOOP_HOME
export HIVE_CONF_DIR=$HIVE_HOME/conf
export HADOOP_HEAPSIZE=$( echo $DAEMON_HEAP | tr -d m )
EOF

# ---------------------------------------------------------------- Spark config (YARN + remote Hive metastore)
cat > "$SPARK_HOME/conf/hive-site.xml" <<EOF
<?xml version="1.0"?>
<configuration>
  <property><name>hive.metastore.uris</name><value>thrift://localhost:9083</value></property>
  <property><name>hive.metastore.warehouse.dir</name><value>/user/hive/warehouse</value></property>
</configuration>
EOF
cat > "$SPARK_HOME/conf/spark-env.sh" <<EOF
export JAVA_HOME=$JAVA_HOME
export HADOOP_CONF_DIR=$HC
export YARN_CONF_DIR=$HC
EOF
cat > "$SPARK_HOME/conf/spark-defaults.conf" <<EOF
spark.master                     yarn
spark.yarn.archive               hdfs://localhost:9000/spark/spark-jars.zip
spark.sql.hive.metastore.version $HIVE_VERSION
spark.sql.hive.metastore.jars    path
spark.sql.hive.metastore.jars.path file://$HIVE_HOME/lib/*.jar
spark.sql.session.timeZone       UTC
spark.ui.showConsoleProgress     false
EOF

# ---------------------------------------------------------------- Python packages
PY_BIN="$(command -v "$PYTHON")"
"$PY_BIN" -m pip install -q -r "$REPO_DIR/requirements.txt"
"$PY_BIN" -c "import pyspark; assert pyspark.__version__ == '$SPARK_VERSION', pyspark.__version__; print('pyspark', pyspark.__version__)"

# ---------------------------------------------------------------- env file
cat > "$STACK_HOME/cosmic_env.sh" <<EOF
export STACK_HOME=$STACK_HOME
export JAVA_HOME=$JAVA_HOME
export HADOOP_HOME=$HADOOP_HOME
export HADOOP_CONF_DIR=$HC
export YARN_CONF_DIR=$HC
export HADOOP_MAPRED_HOME=$HADOOP_HOME
export SPARK_HOME=$SPARK_HOME
export HIVE_HOME=$HIVE_HOME
export HIVE_CONF_DIR=$HIVE_HOME/conf
export DERBY_HOME=$DERBY_HOME
export COSMIC_DATA_DIR=$DATA_DIR
export PYSPARK_PYTHON=$PY_BIN
export PYSPARK_DRIVER_PYTHON=$PY_BIN
export PYTHONPATH=$REPO_DIR/src
export PATH=$JAVA_HOME/bin:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$SPARK_HOME/bin:$HIVE_HOME/bin:$DERBY_HOME/bin:\$PATH
EOF
echo "$STACK_HOME/cosmic_env.sh" > "$REPO_DIR/setup/ENV_LOCATION"
echo "== Install complete. Env file: $STACK_HOME/cosmic_env.sh"
