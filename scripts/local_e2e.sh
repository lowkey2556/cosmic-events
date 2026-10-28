#!/usr/bin/env bash
# Local end-to-end smoke run of the WHOLE step chain on synthetic data (no Colab, no YARN/HDFS/Hive services).
#
# Spark runs in local mode with an embedded Hive metastore, a local folder stands in for HDFS, and the internet
# inputs (ZTF files, ZTF field centres, VSX) are replaced by synthetic files (scripts/make_synthetic.py --ztf-v3-root).
# RGA-Net uses shortened epochs and a 2-point grid so the run fits on a laptop CPU (~20 min on 2 cores).
# Everything is written to synthetic_run/ and $LOCAL_FS; NOTHING here is a result. It checks that the code runs.
#
# Needs: Python 3.12 with requirements.txt + torch, and Java 8, 11 or 17 (Spark 3.5's Arrow does not work on Java 21).
#   JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 PATH=$JAVA_HOME/bin:$PATH bash scripts/local_e2e.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-python}"
LOCAL_FS="${LOCAL_FS:-$PWD/synthetic_run/local_hdfs}"
rm -rf synthetic_run data/synthetic
"$PY" scripts/make_synthetic.py --ztf-v3-root synthetic_run/v3_5GB
ARGS=(--data synthetic --set "spark.master=local[2]" --set "hdfs.namenode=file://$LOCAL_FS"
      --set spark.driver_memory=3g --set spark.shuffle_partitions=4)
FAST=(--set rganet.pretrain_epochs=1 --set rganet.max_epochs=3 --set rganet.patience=2 --set "rganet.grid.lr=[0.0003]"
      --set "rganet.grid.dropout=[0.1]" --set "rganet.grid.hybrid_head=[false,true]" --set rganet.pretrain_batch_size=256)
"$PY" run.py audit "${ARGS[@]}"
"$PY" run.py ingest+preprocess+score+features "${ARGS[@]}"
"$PY" run.py ztf_ingest+ztf_clean+ztf_label "${ARGS[@]}"
"$PY" run.py windows+ztf_windows+akval "${ARGS[@]}"
"$PY" run.py cluster+classify "${ARGS[@]}"
"$PY" run.py rganet "${ARGS[@]}" "${FAST[@]}"
"$PY" run.py infer "${ARGS[@]}"
"$PY" run.py compare "${ARGS[@]}"
"$PY" run.py dashboard "${ARGS[@]}"
"$PY" run.py report "${ARGS[@]}"
echo "LOCAL END-TO-END RUN FINISHED: synthetic_run/outputs (figures/, DATA_FLOW.txt, step_timings.csv)"
