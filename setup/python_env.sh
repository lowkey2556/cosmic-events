#!/usr/bin/env bash
# Python environments for the pipeline (run once per Colab session by the notebook's setup cell).
#
# 1. /content/cosmic_py : a Python 3.12 virtual environment for every Spark step. PySpark 3.5 does
#    not support Python 3.13 (the current Colab default), so the Spark steps never use Colab's own
#    Python. It holds requirements.txt plus a CPU build of PyTorch (Spark executors use it to apply
#    the trained RGA-Net in the distributed-inference step). Made with `uv`, which also downloads
#    Python 3.12 itself if the machine does not have it.
# 2. Colab's own Python (python3) gets requirements-notebook.txt: the packages used by the notebook
#    and by the steps that run without Spark (audit, ZTF download/audit, RGA-Net training on the GPU
#    with Colab's CUDA PyTorch, comparison, charts).
# Safe to re-run: an existing environment is reused and only missing packages are installed.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${COSMIC_SPARK_VENV:-/content/cosmic_py}"
SYS_PY="${SYS_PY:-python3}"

command -v uv >/dev/null 2>&1 || "$SYS_PY" -m pip install -q uv
if [ ! -x "$VENV/bin/python" ]; then
  echo "== Creating the Python 3.12 environment for Spark at $VENV"
  uv venv --quiet --seed --python 3.12 "$VENV"
fi
"$VENV/bin/python" --version
echo "== Spark-side packages (requirements.txt + CPU PyTorch)"
uv pip install --quiet --python "$VENV/bin/python" -r "$REPO_DIR/requirements.txt"
"$VENV/bin/python" -c "import torch" 2>/dev/null || \
  uv pip install --quiet --python "$VENV/bin/python" torch --index-url https://download.pytorch.org/whl/cpu
echo "== Notebook-side packages for $("$SYS_PY" --version)"
"$SYS_PY" -m pip install -q -r "$REPO_DIR/requirements-notebook.txt"
"$VENV/bin/python" -c "import pyspark, torch, pandas; print('spark env: pyspark', pyspark.__version__, '| torch', torch.__version__, '| pandas', pandas.__version__)"
"$SYS_PY" -c "import plotly, pandas, pyarrow; print('notebook env: plotly', plotly.__version__, '| pandas', pandas.__version__, '| pyarrow', pyarrow.__version__)"
