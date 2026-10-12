"""Architecture diagram (Review 2, slide 3), drawn with Graphviz.

What it does: writes the pipeline as a Graphviz DOT graph — Drive CSVs → audit → HDFS →
Spark ingestion (Parquet + Hive) → Spark SQL preprocessing → observer scoring → Spark
windowing → (a) per-window features → Spark MLlib NB / DT / RF / LR and (b) token
sequences → PyTorch RGA-Net on the Colab GPU → comparison → Plotly figures — labelling the
tool used at each stage, and renders it with the `dot` program to a 300-dpi PNG and an SVG.
The DOT source is saved next to them.

Why: one picture of which tool does what, generated from code so it is reproducible and can
be regenerated when the pipeline changes. It contains no results.
"""
import os
import shutil
import subprocess

# fill colour per tool family (legend at the bottom of the diagram)
SPARK, STORE, PY, TORCH, VIZ, LATER = "#e8f0fb", "#fdf0e9", "#f3f2ee", "#efeafb", "#fef6e0", "#ffffff"

NODES = [
    # id, tool (bold), what it does, fill
    ("drive", "Google Drive", "AAVSO CSV exports (cosmic_roja/raw, existing)<br/>+ VSX types (VizieR)", PY),
    ("audit", "Python — Step 0", "dataset audit: MB, rows, stars,<br/>50 MB check, header check", PY),
    ("hdfs", "Hadoop HDFS", "/cosmic/raw/&lt;sub-folder&gt;/*.csv", STORE),
    ("ingest", "Spark on YARN — Step 1", "explicit-schema read → Parquet + Snappy<br/>partitioned by obs_year / sky_region", SPARK),
    ("prep", "Spark SQL — Step 2", "dedup · null keys · comment codes<br/>upper limits · thinning · band normalisation", SPARK),
    ("score", "Spark SQL — Step 3", "observer agreement score<br/>leave-one-out consensus self-join → A_k", SPARK),
    ("win", "Spark — Step 5 windowing", "100-day windows (stride 50 d, ≥ 20 obs)<br/>label + auid; 5 star-grouped splits", SPARK),
    ("feat", "Spark — window features", "amplitude · moments · Lomb–Scargle<br/>cadence · observer features", SPARK),
    ("seq", "Spark → Parquet → npz", "token sequences (L = 200, masked)<br/>mag_norm, Δt, band, type, A_k, offset", SPARK),
    ("mllib", "Spark MLlib — Step 7", "Naive Bayes · Decision Tree<br/>Random Forest · Logistic Regression", SPARK),
    ("rga", "PyTorch (Colab GPU) — Step 8", "RGA-Net: reliability-gated attention<br/>+ ablations (no gate, observer-blind)", TORCH),
    ("cmp", "Comparison — Step 9", "same 5 splits for all models; thresholds<br/>from validation; mean ± std on test", PY),
    ("fig", "Plotly", "tables and figures<br/>PNG + interactive HTML", VIZ),
    ("hive", "Hive metastore", "observations_raw (external), obs_clean, observer_scores,<br/>windows, window_features, splits, ml_predictions", STORE),
]
# The second row runs right to left (a "snake" that fits a 16:9 slide): those edges are written
# reversed with dir=back, so Graphviz places them right-to-left but the arrows still follow the data.
BACK = {("prep", "score"), ("score", "win")}
EDGES = [("drive", "audit"), ("audit", "hdfs"), ("hdfs", "ingest"), ("ingest", "prep"), ("prep", "score"),
         ("score", "win"), ("win", "feat"), ("win", "seq"), ("feat", "mllib"), ("seq", "rga"),
         ("mllib", "cmp"), ("rga", "cmp"), ("cmp", "fig")]
HIVE_LINKS = ["ingest", "prep", "score", "win"]
ROWS = [["drive", "audit", "hdfs", "ingest"], ["win", "score", "prep", "hive"], ["feat", "seq"],
        ["mllib", "rga"], ["cmp", "fig", "legend"]]
LEGEND = [("Spark / Spark SQL / MLlib, submitted to YARN", SPARK), ("HDFS / Hive storage", STORE),
          ("Plain Python on the Colab VM", PY), ("PyTorch on the GPU", TORCH), ("Plotly", VIZ)]


def dot_source():
    out = ['digraph cosmic {',
           '  graph [rankdir=TB, newrank=true, nodesep=0.45, ranksep=0.55, bgcolor="white", pad=0.3, fontname="Helvetica",',
           '         labelloc=t, fontsize=22, label=<<b>Cosmic Event Crowdsourcing Platform — pipeline architecture</b>>];',
           '  node  [shape=box, style="rounded,filled", color="#52514e", penwidth=1.2, fontname="Helvetica",',
           '         fontsize=11, fontcolor="#0b0b0b", margin="0.18,0.10", width=3.1];',
           '  edge  [color="#2a78d6", penwidth=1.8, arrowsize=0.8];']
    for nid, tool, what, fill in NODES:
        out.append(f'  {nid} [fillcolor="{fill}", label=<<b><font point-size="13">{tool}</font></b><br/>'
                   f'<font color="#52514e">{what}</font>>];')
    for row in ROWS:
        out.append("  { rank=same; " + "; ".join(row) + "; }")
    for a, b in EDGES:
        out.append(f"  {b} -> {a} [dir=back];" if (a, b) in BACK else f"  {a} -> {b};")
    out.append("  prep -> hive [style=invis];")
    for n in HIVE_LINKS:
        out.append(f'  {n} -> hive [style=dashed, color="#eb6834", arrowsize=0.6, constraint=false];')
    rows = "".join(f'<tr><td bgcolor="{fill}" width="22"> </td><td align="left">{text}</td></tr>' for text, fill in LEGEND)
    out.append('  legend [shape=plaintext, style="", fontsize=11, label=<<table border="0" cellspacing="4">'
               f'{rows}<tr><td></td><td align="left"><font color="#eb6834">orange dashed arrow</font>: table registered in Hive</td></tr>'
               '<tr><td></td><td align="left"><i>Single-node pseudo-distributed cluster on Colab: HDFS, YARN, '
               'Hive metastore.</i></td></tr></table>>];')
    out.append("  fig -> legend [style=invis, minlen=2];")
    out.append("}")
    return "\n".join(out) + "\n"


def draw(out_path, dpi=300):
    """Write <out>.dot and render <out>.png (at `dpi`) and <out>.svg with Graphviz `dot`."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    base = os.path.splitext(out_path)[0]
    with open(base + ".dot", "w", encoding="utf-8") as fh:
        fh.write(dot_source())
    print(f"saved {base}.dot")
    if shutil.which("dot") is None:
        raise RuntimeError("Graphviz `dot` not found. In Colab run:  !apt-get -qq install -y graphviz")
    for fmt, extra in (("png", [f"-Gdpi={dpi}"]), ("svg", [])):
        subprocess.run(["dot", f"-T{fmt}", *extra, base + ".dot", "-o", f"{base}.{fmt}"], check=True)
        print(f"saved {base}.{fmt}")
    return base + ".png"


if __name__ == "__main__":
    draw(os.path.join("docs", "figures", "architecture.png"))
