"""Architecture diagram (Review 2, slide 3), drawn with Matplotlib at 300 dpi.

What it does: draws the data flow AAVSO CSVs → HDFS → Spark SQL preprocessing →
Hive tables → observer scoring → feature engineering → MLlib (K-Means / NB / DT)
→ Matplotlib + Tableau, labelling the tool used at each stage.

Why: one picture of which Big Data tool does what, generated from code so it is
reproducible and can be regenerated if the pipeline changes. It contains no results.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from .plots import INK, INK2, SERIES  # noqa: E402

STAGES = [
    ("AAVSO CSV exports\n+ VSX labels (VizieR)", "Google Drive", "#f3f2ee"),
    ("Raw storage\nParquet + Snappy, partitioned\nobs_year / sky_region", "Hadoop HDFS", "#e8f0fb"),
    ("Preprocessing\ndedup · null keys · comment codes\nupper limits · band normalisation", "Spark SQL (on YARN)", "#e8f0fb"),
    ("Tables: observations_raw (external),\nobs_clean, observer_scores,\nstar_features, model_metrics", "Hive metastore", "#fdf0e9"),
    ("Observer agreement score\nleave-one-out consensus self-join\nσ_k (MAD) → A_k = exp(−σ_k/σ_ref)", "PySpark / Spark SQL", "#e8f0fb"),
    ("Feature engineering (1 row / star)\namplitude · moments · Lomb–Scargle\ncadence · W_s · observers · offsets", "PySpark applyInPandas", "#e8f0fb"),
    ("Models\nK-Means · Naive Bayes · Decision Tree\n(+ LR, RF for comparison)", "Spark MLlib", "#e7f6ef"),
    ("Figures · tables · dashboards", "Matplotlib + Tableau", "#fef6e0"),
]


def draw(out_path, dpi=300):
    fig, ax = plt.subplots(figsize=(14, 5.4))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 2.25)
    ax.axis("off")
    # two rows of four, snake order: left→right on top, right→left on bottom
    pos = [(0.5, 1.65), (1.5, 1.65), (2.5, 1.65), (3.5, 1.65), (3.5, 0.55), (2.5, 0.55), (1.5, 0.55), (0.5, 0.55)]
    w, h = 0.84, 0.8
    for (title, tool, fill), (x, y) in zip(STAGES, pos):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.04",
                                    facecolor=fill, edgecolor=INK2, linewidth=1))
        ax.text(x, y + h / 2 - 0.1, tool, ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)
        ax.text(x, y - 0.06, title, ha="center", va="center", fontsize=7.8, color=INK2, linespacing=1.4)
    for (x0, y0), (x1, y1) in zip(pos[:-1], pos[1:]):
        if y0 == y1:
            s = 1 if x1 > x0 else -1
            a, b = (x0 + s * w / 2, y0), (x1 - s * w / 2, y1)
        else:
            a, b = (x0, y0 - h / 2), (x1, y1 + h / 2)
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color=SERIES[0], linewidth=1.6))
    ax.text(2.0, 2.2, "Cosmic Event Crowdsourcing Platform — pipeline architecture",
            ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
    ax.text(2.0, 0.04, "Single-node pseudo-distributed cluster on Google Colab: HDFS (NameNode + DataNode), "
            "YARN (ResourceManager + NodeManager), Hive metastore on Derby network server; "
            "the same spark-submit runs unchanged on a multi-node cluster.",
            ha="center", va="center", fontsize=7.5, color=INK2, style="italic")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_path}")
    return out_path


if __name__ == "__main__":
    draw(os.path.join("docs", "figures", "architecture.png"))
