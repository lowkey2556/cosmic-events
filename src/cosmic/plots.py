"""Matplotlib figures for the report.

What it does: one function per figure. Every function takes a SMALL pandas table
that Spark has already aggregated (or one star's rows), draws it, and saves a
300-dpi PNG into the run's output folder.

Why: keeping all plotting in one module gives every figure the same look
(a fixed, colour-blind-checked palette; magnitudes always plotted with the
y-axis inverted because lower magnitude = brighter) and keeps Spark code free
of plotting details.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER = "#b8b7b1"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3de"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f3f7fd", "#9cc1ee", "#2a78d6", "#123e73"])

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 100, "savefig.bbox": "tight",
})


def _save(fig, out_dir, name, dpi=300):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"saved {path}")
    return path


def bar_obs_per_star(pdf, out_dir):
    """pdf: columns name, label_name, n — observations per star, coloured by label."""
    pdf = pdf.sort_values("n", ascending=False)
    colours = {"anomalous": SERIES[1], "normal": SERIES[0], "unlabelled": OTHER}
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(range(len(pdf)), pdf["n"], color=[colours[c] for c in pdf["label_name"]], width=0.75)
    ax.set_xticks(range(len(pdf)))
    ax.set_xticklabels(pdf["name"], rotation=90, fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("Observations (log scale)")
    ax.set_title("Observations per star")
    ax.grid(axis="x", visible=False)
    for k, c in colours.items():
        ax.bar([0], [0], color=c, label=k)
    ax.legend(loc="upper right", ncol=3)
    return _save(fig, out_dir, "fig_obs_per_star.png")


def funnel(pdf, out_dir):
    """pdf: columns step, rows (ordered)."""
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(pdf) + 1))
    y = np.arange(len(pdf))[::-1]
    ax.barh(y, pdf["rows"], color=SERIES[0], height=0.6)
    for yi, v in zip(y, pdf["rows"]):
        ax.text(v, yi, f" {v:,}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(pdf["step"])
    ax.set_xlabel("Rows remaining")
    ax.set_title("Preprocessing: rows after each step")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, pdf["rows"].max() * 1.18)
    return _save(fig, out_dir, "fig_preprocessing_funnel.png")


def pruning(pdf, out_dir):
    """pdf: columns table, files_read, bytes_read."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    for ax, col, title in [(axes[0], "files_read", "Files scanned"), (axes[1], "bytes_read", "MB scanned")]:
        vals = pdf[col] / (1e6 if col == "bytes_read" else 1)
        ax.bar(pdf["table"], vals, color=[SERIES[0], OTHER], width=0.55)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:,.2f}" if col == "bytes_read" else f"{int(v)}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Partition pruning: same year-restricted query", fontsize=10)
    return _save(fig, out_dir, "fig_partition_pruning.png")


def _observer_colours(observers):
    return {o: SERIES[i] for i, o in enumerate(sorted(observers)[:len(SERIES)])}


def lightcurve_before_after(pdf, star_name, top_observers, out_dir):
    """pdf: one star's rows: jd, mag, mag_norm, band, observer."""
    colours = _observer_colours(top_observers)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
    for ax, col, title in [(axes[0], "mag", "Before: raw magnitude (all bands)"),
                           (axes[1], "mag_norm", "After: band-normalised (mag − median per band)")]:
        other = pdf[~pdf["observer"].isin(colours)]
        ax.scatter(other["jd"] - 2400000, other[col], s=6, color=OTHER, label="other observers", linewidths=0)
        for o, c in colours.items():
            d = pdf[pdf["observer"] == o]
            ax.scatter(d["jd"] - 2400000, d[col], s=8, color=c, label=o, linewidths=0)
        ax.invert_yaxis()
        ax.set_xlabel("JD − 2400000")
        ax.set_ylabel("Magnitude (brighter ↑)" if col == "mag" else "Normalised magnitude (brighter ↑)")
        ax.set_title(title)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, markerscale=1.5)
    fig.suptitle(f"Light curve of {star_name}, coloured by observer", fontsize=10)
    return _save(fig, out_dir, "fig_lightcurve_before_after.png")


def score_histogram(pdf, out_dir):
    """pdf: observer, score, obs_type (dominant), low_n."""
    types = [t for t in ["Visual", "CCD", "DSLR"] if t in set(pdf["obs_type"])]
    groups = [pdf.loc[pdf["obs_type"] == t, "score"] for t in types]
    labels = list(types)
    rest = pdf.loc[~pdf["obs_type"].isin(types), "score"]
    colours = SERIES[:len(types)]
    if len(rest):
        groups.append(rest)
        labels.append("other")
        colours.append(OTHER)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.hist(groups, bins=np.linspace(0, 1, 26), stacked=True, color=colours, label=labels,
            edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Observer agreement score  A_k  (1 = agrees perfectly with peers)")
    ax.set_ylabel("Observers")
    ax.set_title(f"Observer agreement scores (n = {len(pdf)} observers; low-n observers set to the median)")
    ax.legend(title="Dominant observation type")
    ax.grid(axis="x", visible=False)
    return _save(fig, out_dir, "fig_observer_score_hist.png")


def _cm_axes(ax, cm, title):
    ax.imshow(cm, cmap=SEQ, vmin=0, vmax=max(cm.max(), 1))
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(int(v)), ha="center", va="center", fontsize=12,
                color="white" if v > cm.max() * 0.6 else INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["normal", "anomalous"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["normal", "anomalous"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontsize=9)
    ax.grid(False)


def confusion(cm, title, out_dir, name):
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    _cm_axes(ax, np.asarray(cm), title)
    return _save(fig, out_dir, name)


def confusion_side_by_side(cm_a, title_a, cm_b, title_b, out_dir, name):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    _cm_axes(axes[0], np.asarray(cm_a), title_a)
    _cm_axes(axes[1], np.asarray(cm_b), title_b)
    return _save(fig, out_dir, name)


def grouped_metrics(pdf, out_dir):
    """pdf: model, precision, recall, f1, balanced_accuracy."""
    metrics = ["precision", "recall", "f1", "balanced_accuracy"]
    x = np.arange(len(pdf))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1.5) * w, pdf[m].fillna(0), width=w * 0.92, color=SERIES[i], label=m.replace("_", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(pdf["model"], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score on test stars (anomalous = positive class)")
    ax.set_title("Existing observer-agnostic classifiers vs observer-aware models (one seed)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="x", visible=False)
    return _save(fig, out_dir, "fig_model_comparison.png")


def silhouette(pdf, best_k, out_dir):
    """pdf: k, silhouette."""
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(pdf["k"], pdf["silhouette"], color=SERIES[0], linewidth=2, marker="o", markersize=6)
    b = pdf[pdf["k"] == best_k]
    ax.scatter(b["k"], b["silhouette"], s=90, color=SERIES[1], zorder=3, label=f"chosen k = {best_k}")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("K-Means: silhouette by k")
    ax.legend()
    return _save(fig, out_dir, "fig_kmeans_silhouette.png")
