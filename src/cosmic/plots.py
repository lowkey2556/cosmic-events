"""Plotly figures for the report (no Matplotlib anywhere in the project).

What it does: one function per figure. Every function takes a SMALL pandas table that
Spark (or PyTorch) has already produced, builds a Plotly figure and saves it three ways in
the run's output folder:
  * <name>.png  — static image for the slides (Plotly's `kaleido` engine);
  * <name>.html — interactive version (zoom, hover), opens in any browser;
  * <name>.json — the figure itself, so the Colab notebook can show it inline with
                  `plotly.io.read_json(...).show()`.
If PNG export fails (e.g. kaleido missing), the HTML and JSON are still written, the error is
printed, and the failure is recorded in figure_export_log.csv.

Why: one module and one template ("cosmic") give every figure the same clean look, sized
to stay readable on a projector: large fonts, white background, light grid, a fixed
colour-blind-checked palette. Magnitude axes are always reversed (brighter = up).
"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER = "#b8b7b1"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3de"
SEQ = [[0.0, "#f3f7fd"], [0.35, "#9cc1ee"], [0.7, "#2a78d6"], [1.0, "#123e73"]]
FONT = "Arial, Helvetica, sans-serif"

_axis = dict(showline=True, linecolor=INK2, linewidth=1.5, ticks="outside", tickcolor=INK2, ticklen=6,
             zeroline=False, gridcolor=GRID, gridwidth=1, title=dict(standoff=14))
pio.templates["cosmic"] = go.layout.Template(layout=dict(
    font=dict(family=FONT, size=18, color=INK),
    title=dict(font=dict(size=24, color=INK), x=0.01, xanchor="left", y=0.97, yanchor="top"),
    paper_bgcolor="white", plot_bgcolor="white", colorway=SERIES,
    xaxis=dict(_axis, showgrid=False), yaxis=dict(_axis, showgrid=True),
    legend=dict(bgcolor="rgba(255,255,255,0)", font=dict(size=16)),
    hoverlabel=dict(font=dict(family=FONT, size=15)),
    margin=dict(l=90, r=40, t=110, b=80),
))
pio.templates.default = "plotly_white+cosmic"

# Export settings; run.py passes the `plots` section of settings.yaml to set_export().
EXPORT = dict(png_scale=2, html_plotlyjs=True)


def set_export(png_scale=None, html_plotlyjs=None, **_):
    if png_scale is not None:
        EXPORT["png_scale"] = png_scale
    if html_plotlyjs is not None:
        EXPORT["html_plotlyjs"] = html_plotlyjs


def _log(out_dir, name, png_ok, error=""):
    path = os.path.join(out_dir, "figure_export_log.csv")
    row = pd.DataFrame([dict(figure=name, html=True, json=True, png=png_ok, png_error=error[:300])])
    old = pd.read_csv(path) if os.path.exists(path) else row.iloc[:0]
    pd.concat([old[old["figure"] != name], row], ignore_index=True).to_csv(path, index=False)


def save(fig, out_dir, name, width=1600, height=900):
    """Write <name>.html, <name>.json and (if kaleido works) <name>.png; return the PNG (or HTML) path."""
    name = os.path.splitext(name)[0]
    base = os.path.join(out_dir, name)
    fig.update_layout(width=width, height=height)
    fig.update_annotations(font=dict(family=FONT, size=20, color=INK))  # subplot titles
    fig.write_html(base + ".html", include_plotlyjs=EXPORT["html_plotlyjs"], full_html=True)
    fig.write_json(base + ".json")
    try:
        fig.write_image(base + ".png", width=width, height=height, scale=EXPORT["png_scale"])
    except Exception as exc:  # noqa: BLE001 - any export failure is reported; the HTML is kept
        msg = f"{type(exc).__name__}: {exc}"
        _log(out_dir, name, False, msg)
        print(f"WARNING: PNG export failed for {name} ({msg}). Kept {base}.html and {base}.json")
        return base + ".html"
    _log(out_dir, name, True)
    print(f"saved {base}.png / .html / .json")
    return base + ".png"


def bar_obs_per_star(pdf, out_dir):
    """pdf: columns name, label_name, n — observations per star, coloured by label."""
    pdf = pdf.sort_values("n", ascending=False)
    colours = {"anomalous": SERIES[1], "normal": SERIES[0], "unlabelled": OTHER}
    fig = go.Figure()
    for k, c in colours.items():
        d = pdf[pdf["label_name"] == k]
        fig.add_bar(x=d["name"], y=d["n"], name=k, marker_color=c)
    fig.update_xaxes(categoryorder="array", categoryarray=list(pdf["name"]), tickangle=-90, tickfont=dict(size=13))
    fig.update_yaxes(type="log", title="Observations (log scale)")
    fig.update_layout(title="Observations per star", barmode="overlay", bargap=0.2,
                      legend=dict(orientation="h", x=1, xanchor="right", y=1.06))
    return save(fig, out_dir, "fig_obs_per_star", height=800)


def funnel(pdf, out_dir):
    """pdf: columns step, rows (ordered), optional pct_removed_by_step — rows remaining after each step.
    The thinning step is drawn in a second colour so it stands out as its own step."""
    pct = pdf["pct_removed_by_step"] if "pct_removed_by_step" in pdf else pd.Series([0.0] * len(pdf))
    text = [f"{v:,}" + (f"  (−{p:.1f}%)" if i and p > 0 else "") for i, (v, p) in enumerate(zip(pdf["rows"], pct))]
    colours = [SERIES[1] if "thinning" in s else SERIES[0] for s in pdf["step"]]
    fig = go.Figure(go.Bar(x=pdf["rows"], y=pdf["step"], orientation="h", marker_color=colours,
                           text=text, textposition="outside", cliponaxis=False))
    fig.update_yaxes(autorange="reversed", showgrid=False, automargin=True)
    fig.update_xaxes(title="Rows remaining (in brackets: % removed by that step)",
                     range=[0, pdf["rows"].max() * 1.35], showgrid=True)
    fig.update_layout(title="Preprocessing: rows after each step")
    return save(fig, out_dir, "fig_preprocessing_funnel", height=max(550, 80 * len(pdf) + 220))


def pruning(pdf, out_dir):
    """pdf: columns table, files_read, bytes_read."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Files scanned", "MB scanned"), horizontal_spacing=0.15)
    cols = [SERIES[0], OTHER]
    fig.add_bar(x=pdf["table"], y=pdf["files_read"], marker_color=cols, text=pdf["files_read"].astype(int),
                textposition="outside", cliponaxis=False, row=1, col=1)
    mb = pdf["bytes_read"] / 1e6
    fig.add_bar(x=pdf["table"], y=mb, marker_color=cols, text=[f"{v:,.2f}" for v in mb],
                textposition="outside", cliponaxis=False, row=1, col=2)
    fig.update_layout(title="Partition pruning: the same year-restricted query", showlegend=False)
    return save(fig, out_dir, "fig_partition_pruning", width=1400, height=750)


def lightcurve_before_after(pdf, star_name, top_observers, out_dir):
    """pdf: one star's rows: jd, mag, mag_norm, band, observer. Magnitude axes reversed (brighter = up)."""
    colours = {o: SERIES[i] for i, o in enumerate(sorted(top_observers)[:len(SERIES)])}
    fig = make_subplots(rows=1, cols=2, shared_xaxes=True, horizontal_spacing=0.09,
                        subplot_titles=("Before: raw magnitude (all bands)",
                                        "After: band-normalised (mag − median per band)"))
    groups = [("other observers", pdf[~pdf["observer"].isin(colours)], OTHER, 5)] + \
             [(o, pdf[pdf["observer"] == o], c, 7) for o, c in colours.items()]
    for col, ycol in ((1, "mag"), (2, "mag_norm")):
        for label, d, c, size in groups:
            fig.add_trace(go.Scatter(x=d["jd"] - 2400000, y=d[ycol], mode="markers", name=label, legendgroup=label,
                                     showlegend=col == 1, marker=dict(color=c, size=size, opacity=0.85),
                                     customdata=d[["band", "observer"]],
                                     hovertemplate="JD−2400000 %{x:.3f}<br>%{y:.3f} mag<br>%{customdata[0]}"
                                                   "<br>observer %{customdata[1]}<extra></extra>"),
                          row=1, col=col)
        fig.update_yaxes(autorange="reversed", row=1, col=col)
    fig.update_xaxes(title="JD − 2400000", tickformat="d")
    fig.update_yaxes(title="Magnitude (brighter ↑)", row=1, col=1)
    fig.update_yaxes(title="Normalised magnitude (brighter ↑)", row=1, col=2)
    fig.update_layout(title=f"Light curve of {star_name}, coloured by observer",
                      legend=dict(title="Observer", x=1.01, y=0.5, yanchor="middle"))
    return save(fig, out_dir, "fig_lightcurve_before_after", width=1800, height=800)


def score_histogram(pdf, out_dir):
    """pdf: observer, score, obs_type (dominant), low_n."""
    types = [t for t in ["Visual", "CCD", "DSLR"] if t in set(pdf["obs_type"])]
    groups = [(t, pdf.loc[pdf["obs_type"] == t, "score"], SERIES[i]) for i, t in enumerate(types)]
    rest = pdf.loc[~pdf["obs_type"].isin(types), "score"]
    if len(rest):
        groups.append(("other", rest, OTHER))
    fig = go.Figure()
    for label, s, c in groups:
        fig.add_trace(go.Histogram(x=s, name=label, marker=dict(color=c, line=dict(color="white", width=1)),
                                   xbins=dict(start=0, end=1.0001, size=0.04)))
    fig.update_layout(barmode="stack", legend=dict(title="Dominant observation type"),
                      title=f"Observer agreement scores (n = {len(pdf)} observers; low-n observers set to the median)")
    fig.update_xaxes(title="Observer agreement score A_k (1 = agrees perfectly with peers)", range=[0, 1])
    fig.update_yaxes(title="Observers", rangemode="tozero", tickformat="d")
    return save(fig, out_dir, "fig_observer_score_hist")


def _cm_trace(cm):
    cm = np.asarray(cm, dtype=float)
    labels = ["normal", "anomalous"]
    return go.Heatmap(z=cm, x=labels, y=labels, colorscale=SEQ, zmin=0, zmax=max(cm.max(), 1),
                      text=cm.astype(int).astype(str), texttemplate="%{text}", textfont=dict(size=26),
                      showscale=False, hovertemplate="true %{y}<br>predicted %{x}<br>%{z:d}<extra></extra>")


def confusion_grid(items, out_dir, name, title, cols=None):
    """items: list of (title, 2x2 matrix [[tn, fp], [fn, tp]]). One heatmap per model, same style for all."""
    n = len(items)
    cols = cols or min(n, 5)
    rows = int(np.ceil(n / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[t for t, _ in items],
                        horizontal_spacing=0.07, vertical_spacing=0.22)
    for i, (_, cm) in enumerate(items):
        r, c = divmod(i, cols)
        fig.add_trace(_cm_trace(cm), row=r + 1, col=c + 1)
        fig.update_yaxes(autorange="reversed", showgrid=False, row=r + 1, col=c + 1, title="True" if c == 0 else None)
        fig.update_xaxes(showgrid=False, row=r + 1, col=c + 1, title="Predicted")
    fig.update_layout(title=title)
    return save(fig, out_dir, name, width=max(750, 430 * cols + 160), height=480 * rows + 170)


def confusion(cm, title, out_dir, name):
    return confusion_grid([(title, cm)], out_dir, name, title="Confusion matrix")


def confusion_side_by_side(cm_a, title_a, cm_b, title_b, out_dir, name):
    return confusion_grid([(title_a, cm_a), (title_b, cm_b)], out_dir, name, title="Confusion matrices")


def grouped_metrics(pdf, out_dir, metrics=("precision", "recall", "f1", "balanced_accuracy"),
                    title="Existing observer-agnostic classifiers vs observer-aware models (one seed)",
                    name="fig_model_comparison", std=None, labels=None):
    """pdf: column model + one column per metric. Optional std (same columns): drawn as error bars.
    labels: optional display name per metric (e.g. {"roc_auc": "ROC-AUC"})."""
    fig = go.Figure()
    for i, m in enumerate(metrics):
        err = None if std is None else dict(type="data", array=std[m].fillna(0), visible=True, thickness=1.5, width=5)
        fig.add_bar(x=pdf["model"], y=pdf[m].fillna(0), name=(labels or {}).get(m, m.replace("_", " ")),
                    marker_color=SERIES[i % len(SERIES)], error_y=err)
    fig.update_yaxes(range=[0, 1.1], title="Score (anomalous = positive class)")
    fig.update_layout(title=title, barmode="group", bargap=0.25,
                      legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.15))
    return save(fig, out_dir, name, height=850)


def silhouette(pdf, best_k, out_dir):
    """pdf: k, silhouette."""
    fig = go.Figure(go.Scatter(x=pdf["k"], y=pdf["silhouette"], mode="lines+markers", name="silhouette",
                               line=dict(color=SERIES[0], width=3), marker=dict(size=11)))
    b = pdf[pdf["k"] == best_k]
    fig.add_trace(go.Scatter(x=b["k"], y=b["silhouette"], mode="markers", name=f"chosen k = {best_k}",
                             marker=dict(color=SERIES[1], size=20)))
    fig.update_xaxes(title="k (number of clusters)", dtick=1)
    fig.update_yaxes(title="Silhouette score")
    fig.update_layout(title="K-Means: silhouette by k")
    return save(fig, out_dir, "fig_kmeans_silhouette", width=1200, height=750)


def roc_pr(curves, out_dir, name, title):
    """curves: list of dict(label, fpr, tpr, roc_auc, recall, precision, pr_auc); positive-class prevalence
    as `prevalence` (the no-skill PR line). Both panels use the same 0-1 axes for every model."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ROC curve (test windows)", "Precision-recall curve (test windows)"),
                        horizontal_spacing=0.1)
    for i, c in enumerate(curves):
        col = SERIES[i % len(SERIES)]
        width = 4 if i == 0 else 2.5
        fig.add_trace(go.Scatter(x=c["fpr"], y=c["tpr"], mode="lines", name=f"{c['label']} (AUC {c['roc_auc']:.2f})",
                                 legendgroup=c["label"], line=dict(color=col, width=width)), row=1, col=1)
        fig.add_trace(go.Scatter(x=c["recall"], y=c["precision"], mode="lines", line_shape="hv",
                                 name=f"{c['label']} (AP {c['pr_auc']:.2f})", legendgroup=c["label"],
                                 line=dict(color=col, width=width)), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance", line=dict(color=OTHER, dash="dash")),
                  row=1, col=1)
    if curves:
        prev = curves[0]["prevalence"]
        fig.add_trace(go.Scatter(x=[0, 1], y=[prev, prev], mode="lines", name=f"no skill (prevalence {prev:.2f})",
                                 line=dict(color=OTHER, dash="dot")), row=1, col=2)
    fig.update_xaxes(range=[0, 1], showgrid=True)
    fig.update_yaxes(range=[0, 1.02])
    fig.update_xaxes(title="False positive rate", row=1, col=1)
    fig.update_yaxes(title="True positive rate (recall)", row=1, col=1)
    fig.update_xaxes(title="Recall", row=1, col=2)
    fig.update_yaxes(title="Precision", row=1, col=2)
    fig.update_layout(title=title, legend=dict(x=1.01, y=0.5, yanchor="middle", font=dict(size=14)))
    return save(fig, out_dir, name, width=1900, height=850)


def training_curves(hist, out_dir, name, title):
    """hist: epoch, train_loss, val_loss, val_pr_auc, seed (one model variant). One line per seed."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss (class-weighted BCE)", "Validation PR-AUC"),
                        horizontal_spacing=0.1)
    for i, (seed, h) in enumerate(hist.groupby("seed")):
        col = SERIES[i % len(SERIES)]
        fig.add_trace(go.Scatter(x=h["epoch"], y=h["train_loss"], mode="lines", name=f"seed {seed}: train",
                                 legendgroup=str(seed), line=dict(color=col, width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=h["epoch"], y=h["val_loss"], mode="lines", name=f"seed {seed}: validation",
                                 legendgroup=str(seed), line=dict(color=col, width=2.5, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=h["epoch"], y=h["val_pr_auc"], mode="lines", name=f"seed {seed}: val PR-AUC",
                                 legendgroup=str(seed), showlegend=False, line=dict(color=col, width=2.5)), row=1, col=2)
        b = h.loc[h["val_pr_auc"].idxmax()]
        fig.add_trace(go.Scatter(x=[b["epoch"]], y=[b["val_pr_auc"]], mode="markers", showlegend=False,
                                 legendgroup=str(seed), marker=dict(color=col, size=12, symbol="star")), row=1, col=2)
    fig.update_xaxes(title="Epoch")
    fig.update_yaxes(title="Loss", row=1, col=1)
    fig.update_yaxes(title="PR-AUC (star = best epoch, weights kept)", range=[0, 1.02], row=1, col=2)
    fig.update_layout(title=title, legend=dict(x=1.01, y=0.5, yanchor="middle", font=dict(size=13)))
    return save(fig, out_dir, name, width=1900, height=850)


def scalability(df, cores, out_dir):
    """df: executors, time_mean, time_std, speedup. Time with error bars, and speedup vs ideal."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Run time (mean ± std of the runs)", "Speedup vs 1 executor"),
                        horizontal_spacing=0.12)
    x = [str(e) for e in df["executors"]]
    fig.add_bar(x=x, y=df["time_mean"], marker_color=SERIES[0], name="time",
                error_y=dict(type="data", array=df["time_std"].fillna(0), visible=True, thickness=1.5, width=6),
                text=[f"{v:.0f} s" for v in df["time_mean"]], textposition="outside", cliponaxis=False, row=1, col=1)
    fig.add_trace(go.Scatter(x=df["executors"], y=df["speedup"], mode="lines+markers", name="measured",
                             line=dict(color=SERIES[1], width=3), marker=dict(size=12)), row=1, col=2)
    fig.add_trace(go.Scatter(x=df["executors"], y=df["executors"] / df["executors"].min(), mode="lines", name="ideal (linear)",
                             line=dict(color=OTHER, dash="dash")), row=1, col=2)
    fig.update_xaxes(title="Spark executors on YARN", row=1, col=1)
    fig.update_xaxes(title="Spark executors on YARN", tickvals=list(df["executors"]), row=1, col=2)
    fig.update_yaxes(title="Seconds", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title="Speedup", rangemode="tozero", row=1, col=2)
    fig.update_layout(title=f"Scalability: Spark preprocessing + windowing through YARN ({cores} CPU cores)",
                      legend=dict(x=1.01, y=0.5))
    return save(fig, out_dir, "fig_scalability", width=1700, height=800)
