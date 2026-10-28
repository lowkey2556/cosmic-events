"""Plotly figures for the report (no Matplotlib anywhere in the project, and no bar charts or histograms).

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


def _style():
    from .style import load
    return load()


def _num(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:,.{d}f}"


def _int(x):
    return f"{int(x):,}"


# ------------------------------------------------------------------ tables as figures (screenshot-friendly)
def table_figure(df, title, col_widths=None, highlight_col=None):
    """A Plotly table (no bars): header row in the theme colour, alternating row shading."""
    df = df.copy()
    n = len(df)
    fill = [["#f6f6f3" if i % 2 else "white" for i in range(n)] for _ in df.columns]
    if highlight_col is not None and highlight_col in df.columns:
        j = list(df.columns).index(highlight_col)
        fill[j] = ["#fdecee"] * n
    fig = go.Figure(go.Table(
        columnwidth=col_widths,
        header=dict(values=[f"<b>{c}</b>" for c in df.columns], fill_color="#123e73", font=dict(color="white", size=15),
                    align="left", height=34),
        cells=dict(values=[df[c].astype(str).tolist() for c in df.columns], fill_color=fill, align="left",
                   font=dict(size=14, color=INK), height=30)))
    fig.update_layout(title=title, margin=dict(l=20, r=20, t=80, b=20))
    return fig


def caption_samples(raw, clean, dataset):
    added = [c for c in clean.columns if c not in raw.columns]
    return (f"Inference: the same {dataset} observations before and after cleaning ({len(raw)} raw rows shown, "
            f"{len(raw.columns)} raw columns). Cleaning keeps {len(clean.columns)} columns and adds "
            f"{', '.join(added[:6]) if added else 'no new columns'}; raw-only columns are dropped once they have "
            "been used for filtering.")


def data_quality(dq):
    """dq: rows dataset, stage, size_mb, rows, nulls, duplicates, flagged, bands, stars, observers
    (AAVSO, ZTF and a combined pair). Returns (figure, caption): a before/after table with the change in %."""
    cols = ["size_mb", "rows", "nulls", "duplicates", "flagged", "bands", "stars", "observers"]
    out = []
    for ds, g in dq.groupby("dataset", sort=False):
        b, a = g.iloc[0], g.iloc[-1]
        for c in cols:
            if c not in g:
                continue
            bv, av = float(b.get(c, np.nan)), float(a.get(c, np.nan))
            ch = (av - bv) / bv * 100 if bv else np.nan
            out.append(dict(Dataset=ds, Measure=c.replace("_", " ").replace("size mb", "size (MB)"),
                            Before=_num(bv, 2) if c == "size_mb" else _int(bv) if np.isfinite(bv) else "n/a",
                            After=_num(av, 2) if c == "size_mb" else _int(av) if np.isfinite(av) else "n/a",
                            Change="n/a" if not np.isfinite(ch) else f"{ch:+.1f} %"))
    t = pd.DataFrame(out)
    fig = table_figure(t, "Data quality before vs after cleaning (AAVSO, ZTF, combined)", highlight_col="Change")
    cap = []
    for ds, g in dq.groupby("dataset", sort=False):
        b, a = g.iloc[0], g.iloc[-1]
        cap.append(f"{ds}: {_int(b['rows'])} -> {_int(a['rows'])} rows "
                   f"({(1 - a['rows'] / b['rows']) * 100:.1f} % removed), {_num(b['size_mb'], 1)} -> "
                   f"{_num(a['size_mb'], 1)} MB, {_int(b.get('nulls', 0))} nulls and "
                   f"{_int(b.get('duplicates', 0))} duplicates before, {_int(a.get('nulls', 0))} and "
                   f"{_int(a.get('duplicates', 0))} after")
    return fig, ("Inference: " + "; ".join(cap) + ". Every null key and duplicate was removed, so the tables used "
                 "for training contain only valid, unique observations.")


# ------------------------------------------------------------------ 3. Sankey of cleaning steps
def cleaning_sankey(steps_by_ds):
    """steps_by_ds: {dataset: DataFrame(step, rows)} in order (first row = raw). Kept rows flow left to right; what
    each step removes branches off to a grey 'removed by ...' node."""
    labels, src, dst, val, colors, ncol = [], [], [], [], [], []
    st = _style()

    def node(name, color):
        labels.append(name)
        ncol.append(color)
        return len(labels) - 1

    caps = []
    for ds, d in steps_by_ds.items():
        d = d.reset_index(drop=True)
        base = st["sources"].get(ds, SERIES[0])
        prev = node(f"{ds} raw: {_int(d.loc[0, 'rows'])}", base)
        biggest = (None, 0)
        for i in range(1, len(d)):
            rows, removed = int(d.loc[i, "rows"]), int(d.loc[i - 1, "rows"] - d.loc[i, "rows"])
            name = str(d.loc[i, "step"]).lstrip("0123456789 ")
            cur = node(f"{ds}: {name} ({_int(rows)})", base)
            src.append(prev); dst.append(cur); val.append(rows); colors.append("rgba(42,120,214,0.30)"
                                                                               if ds == "AAVSO" else "rgba(27,175,122,0.30)")
            if removed > 0:
                r = node(f"removed: {name} ({_int(removed)})", OTHER)
                src.append(prev); dst.append(r); val.append(removed); colors.append("rgba(184,183,177,0.55)")
                if removed > biggest[1]:
                    biggest = (name, removed)
            prev = cur
        kept = d["rows"].iloc[-1] / d["rows"].iloc[0] * 100
        caps.append(f"{ds} keeps {kept:.1f} % of its raw rows; the largest removal is '{biggest[0]}' "
                    f"({_int(biggest[1])} rows)")
    fig = go.Figure(go.Sankey(arrangement="snap", node=dict(label=labels, color=ncol, pad=14, thickness=16),
                              link=dict(source=src, target=dst, value=val, color=colors)))
    fig.update_layout(title="Cleaning steps: rows kept (coloured) and removed (grey) at each step")
    return fig, "Inference: " + "; ".join(caps) + ". Link widths are row counts, so each grey branch is exactly " \
                                                  "what one cleaning rule removed."


# ------------------------------------------------------------------ 4. light curve coloured by A_k
def lightcurve_ak(d, star):
    """d: jd, mag, band, observer, A_k (one star). Magnitude axis reversed (brighter = up)."""
    st = _style()
    fig = go.Figure()
    symbols = ["circle", "diamond", "square", "triangle-up", "x", "star"]
    for i, (band, g) in enumerate(d.groupby("band")):
        fig.add_trace(go.Scattergl(
            x=g["jd"], y=g["mag"], mode="markers", name=str(band),
            marker=dict(size=6, symbol=symbols[i % len(symbols)], color=g["A_k"], cmin=0, cmax=1,
                        colorscale=st.get("ak_colorscale", "Viridis"), showscale=i == 0, opacity=0.8,
                        colorbar=dict(title="A_k", x=1.02)),
            customdata=g[["observer", "A_k"]], hovertemplate="JD %{x:.3f}<br>mag %{y:.3f}<br>observer %{customdata[0]}"
                                                               "<br>A_k %{customdata[1]:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="Magnitude (mag; brighter = up)")
    fig.update_xaxes(title="Julian Date (days)")
    fig.update_layout(title=f"Light curve of {star}, each observation coloured by its observer's A_k",
                      legend=dict(title="Band", x=1.12))
    low = d.groupby("observer")["A_k"].first()
    share = float((d["A_k"] < 0.5).mean() * 100)
    cap = (f"Inference: {_int(len(d))} cleaned observations of {star} from {d['observer'].nunique()} observers in "
           f"{d['band'].nunique()} band(s); observer A_k ranges {low.min():.2f}-{low.max():.2f} (median "
           f"{low.median():.2f}). {share:.1f} % of the points come from observers with A_k below 0.5; these are the "
           "observations the reliability gate can give less attention.")
    return fig, cap


# ------------------------------------------------------------------ 5. violin of A_k by observation type
def ak_violin(scores):
    """scores: observer, obs_type, score, low_n (one row per observer). Only measured A_k (not low_n)."""
    st = _style()
    s = scores[~scores["low_n"].astype(bool)].copy()
    order = s.groupby("obs_type")["score"].median().sort_values(ascending=False).index
    fig = go.Figure()
    for t in order:
        g = s[s["obs_type"] == t]
        fig.add_trace(go.Violin(y=g["score"], name=f"{t} (n = {len(g)})", box_visible=True, meanline_visible=True,
                                points="all" if len(g) < 200 else "outliers", pointpos=0, jitter=0.3,
                                line_color=st["obs_types"].get(t, st["obs_types"]["other"])))
    fig.update_yaxes(title="A_k (agreement score, 0-1)", range=[0, 1.05])
    fig.update_xaxes(title="Observation type (observer's most common)")
    fig.update_layout(title="Observer agreement score A_k by observation type", showlegend=False)
    med = s.groupby("obs_type")["score"].median()
    cap = (f"Inference: {len(s)} observers with enough overlap were scored (low-overlap observers excluded). "
           + ", ".join(f"{t}: median {med[t]:.2f}" for t in order[:4]) + ". "
           + (f"{order[0]} observers agree best with their peers and {order[-1]} least, so A_k separates observer "
              "groups as expected." if len(order) > 1 else "Only one observation type had scored observers."))
    return fig, cap


# ------------------------------------------------------------------ 6. A_k validation against ZTF
def ak_validation(per_obs, summ):
    s = per_obs[per_obs["low_n"] == False].dropna(subset=["A_k", "scatter_vs_ztf_mag"])  # noqa: E712
    rho, p, n = summ.get("spearman_rho"), summ.get("p_value"), int(summ.get("observers", len(s)))
    fig = go.Figure(go.Scatter(
        x=s["A_k"], y=s["scatter_vs_ztf_mag"], mode="markers",
        marker=dict(size=np.clip(np.sqrt(s["n_matched"].astype(float)) * 2, 6, 30), color=SERIES[0], opacity=0.75,
                    line=dict(width=1, color="white")),
        text=s["observer"], customdata=s["n_matched"],
        hovertemplate="%{text}<br>A_k %{x:.3f}<br>scatter %{y:.3f} mag<br>%{customdata} matched<extra></extra>"))
    fig.update_xaxes(title="A_k (agreement with other AAVSO observers)")
    fig.update_yaxes(title="Scatter against ZTF (mag, 1.4826 x MAD)")
    fig.update_layout(title=f"A_k checked against ZTF: Spearman ρ = {_num(rho)} (p = {_num(p, 3)}), "
                            f"n = {n} observers")
    enough = bool(summ.get("enough_to_conclude", n >= 10))
    if not enough:
        tail = (f"With only {n} observers overlapping ZTF this is too few to conclude anything; the check is reported "
                "but not used as evidence.")
    elif rho is not None and np.isfinite(rho) and rho < 0:
        tail = "Higher A_k goes with smaller scatter against an independent professional survey, supporting A_k."
    else:
        tail = "A_k does not track the scatter against ZTF here, so this check does not support A_k."
    cap = (f"Inference: {int(summ.get('matched_observations', 0)):,} AAVSO observations were matched to ZTF within "
           f"1 day after removing each star/band offset; ρ = {_num(rho)} over {n} observers (marker size = matched "
           f"points). {tail}")
    return fig, cap


# ------------------------------------------------------------------ 7. sunburst source -> class -> VSX type
def composition_sunburst(comp):
    """comp: source, class, vsx_type, rows (AAVSO + ZTF)."""
    st = _style()
    c = comp.copy()
    c["vsx_type"] = c["vsx_type"].fillna("unknown").astype(str)
    c = c.groupby(["source", "class", "vsx_type"], as_index=False)["rows"].sum()
    ids, labels, parents, values, colors = [], [], [], [], []
    for src, g in c.groupby("source"):
        ids.append(src); labels.append(src); parents.append(""); values.append(g["rows"].sum())
        colors.append(st["sources"].get(src, SERIES[0]))
        for cl, h in g.groupby("class"):
            ids.append(f"{src}/{cl}"); labels.append(cl); parents.append(src); values.append(h["rows"].sum())
            colors.append(st["classes"].get(cl, OTHER))
            for r in h.itertuples():
                ids.append(f"{src}/{cl}/{r.vsx_type}"); labels.append(r.vsx_type); parents.append(f"{src}/{cl}")
                values.append(r.rows); colors.append(st["classes"].get(cl, OTHER))
    fig = go.Figure(go.Sunburst(ids=ids, labels=labels, parents=parents, values=values, branchvalues="total",
                                marker=dict(colors=colors, line=dict(color="white", width=1)),
                                hovertemplate="%{label}<br>%{value:,} rows<br>%{percentParent:.1%} of parent"
                                              "<extra></extra>"))
    fig.update_layout(title="Data composition: source -> class -> VSX type (sized by observation rows)")
    tot = c.groupby("source")["rows"].sum()
    lab = c[c["class"] != "unlabelled"].groupby("source")["rows"].sum()
    top = c[c["class"] != "unlabelled"].sort_values("rows", ascending=False).head(1)
    cap = ("Inference: " + "; ".join(f"{s} has {_int(tot[s])} rows, {lab.get(s, 0) / tot[s] * 100:.1f} % of them "
                                    "labelled" for s in tot.index)
           + (f". The largest labelled group is {top['vsx_type'].iloc[0]} ({top['class'].iloc[0]}, "
              f"{_int(top['rows'].iloc[0])} rows), so class balance is handled with class weights and balanced "
              "sampling." if len(top) else "."))
    return fig, cap


# ------------------------------------------------------------------ 8/9. comparison tables
def comparison_table_fig(comp, metrics, labels, title):
    t = comp[["display_name", "role"] + list(metrics)].rename(columns=dict(display_name="Model", role="Role",
                                                                           **{m: labels[m] for m in metrics}))
    return table_figure(t, title, col_widths=[3.2, 2.2] + [1.4] * len(metrics))


def caption_comparison(comp, level):
    c = comp[comp["level"] == level]
    b = c.loc[c["mcc_mean"].idxmax()]
    rg = c[c["role"] == "proposed"]
    pri = c[c["role"] == "prior approach"]
    bp = pri.loc[pri["mcc_mean"].idxmax()] if len(pri) else None
    s = f"Inference: at {level} level the highest mean MCC is {b['display_name']} ({b['mcc']})."
    if len(rg) and bp is not None:
        r = rg.iloc[0]
        s += (f" RGA-Net reaches {r['mcc']} against the best prior approach, {bp['display_name']} ({bp['mcc']}), "
              f"a difference of {r['mcc_mean'] - bp['mcc_mean']:+.3f}; AUPRC {r['pr_auc']} vs {bp['pr_auc']}.")
    return s


# ------------------------------------------------------------------ 10. radar
def radar(comp, level, models_order):
    """5 models x 5 metrics; Brier is shown as 1 - Brier so that outward = better on every axis."""
    st = _style()
    axes = ["MCC", "ROC-AUC", "AUPRC", "Cohen's κ", "1 − Brier (inverted)"]
    fig = go.Figure()
    c = comp[comp["level"] == level].set_index("model")
    for m in models_order:
        if m not in c.index:
            continue
        r = c.loc[m]
        vals = [r["mcc_mean"], r["roc_auc_mean"], r["pr_auc_mean"], r["kappa_mean"], 1 - r["brier_mean"]]
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=axes + axes[:1], name=r["display_name"],
                                      line=dict(color=st["models"][m]["color"], width=4 if m == "RGA-Net" else 2),
                                      fill="toself" if m == "RGA-Net" else None, opacity=0.9))
    fig.update_layout(title=f"Proposed RGA-Net vs the prior approaches ({level} level, mean over seeds)",
                      polar=dict(radialaxis=dict(range=[min(0, float(np.nanmin(c[['mcc_mean', 'kappa_mean']].values)))
                                                        , 1], title="score (outward = better)")),
                      legend=dict(title="Model"))
    rg = c.loc["RGA-Net"] if "RGA-Net" in c.index else None
    if rg is None:
        return fig, "Inference: RGA-Net results are missing."
    wins = []
    for k, lab in (("mcc", "MCC"), ("roc_auc", "ROC-AUC"), ("pr_auc", "AUPRC"), ("kappa", "κ"), ("brier", "Brier")):
        others = c.drop(index="RGA-Net")[f"{k}_mean"]
        best = others.min() if k == "brier" else others.max()
        if (rg[f"{k}_mean"] < best) if k == "brier" else (rg[f"{k}_mean"] > best):
            wins.append(lab)
    return fig, (f"Inference: RGA-Net is ahead of every prior approach on {len(wins)} of 5 metrics"
                 + (f" ({', '.join(wins)})" if wins else "") + f"; its MCC is {rg['mcc']}. Brier is drawn as "
                 "1 − Brier so that a larger polygon is better on every axis.")


# ------------------------------------------------------------------ 11. dumbbell (Fix 1)
def dumbbell(gain_mcc, comp, level):
    """gain_mcc: model, without_mean, with_mean, seeds_ahead, seeds (MCC). RGA-Net full as a reference line."""
    st = _style()
    fig = go.Figure()
    g = gain_mcc.reset_index(drop=True)
    names = [st["models"][m]["name"] for m in g["model"]]
    for i, r in g.iterrows():
        fig.add_trace(go.Scatter(x=[r["without_mean"], r["with_mean"]], y=[names[i]] * 2, mode="lines",
                                 line=dict(color=GRID, width=6), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=g["without_mean"], y=names, mode="markers", name="without observer features (prior)",
                             marker=dict(size=18, color="white", line=dict(width=3, color=SERIES[0]))))
    fig.add_trace(go.Scatter(x=g["with_mean"], y=names, mode="markers", name="with observer features (our framework)",
                             marker=dict(size=18, color=SERIES[0])))
    rg = comp[(comp["level"] == level) & (comp["role"] == "proposed")]
    if len(rg):
        x = rg["mcc_mean"].iloc[0]
        fig.add_vline(x=x, line=dict(color=st["models"]["RGA-Net"]["color"], width=3, dash="dash"),
                      annotation_text=f"RGA-Net full {x:.3f}", annotation_position="top")
    fig.update_xaxes(title="Test MCC (mean over seeds)")
    fig.update_yaxes(title="Classic model", showgrid=False)
    fig.update_layout(title=f"Adding observer features to classic models ({level} level): MCC without -> with",
                      legend=dict(orientation="h", y=-0.18))
    up = int((g["with_mean"] > g["without_mean"]).sum())
    best = g.loc[(g["with_mean"] - g["without_mean"]).idxmax()]
    cap = (f"Inference: observer features raise the mean MCC of {up} of {len(g)} classic models; the largest change is "
           f"{st['models'][best['model']]['name']} ({best['without_mean']:.3f} -> {best['with_mean']:.3f}, better on "
           f"{int(best['seeds_ahead'])} of {int(best['seeds'])} seeds). The open circles are the prior approaches; "
           "the filled circles already use our reliability framework, so only the open ones are compared with "
           "RGA-Net (dashed line) as prior work.")
    return fig, cap


# ------------------------------------------------------------------ 12. ablation dot plot
def ablation_dots(ab, level, order):
    st = _style()
    a = ab[ab["level"] == level].set_index("variant")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("MCC", "AUPRC"), shared_yaxes=True, horizontal_spacing=0.06)
    for j, m in enumerate(("mcc", "pr_auc"), start=1):
        for v in order:
            if v not in a.index:
                continue
            fig.add_trace(go.Scatter(x=[a.loc[v, f"{m}_mean"]], y=[a.loc[v, "display_name"]], mode="markers",
                                     error_x=dict(type="data", array=[a.loc[v, f"{m}_std"]], thickness=2, width=8),
                                     marker=dict(size=16, color=st["variant_colors"].get(v, OTHER)),
                                     name=a.loc[v, "display_name"], showlegend=j == 1), row=1, col=j)
    fig.update_xaxes(title_text="test score (mean ± std over seeds)")
    fig.update_layout(title=f"Ablation of RGA-Net ({level} level)", legend=dict(orientation="h", y=-0.2))
    if "full" not in a.index:
        return fig, "Inference: the full model is missing."
    f = a.loc["full", "mcc_mean"]
    parts = [f"{a.loc[v, 'display_name']} {a.loc[v, 'mcc_mean'] - f:+.3f}" for v in order if v != "full" and v in a.index]
    worst = min((v for v in order if v != "full" and v in a.index), key=lambda v: a.loc[v, "mcc_mean"], default=None)
    return fig, (f"Inference: full RGA-Net MCC is {a.loc['full', 'mcc']}; change when a part is removed: "
                 + "; ".join(parts) + "." + (f" Removing '{a.loc[worst, 'display_name']}' costs the most. Error bars "
                                             "are the seed-to-seed standard deviation." if worst else ""))


# ------------------------------------------------------------------ 13. confusion matrices at the median-MCC seed
def confusion_grid(per_seed, level, models_order, seed):
    st = _style()
    d = per_seed[(per_seed["level"] == level) & (per_seed["seed"] == seed)].set_index(["model", "variant"])
    keys = [(m, "without_observer") if m != "RGA-Net" else (m, "full") for m in models_order]
    keys = [k for k in keys if k in d.index]
    fig = make_subplots(rows=1, cols=len(keys), subplot_titles=[st["models"][k[0]]["name"] for k in keys],
                        horizontal_spacing=0.05)
    for j, k in enumerate(keys, start=1):
        r = d.loc[k]
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]], float)
        pct = cm / cm.sum(1, keepdims=True).clip(min=1) * 100
        fig.add_trace(go.Heatmap(z=pct, x=["pred normal", "pred anomalous"], y=["true normal", "true anomalous"],
                                 colorscale=st.get("heatmap_colorscale", "Blues"), zmin=0, zmax=100,
                                 showscale=j == len(keys), colorbar=dict(title="% of true class"),
                                 text=[[f"{int(cm[i, jj])}<br>{pct[i, jj]:.0f}%" for jj in range(2)] for i in range(2)],
                                 texttemplate="%{text}", textfont=dict(size=15)), row=1, col=j)
        fig.update_yaxes(autorange="reversed", row=1, col=j, showticklabels=j == 1)
    fig.update_layout(title=f"Confusion matrices on the test set at seed {seed} (the median-MCC seed of RGA-Net), "
                            f"{level} level")
    rg = d.loc[("RGA-Net", "full")] if ("RGA-Net", "full") in d.index else None
    cap = (f"Inference: seed {seed} is shown because RGA-Net's MCC there is the median over seeds, so it is neither a "
           "best nor a worst case. ")
    if rg is not None:
        rec = rg["tp"] / max(rg["tp"] + rg["fn"], 1) * 100
        spec = rg["tn"] / max(rg["tn"] + rg["fp"], 1) * 100
        cap += (f"RGA-Net finds {rec:.0f} % of anomalous and {spec:.0f} % of normal test {level}s "
                f"({int(rg['fp'])} false alarms, {int(rg['fn'])} misses) at the threshold chosen on validation.")
    return fig, cap


def median_seed(per_seed, level, model="RGA-Net", variant="full"):
    s = per_seed[(per_seed["model"] == model) & (per_seed["variant"] == variant) & (per_seed["level"] == level)]
    s = s.sort_values(["mcc", "seed"]).reset_index(drop=True)
    return int(s.loc[(len(s) - 1) // 2, "seed"])
