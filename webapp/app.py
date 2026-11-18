"""Cosmic Event Crowdsourcing Platform: interactive web dashboard (Streamlit + Plotly).

Reads only the tidy tables the pipeline's `dashboard` step wrote (dashboard_data/: CSV), so nothing is retrained
or recomputed beyond simple means. Charts follow the project rule: no bar, pie or line charts.

Run:  streamlit run app.py
Data: webapp/dashboard_data/ (copied from V3_ROOT/dashboard_data), or the COSMIC_DASHBOARD_DATA environment variable.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = os.environ.get("COSMIC_DASHBOARD_DATA", str(HERE / "dashboard_data"))

RED, BLUE, GREEN, GOLD, PURPLE, GREY = "#d1495b", "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#9a9a94"
ROLE_COLOR = {"proposed": RED, "prior approach": BLUE, "our framework on a classic model": GREEN, "ablation": GOLD}
CLASS_COLOR = {"anomalous": RED, "normal": BLUE, "unlabelled": GREY}
TYPE_COLOR = {"CCD": BLUE, "DSLR": GREEN, "Visual": GOLD, "PEP": PURPLE}
SHORT = {"NB": "Naive Bayes", "DT": "Decision Tree", "RF": "Random Forest", "LR": "Logistic Regression"}
VARIANT = {"full": "RGA-Net full", "no_pretrain": "no ZTF pre-training", "no_gate": "no reliability gate",
           "observer_blind": "observer-blind"}
METRICS = [("mcc", "MCC"), ("roc_auc", "ROC-AUC"), ("pr_auc", "AUPRC"), ("kappa", "Cohen's κ"), ("brier", "1 − Brier")]
FONT = "Inter, Segoe UI, Arial, sans-serif"

st.set_page_config(page_title="Cosmic Events Dashboard", page_icon="✨", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  .hero {padding: 1.4rem 1.6rem; border-radius: 16px; margin-bottom: 1.2rem;
         background: linear-gradient(120deg, #0f1b3d 0%, #1d3470 55%, #3b2a6b 100%); color: #fff;}
  .hero h1 {color: #fff; margin: 0 0 .35rem 0; font-size: 2.0rem;}
  .hero p {color: #d9def0; margin: 0; font-size: 1.02rem;}
  .kpi {border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: .9rem 1rem; height: 100%;
        background: rgba(127,127,127,.04);}
  .kpi .v {font-size: 1.65rem; font-weight: 700; line-height: 1.15;}
  .kpi .l {font-size: .82rem; opacity: .75; margin-top: .15rem;}
  .kpi .s {font-size: .78rem; opacity: .6; margin-top: .25rem;}
  .note {font-size: .9rem; opacity: .85; border-left: 3px solid #2a78d6; padding: .35rem .8rem; margin: .4rem 0 1rem 0;}
  .stTabs [data-baseweb="tab-list"] {gap: .4rem;}
  .stTabs [data-baseweb="tab"] {padding: .5rem 1rem; border-radius: 10px 10px 0 0;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner="Loading the dashboard tables ...")
def load(folder):
    f = Path(folder)
    need = ["dim_star", "dim_observer", "fact_model_metrics", "fact_window_predictions", "fact_cleaning_steps",
            "fact_observations_sample"]
    missing = [n for n in need if not (f / f"{n}.csv").exists()]
    if missing:
        return None, missing
    d = {n: pd.read_csv(f / f"{n}.csv") for n in need}
    s = d["dim_star"]
    s["class"] = s["label"].map({1: "anomalous", 0: "normal"}).fillna("unlabelled")
    s["star"] = s["name"].fillna(s["auid"])
    names = s.set_index("auid")[["star", "class"]]
    m = d["fact_model_metrics"]
    m["label"] = [(f"RGA-Net (proposed)" if v == "full" else VARIANT[v]) if a == "RGA-Net" else
                  f"{SHORT[a]}{'' if v == 'without_observer' else ' + observer'}" for a, v in zip(m["model"], m["variant"])]
    p = d["fact_window_predictions"].join(names, on="auid")
    p["window"] = p["window_id"].str.split("#").str[1].astype(int)
    p["label_short"] = [("RGA-Net" if a == "RGA-Net" else f"{a}{'' if v == 'without_observer' else ' + obs'}")
                        for a, v in zip(p["model"], p["variant"])]
    o = d["fact_observations_sample"].join(names, on="auid")
    o["date"] = pd.to_datetime(o["jd"] - 2440587.5, unit="D")
    d.update(dim_star=s, fact_model_metrics=m, fact_window_predictions=p, fact_observations_sample=o)
    return d, []


def style(fig, h=420, legend=True):
    fig.update_layout(polar_bgcolor="rgba(0,0,0,0)")
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), font=dict(family=FONT, size=13),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
                      legend=dict(orientation="h", y=-0.15, x=0), hoverlabel=dict(font_family=FONT))
    fig.update_xaxes(gridcolor="rgba(128,128,128,.18)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.18)", zeroline=False)
    return fig


def short_step(step):
    t = str(step).lower()
    for k, v in (("dedup", "Repeated rows"), ("null", "Missing keys"), ("comment", "Flagged comments"),
                 ("fainter", "Upper limits"), ("thin", "Night thinning"), ("band", "Small band groups"),
                 ("valid filter", "Invalid filters"), ("missing", "Invalid values"), ("quality", "Quality flags"),
                 ("saturat", "Saturated"), ("repeat", "Repeated epochs"), ("light curve", "Short light curves")):
        if k in t:
            return v
    return str(step).lstrip("0123456789 ")[:24]


def kpi(col, value, label, sub=""):
    col.markdown(f"<div class='kpi'><div class='v'>{value}</div><div class='l'>{label}</div>"
                 f"<div class='s'>{sub}</div></div>", unsafe_allow_html=True)


def mean_table(m, level):
    t = m[m["level"] == level].groupby(["model", "variant", "label", "role"])[[k for k, _ in METRICS]]
    return t.mean().reset_index(), t.std().reset_index()


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### ✨ Cosmic Events")
    data, missing = load(DEFAULT_DATA)
    if data is None:
        st.error("Data tables not found: " + ", ".join(f"{m}.csv" for m in missing))
        st.stop()
    level = st.radio("Evaluation level", ["window", "star"], horizontal=True,
                     help="Window: each 100-day window is scored. Star: a star's probability is the mean of its windows.")
    st.caption("All numbers come from the saved pipeline outputs (10 star-grouped splits, thresholds chosen on "
               "validation). Nothing here is retrained.")

S, OB, M, P, C, O = (data[k] for k in ("dim_star", "dim_observer", "fact_model_metrics", "fact_window_predictions",
                                        "fact_cleaning_steps", "fact_observations_sample"))
mean, std = mean_table(M, level)
rg = mean[(mean["model"] == "RGA-Net") & (mean["variant"] == "full")].iloc[0]
prior = mean[mean["role"] == "prior approach"].sort_values("mcc", ascending=False).iloc[0]

st.markdown(
    "<div class='hero'><h1>Detecting anomalous variable stars in crowdsourced observations</h1>"
    "<p>AAVSO amateur observations + ZTF survey light curves · Hadoop, Spark, Hive, MLlib · RGA-Net, a "
    "Transformer whose attention is gated by each observer's reliability A<sub>k</sub></p></div>",
    unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🛰️  Overview", "🏆  Models", "🔭  Star explorer", "👥  Observers"])

# ------------------------------------------------------------------ 1 overview
with tab1:
    a = C[C["dataset"] == "AAVSO"].sort_values("step_order")
    z = C[C["dataset"] == "ZTF"].sort_values("step_order")
    k = st.columns(5)
    kpi(k[0], f"{int(a['rows'].iloc[0]):,}", "AAVSO raw observations", f"{int(a['rows'].iloc[-1]):,} after cleaning")
    kpi(k[1], f"{int(z['rows'].iloc[0]) / 1e6:,.0f} M", "ZTF raw epochs", f"{int(z['rows'].iloc[-1]) / 1e6:,.0f} M after cleaning")
    kpi(k[2], f"{len(S)}", "AAVSO stars", f"{(S['class'] == 'anomalous').sum()} anomalous · {(S['class'] == 'normal').sum()} normal")
    kpi(k[3], f"{len(OB):,}", "observers", f"{(~OB['low_n'].astype(bool)).sum()} with a measured A_k")
    kpi(k[4], f"{rg['mcc']:.3f}", f"RGA-Net MCC ({level})", f"best prior approach {prior['mcc']:.3f}")
    st.write("")
    left, right = st.container(), st.container()
    with left:
        st.markdown("#### Where the rows go: cleaning steps")
        ds = st.radio("Dataset", ["AAVSO", "ZTF"], horizontal=True, label_visibility="collapsed")
        d = C[C["dataset"] == ds].sort_values("step_order").reset_index(drop=True)
        labels, src, tgt, val, col = [f"Raw {int(d.loc[0, 'rows']):,}"], [], [], [], []
        node_col = [BLUE if ds == "AAVSO" else GREEN]
        prev = 0
        for i in range(1, len(d)):
            removed = int(d.loc[i, "removed"])
            if removed <= 0:
                continue
            step = short_step(d.loc[i, "step"])
            labels += [f"Kept {int(d.loc[i, 'rows']):,}", f"{step}  −{removed:,}"]
            node_col += [node_col[0], GREY]
            src += [prev, prev]; tgt += [len(labels) - 2, len(labels) - 1]
            val += [int(d.loc[i, "rows"]), removed]
            col += ["rgba(42,120,214,.30)" if ds == "AAVSO" else "rgba(27,175,122,.30)", "rgba(160,160,155,.45)"]
            prev = len(labels) - 2
        fig = go.Figure(go.Sankey(arrangement="snap", node=dict(label=labels, color=node_col, pad=18, thickness=16),
                                  link=dict(source=src, target=tgt, value=val, color=col)))
        st.plotly_chart(style(fig, 430, False), use_container_width=True)
        kept = d["rows"].iloc[-1] / d["rows"].iloc[0] * 100
        big = d.loc[d["removed"].idxmax()]
        st.markdown(f"<div class='note'>{ds} keeps <b>{kept:.1f}%</b> of its raw rows. The largest removal is "
                    f"<b>{short_step(big['step']).lower()}</b> ({int(big['removed']):,} rows).</div>",
                    unsafe_allow_html=True)
    with right:
        st.markdown("#### The 60 AAVSO stars on the sky")
        s = S.dropna(subset=["ra", "dec"])
        fig = go.Figure()
        for c, g in s.groupby("class"):
            fig.add_trace(go.Scatterpolar(
                r=90 - g["dec"], theta=g["ra"], mode="markers", name=c,
                marker=dict(size=np.clip(np.sqrt(g["n_obs"]) / 6, 6, 34), color=CLASS_COLOR.get(c, GREY),
                            opacity=.8, line=dict(width=1, color="white")),
                text=g["star"], customdata=np.c_[g["n_obs"], g["vsx_type"].fillna("")],
                hovertemplate="<b>%{text}</b><br>%{customdata[1]}<br>%{customdata[0]:,} observations<extra></extra>"))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 180], tickvals=[0, 60, 90, 120, 180],
                                                     ticktext=["+90°", "+30°", "0°", "−30°", "−90°"], angle=90,
                                                     tickfont=dict(size=10), showline=False),
                                     angularaxis=dict(direction="clockwise", rotation=90)))
        st.plotly_chart(style(fig, 470), use_container_width=True)
        st.caption("Polar sky chart: angle = right ascension, distance from centre = 90° − declination "
                   "(north pole in the middle). Bubble size = number of observations.")

# ------------------------------------------------------------------ 2 models
with tab2:
    show = mean[mean["role"].isin(["prior approach", "proposed"])].copy()
    show["brier"] = 1 - show["brier"]
    c1, c2 = st.container(), st.container()
    with c1:
        st.markdown("#### Proposed vs prior approaches")
        fig = go.Figure()
        cats = [n for _, n in METRICS]
        for _, r in show.iterrows():
            vals = [r[k] for k, _ in METRICS]
            is_rg = r["model"] == "RGA-Net"
            fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=cats + cats[:1], name=r["label"],
                                          fill="toself" if is_rg else None, opacity=.95 if is_rg else .75,
                                          line=dict(width=3.5 if is_rg else 1.6,
                                                    color=RED if is_rg else None)))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], tickfont=dict(size=10))))
        st.plotly_chart(style(fig, 470), use_container_width=True)
        st.caption("Outward = better on every axis (Brier is shown as 1 − Brier). Prior approaches = classic models "
                   "without observer features.")
    with c2:
        st.markdown("#### What observer features add to classic models (MCC)")
        rows = []
        for a in ["NB", "DT", "LR", "RF"]:
            w = mean[(mean["model"] == a) & (mean["variant"] == "with_observer")]["mcc"].iloc[0]
            wo = mean[(mean["model"] == a) & (mean["variant"] == "without_observer")]["mcc"].iloc[0]
            rows.append((SHORT[a], wo, w))
        fig = go.Figure()
        for n, wo, w in rows:
            fig.add_trace(go.Scatter(x=[wo, w], y=[n, n], mode="lines", line=dict(color="rgba(128,128,128,.4)", width=6),
                                     showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[r[1] for r in rows], y=[r[0] for r in rows], mode="markers", name="prior (without)",
                                 marker=dict(size=17, color="white", line=dict(width=3, color=BLUE))))
        fig.add_trace(go.Scatter(x=[r[2] for r in rows], y=[r[0] for r in rows], mode="markers",
                                 name="+ observer features", marker=dict(size=17, color=GREEN)))
        fig.add_vline(x=rg["mcc"], line=dict(color=RED, width=2.5, dash="dash"),
                      annotation_text=f"RGA-Net {rg['mcc']:.3f}", annotation_position="top")
        fig.update_xaxes(title=f"Test MCC (mean over seeds, {level} level)")
        st.plotly_chart(style(fig, 470), use_container_width=True)
        best = max(rows, key=lambda r: r[2] - r[1])
        st.caption(f"Observer features help all four classic models; the biggest gain is {best[0]} "
                   f"({best[1]:.3f} → {best[2]:.3f}). The filled dots already use our framework, so only the open "
                   "dots are prior work.")

    c3, c4 = st.container(), st.container()
    with c3:
        st.markdown("#### Every split, every model (MCC per seed)")
        mm = M[(M["level"] == level) & M["role"].isin(["prior approach", "proposed", "our framework on a classic model"])]
        order = mm.groupby("label")["mcc"].mean().sort_values().index.tolist()
        fig = go.Figure()
        for role, g in mm.groupby("role"):
            fig.add_trace(go.Box(x=g["mcc"], y=g["label"], orientation="h", name=role, boxpoints="all", jitter=.4,
                                 pointpos=0, marker=dict(size=6, color=ROLE_COLOR[role]), line=dict(color=ROLE_COLOR[role]),
                                 fillcolor="rgba(0,0,0,0)"))
        fig.update_yaxes(categoryorder="array", categoryarray=order)
        fig.update_xaxes(title="Test MCC")
        st.plotly_chart(style(fig, 470), use_container_width=True)
        st.caption("Box-and-whisker with every seed shown: the spread is how much a result depends on which stars "
                   "land in the test set.")
    with c4:
        st.markdown("#### Ablation: which part of RGA-Net matters")
        ab = M[(M["level"] == level) & (M["model"] == "RGA-Net")].copy()
        ab["variant_name"] = ab["variant"].map(VARIANT)
        fig = go.Figure()
        vcol = {"RGA-Net full": RED, "no ZTF pre-training": GOLD, "no reliability gate": BLUE, "observer-blind": GREY}
        for v, g in ab.groupby("variant_name"):
            fig.add_trace(go.Violin(y=g["mcc"], name=v, box_visible=True, meanline_visible=True, points="all",
                                    line_color=vcol[v], fillcolor=vcol[v], opacity=.55))
        fig.update_yaxes(title="Test MCC")
        st.plotly_chart(style(fig, 470, False), use_container_width=True)
        av = ab.groupby("variant_name")["mcc"].mean()
        st.caption(f"Removing observer information costs the most ({av['observer-blind'] - av['RGA-Net full']:+.3f} MCC). "
                   f"ZTF pre-training did not help here: without it MCC is {av['no ZTF pre-training']:.3f}.")

    st.markdown("#### Full results table (mean ± std over seeds)")
    tbl = mean.merge(std, on=["model", "variant", "label", "role"], suffixes=("", "_sd"))
    out = pd.DataFrame({"Model": tbl["label"], "Role": tbl["role"]})
    for k, n in METRICS:
        src = k
        out[n.replace("1 − ", "")] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(tbl[src], tbl[src + "_sd"])]
    st.dataframe(out.sort_values("Role"), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ 3 star explorer
with tab3:
    stars = S[S["star"].isin(P["star"].dropna().unique())].sort_values("star")
    c1, c2, c3 = st.columns([1.4, 1, 1])
    star = c1.selectbox("Star", stars["star"], index=int(np.flatnonzero(stars["star"].to_numpy() == "AM Her")[0])
                        if "AM Her" in set(stars["star"]) else 0)
    info = S[S["star"] == star].iloc[0]
    seeds = sorted(P.loc[P["star"] == star, "seed"].unique())
    seed = c2.selectbox("Split seed", seeds)
    sset = P[(P["star"] == star) & (P["seed"] == seed)]["set"].iloc[0]
    c3.markdown(f"<div class='kpi'><div class='v' style='color:{CLASS_COLOR[info['class']]}'>{info['class']}</div>"
                f"<div class='l'>true class · VSX {info['vsx_type']}</div><div class='s'>{sset} set in seed {seed} · "
                f"{int(info['n_obs']):,} obs · {int(info['n_observers'])} observers</div></div>", unsafe_allow_html=True)
    st.write("")
    o = O[O["star"] == star]
    l, r = st.container(), st.container()
    with l:
        st.markdown("#### Light curve, coloured by observer reliability")
        if len(o):
            fig = go.Figure(go.Scattergl(
                x=o["date"], y=o["mag"], mode="markers",
                marker=dict(size=5, color=o["A_k"], colorscale="Viridis", cmin=0, cmax=1, opacity=.8,
                            colorbar=dict(title="A_k", thickness=12)),
                customdata=np.c_[o["observer"], o["band"], o["obs_type"]],
                hovertemplate="%{x|%Y-%m-%d}<br>mag %{y:.2f}<br>%{customdata[0]} · %{customdata[2]} · "
                              "%{customdata[1]}<br>A_k %{marker.color:.2f}<extra></extra>"))
            fig.update_yaxes(autorange="reversed", title="Magnitude (brighter = up)")
            fig.update_xaxes(title="Date")
            st.plotly_chart(style(fig, 440, False), use_container_width=True)
            st.caption(f"{len(o):,} observations in the dashboard sample (a seeded random sample of the cleaned data).")
        else:
            st.info("This star has no rows in the observation sample.")
    with r:
        st.markdown("#### What each model said, window by window")
        pp = P[(P["star"] == star) & (P["seed"] == seed)]
        hm = pp.pivot_table(index="label_short", columns="window", values="prob", aggfunc="mean")
        fig = go.Figure(go.Heatmap(z=hm.values, x=[str(c) for c in hm.columns], y=hm.index, colorscale="RdBu_r",
                                   zmin=0, zmax=1, colorbar=dict(title="P(anomalous)", thickness=12),
                                   hovertemplate="%{y}<br>window %{x}<br>P(anomalous) %{z:.2f}<extra></extra>"))
        fig.update_xaxes(title="Window (100 days each)", type="category")
        st.plotly_chart(style(fig, 440, False), use_container_width=True)
        rgp = pp[pp["model"] == "RGA-Net"]["prob"].mean()
        st.caption(f"Red = model leans anomalous, blue = normal. RGA-Net's mean probability for this star: "
                   f"{rgp:.2f} (true class: {info['class']}).")

# ------------------------------------------------------------------ 4 observers
with tab4:
    ob = OB[~OB["low_n"].astype(bool)].copy()
    k = st.columns(4)
    for i, t in enumerate(["CCD", "DSLR", "Visual", "PEP"]):
        g = ob[ob["obs_type"] == t]["A_k"]
        if len(g):
            kpi(k[i], f"{g.median():.2f}", f"median A_k · {t}", f"{len(g)} observers")
    st.write("")
    l, r = st.container(), st.container()
    with l:
        st.markdown("#### Observer agreement A_k by observation type")
        fig = go.Figure()
        for t, g in ob.groupby("obs_type"):
            fig.add_trace(go.Violin(x=[t] * len(g), y=g["A_k"], name=t, box_visible=True, meanline_visible=True,
                                    points="all", pointpos=0, jitter=.35, line_color=TYPE_COLOR.get(t, GREY),
                                    fillcolor=TYPE_COLOR.get(t, GREY), opacity=.5, text=g["observer"],
                                    hovertemplate="%{text}<br>A_k %{y:.2f}<extra></extra>"))
        fig.update_yaxes(title="A_k (0 = disagrees, 1 = agrees with other observers)", range=[0, 1.05])
        st.plotly_chart(style(fig, 450, False), use_container_width=True)
        st.caption("A_k compares each observation with the median of other observers of the same star, band and "
                   "night. It was independently checked against ZTF: Spearman ρ = −0.311 (p = 0.0053, 79 observers).")
    with r:
        st.markdown("#### Who observes the most, and how reliably")
        fig = go.Figure()
        for t, g in ob.groupby("obs_type"):
            fig.add_trace(go.Scatter(
                x=g["n_obs"], y=g["A_k"], mode="markers", name=t, text=g["observer"],
                marker=dict(size=np.clip(np.sqrt(g["n_stars"]) * 7, 6, 30), color=TYPE_COLOR.get(t, GREY), opacity=.7,
                            line=dict(width=1, color="white")),
                customdata=np.c_[g["n_stars"], g["offset_mag"].round(3)],
                hovertemplate="<b>%{text}</b><br>%{x:,} observations · %{customdata[0]} stars<br>A_k %{y:.2f}"
                              "<br>offset %{customdata[1]} mag<extra></extra>"))
        fig.update_xaxes(type="log", title="Observations (log scale)")
        fig.update_yaxes(title="A_k", range=[0, 1.05])
        st.plotly_chart(style(fig, 450), use_container_width=True)
        st.caption("Bubble size = number of stars observed. Many observations do not guarantee high agreement.")
    st.markdown("#### Observer lookup")
    q = st.text_input("Search an observer code", "")
    t = OB if not q else OB[OB["observer"].str.contains(q.strip(), case=False, na=False)]
    st.dataframe(t.sort_values("A_k", ascending=False).rename(columns={
        "observer": "Observer", "obs_type": "Type", "A_k": "A_k", "offset_mag": "Offset (mag)", "n_obs": "Observations",
        "n_stars": "Stars", "n_residuals": "Compared obs.", "low_n": "Too little overlap"}),
        hide_index=True, use_container_width=True, height=320)
