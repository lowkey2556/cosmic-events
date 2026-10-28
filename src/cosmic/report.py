"""Step 11 — Report: every Review 2 figure and caption from the saved result files (no Spark, no training).

What it does: reads the CSV files the pipeline saved in V3_ROOT/outputs/ and builds the 13 result items, each as
PNG + HTML + JSON in outputs/figures/ with a GENERATED "Inference:" caption (2-3 sentences computed from the
numbers; nothing is hand-written). Captions go to outputs/figure_captions.md and figure_captions.json, so the
RESULTS cells only display saved files. Also writes outputs/DATA_FLOW.txt from the measured data-flow records and
the small tables (λ, A_k summary, scalability) the RESULTS section shows.
  01 raw vs cleaned samples (AAVSO, ZTF)        08 main comparison table
  02 data quality before vs after               09 proposed vs prior approaches
  03 Sankey of the cleaning steps               10 radar: 5 models x 5 metrics
  04 light curve coloured by A_k                11 dumbbell: classic models without -> with observer features
  05 violin of A_k by observation type          12 ablation dot plot (MCC, AUPRC)
  06 A_k validation against ZTF                 13 confusion matrices at the median-MCC seed
  07 sunburst: source -> class -> VSX type
A figure whose input is missing is skipped and listed as such (it is never faked).
"""
import json
import os

import numpy as np
import pandas as pd

from . import dataflow, plots
from .evaluation import LABELS, METRICS

ORDER = ["NB", "DT", "RF", "LR", "RGA-Net"]


def _read(out, name):
    p = os.path.join(out, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def build(cfg):
    out, figs = cfg.output_dir, cfg.figures_dir
    level = cfg.section("metrics").get("main_level", "window")
    captions, missing = {}, []

    def emit(name, title, fn, *inputs):
        if any(i is None or (hasattr(i, "__len__") and len(i) == 0) for i in inputs):
            missing.append(name)
            print(f"skipped {name}: its input file is missing (run the step that makes it)")
            return
        fig, cap = fn()
        plots.save(fig, figs, name)
        captions[name] = dict(title=title, caption=cap)

    # 1. samples
    for ds, key in (("AAVSO", "aavso"), ("ZTF", "ztf")):
        raw, clean = _read(out, f"sample_{key}_raw.csv"), _read(out, f"sample_{key}_clean.csv")
        emit(f"01_sample_{key}_raw", f"{ds} sample: raw", lambda raw=raw, clean=clean, ds=ds: (
            plots.table_figure(raw.head(10), f"{ds}: 10 raw rows (as received)"), plots.caption_samples(raw, clean, ds)),
             raw, clean)
        emit(f"01_sample_{key}_clean", f"{ds} sample: cleaned", lambda raw=raw, clean=clean, ds=ds: (
            plots.table_figure(clean.head(10).round(4), f"{ds}: the same observations after cleaning"),
            plots.caption_samples(raw, clean, ds)), raw, clean)
    # 2. data quality
    dqa, dqz = _read(out, "data_quality_aavso.csv"), _read(out, "data_quality_ztf.csv")
    if dqa is not None:
        parts = [dqa] + ([dqz] if dqz is not None else [])
        dq = pd.concat(parts, ignore_index=True)
        if dqz is not None:
            num = ["size_mb", "rows", "nulls", "duplicates", "flagged", "bands", "stars", "observers"]
            comb = [dict(dataset="AAVSO + ZTF", stage=s, **{c: sum(float(p.iloc[i].get(c, 0) or 0) for p in parts)
                                                           for c in num}) for i, s in ((0, "before"), (-1, "after"))]
            dq = pd.concat([dq, pd.DataFrame(comb)], ignore_index=True)
        emit("02_data_quality", "Data quality before vs after", lambda: plots.data_quality(dq), dq)
    else:
        missing.append("02_data_quality")
    # 3. Sankey
    steps = {}
    a = _read(out, "preprocess_row_counts.csv")
    if a is not None:
        steps["AAVSO"] = a[["step", "rows"]]
    z = _read(out, "ztf_clean_steps.csv")
    if z is not None:
        steps["ZTF"] = z[["step", "rows"]]
    emit("03_cleaning_sankey", "Cleaning steps (Sankey)", lambda: plots.cleaning_sankey(steps), steps or None)
    # 4-6. A_k
    lc = _read(out, "lightcurve_ak_sample.csv")
    emit("04_lightcurve_ak", "Light curve coloured by A_k",
         lambda: plots.lightcurve_ak(lc, str(lc["name"].iloc[0])), lc)
    sc = _read(out, "observer_scores.csv")
    emit("05_ak_violin", "A_k by observation type", lambda: plots.ak_violin(sc), sc)
    po, ps = _read(out, "ak_vs_ztf_observers.csv"), _read(out, "ak_validation_summary.csv")
    emit("06_ak_validation", "A_k validation against ZTF", lambda: plots.ak_validation(po, ps.iloc[0].to_dict()), po, ps)
    # 7. sunburst
    comps = [c for c in (_read(out, "composition_aavso.csv"), _read(out, "composition_ztf.csv")) if c is not None]
    comp_all = pd.concat(comps, ignore_index=True) if comps else None
    emit("07_composition_sunburst", "Data composition", lambda: plots.composition_sunburst(comp_all), comp_all)
    # 8-13. results
    comp = _read(out, "comparison_table.csv")
    lv = comp[comp["level"] == level] if comp is not None else None
    emit("08_comparison_table", "Main comparison table", lambda: (
        plots.comparison_table_fig(lv, METRICS, LABELS, f"Test results, mean ± std over seeds ({level} level)"),
        plots.caption_comparison(comp, level)), lv)
    pvp = _read(out, "prior_vs_proposed_paired.csv")

    def pvp_fig():
        p = pvp[(pvp["level"] == level) & (pvp["metric"].isin(["mcc", "pr_auc"]))]
        rows = []
        c = lv.set_index("model")
        for m in [x for x in ORDER if x != "RGA-Net"]:
            q = p[p["prior_model"] == m].set_index("metric")
            if not len(q):
                continue
            rows.append({"Prior approach": c.loc[m, "display_name"] if m in c.index else m,
                         "MCC prior": c.loc[m, "mcc"], "MCC RGA-Net": c.loc["RGA-Net", "mcc"],
                         "MCC difference": f"{q.loc['mcc', 'mean_difference']:+.3f}",
                         "Seeds RGA-Net ahead": f"{int(q.loc['mcc', 'seeds_ahead'])}/{int(q.loc['mcc', 'seeds'])}",
                         "Wilcoxon p (MCC)": plots._num(q.loc["mcc", "wilcoxon_p"], 3),
                         "AUPRC difference": f"{q.loc['pr_auc', 'mean_difference']:+.3f}"})
        t = pd.DataFrame(rows)
        won = int((p[p["metric"] == "mcc"]["mean_difference"] > 0).sum())
        n = int((p["metric"] == "mcc").sum())
        cap = (f"Inference: RGA-Net's mean MCC is higher than {won} of {n} prior approaches (classic models WITHOUT "
               "observer features). 'Seeds ahead' counts the splits where RGA-Net wins; with 10 seeds a Wilcoxon "
               "p below 0.05 needs a consistent win, so a larger p means the difference is not reliable.")
        return plots.table_figure(t, f"Proposed RGA-Net vs prior approaches (paired over the same seeds, {level} level)",
                                  highlight_col="MCC difference"), cap
    emit("09_prior_vs_proposed", "Proposed vs prior approaches", pvp_fig, pvp, lv)
    radar_in = lv[lv["role"].isin(["prior approach", "proposed"])] if lv is not None else None
    emit("10_radar", "Radar: 5 models x 5 metrics", lambda: plots.radar(radar_in, level, ORDER), radar_in)
    gain = _read(out, "observer_feature_gain.csv")
    g = gain[(gain["level"] == level) & (gain["metric"] == "mcc")] if gain is not None and len(gain) else None
    emit("11_observer_feature_dumbbell", "Observer features on classic models (Fix 1)",
         lambda: plots.dumbbell(g, comp, level), g, comp)
    ab = _read(out, "ablation_table.csv")
    emit("12_ablation", "Ablation", lambda: plots.ablation_dots(ab, level, ["full", "no_pretrain", "no_gate",
                                                                             "observer_blind"]), ab)
    ps_m = _read(out, "per_seed_metrics.csv")
    emit("13_confusion_grid", "Confusion matrices at the median-MCC seed",
         lambda: plots.confusion_grid(ps_m, level, ORDER, plots.median_seed(ps_m, level)), ps_m)

    with open(os.path.join(out, "figure_captions.json"), "w", encoding="utf-8") as fh:
        json.dump(captions, fh, indent=1, ensure_ascii=False)
    lines = ["# Figure captions (generated from the saved numbers by report.py)\n"]
    for name, c in captions.items():
        lines += [f"## {name}: {c['title']}\n", f"figures/{name}.png\n", c["caption"] + "\n"]
    if missing:
        lines.append("Not made (input missing): " + ", ".join(missing) + "\n")
    with open(os.path.join(out, "figure_captions.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return captions, missing


def small_tables(cfg):
    """λ, A_k summary and scalability tables for the RESULTS section (saved as CSV; shown as tables only)."""
    out = cfg.output_dir
    s = _read(out, "observer_scores.csv")
    if s is not None:
        m = s[~s["low_n"].astype(bool)]
        t = m.groupby("obs_type")["score"].describe()[["count", "mean", "25%", "50%", "75%", "min", "max"]].round(3)
        t = t.rename(columns={"count": "observers", "50%": "median"}).reset_index()
        t.to_csv(os.path.join(out, "ak_summary_by_type.csv"), index=False)


def run(cfg):
    captions, missing = build(cfg)
    small_tables(cfg)
    files = [os.path.join(dp, f) for d in (cfg.output_dir, cfg.dashboard_dir) for dp, _, fs in os.walk(d) for f in fs]
    dataflow.record(cfg, "results", "Result tables, the 13 figures (PNG + HTML + JSON), generated captions, "
                    "COMPARISON.md and the dashboard tables.", f"{cfg.output_dir}; {cfg.dashboard_dir}",
                    "CSV, PNG, HTML, JSON, Markdown, Parquet", "pandas + Plotly (no Spark)",
                    size_bytes=sum(os.path.getsize(f) for f in files), files=len(files),
                    changes=f"{len(captions)} figures with captions")
    path = dataflow.write(cfg)
    print(f"\n{len(captions)} figures written to {cfg.figures_dir}; captions in figure_captions.md")
    if missing:
        print("not made (input missing): " + ", ".join(missing))
    print(f"data flow: {path}")
