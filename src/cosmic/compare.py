"""Step 9 — Comparison: the proposed RGA-Net vs the prior approaches, from SAVED PREDICTION FILES ONLY. Plain Python.

Inputs (nothing is retrained and no Spark is needed): work/predictions_ml.csv, work/predictions_rganet.csv and
work/rganet_runs.csv (learned λ). One metric implementation for every model (cosmic/evaluation.py): per model and
seed the MCC threshold is chosen on validation, and the test metrics are MCC, ROC-AUC, AUPRC, Cohen's κ and Brier,
at window level and at star level; mean ± std over the 10 seeds.

Who is who (Fix 1):
  * PRIOR APPROACHES = the four classic ML models (NB, DT, RF, LR) WITHOUT observer features.
  * PROPOSED = RGA-Net full (pre-trained on ZTF + reliability gate).
  * The ML models WITH observer features are "our reliability framework applied to classic models"; they are
    reported separately (observer_feature_gain.csv) and never called prior approaches.
Outputs (V3_ROOT/outputs/):
  per_seed_metrics.csv        every model x variant x seed x level (incl. confusion counts and threshold)
  comparison_table.csv        main table: 4 prior approaches + 4 framework-on-classic + RGA-Net full
  prior_vs_proposed_table.csv RGA-Net full vs each prior approach (means and differences)
  prior_vs_proposed_paired.csv  per prior model and metric: per-seed differences, seeds RGA-Net is ahead, Wilcoxon p
  observer_feature_gain.csv   each ML model without -> with observer features (gain, seeds improved, Wilcoxon p)
  ablation_table.csv          full / no_pretrain / no_gate / observer_blind
  rganet_lambda.csv           learned λ per layer (mean ± std over seeds, and per seed)
  metrics_appendix.csv        precision, recall, accuracy, F1, balanced accuracy (appendix only)
  COMPARISON.md               generated text: best model per metric, whether gaps exceed the seed spread, proposed
                              vs prior, the effect of pre-training, and the line about the with-observer ML models
"""
import os

import numpy as np
import pandas as pd

from .evaluation import APPENDIX, LABELS, LOWER_IS_BETTER, METRICS, evaluate_all, mean_std
from .style import model_name, variant_name

ML = ["NB", "DT", "RF", "LR"]
PRIOR = "without_observer"
FRAMEWORK = "with_observer"


def load_predictions(work_dir):
    rg = pd.read_csv(os.path.join(work_dir, "predictions_rganet.csv"))
    ml = pd.read_csv(os.path.join(work_dir, "predictions_ml.csv"))
    cols = ["model", "variant", "seed", "set", "window_id", "auid", "label", "prob"]
    return pd.concat([rg[cols], ml[cols]], ignore_index=True)


def role(model, variant):
    if model == "RGA-Net":
        return "proposed" if variant == "full" else "ablation"
    return "prior approach" if variant == PRIOR else "our framework on a classic model"


def display(model, variant):
    if model == "RGA-Net":
        return model_name("RGA-Net") if variant == "full" else variant_name(variant)
    return f"{model_name(model)} ({'without' if variant == PRIOR else 'with'} observer features)"


def better(metric, a, b):
    return a < b if metric in LOWER_IS_BETTER else a > b


def wilcoxon_p(d):
    """Two-sided Wilcoxon signed-rank p-value of paired differences (NaN when undefined, e.g. all zero)."""
    from scipy.stats import wilcoxon
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2 or np.allclose(d, 0):
        return float("nan")
    try:
        return float(wilcoxon(d).pvalue)
    except ValueError:
        return float("nan")


def paired(per_seed, a, b, level, metrics=METRICS):
    """a, b: (model, variant). Per metric: mean of (a - b) over common seeds, its std, seeds where a is better,
    the number of seeds, the Wilcoxon p-value, and the per-seed differences as text."""
    rows = []
    sa = per_seed[(per_seed["model"] == a[0]) & (per_seed["variant"] == a[1]) & (per_seed["level"] == level)]
    sb = per_seed[(per_seed["model"] == b[0]) & (per_seed["variant"] == b[1]) & (per_seed["level"] == level)]
    m = sa.merge(sb, on="seed", suffixes=("_a", "_b"))
    for metric in metrics:
        d = (m[f"{metric}_a"] - m[f"{metric}_b"]).to_numpy(float)
        ahead = int(np.sum(d < 0)) if metric in LOWER_IS_BETTER else int(np.sum(d > 0))
        rows.append(dict(level=level, metric=metric, mean_difference=float(np.nanmean(d)) if len(d) else np.nan,
                         std_difference=float(np.nanstd(d, ddof=1)) if len(d) > 1 else np.nan,
                         seeds_ahead=ahead, seeds=len(d), wilcoxon_p=wilcoxon_p(d),
                         per_seed_differences=" ".join(f"{x:+.3f}" for x in d)))
    return pd.DataFrame(rows)


def prior_vs_proposed(per_seed, level):
    """Fix 1: RGA-Net full vs each prior approach. Uses ONLY the ML predictions WITHOUT observer features."""
    prior = per_seed[(per_seed["model"].isin(ML)) & (per_seed["variant"] == PRIOR)]
    ps = pd.concat([per_seed[(per_seed["model"] == "RGA-Net") & (per_seed["variant"] == "full")], prior])
    out = []
    for m in ML:
        if not len(prior[prior["model"] == m]):
            continue
        out.append(paired(ps, ("RGA-Net", "full"), (m, PRIOR), level).assign(prior_model=m))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def observer_gain(per_seed, level):
    out = []
    for m in ML:
        g = paired(per_seed, (m, FRAMEWORK), (m, PRIOR), level)
        if len(g) and g["seeds"].iloc[0]:
            out.append(g.assign(model=m))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def fmt(mean, std):
    return f"{mean:.3f} ± {std:.3f}"


def summary_table(ms, keys, level):
    """Wide table: one row per (model, variant), 'mean ± std' text and numeric columns for every metric."""
    t = ms[ms["level"] == level].copy()
    t = t.set_index(["model", "variant"]).loc[[k for k in keys if k in set(zip(t["model"], t["variant"]))]].reset_index()
    t.insert(0, "display_name", [display(m, v) for m, v in zip(t["model"], t["variant"])])
    t.insert(1, "role", [role(m, v) for m, v in zip(t["model"], t["variant"])])
    return t


def run(cfg):
    out, work = cfg.output_dir, cfg.work_dir
    pred = load_predictions(work)
    per_seed = evaluate_all(pred)
    per_seed["display_name"] = [display(m, v) for m, v in zip(per_seed["model"], per_seed["variant"])]
    per_seed["role"] = [role(m, v) for m, v in zip(per_seed["model"], per_seed["variant"])]
    per_seed.to_csv(os.path.join(out, "per_seed_metrics.csv"), index=False)
    ms = mean_std(per_seed, metrics=METRICS + APPENDIX)
    main_keys = [(m, PRIOR) for m in ML] + [(m, FRAMEWORK) for m in ML] + [("RGA-Net", "full")]
    comp = pd.concat([summary_table(ms, main_keys, lv) for lv in ("window", "star")], ignore_index=True)
    keep = ["display_name", "role", "model", "variant", "level", "seeds"] + METRICS + \
           [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
    comp[keep].to_csv(os.path.join(out, "comparison_table.csv"), index=False)
    app_keep = ["display_name", "role", "model", "variant", "level", "seeds"] + APPENDIX + \
               [f"{m}_{s}" for m in APPENDIX for s in ("mean", "std")]
    all_keys = main_keys + [("RGA-Net", v) for v in ("no_pretrain", "no_gate", "observer_blind")]
    pd.concat([summary_table(ms, all_keys, lv) for lv in ("window", "star")])[app_keep] \
        .to_csv(os.path.join(out, "metrics_appendix.csv"), index=False)
    # Fix 1
    pvp = pd.concat([prior_vs_proposed(per_seed, lv) for lv in ("window", "star")], ignore_index=True)
    pvp.to_csv(os.path.join(out, "prior_vs_proposed_paired.csv"), index=False)
    pvt = comp[(comp["role"].isin(["prior approach", "proposed"]))][["display_name", "role", "model", "level"] +
                                                                     METRICS + [f"{m}_mean" for m in METRICS]]
    pvt.to_csv(os.path.join(out, "prior_vs_proposed_table.csv"), index=False)
    gain = pd.concat([observer_gain(per_seed, lv) for lv in ("window", "star")], ignore_index=True)
    if len(gain):
        w = ms.set_index(["model", "variant", "level"])
        gain["without_mean"] = [w.loc[(m, PRIOR, lv), f"{x}_mean"] for m, lv, x in
                                zip(gain["model"], gain["level"], gain["metric"])]
        gain["with_mean"] = [w.loc[(m, FRAMEWORK, lv), f"{x}_mean"] for m, lv, x in
                             zip(gain["model"], gain["level"], gain["metric"])]
    gain.to_csv(os.path.join(out, "observer_feature_gain.csv"), index=False)
    # ablation
    ab_keys = [("RGA-Net", v) for v in ("full", "no_pretrain", "no_gate", "observer_blind")]
    ab = pd.concat([summary_table(ms, ab_keys, lv) for lv in ("window", "star")], ignore_index=True)
    ab_d = pd.concat([paired(per_seed, ("RGA-Net", v), ("RGA-Net", "full"), lv).assign(variant=v)
                      for v in ("no_pretrain", "no_gate", "observer_blind") for lv in ("window", "star")
                      if len(per_seed[(per_seed["model"] == "RGA-Net") & (per_seed["variant"] == v)])],
                     ignore_index=True)
    ab[["display_name", "variant", "level", "seeds"] + METRICS + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]] \
        .to_csv(os.path.join(out, "ablation_table.csv"), index=False)
    ab_d.to_csv(os.path.join(out, "ablation_paired.csv"), index=False)
    lam = lambda_table(work)
    lam.to_csv(os.path.join(out, "rganet_lambda.csv"), index=False)
    md = write_markdown(cfg, per_seed, ms, comp, pvp, gain, ab, ab_d, lam)
    print(md)
    return comp


def lambda_table(work_dir):
    runs = pd.read_csv(os.path.join(work_dir, "rganet_runs.csv"))
    cols = [c for c in runs.columns if c.startswith("lambda_")]
    gated = runs[runs["variant"].isin(["full", "no_pretrain"])]
    rows = []
    for v, g in gated.groupby("variant"):
        for c in cols:
            rows.append(dict(variant=v, parameter=c.replace("lambda_", "λ "), mean=g[c].mean(), std=g[c].std(ddof=1),
                             min=g[c].min(), max=g[c].max(), seeds=len(g),
                             per_seed=" ".join(f"{x:.3f}" for x in g.sort_values("seed")[c])))
    return pd.DataFrame(rows)


def exceeds_spread(per_seed, a, b, metric, level):
    """'Larger than the seed spread' = |mean difference| > the larger of the two models' seed standard deviations."""
    def stats(k):
        s = per_seed[(per_seed["model"] == k[0]) & (per_seed["variant"] == k[1]) & (per_seed["level"] == level)][metric]
        return s.mean(), s.std(ddof=1)
    (ma, sa), (mb, sb) = stats(a), stats(b)
    return abs(ma - mb) > max(sa, sb), ma - mb, max(sa, sb)


def md_table(t, metrics):
    head = "| Model | Role | " + " | ".join(LABELS[m] for m in metrics) + " |"
    lines = [head, "|" + "---|" * (len(metrics) + 2)]
    for r in t.itertuples():
        lines.append(f"| {r.display_name} | {r.role} | " + " | ".join(getattr(r, m) for m in metrics) + " |")
    return "\n".join(lines)


def p_text(p):
    return "p = n/a" if not np.isfinite(p) else f"p = {p:.3g}"


def write_markdown(cfg, per_seed, ms, comp, pvp, gain, ab, ab_d, lam):
    level = cfg.section("metrics").get("main_level", "window")
    seeds = int(per_seed["seed"].nunique())
    L = [f"# Comparison of RGA-Net with the prior approaches (generated from the saved predictions)\n",
         f"All numbers are test-set means ± standard deviations over {seeds} star-grouped splits (no star is in "
         "more than one of train, validation and test). Thresholds for MCC and κ were chosen on validation by "
         "maximising MCC. Brier score: lower is better. Main level: "
         f"**{level}**; star level (mean window probability per star) is in comparison_table.csv.\n",
         "Prior approaches = the four classic ML models without observer features. Proposed = RGA-Net full "
         "(pre-trained on ZTF, reliability gate active).\n",
         "## Main table\n", md_table(comp[comp["level"] == level], METRICS), ""]
    # best per metric among prior + proposed
    cand = comp[(comp["level"] == level) & comp["role"].isin(["prior approach", "proposed"])]
    L.append("## Best model per metric (prior approaches and the proposed model)\n")
    for m in METRICS:
        col = f"{m}_mean"
        if cand[col].isna().all():
            continue
        i = cand[col].idxmin() if m in LOWER_IS_BETTER else cand[col].idxmax()
        b = cand.loc[i]
        rest = cand.drop(index=i)
        j = rest[col].idxmin() if m in LOWER_IS_BETTER else rest[col].idxmax()
        r2 = rest.loc[j]
        big, diff, spread = exceeds_spread(per_seed, (b["model"], b["variant"]), (r2["model"], r2["variant"]), m, level)
        L.append(f"- **{LABELS[m]}**: {b['display_name']} ({b[m]}); next {r2['display_name']} ({r2[m]}). The gap "
                 f"({abs(diff):.3f}) is {'LARGER' if big else 'NOT larger'} than the seed spread ({spread:.3f}).")
    L.append("\nRule: a gap counts as larger than the seed spread when |difference of means| exceeds the larger of "
             "the two models' standard deviations over seeds.\n")
    # proposed vs prior
    L.append("## Proposed vs prior approaches (paired over the same seeds)\n")
    pv = pvp[pvp["level"] == level] if len(pvp) else pvp
    for m in ML:
        r = pv[(pv["prior_model"] == m) & (pv["metric"] == "mcc")] if len(pv) else pv
        if not len(r):
            continue
        r = r.iloc[0]
        verdict = "better" if r["mean_difference"] > 0 else "worse"
        L.append(f"- vs {model_name(m)}: RGA-Net MCC is {verdict} by {abs(r['mean_difference']):.3f} on average; "
                 f"ahead on {int(r['seeds_ahead'])} of {int(r['seeds'])} seeds; Wilcoxon {p_text(r['wilcoxon_p'])}.")
    if len(pv):
        mcc = pv[pv["metric"] == "mcc"]
        won = int((mcc["mean_difference"] > 0).sum())
        L.append(f"\nOverall: RGA-Net has the higher mean MCC against {won} of {len(mcc)} prior approaches. "
                 + ("It does not beat every prior approach; this is reported as measured." if won < len(mcc) else ""))
    # pre-training
    L.append("\n## Effect of ZTF pre-training and of the gate (ablation)\n")
    for v in ("no_pretrain", "no_gate", "observer_blind"):
        r = ab_d[(ab_d["variant"] == v) & (ab_d["level"] == level) & (ab_d["metric"] == "mcc")] if len(ab_d) else ab_d
        if not len(r):
            continue
        r = r.iloc[0]
        d = -r["mean_difference"]  # full - variant
        L.append(f"- Full vs {variant_name(v)}: full model MCC {'higher' if d > 0 else 'lower'} by {abs(d):.3f} "
                 f"(full ahead on {int(r['seeds']) - int(r['seeds_ahead'])} of {int(r['seeds'])} seeds; Wilcoxon "
                 f"{p_text(r['wilcoxon_p'])}).")
    if len(lam):
        f = lam[lam["variant"] == "full"]
        if len(f):
            L.append("\nLearned λ (full model): " + "; ".join(f"{r.parameter} = {r.mean:.3f} ± {r.std:.3f}"
                                                              for r in f.itertuples()) +
                     ". λ > 0 means attention leans towards observers with higher A_k.")
    # framework on classic models (honest line)
    L.append("\n## Classic models with observer features (our framework applied to classic models)\n")
    if len(gain):
        g = gain[(gain["level"] == level) & (gain["metric"] == "mcc")]
        for r in g.itertuples():
            L.append(f"- {model_name(r.model)}: MCC {r.without_mean:.3f} -> {r.with_mean:.3f} "
                     f"({r.mean_difference:+.3f}; improved on {r.seeds_ahead} of {r.seeds} seeds; Wilcoxon "
                     f"{p_text(r.wilcoxon_p)}).")
        fw = comp[(comp["level"] == level) & (comp["role"] == "our framework on a classic model")]
        rg = comp[(comp["level"] == level) & (comp["role"] == "proposed")]
        if len(fw) and len(rg):
            b = fw.loc[fw["mcc_mean"].idxmax()]
            rgm = rg["mcc_mean"].iloc[0]
            L.append(f"\nHonest note: the best classic model WITH observer features, {b['display_name']}, reaches MCC "
                     f"{b['mcc']}, which is {'above' if b['mcc_mean'] > rgm else 'below'} RGA-Net full "
                     f"({rg['mcc'].iloc[0]}). These models already use our observer-reliability features, so they are "
                     "not prior approaches and are not used to claim an improvement over prior work.")
    text = "\n".join(L) + "\n"
    with open(os.path.join(cfg.output_dir, "COMPARISON.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text
