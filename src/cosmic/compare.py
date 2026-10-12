"""Step 9 — Comparison: RGA-Net vs the four ML models (the main Review 2 result). Plain Python.

What it does: reads the predictions written by RGA-Net (predictions_rganet.csv) and by the
ML models (predictions_ml.csv) — the same windows and the same 5 star-grouped splits — and,
with ONE metric implementation for all models (cosmic/evaluation.py):
  * per model and seed: threshold chosen on validation; test precision, recall, F1, balanced
    accuracy, ROC-AUC and PR-AUC for the anomalous class, at window level and at star level
    (a star's probability = mean of its windows' probabilities);
  * mean ± std over the 5 seeds, plus training time;
  * tables: comparison (RGA-Net vs NB, DT, RF, LR), star-level comparison, ablation
    (full / no-gate / observer-blind), and ML with vs without observer features;
  * Plotly figures: grouped metric bars with error bars, ROC and PR curves of the 5 models
    (one seed, shared axes), a confusion-matrix grid, RGA-Net training/validation curves,
    the ablation bars;
  * COMPARISON.md: which model is best on which metric, by how much, and whether the gap is
    larger than the seed-to-seed spread. Written from the numbers; nothing is hand-edited.

Why: one place that turns predictions into every comparison table and figure, so all models
are scored in exactly the same way.
"""
import os

import numpy as np
import pandas as pd

from . import plots
from .evaluation import (METRICS, at_threshold, average_precision, choose_threshold, evaluate_all, mean_std,
                         pr_points, roc_auc, roc_points)

MAIN = ["RGA-Net", "NB", "DT", "RF", "LR"]
NAMES = {"RGA-Net": "RGA-Net (proposed)", "NB": "Naive Bayes", "DT": "Decision Tree", "RF": "Random Forest",
         "LR": "Logistic Regression"}
VARIANT_NAMES = {"full": "RGA-Net (full)", "no_gate": "RGA-Net, no gate (λ = 0)",
                 "observer_blind": "RGA-Net, observer-blind"}
LABELS = {"precision": "Precision", "recall": "Recall", "f1": "F1", "balanced_accuracy": "Balanced accuracy",
          "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "train_seconds": "Training time (s)"}


def load(cfg):
    w = cfg.work_dir
    rg = pd.read_csv(os.path.join(w, "predictions_rganet.csv"))
    ml = pd.read_csv(os.path.join(w, "predictions_ml.csv"))
    runs = pd.read_csv(os.path.join(w, "rganet_runs.csv"))
    hist = pd.read_csv(os.path.join(w, "rganet_history.csv"))
    params = pd.read_csv(os.path.join(cfg.output_dir, "ml_selected_params.csv"))
    return rg, ml, runs, hist, params


def main_rows(per_seed):
    """RGA-Net (full) + the 4 ML models with observer features."""
    keep = ((per_seed["model"] == "RGA-Net") & (per_seed["variant"] == "full")) | \
           ((per_seed["model"] != "RGA-Net") & (per_seed["variant"] == "with_observer"))
    return per_seed[keep]


def times(runs, params):
    t = pd.concat([runs[runs["variant"] == "full"].assign(model="RGA-Net")[["model", "seed", "train_seconds"]],
                   params[params["variant"] == "with_observer"].rename(columns={"fit_seconds": "train_seconds"})
                   [["model", "seed", "train_seconds"]]])
    g = t.groupby("model")["train_seconds"]
    return pd.DataFrame({"train_seconds_mean": g.mean(), "train_seconds_std": g.std(ddof=1)}).reset_index()


def table(ms, order, level, name_map, key="model"):
    t = ms[ms["level"] == level].set_index(key).reindex([o for o in order if o in set(ms[key])]).reset_index()
    t.insert(1, "name", t[key].map(name_map))
    return t


def gap_text(per_seed, a, b, metric, key="model"):
    """How much a beats b on metric: mean gap, per-seed paired gap, wins out of seeds, and spread check."""
    pa = per_seed[per_seed[key] == a].set_index("seed")[metric]
    pb = per_seed[per_seed[key] == b].set_index("seed")[metric]
    d = (pa - pb).dropna()
    sa, sb = pa.std(ddof=1), pb.std(ddof=1)
    gap = pa.mean() - pb.mean()
    spread = max(sa, sb)
    exceeds = abs(gap) > spread
    return (f"{gap:+.3f} (paired per-seed difference {d.mean():+.3f} ± {d.std(ddof=1):.3f}; ahead in "
            f"{int((d > 0).sum())} of {len(d)} seeds). The gap is "
            + ("larger" if exceeds else "**not** larger")
            + f" than the seed-to-seed spread (largest std {spread:.3f}).")


def best_by_metric(per_seed, ms, level, names, key="model"):
    lines = []
    sub = ms[ms["level"] == level]
    ps = per_seed[per_seed["level"] == level]
    for m in METRICS:
        s = sub.sort_values(f"{m}_mean", ascending=False)
        if len(s) < 2 or s[f"{m}_mean"].isna().all():
            continue
        a, b = s.iloc[0][key], s.iloc[1][key]
        lines.append(f"- **{LABELS[m]}**: best = {names.get(a, a)} ({s.iloc[0][m]}); runner-up = {names.get(b, b)} "
                     f"({s.iloc[1][m]}). Gap {gap_text(ps, a, b, m, key)}")
    return lines


def md_table(t, cols, key_label="Model"):
    head = "| " + key_label + " | " + " | ".join(LABELS.get(c, c) for c in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    rows = ["| " + str(r["name"]) + " | " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in t.iterrows()]
    return "\n".join([head, sep, *rows])


def curves_and_cms(pred, seed):
    curves, cms = [], []
    for m in MAIN:
        d = pred[(pred["model"] == m) & (pred["seed"] == seed)]
        if not len(d):
            continue
        va, te = d[d["set"] == "val"], d[d["set"] == "test"]
        fpr, tpr = roc_points(te["label"], te["prob"])
        prec, rec, _ = pr_points(te["label"], te["prob"])
        curves.append(dict(label=NAMES[m], fpr=fpr, tpr=tpr, roc_auc=roc_auc(te["label"], te["prob"]),
                           recall=np.r_[0.0, rec], precision=np.r_[prec[0], prec],
                           pr_auc=average_precision(te["label"], te["prob"]), prevalence=float(te["label"].mean())))
        c = at_threshold(te["label"], te["prob"], choose_threshold(va["label"], va["prob"]))
        cms.append((NAMES[m], [[c["tn"], c["fp"]], [c["fn"], c["tp"]]]))
    return curves, cms


def run(cfg):
    out = cfg.output_dir
    rg, ml, runs, hist, params = load(cfg)
    pred = pd.concat([rg, ml], ignore_index=True)
    per_seed = evaluate_all(pred)
    per_seed.to_csv(os.path.join(out, "metrics_per_seed.csv"), index=False)
    ms = mean_std(per_seed)
    ms.to_csv(os.path.join(out, "metrics_mean_std_all.csv"), index=False)

    # main comparison (window and star level) + training time
    main = main_rows(ms)
    tt = times(runs, params)
    comp = table(main, MAIN, "window", NAMES).merge(tt, on="model", how="left")
    comp["train_seconds"] = [f"{a:.1f} ± {b:.1f}" for a, b in zip(comp["train_seconds_mean"], comp["train_seconds_std"])]
    comp_star = table(main, MAIN, "star", NAMES)
    cols = METRICS
    comp[["name"] + cols + ["train_seconds"]].to_csv(os.path.join(out, "comparison_table.csv"), index=False)
    comp_star[["name"] + cols].to_csv(os.path.join(out, "comparison_table_star_level.csv"), index=False)
    comp.to_csv(os.path.join(out, "comparison_table_numeric.csv"), index=False)
    print("\nComparison (test windows, mean ± std over seeds):\n" + comp[["name"] + cols + ["train_seconds"]].to_string(index=False))
    print("\nComparison (test stars):\n" + comp_star[["name"] + cols].to_string(index=False))

    # ablation
    ab_ps = per_seed[per_seed["model"] == "RGA-Net"]
    ab = mean_std(ab_ps, by=("variant", "level"))
    ab_w = table(ab, list(VARIANT_NAMES), "window", VARIANT_NAMES, key="variant")
    ab_s = table(ab, list(VARIANT_NAMES), "star", VARIANT_NAMES, key="variant")
    ab_w[["name"] + cols].to_csv(os.path.join(out, "ablation_table.csv"), index=False)
    ab_s[["name"] + cols].to_csv(os.path.join(out, "ablation_table_star_level.csv"), index=False)
    print("\nAblation (test windows):\n" + ab_w[["name"] + cols].to_string(index=False))

    # ML with vs without observer features (supplementary)
    mlv = ms[(ms["model"] != "RGA-Net") & (ms["level"] == "window")].copy()
    mlv["name"] = mlv["model"].map(NAMES) + " — " + mlv["variant"].str.replace("_", " ")
    mlv = mlv.sort_values(["model", "variant"])
    mlv[["name"] + cols].to_csv(os.path.join(out, "ml_observer_features_table.csv"), index=False)

    lam_cols = [c for c in runs.columns if c.startswith("lambda_")]
    lam = runs[runs["variant"] == "full"][["seed"] + lam_cols]
    lam.to_csv(os.path.join(out, "rganet_learned_lambda.csv"), index=False)

    # figures
    def bars(t, key_names, title, name):
        p = pd.DataFrame({"model": t["name"], **{m: t[f"{m}_mean"] for m in cols}})
        std = pd.DataFrame({m: t[f"{m}_std"] for m in cols})
        plots.grouped_metrics(p, out, metrics=cols, title=title, name=name, std=std, labels=LABELS)

    k = per_seed["seed"].nunique()
    bars(comp, NAMES, f"Test windows: RGA-Net vs ML models (mean ± std over {k} star-grouped splits)",
         "fig_comparison_metrics")
    bars(comp_star, NAMES, f"Test stars (mean window probability): RGA-Net vs ML models (mean ± std, {k} splits)",
         "fig_comparison_metrics_star_level")
    bars(ab_w, VARIANT_NAMES, f"RGA-Net ablation on test windows (mean ± std over {k} splits)", "fig_ablation")
    seed = int(cfg.section("split")["curves_seed"])
    curves, cms = curves_and_cms(pred[(pred["model"] != "RGA-Net") & (pred["variant"] == "with_observer") |
                                      (pred["model"] == "RGA-Net") & (pred["variant"] == "full")], seed)
    plots.roc_pr(curves, out, "fig_roc_pr_curves", f"ROC and PR curves, test windows, split seed {seed}")
    plots.confusion_grid(cms, out, "fig_confusion_matrices",
                         f"Confusion matrices, test windows, split seed {seed} (threshold chosen on validation)")
    plots.training_curves(hist[hist["variant"] == "full"], out, "fig_rganet_training_curves",
                          "RGA-Net (full): training and validation curves, one line per split seed")

    write_markdown(cfg, per_seed, ms, comp, comp_star, ab, lam, pred)
    return comp


def write_markdown(cfg, per_seed, ms, comp, comp_star, ab, lam, pred):
    cols = METRICS
    main_ps = main_rows(per_seed)
    ab_ps = per_seed[per_seed["model"] == "RGA-Net"]
    seeds = sorted(per_seed["seed"].unique())
    test_stars = pred[(pred["set"] == "test")].groupby("seed")["auid"].nunique()
    test_windows = pred[(pred["set"] == "test") & (pred["model"] == "RGA-Net") & (pred["variant"] == "full")] \
        .groupby("seed").size()
    best = {m: comp.sort_values(f"{m}_mean", ascending=False).iloc[0]["model"] for m in cols}
    rga_best = [LABELS[m] for m in cols if best[m] == "RGA-Net"]
    rga_not = [LABELS[m] for m in cols if best[m] != "RGA-Net"]
    L = ["# Comparison: RGA-Net vs four ML models",
         "",
         f"*Generated by `python run.py compare` from the predictions of the {cfg.dataset}-data run. "
         "Do not edit by hand; re-run instead.*",
         "",
         f"Setup: {len(seeds)} star-grouped splits (seeds {', '.join(map(str, seeds))}), the same for every model; "
         f"test sets of {test_stars.min()}–{test_stars.max()} stars and {test_windows.min()}–{test_windows.max()} "
         "windows. Thresholds were chosen on validation; the test set was used only for the numbers below. "
         "Positive class = anomalous. Values are mean ± std over the splits.",
         "",
         "## Window level (main result)", "", md_table(comp, cols + ["train_seconds"]), "",
         "## Star level (mean of a star's window probabilities)", "", md_table(comp_star, cols), "",
         "## Which model is best, metric by metric (window level)", "",
         *best_by_metric(main_ps, main_rows(ms), "window", NAMES), "",
         "## Plain summary", ""]
    if not rga_not:
        L.append("RGA-Net has the highest mean on every metric. Check above whether each gap is larger than the "
                 "seed-to-seed spread before calling it a clear win.")
    elif not rga_best:
        L.append("**RGA-Net is not the best model on any metric.** " + "The best model per metric: "
                 + "; ".join(f"{LABELS[m]}: {NAMES[best[m]]}" for m in cols) + ".")
    else:
        L.append(f"RGA-Net has the highest mean on: {', '.join(rga_best)}. **It is not the best on: "
                 f"{', '.join(rga_not)}** (" + "; ".join(f"{LABELS[m]}: {NAMES[best[m]]}" for m in cols
                                                       if best[m] != "RGA-Net") + ").")
    ab_t = ab.copy()
    L += ["", "## Ablation (RGA-Net, window level)", "",
          md_table(table(ab_t, list(VARIANT_NAMES), "window", VARIANT_NAMES, key="variant"), cols), "",
          *best_by_metric(ab_ps, ab_t, "window", VARIANT_NAMES, key="variant"), "",
          "## Learned gate strengths λ (full model)", "",
          "λ = softplus(ρ) ≥ 0; λ = 0 would mean the gate is unused. Per seed:", "",
          "| seed | " + " | ".join(c for c in lam.columns if c != "seed") + " |",
          "|" + "---|" * len(lam.columns),
          *["| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |"
            for r in lam.itertuples(index=False)], "",
          "## Caveats", "",
          "- Test sets are small (a few stars per class per split), so one star can move recall and precision a lot. "
          "That is why every number is reported as mean ± std over 5 different splits.",
          "- Windows of the same star are correlated; the star-level table is the stricter view.",
          "- \"Larger than the seed-to-seed spread\" compares the gap in means with the larger of the two standard "
          "deviations. It is a rough check, not a significance test."]
    path = os.path.join(cfg.output_dir, "COMPARISON.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"saved {path}")
