"""Metrics, used identically for RGA-Net and the four ML models (plain NumPy).

What it does: every model writes the same prediction table — one row per window with model, variant, seed,
set (val/test), window_id, auid, label and prob (the predicted probability of "anomalous"). From it:
  * Main metrics (all reported as mean ± std over the 10 seeds):
      MCC (Matthews correlation coefficient, the headline metric), ROC-AUC, AUPRC (average precision),
      Cohen's kappa, and the Brier score (mean squared error of the probability; lower is better).
  * MCC and kappa need a yes/no decision: the threshold is the one with the HIGHEST VALIDATION MCC, chosen
    separately at window level and at star level; the test set is never used to choose it. ROC-AUC, AUPRC and
    Brier use the probabilities directly.
  * Star level: a star's probability is the mean of its windows' probabilities.
  * Precision, recall, accuracy, F1 and balanced accuracy are computed at the same threshold but go only to the
    appendix table (metrics_appendix.csv).

Why: one implementation for all models makes the comparison like-for-like. MCC uses all four cells of the
confusion matrix, so it is not flattered by the class imbalance.
"""
import numpy as np
import pandas as pd

METRICS = ["mcc", "roc_auc", "pr_auc", "kappa", "brier"]
APPENDIX = ["precision", "recall", "accuracy", "f1", "balanced_accuracy"]
LOWER_IS_BETTER = {"brier"}
LABELS = {"mcc": "MCC", "roc_auc": "ROC-AUC", "pr_auc": "AUPRC", "kappa": "Cohen's κ", "brier": "Brier score",
          "precision": "Precision", "recall": "Recall", "accuracy": "Accuracy", "f1": "F1",
          "balanced_accuracy": "Balanced accuracy"}


def roc_auc(y, p):
    """Area under the ROC curve (Mann-Whitney statistic; ties count one half). NaN if one class is missing."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _cum(y, p):
    """Cumulative TP/FP at every distinct threshold (highest first) and those thresholds."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    order = np.argsort(-p, kind="mergesort")
    y, p = y[order], p[order]
    last = np.r_[np.flatnonzero(np.diff(p)), len(p) - 1]  # last index of each distinct score
    tp = np.cumsum(y)[last].astype(float)
    return tp, (last + 1) - tp, p[last], float(y.sum()), float(len(y) - y.sum())


def pr_points(y, p):
    """Precision and recall at every distinct threshold (highest first), as in scikit-learn."""
    tp, fp, t, P, _ = _cum(y, p)
    return tp / (tp + fp), tp / max(P, 1), t


def average_precision(y, p):
    """AUPRC as average precision: sum over thresholds of (R_n - R_{n-1}) * P_n."""
    if len(y) == 0 or int(np.sum(y)) == 0:
        return float("nan")
    precision, recall, _ = pr_points(y, p)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_points(y, p):
    """False- and true-positive rates at every distinct threshold, starting at (0, 0)."""
    tp, fp, _, P, N = _cum(y, p)
    return np.r_[0.0, fp / max(N, 1)], np.r_[0.0, tp / max(P, 1)]


def brier(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def mcc_from(tp, fp, fn, tn):
    den = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0


def kappa_from(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    if n == 0:
        return float("nan")
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0


def at_threshold(y, p, t):
    """Confusion counts and threshold metrics for 'anomalous if prob >= t'."""
    y, pred = np.asarray(y, int), (np.asarray(p, float) >= t).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    n = tp + fp + fn + tn
    return dict(mcc=mcc_from(tp, fp, fn, tn), kappa=kappa_from(tp, fp, fn, tn), precision=prec, recall=rec,
                accuracy=(tp + tn) / n if n else float("nan"), f1=f1, balanced_accuracy=(rec + spec) / 2,
                tp=tp, fp=fp, fn=fn, tn=tn)


def choose_threshold(y, p):
    """Threshold with the highest validation MCC (ties: the threshold closest to 0.5)."""
    if len(p) == 0:
        return 0.5
    tp, fp, ts, P, N = _cum(y, p)
    fn, tn = P - tp, N - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1.0), 0.0)
    best = np.flatnonzero(mcc == mcc.max())
    return float(ts[best[np.argmin(np.abs(ts[best] - 0.5))]])


def star_level(pred):
    """Mean window probability per star (one row per model, variant, seed, set, auid)."""
    keys = [c for c in ("model", "variant", "setting", "seed", "set", "auid", "label") if c in pred.columns]
    return pred.groupby(keys, as_index=False)["prob"].mean()


def evaluate(pred):
    """pred: one model/variant/seed with set in {val, test}. Returns {level: metrics} on the TEST set."""
    out = {}
    for level, d in (("window", pred), ("star", star_level(pred))):
        va, te = d[d["set"] == "val"], d[d["set"] == "test"]
        t = choose_threshold(va["label"], va["prob"])
        m = at_threshold(te["label"], te["prob"], t)
        m.update(roc_auc=roc_auc(te["label"], te["prob"]), pr_auc=average_precision(te["label"], te["prob"]),
                 brier=brier(te["label"], te["prob"]), threshold=t, n_test=len(te), n_test_pos=int(te["label"].sum()))
        out[level] = m
    return out


def evaluate_all(pred, by=("model", "variant", "seed")):
    """All models/variants/seeds -> long table: <by...>, level, metrics..."""
    rows = []
    for key, d in pred.groupby(list(by)):
        for level, m in evaluate(d).items():
            rows.append(dict(zip(by, key), level=level, **m))
    return pd.DataFrame(rows)


def mean_std(table, by=("model", "variant", "level"), metrics=METRICS):
    """Mean and standard deviation over seeds (sample std, ddof=1) plus a 'mean ± std' text column."""
    g = table.groupby(list(by))[list(metrics)]
    mean, std = g.mean(), g.std(ddof=1)
    out = mean.add_suffix("_mean").join(std.add_suffix("_std")).reset_index()
    for m in metrics:
        out[m] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(out[f"{m}_mean"], out[f"{m}_std"])]
    out["seeds"] = g.size().to_numpy()
    return out
