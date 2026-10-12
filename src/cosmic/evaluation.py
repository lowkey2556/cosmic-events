"""Metrics, used identically for RGA-Net and the four ML models (plain NumPy).

What it does: every model writes the same prediction table — one row per window with
model, variant, seed, set (val/test), window_id, auid, label and prob (the predicted
probability of "anomalous"). From that table this module:
  * picks the decision threshold on the VALIDATION windows (the threshold with the best F1),
    never on test;
  * computes on the TEST windows: precision, recall and F1 for the anomalous class,
    balanced accuracy ((recall + specificity) / 2), ROC-AUC and PR-AUC (average precision);
  * repeats the same at star level: each star's probability is the mean of its windows'
    probabilities, with its own threshold chosen on the validation stars.

Why: one implementation for all models makes the comparison like-for-like, and keeping
the threshold choice on validation keeps the test set untouched.
"""
import numpy as np
import pandas as pd

METRICS = ["precision", "recall", "f1", "balanced_accuracy", "roc_auc", "pr_auc"]


def roc_auc(y, p):
    """Area under the ROC curve (Mann-Whitney statistic; ties count one half). NaN if one class is missing."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pr_points(y, p):
    """Precision and recall at every distinct threshold (highest first), as in scikit-learn."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    order = np.argsort(-p, kind="mergesort")
    y, p = y[order], p[order]
    last = np.r_[np.flatnonzero(np.diff(p)), len(p) - 1]  # last index of each distinct score
    tp = np.cumsum(y)[last]
    fp = (last + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / max(int(y.sum()), 1)
    return precision, recall, p[last]


def average_precision(y, p):
    """PR-AUC as average precision: sum over thresholds of (R_n - R_{n-1}) * P_n."""
    if int(np.sum(y)) == 0:
        return float("nan")
    precision, recall, _ = pr_points(y, p)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_points(y, p):
    """False- and true-positive rates at every distinct threshold, starting at (0, 0)."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    order = np.argsort(-p, kind="mergesort")
    y, p = y[order], p[order]
    last = np.r_[np.flatnonzero(np.diff(p)), len(p) - 1]
    tp = np.cumsum(y)[last]
    fp = (last + 1) - tp
    return np.r_[0.0, fp / max(int((1 - y).sum()), 1)], np.r_[0.0, tp / max(int(y.sum()), 1)]


def at_threshold(y, p, t):
    """Confusion counts and threshold metrics for 'anomalous if prob >= t'."""
    y, pred = np.asarray(y, int), (np.asarray(p, float) >= t).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    return dict(precision=prec, recall=rec, f1=f1, balanced_accuracy=(rec + spec) / 2,
                tp=tp, fp=fp, fn=fn, tn=tn)


def choose_threshold(y, p):
    """Threshold with the best validation F1 (ties: better balanced accuracy, then the higher threshold)."""
    best, best_key = 0.5, None
    for t in np.unique(np.asarray(p, float)):
        m = at_threshold(y, p, t)
        key = (m["f1"], np.nan_to_num(m["balanced_accuracy"]), t)
        if best_key is None or key > best_key:
            best, best_key = float(t), key
    return best


def star_level(pred):
    """Mean window probability per star (one row per seed, set, auid)."""
    keys = [c for c in ("model", "variant", "seed", "set", "auid", "label") if c in pred.columns]
    return pred.groupby(keys, as_index=False)["prob"].mean()


def evaluate(pred):
    """pred: one model/variant/seed with set in {val, test}. Returns window- and star-level test metrics."""
    out = {}
    for level, d in (("window", pred), ("star", star_level(pred))):
        va, te = d[d["set"] == "val"], d[d["set"] == "test"]
        t = choose_threshold(va["label"], va["prob"])
        m = at_threshold(te["label"], te["prob"], t)
        m.update(roc_auc=roc_auc(te["label"], te["prob"]), pr_auc=average_precision(te["label"], te["prob"]),
                 threshold=t, n_test=len(te), n_test_pos=int(te["label"].sum()))
        out[level] = m
    return out


def evaluate_all(pred):
    """All models/variants/seeds -> long table: model, variant, seed, level, metrics..."""
    rows = []
    for (model, variant, seed), d in pred.groupby(["model", "variant", "seed"]):
        for level, m in evaluate(d).items():
            rows.append(dict(model=model, variant=variant, seed=seed, level=level, **m))
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
