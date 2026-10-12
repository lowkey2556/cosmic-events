"""Star-grouped train / validation / test splits (plain Python, shared by every model).

What it does: for each seed, shuffles the labelled stars of each class with that seed and
assigns 70 % of them to train, 15 % to validation and the rest to test (stratified by class,
so every set gets both classes). All windows of a star go to the star's set. The same table
of splits (seed, auid, set) is used by RGA-Net and by the four ML models, so every model
sees exactly the same stars in every set.

Why: windows of one star are strongly correlated; if a star had windows in both train and
test, the test score would measure memorising that star, not generalising to new stars.
"""
import numpy as np
import pandas as pd

SETS = ("train", "val", "test")


def _sizes(n, train, val):
    """Stars per set for one class of n stars; val and test get at least one star each when n >= 3."""
    n_tr, n_va = int(round(train * n)), int(round(val * n))
    if n >= 3:
        n_va = max(n_va, 1)
        n_tr = min(n_tr, n - n_va - 1)
    n_te = n - n_tr - n_va
    return n_tr, n_va, n_te


def star_split(stars, seed, train=0.70, val=0.15):
    """stars: DataFrame with auid, label (0/1). Returns auid, label, set for one seed."""
    rng = np.random.RandomState(int(seed))
    parts = []
    for label in sorted(stars["label"].unique()):
        ids = np.array(sorted(stars.loc[stars["label"] == label, "auid"]))
        rng.shuffle(ids)
        n_tr, n_va, _ = _sizes(len(ids), train, val)
        sets = ["train"] * n_tr + ["val"] * n_va + ["test"] * (len(ids) - n_tr - n_va)
        parts.append(pd.DataFrame({"auid": ids, "label": int(label), "set": sets}))
    out = pd.concat(parts, ignore_index=True)
    assert out["auid"].is_unique, "a star was assigned to more than one set"
    return out


def make_splits(stars, seeds, train=0.70, val=0.15):
    """One split per seed, stacked: seed, auid, label, set."""
    out = pd.concat([star_split(stars, s, train, val).assign(seed=int(s)) for s in seeds], ignore_index=True)
    return out[["seed", "auid", "label", "set"]]


def check_disjoint(splits):
    """Raise if any star is in more than one set for the same seed."""
    n = splits.groupby(["seed", "auid"])["set"].nunique()
    bad = n[n > 1]
    if len(bad):
        raise AssertionError(f"stars in more than one set: {list(bad.index)[:10]}")
