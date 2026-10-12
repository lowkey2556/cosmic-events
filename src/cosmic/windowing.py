"""The time-window rule, as small plain-Python functions (no Spark needed).

What it does: defines how one star's light curve is cut into windows, so the dataset audit
(which runs before Spark) and the Spark windowing step count windows the same way.

  * Window k of a star starts at t0 + k * stride and covers [start, start + length) days,
    where t0 is the star's first observation; k = 0, 1, ... while start <= the last observation.
  * A window is kept only if it holds at least `min_obs` observations.

It also measures "continuous coverage": the longest stretch of a light curve in which no gap
between consecutive observations is longer than `max_gap_days`.

Why: one shared definition keeps the audit's per-star window counts and the real windowing
step consistent, and makes the rule easy to test.
"""
import numpy as np


def window_counts(jd, length, stride):
    """(starts, counts): start time and number of observations of every window of one star."""
    t = np.sort(np.asarray(jd, dtype=float))
    if len(t) == 0:
        return np.array([]), np.array([], dtype=int)
    n = int(np.floor((t[-1] - t[0]) / stride)) + 1
    starts = t[0] + stride * np.arange(n)
    counts = np.searchsorted(t, starts + length, side="left") - np.searchsorted(t, starts, side="left")
    return starts, counts


def count_windows(jd, length, stride, min_obs):
    """Number of windows of one star that hold at least `min_obs` observations."""
    return int((window_counts(jd, length, stride)[1] >= min_obs).sum())


def longest_coverage(jd, max_gap_days):
    """Longest span (days) of observations with no gap between consecutive ones > max_gap_days."""
    t = np.sort(np.asarray(jd, dtype=float))
    if len(t) < 2:
        return 0.0
    breaks = np.flatnonzero(np.diff(t) > max_gap_days)
    first = np.concatenate([[0], breaks + 1])
    last = np.concatenate([breaks, [len(t) - 1]])
    return float((t[last] - t[first]).max())
