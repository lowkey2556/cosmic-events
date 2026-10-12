"""Step 5 — Windowing, in Spark. Builds the samples used by EVERY model.

What it does:
  1. Cuts each star's cleaned light curve (Hive table `obs_clean`) into time windows: window k
     of a star covers [t0 + k·stride, t0 + k·stride + length) days, t0 = the star's first
     detection (defaults: 100-day windows, 50-day stride). An observation belongs to every
     window that covers it. This is a Spark SQL `explode(sequence(...))`, no Python loops.
  2. Keeps windows with at least `min_obs` observations (default 20). Each window inherits its
     star's label and keeps its `auid`, so splits can be grouped by star.
  3. Per-window summary features for the ML models (Spark SQL aggregates, plus Lomb–Scargle
     per window inside Spark with applyInPandas):
       amplitude (p95 − p5), std, skewness, kurtosis of mag_norm; Lomb–Scargle period and power;
       cadence (median and largest gap, observation count, time span);
       observer features: reliability-weighted mean A_k (each observation carries its observer's
       A_k, so this is the observation-weighted mean), distinct observers, mean |offset| of the
       window's scored observers.
  4. Token sequences for RGA-Net (applyInPandas per window): the window's observations sorted
     by time; if there are more than L (default 200), an evenly spaced subsample of L; if fewer,
     padding plus a mask. Each token: mag_norm, days since window start, log1p(gap since the
     previous observation in the window), band index, observation-type index, uncertainty and
     a missing-uncertainty flag, the observer's agreement score A_k and offset.
  5. Writes Hive tables (Parquet on HDFS): `windows`, `window_features`, `window_sequences`,
     `splits`; and exports the sequences to work/windows_seq.npz in Drive for PyTorch.
  6. Makes the 5 star-grouped 70/15/15 splits (one per seed) of the labelled stars that have
     at least one window, logs window counts per class and per split, and writes
     star_attrition.csv: every star in the raw data, the rows it had after every
     preprocessing step, its windows, whether it is used, and if not, exactly why.

Why: RGA-Net and the ML models must be trained and tested on the SAME samples (windows) and
the SAME star-grouped splits, so the comparison is like-for-like.
"""
import json
import os

import numpy as np
import pandas as pd

from .splits import check_disjoint, make_splits
from .spark_session import save_csv, save_hive_table

BASE_FEATURES = ["amp_robust", "std", "skewness", "kurtosis", "log_ls_period", "ls_power",
                 "gap_median", "gap_max", "n_obs", "time_span"]
OBSERVER_FEATURES = ["w_a", "n_observers", "mean_abs_offset"]
NUM_TOKEN_FEATURES = ["mag_norm", "t_days", "logdt", "unc", "unc_missing", "a_k", "obs_offset"]
LS_SCHEMA = "window_id STRING, ls_period DOUBLE, ls_power DOUBLE, ls_ok BOOLEAN"
SEQ_SCHEMA = ("window_id STRING, n_obs INT, seq_len INT, mag_norm ARRAY<FLOAT>, t_days ARRAY<FLOAT>, "
              "logdt ARRAY<FLOAT>, unc ARRAY<FLOAT>, unc_missing ARRAY<FLOAT>, a_k ARRAY<FLOAT>, "
              "obs_offset ARRAY<FLOAT>, band_idx ARRAY<INT>, type_idx ARRAY<INT>")


# ------------------------------------------------------------------ pure functions (run on executors)
def lomb_scargle_window(pdf, p):
    """Best Lomb–Scargle period and power of one window (columns window_id, t_days, mag_norm)."""
    from astropy.timeseries import LombScargle

    out = dict(window_id=pdf["window_id"].iloc[0], ls_period=np.nan, ls_power=np.nan, ls_ok=False)
    if len(pdf) > p["ls_max_points"]:
        pdf = pdf.sample(n=p["ls_max_points"], random_state=p["ls_seed"])
    t, y = pdf["t_days"].to_numpy(float), pdf["mag_norm"].to_numpy(float)
    span = float(t.max() - t.min()) if len(t) else 0.0
    if len(t) >= 3 and span > 0 and np.std(y) > 0:
        fmin = 1.0 / min(p["ls_max_period_days"], span)
        fmax = 1.0 / p["ls_min_period_days"]
        if fmax > fmin:
            freq, power = LombScargle(t, y).autopower(minimum_frequency=fmin, maximum_frequency=fmax,
                                                      samples_per_peak=p["ls_samples_per_peak"])
            i = int(np.nanargmax(power))
            out.update(ls_period=float(1.0 / freq[i]), ls_power=float(power[i]), ls_ok=True)
    return pd.DataFrame([out])


def build_sequence(pdf, L):
    """One window's observations -> one row of fixed-length (L) token arrays plus seq_len.

    Sorted by time; log1p(gap) is the gap to the previous observation of the full window
    (0 for the first). More than L observations: an evenly spaced subsample of L (first and
    last kept). Fewer: zero padding; tokens at positions >= seq_len are padding.
    """
    d = pdf.sort_values(["t_days", "mag_norm"], kind="mergesort").reset_index(drop=True)
    n = len(d)
    t = d["t_days"].to_numpy(float)
    logdt = np.log1p(np.r_[0.0, np.diff(t)])
    idx = np.round(np.linspace(0, n - 1, L)).astype(int) if n > L else np.arange(n)
    m = len(idx)
    unc = d["uncertainty"].to_numpy(float)
    miss = ~np.isfinite(unc)
    cols = dict(mag_norm=d["mag_norm"].to_numpy(float), t_days=t, logdt=logdt,
                unc=np.where(miss, 0.0, unc), unc_missing=miss.astype(float),
                a_k=d["a_k"].to_numpy(float), obs_offset=d["obs_offset"].to_numpy(float),
                band_idx=d["band_idx"].to_numpy(int), type_idx=d["type_idx"].to_numpy(int))
    row = dict(window_id=d["window_id"].iloc[0], n_obs=n, seq_len=m)
    for k, v in cols.items():
        pad = np.zeros(L, dtype=int if k.endswith("_idx") else np.float32)
        pad[:m] = v[idx]
        row[k] = pad.tolist()
    return pd.DataFrame([row])


def vocab(counts, max_n):
    """Index per category: most frequent first, starting at 2 (0 = padding, 1 = any other value)."""
    top = list(counts.sort_values(["n", "value"], ascending=[False, True])["value"].head(max_n))
    return {v: i + 2 for i, v in enumerate(top)}


# ------------------------------------------------------------------ Spark
def assign_windows(spark, cfg):
    """Temp view `w_obs`: one row per (observation, window) for windows with >= min_obs observations."""
    w = cfg.section("windows")
    L, S, n_min = float(w["length_days"]), float(w["stride_days"]), int(w["min_obs"])
    if S > L:
        raise ValueError("windows.stride_days must be <= windows.length_days")
    bands = spark.sql(f"SELECT band AS value, COUNT(*) AS n FROM {cfg.table('obs_clean')} GROUP BY band").toPandas()
    types = spark.sql(f"SELECT obs_type AS value, COUNT(*) AS n FROM {cfg.table('obs_clean')} "
                      "WHERE obs_type IS NOT NULL GROUP BY obs_type").toPandas()
    vb, vt = vocab(bands, int(w["max_bands"])), vocab(types, int(w["max_obs_types"]))
    spark.createDataFrame(pd.DataFrame({"band": list(vb), "band_idx": list(vb.values())}) if vb else
                          pd.DataFrame({"band": ["-"], "band_idx": [1]})).createOrReplaceTempView("w_vb")
    spark.createDataFrame(pd.DataFrame({"obs_type": list(vt), "type_idx": list(vt.values())}) if vt else
                          pd.DataFrame({"obs_type": ["-"], "type_idx": [1]})).createOrReplaceTempView("w_vt")
    spark.sql(f"""
        WITH o AS (
          SELECT c.auid, c.jd, c.mag_norm, c.band, c.obs_type, c.observer, c.uncertainty,
                 COALESCE(s.score, 0.0) AS a_k,
                 CASE WHEN s.low_n OR s.median_residual IS NULL THEN 0.0 ELSE s.median_residual END AS obs_offset,
                 COALESCE(s.low_n, true) AS low_n,
                 MIN(c.jd) OVER (PARTITION BY c.auid) AS t0
          FROM {cfg.table('obs_clean')} c LEFT JOIN {cfg.table('observer_scores')} s ON c.observer = s.observer),
        e AS (
          SELECT o.*, explode(sequence(GREATEST(0, CAST(FLOOR((jd - t0 - {L}) / {S}) AS INT) + 1),
                                       CAST(FLOOR((jd - t0) / {S}) AS INT))) AS k
          FROM o),
        f AS (
          SELECT e.*, jd - (t0 + k * {S}) AS t_days FROM e),
        g AS (
          SELECT f.*, COUNT(*) OVER (PARTITION BY auid, k) AS n_in_window
          FROM f WHERE jd - (t0 + k * {S}) >= 0 AND jd - (t0 + k * {S}) < {L})
        SELECT CONCAT(g.auid, '#', LPAD(CAST(g.k AS STRING), 4, '0')) AS window_id, g.*,
               COALESCE(vb.band_idx, 1) AS band_idx, COALESCE(vt.type_idx, 1) AS type_idx
        FROM g LEFT JOIN w_vb vb ON g.band = vb.band LEFT JOIN w_vt vt ON g.obs_type = vt.obs_type
        WHERE g.n_in_window >= {n_min}""").cache().createOrReplaceTempView("w_obs")
    return vb, vt


def window_features(spark, cfg):
    f = cfg.section("features")
    w = cfg.section("windows")
    p = dict(ls_max_points=int(w["ls_max_points"]), ls_seed=int(w["ls_seed"]),
             ls_min_period_days=float(f["ls_min_period_days"]), ls_max_period_days=float(f["ls_max_period_days"]),
             ls_samples_per_peak=int(f["ls_samples_per_peak"]))
    ls = spark.table("w_obs").select("window_id", "t_days", "mag_norm").groupBy("window_id") \
        .applyInPandas(lambda d: lomb_scargle_window(d, p), schema=LS_SCHEMA)
    ls.createOrReplaceTempView("w_ls")
    return spark.sql(f"""
        WITH g AS (
          SELECT window_id, t_days, mag_norm, observer, a_k,
                 t_days - LAG(t_days) OVER (PARTITION BY window_id ORDER BY t_days) AS gap
          FROM w_obs),
        agg AS (
          SELECT window_id,
                 percentile(mag_norm, 0.95) - percentile(mag_norm, 0.05) AS amp_robust,
                 stddev(mag_norm) AS std, skewness(mag_norm) AS skewness, kurtosis(mag_norm) AS kurtosis,
                 percentile(gap, 0.5) AS gap_median, MAX(gap) AS gap_max,
                 CAST(COUNT(*) AS DOUBLE) AS n_obs, MAX(t_days) - MIN(t_days) AS time_span,
                 AVG(a_k) AS w_a, CAST(COUNT(DISTINCT observer) AS DOUBLE) AS n_observers
          FROM g GROUP BY window_id),
        offs AS (
          SELECT window_id, AVG(ABS(obs_offset)) AS mean_abs_offset
          FROM (SELECT DISTINCT window_id, observer, obs_offset FROM w_obs WHERE NOT low_n) GROUP BY window_id),
        idx AS (
          SELECT DISTINCT window_id, auid, k, t0 + k * {float(w['stride_days'])} AS window_start_jd FROM w_obs)
        SELECT idx.window_id, idx.auid, idx.k, idx.window_start_jd, l.label,
               COALESCE(l.name, l.data_name) AS name, l.vsx_type,
               agg.amp_robust, agg.std, agg.skewness, agg.kurtosis,
               CASE WHEN ls.ls_ok THEN LOG10(ls.ls_period) END AS log_ls_period,
               CASE WHEN ls.ls_ok THEN ls.ls_power END AS ls_power,
               agg.gap_median, agg.gap_max, agg.n_obs, agg.time_span,
               agg.w_a, agg.n_observers, offs.mean_abs_offset
        FROM idx JOIN agg ON idx.window_id = agg.window_id
        LEFT JOIN w_ls ls ON idx.window_id = ls.window_id
        LEFT JOIN offs ON idx.window_id = offs.window_id
        LEFT JOIN {cfg.table('star_labels')} l ON idx.auid = l.auid""")


def window_sequences(spark, cfg):
    L = int(cfg.section("windows")["seq_len"])
    cols = ["window_id", "t_days", "mag_norm", "uncertainty", "a_k", "obs_offset", "band_idx", "type_idx"]
    return spark.table("w_obs").select(*cols).groupBy("window_id") \
        .applyInPandas(lambda d: build_sequence(d, L), schema=SEQ_SCHEMA)


def export_npz(spark, cfg, vb, vt):
    """window_sequences (+ labels) -> work/windows_seq.npz (float32 arrays, N x L x features)."""
    seq = spark.sql(f"""SELECT s.*, f.auid, f.label FROM {cfg.table('window_sequences')} s
                        JOIN {cfg.table('window_features')} f ON s.window_id = f.window_id
                        ORDER BY s.window_id""").toPandas()
    L = int(cfg.section("windows")["seq_len"])
    num = np.stack([np.stack([np.asarray(v, np.float32) for v in seq[c]]) for c in NUM_TOKEN_FEATURES], axis=-1)
    mask = np.arange(L)[None, :] < seq["seq_len"].to_numpy()[:, None]
    path = os.path.join(cfg.work_dir, "windows_seq.npz")
    np.savez_compressed(path, num=num, mask=mask,
                        band=np.stack([np.asarray(v, np.int64) for v in seq["band_idx"]]),
                        otype=np.stack([np.asarray(v, np.int64) for v in seq["type_idx"]]),
                        label=seq["label"].fillna(-1).astype(int).to_numpy(),
                        window_id=seq["window_id"].astype(str).to_numpy(), auid=seq["auid"].astype(str).to_numpy(),
                        feature_names=np.array(NUM_TOKEN_FEATURES))
    with open(os.path.join(cfg.work_dir, "vocab.json"), "w") as fh:
        json.dump(dict(band=vb, obs_type=vt, n_band=len(vb) + 2, n_type=len(vt) + 2), fh, indent=1)
    print(f"saved {path}: {num.shape[0]} windows x {L} tokens x {num.shape[2]} numeric features")
    return path


def star_attrition(spark, cfg, per_star_windows, splits):
    """Every star in the raw data: rows after each step, windows, used or not, and why."""
    stars = spark.sql(f"""SELECT o.auid, COALESCE(l.name, l.data_name) AS name, l.vsx_type, l.primary_type, l.label
                          FROM (SELECT DISTINCT auid FROM {cfg.table('observations_raw')} WHERE auid IS NOT NULL) o
                          LEFT JOIN {cfg.table('star_labels')} l ON o.auid = l.auid""").toPandas()
    steps = spark.table(cfg.table("preprocess_star_log")).toPandas()
    out = stars.merge(steps, on="auid", how="left").merge(per_star_windows, on="auid", how="left")
    step_cols = [c for c in steps.columns if c.startswith("rows_")]
    out[step_cols + ["windows"]] = out[step_cols + ["windows"]].fillna(0).astype("int64")
    first_seed = splits[splits["seed"] == splits["seed"].min()].set_index("auid")["set"]
    out["set_first_seed"] = out["auid"].map(first_seed)
    w = cfg.section("windows")
    names = {"rows_after_dedup": "deduplication", "rows_after_null_keys": "dropping null keys",
             "rows_after_comment_codes": "dropping unreliable comment codes",
             "rows_detections": "removing fainter-than rows (only upper limits left)",
             "rows_after_thinning": "thinning", "rows_after_band_groups":
             f"band normalisation (every band group < {cfg.section('preprocess')['min_group_size']} rows)"}

    def reason(r):
        if pd.isna(r["label"]):
            why = "no VSX match" if pd.isna(r["vsx_type"]) else f"VSX type {r['vsx_type']} not in label map"
            return "not used", f"unlabelled ({why})"
        for c in step_cols[1:]:
            if r[c] == 0:
                return "dropped", f"all rows removed at: {names.get(c, c)}"
        if r["windows"] == 0:
            return "dropped", (f"no {w['length_days']}-day window with >= {w['min_obs']} observations "
                               f"({r['rows_after_band_groups']} clean detections in total)")
        return "used", ""

    out[["status", "reason"]] = out.apply(lambda r: pd.Series(reason(r)), axis=1)
    out["label"] = out["label"].astype("Int64")
    return out.sort_values(["status", "auid"]).reset_index(drop=True)


def run(spark, cfg):
    w = cfg.section("windows")
    sp = cfg.section("split")
    vb, vt = assign_windows(spark, cfg)
    feats = window_features(spark, cfg)
    save_hive_table(feats, cfg, "window_features")
    feats = spark.table(cfg.table("window_features"))
    save_hive_table(feats.select("window_id", "auid", "k", "window_start_jd", "label", "name", "n_obs"), cfg, "windows")
    save_hive_table(window_sequences(spark, cfg), cfg, "window_sequences")

    idx = feats.select("window_id", "auid", "label").toPandas()
    per_star = idx.groupby("auid").size().rename("windows").reset_index()
    labelled = idx.dropna(subset=["label"]).drop_duplicates("auid")[["auid", "label"]]
    labelled["label"] = labelled["label"].astype(int)
    splits = make_splits(labelled, sp["seeds"], float(sp["train"]), float(sp["val"]))
    check_disjoint(splits)
    save_hive_table(spark.createDataFrame(splits), cfg, "splits")
    save_csv(splits, cfg, "splits.csv")
    splits.to_csv(os.path.join(cfg.work_dir, "splits.csv"), index=False)

    # window counts per class, and per class and split for every seed
    lab = idx.assign(cls=idx["label"].map({1: "anomalous", 0: "normal"}).fillna("unlabelled"))
    by_class = lab.groupby("cls").agg(windows=("window_id", "count"), stars=("auid", "nunique")).reset_index()
    print("\nWindows per class (all stars):\n" + by_class.to_string(index=False))
    save_csv(by_class, cfg, "window_counts_by_class.csv")
    ws = idx.dropna(subset=["label"]).merge(splits[["seed", "auid", "set"]], on="auid")
    by_split = ws.assign(cls=ws["label"].map({1: "anomalous", 0: "normal"})) \
        .groupby(["seed", "set", "cls"]).agg(windows=("window_id", "count"), stars=("auid", "nunique")).reset_index()
    print("\nWindows per split and class:\n" + by_split.to_string(index=False))
    save_csv(by_split, cfg, "window_counts_by_split.csv")

    att = star_attrition(spark, cfg, per_star, splits)
    save_csv(att, cfg, "star_attrition.csv")
    dropped = att[att["status"] != "used"]
    print(f"\nStars in the raw data: {len(att)}; used: {int((att['status'] == 'used').sum())}; "
          f"not used: {len(dropped)}")
    if len(dropped):
        print(dropped[["auid", "name", "vsx_type", "status", "reason"]].to_string(index=False))

    summary = pd.DataFrame([dict(windows=len(idx), stars_with_windows=idx["auid"].nunique(),
                                 labelled_stars_in_splits=len(labelled), length_days=w["length_days"],
                                 stride_days=w["stride_days"], min_obs=w["min_obs"], seq_len=w["seq_len"],
                                 bands_indexed=len(vb), obs_types_indexed=len(vt))])
    print("\n" + summary.T.to_string(header=False))
    save_csv(summary, cfg, "window_summary.csv")
    if w.get("export_for_pytorch", True):
        export_npz(spark, cfg, vb, vt)
    spark.catalog.dropTempView("w_obs")
    return summary
