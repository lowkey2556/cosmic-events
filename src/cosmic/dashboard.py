"""Step 10b — Dashboard data: tidy star-schema tables for the Review 3 dashboard (Spark).

What it does: builds six tidy tables from the Hive tables and the saved result files, and writes each one three
ways: CSV and Parquet in V3_ROOT/dashboard_data/, and a Hive table `dash_<name>` (Parquet on HDFS).
  dim_star                  one row per AAVSO star: name, VSX type, class label, position, observations,
                            observers, windows, the star's set in split seed 1
  dim_observer              one row per observer: A_k, offset, observation type, observations, stars
  fact_window_predictions   one row per (model, variant, seed, window): set, label, probability
                            (RGA-Net from the Spark inference table; the ML models from their saved predictions)
  fact_model_metrics        one row per (model, variant, seed, level): every metric, threshold, confusion counts
  fact_cleaning_steps       one row per cleaning step and dataset (AAVSO, ZTF): rows kept and removed
  fact_observations_sample  a seeded random sample of cleaned AAVSO observations with the observer's A_k
Also writes DATA_DICTIONARY.md (every column with its type and meaning). Colours and display names for a
dashboard come from config/style.yaml (copied next to the tables).
"""
import os
import shutil

import pandas as pd

from .config import REPO_ROOT
from .spark_session import save_hive_table

DESCRIPTIONS = {
    "auid": "AAVSO unique star identifier (ZTF light curves: 'ZTF' + objectid)",
    "name": "star name", "vsx_type": "VSX variability type (full)", "primary_type": "first VSX type code",
    "label": "1 = anomalous (eruptive/cataclysmic), 0 = normal variable", "ra": "right ascension (deg)",
    "dec": "declination (deg)", "sky_region": "declination band", "n_obs": "cleaned observations",
    "n_observers": "distinct observers", "windows": "100-day windows used", "set_seed1": "train/val/test in seed 1",
    "observer": "AAVSO observer code", "obs_type": "observation type (most common for the observer)",
    "A_k": "observer agreement score (0-1, higher = agrees better with other observers)",
    "offset_mag": "observer's median offset from the consensus (mag)", "n_stars": "stars observed",
    "n_residuals": "observations with a consensus to compare with", "low_n": "too little overlap: A_k is a fill-in",
    "model": "model key (NB, DT, RF, LR, RGA-Net)", "model_name": "display name", "role": "prior approach / "
    "proposed / framework on classic model / ablation", "variant": "feature set or RGA-Net variant",
    "seed": "split seed", "set": "train/val/test (none = unlabelled star)", "window_id": "star#window index",
    "prob": "predicted probability of 'anomalous'", "level": "window or star", "mcc": "Matthews correlation",
    "roc_auc": "ROC-AUC", "pr_auc": "AUPRC (average precision)", "kappa": "Cohen's kappa", "brier": "Brier score",
    "threshold": "decision threshold chosen on validation (max MCC)", "tp": "true positives", "fp": "false positives",
    "fn": "false negatives", "tn": "true negatives", "dataset": "AAVSO or ZTF", "step": "cleaning step",
    "rows": "rows (AAVSO observations / ZTF epochs) after the step", "removed": "rows removed by the step",
    "jd": "Julian Date (days)", "mag": "magnitude", "band": "photometric band", "mag_norm": "magnitude minus the "
    "star-band median", "uncertainty": "reported uncertainty (mag)",
}


def _write(spark, cfg, pdf, name):
    d = cfg.dashboard_dir
    pdf.to_csv(os.path.join(d, f"{name}.csv"), index=False)
    pdf.to_parquet(os.path.join(d, f"{name}.parquet"), index=False)
    save_hive_table(spark.createDataFrame(pdf), cfg, f"dash_{name}")
    print(f"  {name}: {len(pdf):,} rows -> CSV, Parquet, Hive {cfg.table('dash_' + name)}")
    return pdf


def run(spark, cfg):
    from .compare import display, role
    out, work = cfg.output_dir, cfg.work_dir
    t = cfg.table
    tables = {}
    tables["dim_star"] = spark.sql(f"""
        SELECT l.auid, COALESCE(l.name, l.data_name) AS name, l.type AS vsx_type, l.primary_type,
               CAST(l.label AS INT) AS label, l.ra, l.dec, l.sky_region,
               COALESCE(o.n_obs, 0) AS n_obs, COALESCE(o.n_observers, 0) AS n_observers,
               COALESCE(w.windows, 0) AS windows, s.`set` AS set_seed1
        FROM {t('star_labels')} l
        LEFT JOIN (SELECT auid, COUNT(*) AS n_obs, COUNT(DISTINCT observer) AS n_observers FROM {t('obs_clean')}
                   GROUP BY auid) o ON l.auid = o.auid
        LEFT JOIN (SELECT auid, COUNT(*) AS windows FROM {t('window_features')} GROUP BY auid) w ON l.auid = w.auid
        LEFT JOIN (SELECT auid, `set` FROM {t('splits')} WHERE seed = 1) s ON l.auid = s.auid""").toPandas()
    tables["dim_observer"] = spark.sql(f"""
        SELECT observer, obs_type, score AS A_k, median_residual AS offset_mag, n_obs, n_stars, n_residuals, low_n
        FROM {t('observer_scores')}""").toPandas()
    parts = []
    if spark.catalog.tableExists("window_predictions", cfg.db):
        parts.append(spark.sql(f"""SELECT 'RGA-Net' AS model, variant, seed, `set`, window_id, auid, label, prob
                                   FROM {t('window_predictions')}""").toPandas())
    ml = os.path.join(work, "predictions_ml.csv")
    if os.path.exists(ml):
        parts.append(pd.read_csv(ml)[["model", "variant", "seed", "set", "window_id", "auid", "label", "prob"]])
    if parts:
        f = pd.concat(parts, ignore_index=True)
        f["model_name"] = [display(m, v) for m, v in zip(f["model"], f["variant"])]
        f["role"] = [role(m, v) for m, v in zip(f["model"], f["variant"])]
        tables["fact_window_predictions"] = f
    ps = os.path.join(out, "per_seed_metrics.csv")
    if os.path.exists(ps):
        tables["fact_model_metrics"] = pd.read_csv(ps).rename(columns={"display_name": "model_name"})
    steps = []
    for ds, fname in (("AAVSO", "preprocess_row_counts.csv"), ("ZTF", "ztf_clean_steps.csv")):
        p = os.path.join(out, fname)
        if os.path.exists(p):
            d = pd.read_csv(p)
            steps.append(d[["step", "rows", "removed"]].assign(dataset=ds, step_order=range(len(d))))
    if steps:
        tables["fact_cleaning_steps"] = pd.concat(steps, ignore_index=True)[["dataset", "step_order", "step", "rows",
                                                                             "removed"]]
    n = spark.table(t("obs_clean")).count()
    frac = min(1.0, 100000 / max(n, 1))
    tables["fact_observations_sample"] = spark.sql(f"""
        SELECT c.auid, c.jd, c.mag, c.mag_norm, c.band, c.obs_type, c.observer, c.uncertainty, s.score AS A_k
        FROM {t('obs_clean')} c LEFT JOIN {t('observer_scores')} s ON c.observer = s.observer""") \
        .sample(fraction=frac, seed=7).toPandas()
    os.makedirs(cfg.dashboard_dir, exist_ok=True)
    for name, pdf in tables.items():
        _write(spark, cfg, pdf, name)
    shutil.copy(os.path.join(REPO_ROOT, "config", "style.yaml"), os.path.join(cfg.dashboard_dir, "style.yaml"))
    dictionary(cfg, tables)


def dictionary(cfg, tables):
    lines = ["# Data dictionary — dashboard_data (generated by the dashboard step)\n",
             "Each table is saved as CSV and Parquet in V3_ROOT/dashboard_data/ and as the Hive table "
             f"`{cfg.db}.dash_<table>`. Colours and display names: style.yaml (same folder).\n"]
    for name, pdf in tables.items():
        lines += [f"## {name} ({len(pdf):,} rows)\n", "| Column | Type | Meaning |", "|---|---|---|"]
        for c in pdf.columns:
            lines.append(f"| {c} | {pdf[c].dtype} | {DESCRIPTIONS.get(c, '')} |")
        lines.append("")
    path = os.path.join(cfg.dashboard_dir, "DATA_DICTIONARY.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    shutil.copy(path, cfg.repo_path(os.path.join("docs", "DATA_DICTIONARY.md")))
    print(f"saved {path}")
