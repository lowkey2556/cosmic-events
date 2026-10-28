"""Which Hive tables each Spark step needs and makes, so any step can run on a fresh machine.

What it does: on a brand-new Colab machine HDFS and the Hive metastore are empty. Before a Spark
step runs, `plan_steps` checks which tables it needs are missing and puts the steps that make them
in front of it, recursively and in the right order (e.g. distributed inference on a fresh machine
first runs ingest -> preprocess -> score -> windows). Steps whose tables already exist are not rerun.

Why: last time, distributed inference and the scalability test stopped with
TABLE_OR_VIEW_NOT_FOUND when the earlier steps had not been run in that session.
"""

# tables each Spark step reads (Hive tables in the dataset's database)
NEEDS = {
    "ingest": [],
    "preprocess": ["observations_raw", "star_labels"],
    "score": ["obs_clean"],
    "features": ["obs_clean", "obs_upper_limits", "observer_scores", "star_labels"],
    "windows": ["observations_raw", "star_labels", "obs_clean", "preprocess_star_log", "observer_scores"],
    "cluster": ["star_features"],
    "classify": ["window_features", "splits"],
    "infer": ["window_sequences", "window_features", "splits"],
    "ztf_ingest": [],
    "ztf_clean": ["star_labels"],          # ingests the raw ZTF files itself when needed (or restores from Drive)
    "ztf_label": ["ztf_lightcurves_clean", "star_labels"],
    "ztf_windows": ["ztf_labels", "obs_clean"],
    "akval": ["ztf_lightcurves_clean", "star_labels", "obs_clean", "observer_scores"],
    "dashboard": ["star_labels", "obs_clean", "observer_scores", "window_features", "splits"],
}
# tables each Spark step writes
PRODUCES = {
    "ingest": ["observations_raw", "observations_flat", "star_labels"],
    "preprocess": ["obs_clean", "obs_upper_limits", "preprocess_log", "preprocess_star_log"],
    "score": ["observer_residuals", "observer_scores"],
    "features": ["star_features"],
    "windows": ["window_features", "windows", "window_sequences", "splits"],
    "cluster": ["star_clusters"],
    "classify": ["ml_predictions"],
    "infer": ["window_predictions"],
    "ztf_ingest": ["ztf_lightcurves_raw"],
    "ztf_clean": ["ztf_lightcurves_clean"],
    "ztf_label": ["ztf_labels"],
    "ztf_windows": ["ztf_window_features", "ztf_window_sequences"],
    "akval": [],
    "dashboard": [],
}
PRODUCER = {t: step for step, tables in PRODUCES.items() for t in tables}


def plan_steps(steps, exists, only_missing=False):
    """Steps to run, in order: every requested step (unless only_missing), preceded by the steps that make
    any table it needs that does not exist yet. `exists(table)` -> bool. Returns [(step, reason)]."""
    plan, have = [], {t for t in PRODUCER if exists(t)}

    def add(step, reason):
        for t in NEEDS.get(step, []):
            if t not in have:
                if PRODUCER[t] not in [s for s, _ in plan]:
                    add(PRODUCER[t], f"table {t} is missing (needed by {step})")
                have.add(t)
        if step not in [s for s, _ in plan]:
            plan.append((step, reason))
        have.update(PRODUCES.get(step, []))

    for s in steps:
        if only_missing:
            for t in NEEDS.get(s, []):
                if t not in have:
                    add(PRODUCER[t], f"table {t} is missing (needed by {s})")
        else:
            add(s, "requested")
    return plan
