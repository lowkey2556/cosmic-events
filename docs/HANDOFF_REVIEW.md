# Independent Phase 2–4 handoff review — 2026-10-29

The handoff bundle was verified against `c7d9f3ea708772e6b84d339c76ae7be4768da804` and imported as the existing
`v3-phase2-4-verified` branch. Its two commits are `9ccb1f0` and `0937349`. The original implementation and
architecture are preserved. The branch name does not mean real-data Colab results have been verified.

## Findings and corrections

| Area | Finding | Correction |
|---|---|---|
| Notebook setup | The new notebook cloned the old default branch | Explicitly clone/switch to the reviewed branch and update by fast-forward only |
| Notebook execution | IPython shell cells did not raise Python exceptions on command failures | Pipeline, stack setup and smoke-test commands use the checked `sh` helper |
| RGA-Net selection | A setting chosen on seed 1's validation stars was applied to other seeds, whose test stars can overlap that validation set | Tune separately within each split; share the setting across variants only within that split; do not predict test during tuning |
| Training resume | Completed fine-tuning runs depended on configuration but not the data or splits; model files could be overwritten separately | Hash window inputs, splits, vocabulary and training code; include configuration in cache keys and verify the checkpoint hash before reusing predictions |
| ZTF resume | Cleaned data was restored based only on the raw manifest | Include quality cuts and cleaning code in the cache key; old markers are rebuilt from the existing raw files |
| Figures | JSON export failed on Windows for Unicode metric labels | Write Plotly JSON explicitly as UTF-8 |
| Comparison | Ablation ties were counted as wins for the full model | Count strict wins, losses and ties separately; exclude non-finite paired metrics |
| Interpretation | Captions overstated what a large Wilcoxon p-value establishes | State that overlapping repeated splits make these p-values descriptive, and that a large p-value does not establish equivalence |
| Tests | A simulated catalogue outage still contacted the real HEASARC service | Mock the VSX backend boundary so the test is independent of network availability |
| Windows tests | Spark's batch scripts failed on spaces/parentheses in the workspace path | Use Windows short paths for the test Python and Spark installation; Linux/Colab behavior is unchanged |

## Phase coverage and limits

Phase 2 contains the expected Spark ingestion, array cleaning, Hive partition registration, VSX cross-match,
AAVSO exclusion, window creation and A_k validation steps. Phase 3 contains the four MLlib baselines, the two
observer feature settings, ZTF training/pre-training, RGA-Net variants, shared metrics, validation thresholds,
distributed inference and prediction-based comparisons. Phase 4 contains the figure/caption builder, measured
data-flow records and dashboard exports. Unit and synthetic workflow checks exercise these components; their
presence does not establish real-data success.

The current observer calibration uses the complete unlabelled archive before the star split. Held-out magnitudes
can therefore influence observer scores and offsets. Interpret the experiment as transductive archive analysis,
not as a strictly inductive estimate for unseen stars. Whole-light-curve centring also assumes the complete
curve is available, rather than evaluating causal real-time alerts. These boundaries are now explicit in
PROPOSED_MODEL, PREPROCESSING, CLASSIFICATION and REVIEW2_CHECKLIST.

## What remains to verify on real data

The shared Drive listing exposes `cosmic_roja/data` and `cosmic_roja/v3_5GB`, matching the configured layout.
The shared Colab notebook could be downloaded and declares a T4 runtime, but contains no execution outputs.
The full ZTF dataset was not downloaded again during this review.

Run the reviewed notebook on Colab, using the existing Drive data. Check the measured cleaning size rather than
assuming that 8.53 GB raw yields at least 5 GB clean. Inspect label coverage, class counts, excluded overlaps,
split membership, observer overlap and the A_k correlation. Confirm Spark/PyTorch inference agreement on the
real windows. Then inspect all model comparisons, variants, thresholds, charts, captions and dashboard tables.
Only this run can establish real GPU timings and model quality. `ALL RESULTS SHOWN` is a display-completeness
marker, not a verdict that all scientific or dataset requirements passed.

The earlier 17 min 57 s end-to-end timing belongs to the previous developer's shortened synthetic run. It is
not an independently reproduced timing or an estimate for this revised per-split grid on real data.

## Executed validation

- Full collected suite: **73 passed**, 281 dependency deprecation warnings, **512.03 seconds**. Command:
  `python -u -m pytest -vv tests --tb=short`, using Python 3.12.3, Java 8u411, PySpark 3.5.9 and PyTorch 2.10.0 CPU
  on Windows. This covers Spark preprocessing, windows, the MLlib classifiers with both data settings, leakage
  checks, metrics, distributed inference, reports, captions and six new review regressions.
- One further notebook failure-handling regression was added after full-suite collection and run separately:
  **1 passed in 2.71 seconds**. The final tree therefore has **74 distinct passing tests across those runs**.
- The RGA-Net workflow regression actually performs tiny ZTF pre-training, selection and fine-tuning on two
  synthetic star splits, checks unchanged-run reuse and rejects a mismatched checkpoint. It is not a real-data
  performance result.
- Python compilation and `git diff --check` passed. The generated 51-cell notebook matches its builder.
- The full command-line synthetic end-to-end script was not rerun during this independent review. Colab,
  HDFS/YARN/Hive services, full-data GPU training and real-data outputs remain unverified here.
