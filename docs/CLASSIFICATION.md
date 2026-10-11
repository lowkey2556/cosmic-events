# Classification, in plain English

*(Review 2, slide 6.)*

## The task

For each **star**, predict whether it is **anomalous** (eruptive or cataclysmic: dwarf novae,
novae, nova-likes, symbiotics, R CrB stars, FU Ori stars, flare stars, polars) or **normal**
(regular periodic: Miras, semiregulars, eclipsing binaries, RR Lyrae, Cepheids, δ Scuti,
BY Dra). The truth comes from the star's VSX type and `config/label_map.yaml`, matched on the
primary type (text before any `:`, `+` or `|`). Stars whose type is not in the map are left out
of training and testing, but are kept for K-Means.

## Features (one row per star)

**Baseline** (what existing observer-agnostic approaches use):

| Feature | What it captures |
|---|---|
| `amp_robust` = p95 − p5 of `mag_norm` | Size of the brightness changes, ignoring outliers |
| `std`, `skewness`, `kurtosis` of `mag_norm` | Spread and shape. Outbursts make the distribution lopsided toward bright values. |
| `log_ls_period`, `ls_power` | Best Lomb–Scargle period (log₁₀ days) and how strong that period is. Periodic stars have strong peaks. |
| `gap_median`, `gap_max` | Typical and longest gap between observations |
| `n_obs`, `time_span` | How much and how long the star was observed |

**Observer features** (the proposed addition; see [PROPOSED_MODEL.md](PROPOSED_MODEL.md)):
`w_s` (observation-weighted mean agreement score of the star's observers), `n_observers`, and
`mean_abs_offset`.

Missing values (for example, stars too sparse for Lomb–Scargle) are filled with the median of
the **training** stars. Every feature is then standardised (mean 0, sd 1) using **training**
stars only.

## Split strategy: by star, never by row

Train/test assignment is made **per star (`auid`)**: 70% train, 30% test, stratified so both
classes appear in both sets. If rows from the same star landed in both train and test, the
model would be tested on a star it had already seen, and the scores would be inflated. The
code asserts that no `auid` is in both sets. The order is a hash of `(auid, seed)`, so the
split can be reproduced.

**Class weights:** each training star gets weight `n_train / (2 × n_its_class)`, so the smaller
class counts as much as the larger one (`weightCol` in MLlib).

## Models (Spark MLlib)

| Model | Features | Role |
|---|---|---|
| Gaussian Naive Bayes | baseline | existing |
| Decision Tree, `maxDepth` ∈ {3, 5} by 3-fold CV on the training stars | baseline | existing |
| Logistic Regression (L2, regParam 0.01) | baseline | existing |
| Random Forest (100 trees) | baseline | existing |
| Gaussian Naive Bayes | baseline + observer | observer-aware, for a symmetric comparison |
| **Decision Tree, `maxDepth` ∈ {3, 5}** | **baseline + observer** | **proposed** |

Each is one MLlib `Pipeline` (Imputer → VectorAssembler → StandardScaler → classifier), fitted
on the training stars only.

## Metrics (on the test stars)

The positive class is *anomalous*.

- **Precision**: of the stars flagged anomalous, the fraction that really are.
- **Recall**: of the truly anomalous stars, the fraction we flagged.
- **F1**: the harmonic mean of precision and recall.
- **Balanced accuracy** = (recall + specificity) / 2. Unlike plain accuracy, it cannot be
  inflated by always predicting the bigger class.
- **Confusion matrix**: the four counts (TN, FP, FN, TP).

Plain accuracy is deliberately not headlined.

## Honesty about Milestone 1

Milestone 1 runs **one fixed seed** (42) on a test set of only 10 stars (5 anomalous), so one
star changes recall by 20 points. These results are **preliminary**. Milestone 2 repeats
everything over 5 seeds and reports mean ± std.
