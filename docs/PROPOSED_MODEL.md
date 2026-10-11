# Proposed model: an observer-aware classifier

*(Review 2, slide 2. Plain-English source for the slides.)*

## The problem

The AAVSO archive is built by thousands of amateur observers. When several of them look at
the **same star on the same night**, they often report different magnitudes. Some differences
are random (a less experienced eye, a poor sky, a noisy camera); some are systematic (one
observer is consistently 0.3 mag too faint because of their comparison chart or filter).

Existing approaches either average all observations together or ignore who made them. Both
treat a careful CCD observer and a careless one as equally trustworthy.

## The idea

We **measure each observer's reliability from the archive itself**, then hand it to the
classifier as an explicit feature. We need no external ground truth about observers. The
yardstick is how well each observer agrees with the *other* observers who watched the same star
at the same time.

## The agreement score, step by step

Notation: star *s*, band *b*, time *t* (Julian Date) and observer *k*. `mag_norm` is the
magnitude after band normalisation (Step 2).

1. **Leave-one-out consensus.** For each observation *i* made by observer *k*:

   ```
   consensus_i = median{ mag_norm_j : star_j = s, band_j = b, |t_j − t_i| ≤ Δt, observer_j ≠ k }
   ```

   The default is Δt = 1 day. The median uses **every** observation in that window *except*
   observer *k*'s own, so *k* cannot agree with themselves. In Spark this is a self-join on
   `(auid, band)` with the time-window and `observer ≠ observer` conditions.

2. **Residual:** `r_i = mag_norm_i − consensus_i`.

3. **Robust scatter of observer k** (the scaled median absolute deviation):

   ```
   σ_k = 1.4826 × median_i( | r_i − median(r_k) | )
   ```

   The constant 1.4826 makes σ_k equal the standard deviation when the errors are Gaussian.
   Because it is median-based, a few wild points do not dominate it. It is measured around the
   observer's *own* median, so a constant offset is **not** counted as scatter.

4. **Agreement score:**

   ```
   A_k = exp( − σ_k / σ_ref ),     σ_ref = median over observers of σ_k
   ```

   The score lies between 0 and 1. A_k → 1 means the observer's reports track the consensus
   closely. An observer with typical scatter gets e⁻¹ ≈ 0.37, and one twice as scattered as
   typical gets e⁻² ≈ 0.14.

5. **Too little evidence.** Some observers have fewer than 20 residuals, so their scatter
   cannot be estimated reliably. They receive the population-median score and are flagged
   `low_n`.

6. **Systematic offset.** `offset_k = median(r_k)` is stored separately. It reports, for
   example, that an observer reads 0.6 mag too faint, without lowering their agreement score.

## From observers to stars (the features the classifier sees)

The classifier works one row per star. For star *s*, with n_k = the number of *s*'s
observations made by observer *k*:

| Feature | Formula | Meaning |
|---|---|---|
| `w_s` | Σ_k A_k·n_k / Σ_k n_k | How trustworthy, on average, the data for this star are |
| `n_observers` | number of distinct observers of *s* | How crowd-sourced the star's record is |
| `mean_abs_offset` | mean over well-scored observers of \|offset_k\| | How much systematic disagreement there is |

## The models compared

| | Features | Models (Spark MLlib) |
|---|---|---|
| **Existing, observer-agnostic** | baseline light-curve features only | Gaussian Naive Bayes, Decision Tree, Logistic Regression, Random Forest |
| **Proposed, observer-aware** | baseline + `w_s`, `n_observers`, `mean_abs_offset` | **Decision Tree** (headline model); Naive Bayes also, for a symmetric comparison |

Every model uses the same split by star, the same preprocessing and imputation (fitted on the
training stars), the same class weights and the same metrics. The **only** difference between
"DT (baseline)" and "DT + observer (proposed)" is the three observer features.

## What would support, or not support, the claim

The claim is supported if the observer-aware models beat their observer-agnostic counterparts
(DT vs DT, NB vs NB) on anomalous-class F1 and balanced accuracy, and if the tree actually splits
on observer features. If they don't help, we report that plainly. Milestone 1 uses one seed on
a small test set, so its numbers are preliminary. Milestone 2 repeats the comparison over 5 seeds.

**A caveat we state openly.** Agreement is measured within a 1-day window, so stars that change
brightness within hours (outbursts, fast eclipses) naturally produce larger residuals for
*everyone*. `w_s` can therefore partly encode how fast a star varies, as well as observer
quality. Milestone 2's interpretation step checks how the tree uses it.
