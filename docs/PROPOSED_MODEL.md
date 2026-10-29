# Proposed model: RGA-Net (Reliability-Gated Attention Network)

*(Review 2, "Proposed model" slides. Plain-English source for the slides. Results are never typed here: they are
in `v3_5GB/outputs/` after a run — `COMPARISON.md`, `comparison_table.csv`, `ablation_table.csv`, `rganet_lambda.csv`.)*

## 1. The problem

The AAVSO archive is built by thousands of amateur observers. When several of them observe the
**same star on the same night**, they often report different magnitudes. Some differences are
random (a less experienced eye, a poor sky, a noisy camera); some are systematic (one observer
is always 0.3 mag too faint because of their comparison chart or filter).

We want to tell **anomalous (eruptive) stars** — dwarf novae, novae, symbiotic and R CrB
stars — from **normal periodic** variables (Miras, semiregulars, eclipsing binaries, Cepheids,
RR Lyrae stars), using 100-day windows of their light curves. The data are:

- **irregularly sampled**: observations arrive whenever someone looks, with gaps of hours to weeks;
- **of mixed quality**: every observation comes from a known observer whose reliability differs;
- **without a known period**: eruptive stars have no stable period, and a 100-day window is
  often shorter than a Mira's period.

## 2. Step one: measure each observer's reliability (A_k)

Before any model runs, Spark measures every observer's reliability **from the archive itself**
(Step 3 of the pipeline). For observation *i* by observer *k* of star *s* in band *b* at time *t*
(`mag_norm` = magnitude after band normalisation):

1. **Leave-one-out consensus:**
   `consensus_i = median{ mag_norm_j : same star, same band, |t_j − t_i| ≤ 1 day, observer_j ≠ k }`
   (a Spark SQL self-join; observer *k* never agrees with themselves).
2. **Residual:** `r_i = mag_norm_i − consensus_i`.
3. **Robust scatter:** `σ_k = 1.4826 × median_i | r_i − median(r_k) |` (scaled MAD, measured
   around the observer's own median, so a constant offset is not counted as scatter).
4. **Agreement score:** `A_k = exp(−σ_k / σ_ref)`, `σ_ref` = median of all `σ_k`. A_k lies in
   (0, 1]; a typical observer scores e⁻¹ ≈ 0.37. Observers with fewer than 20 residuals get
   the population median and are flagged `low_n`.
5. **Offset:** `offset_k = median(r_k)`, the observer's systematic shift, kept separately.

Every observation then carries its observer's `A_k` and `offset_k`.

## 3. RGA-Net, in plain English

RGA-Net reads one 100-day window as a **sequence of observations** (up to L = 200; an evenly
spaced subsample if there are more, padding with a mask if fewer) and outputs the probability
that the star is anomalous.

**What it is built on, and what is ours.** RGA-Net is a **Transformer encoder** (Vaswani et al.,
2017): multi-head scaled dot-product self-attention, feed-forward layers, residual connections
and layer normalisation, unchanged in structure. Two parts are adapted from existing work: the
continuous-time encoding that replaces integer positions (the same idea as time-aware attention
models such as mTAN and ASTROMER), and attention pooling into one window vector. **Our
contribution is the reliability-gated attention modification**: an additive term
`λ · log(A_j + ε)` in every attention score (and in the pooling score), where A_j is the measured
reliability of the observer who made observation j and λ ≥ 0 is learned. In plain words,
**observations from more reliable observers get more attention, by an amount the network learns.**
Remove the gate (λ = 0) and RGA-Net is an ordinary Transformer encoder with time encoding; that
is exactly the "no gate" ablation.

### Inputs (one token per observation)

`mag_norm`, days since the window start, `log1p` of the gap since the previous observation,
band, observation type (visual / CCD / DSLR / …), uncertainty plus a "missing uncertainty" flag,
the observer's agreement score `A_k`, and the observer's offset.

### Architecture

1. **Token embedding.** `x_i = W·u_i + E_band[band_i] + E_type[type_i]` — a linear layer on the
   numeric features `u_i` (standardised with the training windows' mean and standard deviation)
   plus learned embeddings for band and observation type.
2. **Continuous-time encoding (no position indices).** With *t_i* the actual number of days since
   the window start and 16 periods *P_m* log-spaced from 0.1 to 1000 days:
   `τ(t_i) = W_t · [ sin(2π t_i / P_m), cos(2π t_i / P_m) ]_{m=1..16}`, added to `x_i`.
   Two observations 3 days apart get the same relative encoding wherever they are in the
   sequence, and gaps are represented as real time, not as "next token".
3. **Reliability-gated self-attention** (2 layers, 4 heads, width 64, pre-LayerNorm, residual
   connections, feed-forward width 128):

   ```
   score(i, j) = (q_i · k_j) / √d_head  +  λ · log(A_j + ε)        λ = softplus(ρ) ≥ 0,  ε = 0.001
   α(i, j)     = softmax_j( score(i, j) )   over real (non-padding) tokens only
   ```

   Because `exp(λ · log A_j) = A_j^λ`, this is the same as multiplying every attention weight
   by `A_j^λ` and renormalising: an observation from an observer with A = 0.8 gets
   (0.8 / 0.2)^λ times the weight of an otherwise identical one from an observer with A = 0.2.
   **λ is learned** (one per layer, shared by its heads): λ = 0 switches the gate off; the
   network decides from the data how strongly to trust reliable observers.
4. **Reliability-gated attention pooling.** One learned query vector *w* scores every token,
   `s_i = w·h_i / √d + λ_pool · log(A_i + ε)`, and the window vector is `z = Σ_i softmax(s)_i h_i`,
   with its own learned `λ_pool`.
5. **Head.** `MLP(z)` (64 → 64 → 1, GELU, dropout 0.2) gives one logit; `sigmoid(logit)` is the
   probability of "anomalous".

About 75,000 trainable parameters with the default settings (74,964 for the full model with a 14-band /
8-type vocabulary; +832 with the hybrid head): deliberately small, because there are only ~60 AAVSO stars.
**Optional hybrid head** (chosen on validation, see below): the window's 13 summary features (imputed and standardised
with training-window statistics) are concatenated to `z` before the MLP.

### Training (Phase 3)

1. **Pre-training once on ZTF** (`ztf_windows` → `work/ztf_windows_seq.npz`): up to 50,000 class-balanced windows of
   labelled ZTF light curves, cut exactly like the AAVSO windows. ZTF is one professional survey, so every token has
   A = 1 and offset = 0: the gate has nothing to act on, and the pre-trained encoder has **no λ parameters**. No ZTF
   object within 10″ of any AAVSO star is in this pool (`ztf_label`), so no test star is seen in pre-training. 10 % of
   the ZTF light curves (by object hash) are held out for early stopping (patience 3, up to 12 epochs). A second encoder
   without the A_k/offset token features is pre-trained the same way for the observer-blind ablation.
2. **Validation grid, within each split:** learning rate {1e-4, 3e-4} × dropout {0.1, 0.3} × hybrid head {no, yes},
   each fine-tuned on that split's training stars and judged only on its validation stars (best validation AUPRC).
   The chosen setting is shared by the four variants within that split (`rganet_grid.csv`). Settings are never
   selected using another split: its validation stars could be test stars here. Grid search does not predict test.
3. **Fine-tuning per seed** on the AAVSO training stars: all layers trained, gate active (λ starts at softplus(0) =
   0.69), input statistics taken from pre-training so the encoder sees inputs on the scale it learned. Class-weighted
   binary cross-entropy (weight of "anomalous" = negatives / positives in the training windows), AdamW (weight decay
   0.01), gradient-norm clipping at 1.0, batches of 64, up to 60 epochs, **early stopping on validation AUPRC**
   (patience 10; the best epoch's weights are kept), fixed seeds.
4. **Training-only augmentation** (never on validation or test windows; enforced by a test): random observation
   dropout (10 %), a random time shift of the whole window (±5 days), and magnitude noise ~ N(0, uncertainty) per
   observation (0.05 mag where none was reported).
5. **10 star-grouped 70/15/15 splits** (seeds 1–10), the same for every model. Thresholds for MCC and κ are chosen on
   validation (maximum MCC); the test windows are used once, for the final numbers.
6. **Resumable:** every finished pre-training, grid point and (variant, seed) run is saved in `work/` and reused if
   Colab disconnects. Content hashes of the window data, splits, vocabulary and training code, together with the
   training configuration, invalidate stale caches. A reused run must also match its saved checkpoint's hash.

**Evaluation boundary:** the observer scores and offsets are currently computed from the whole unlabelled AAVSO
archive before the star split. This is a transductive archive experiment: held-out magnitudes can influence those
features, although held-out labels do not enter training. It must not be described as a strictly inductive estimate
for unseen stars. A claim about deployment to new stars requires scores calibrated on separate history or on each
split's training stars, with a documented fallback for unseen observers. Per-star band centring likewise assumes
the complete observed light curve is available; this is not a causal real-time alert evaluation.

### Inference at scale (Spark)

Training runs once, on the GPU. Applying the trained model is then done **by Spark** (Step 8b):
the saved models (one per split seed) are broadcast to the executors, and `mapInPandas` (a pandas
UDF over batches of windows) scores every window of every star in parallel, with exactly the
same input preparation as in training. The results go to the Hive table `window_predictions`,
and the Spark probabilities on the validation/test windows are checked against PyTorch's own
(`distributed_inference_check.csv`).

### Ablations (identical splits, seeds and chosen setting)

| Variant | ZTF pre-training | Gate | A_k and offset as token features |
|---|---|---|---|
| RGA-Net full (proposed) | yes | learned λ (attention and pooling) | yes |
| no ZTF pre-training | no (trained from scratch on AAVSO) | learned λ | yes |
| no gate | yes | λ fixed at 0 | yes |
| observer-blind | yes (blind encoder) | λ fixed at 0 | **no** |

*Full vs no-pretrain* measures what ZTF pre-training adds. *Full vs no-gate* isolates the gate itself. *No-gate vs
observer-blind* shows whether reliability information helps even as a plain input. Paired per-seed differences and
Wilcoxon p-values are in `ablation_paired.csv`; learned λ per layer in `rganet_lambda.csv`.

## 4. Why RGA-Net?

The task needs a model that **(1)** is deep learning (a faculty requirement), **(2)** handles
irregular timing, **(3)** needs no known period, and **(4)** uses observer reliability.

| Candidate | Where it falls short |
|---|---|
| **1D CNN** (e.g. on phase-folded curves) | Convolutions assume evenly spaced samples, so irregular data must first be binned or interpolated (fails 2). Phase-folding needs a known period (fails 3), which eruptive stars do not have. No notion of who made each point (fails 4). |
| **LSTM / GRU** | Reads observations as an ordered list; the time gaps are lost unless added by hand, and long gaps are handled poorly (weak on 2). No mechanism to weight observations by reliability (fails 4). |
| **Plain MLP** on summary features | Not a sequence model: it only sees hand-made summaries (amplitude, period, …), so it cannot learn from the light curve itself; the period feature itself depends on the window (weak on 2 and 3). Reliability can only enter as one more averaged feature (weak on 4). |
| **Standard Transformer** | Uses integer position indices, which treat a 1-hour gap and a 3-week gap the same (fails 2 unless the positions are replaced by real time). All observations compete for attention on content alone; nothing says that a careful observer's point should count more (fails 4). |
| **RGA-Net** | Deep (1). Continuous-time encoding of real elapsed days (2). Works on raw windows; no period is needed (3). Observer reliability enters attention directly, with a learned strength λ (4). |

## 5. Related models, compared by design (not by performance)

| Model | Handles irregular timing | Uses observer reliability | Needs a known period |
|---|---|---|---|
| 1D CNN on phase-folded light curves (e.g. Shallue & Vanderburg 2018) | No (folding/binning first) | No | **Yes** |
| LSTM / GRU (Hochreiter & Schmidhuber 1997; Cho et al. 2014); RNNs on light curves (e.g. Charnock & Moss 2017; Naul et al. 2018) | Partly (time gaps as extra inputs) | No | No |
| GRU-D (Che et al. 2018) | Yes (learned decay over time gaps) | No | No |
| Phased LSTM (Neil et al. 2016) | Yes (time gate driven by timestamps) | No | No |
| mTAN, multi-time attention (Shukla & Marlin 2021) | Yes (continuous-time attention) | No | No |
| ASTROMER (Donoso-Oliva et al. 2023) | Yes (time-based positional encoding) | No | No |
| ATAT (Cádiz-Leyton et al.) | Yes | No | No |
| Time-series transformer of Allam & McEwen | Yes | No | No |
| Crowd-learning models: Dawid & Skene 1979; Raykar et al. 2010; Rodrigues & Pereira 2018 | n/a (not time-series models) | Yes — **reliability of the people who give the labels** | n/a |
| **RGA-Net (this work)** | Yes (continuous-time encoding) | Yes — **reliability of the people who make the measurements, inside attention** | No |

**The key distinction.** Crowd-learning models deal with *label* reliability: many annotators
assign a class to an item, some annotators are better than others, and the model learns whom to
believe about the label. RGA-Net deals with *measurement* reliability: the label comes from the
VSX catalogue, but every input measurement was made by a different person, and the model learns
how much each person's measurements should count while it reads the light curve.

**Novelty, stated cautiously.** The encoder itself is the standard Transformer encoder of
Vaswani et al. (2017); the contribution is the reliability-gated attention modification. To our
knowledge, among the reviewed literature, no prior work gates attention by a learned
observer-reliability score. We do not claim that RGA-Net is more
accurate than other models: the comparison with four ML models on identical splits, and the
ablation, are reported as measured in `v3_5GB/outputs/COMPARISON.md`, including where RGA-Net
is not the best.

**A caveat.** A_k is measured within a 1-day window, so stars that change brightness within
hours produce larger residuals for everyone; A_k can therefore partly reflect how fast the
observed stars vary, as well as observer quality. The ablation (full vs no-gate vs
observer-blind) shows what the reliability information actually contributes.

## 6. References

None of these references has been checked against the original publication in this project.
**Every entry is marked VERIFY BEFORE CITING**: confirm authors, title, venue and year (and add
volume/pages yourself) before putting it on a slide or in the report.

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., Polosukhin, I. (2017). *Attention Is All You Need.* Advances in Neural Information Processing Systems (NeurIPS 2017). — **VERIFY BEFORE CITING**
2. Hochreiter, S., Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation. — **VERIFY BEFORE CITING**
3. Cho, K., et al. (2014). *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation.* EMNLP 2014 (introduces the GRU). — **VERIFY BEFORE CITING**
4. Che, Z., Purushotham, S., Cho, K., Sontag, D., Liu, Y. (2018). *Recurrent Neural Networks for Multivariate Time Series with Missing Values.* Scientific Reports (GRU-D). — **VERIFY BEFORE CITING**
5. Neil, D., Pfeiffer, M., Liu, S.-C. (2016). *Phased LSTM: Accelerating Recurrent Network Training for Long or Event-based Sequences.* NeurIPS 2016. — **VERIFY BEFORE CITING**
6. Shukla, S. N., Marlin, B. M. (2021). *Multi-Time Attention Networks for Irregularly Sampled Time Series.* ICLR 2021 (mTAN). — **VERIFY BEFORE CITING**
7. Donoso-Oliva, C., Becker, I., Protopapas, P., Cabrera-Vives, G., et al. (2023). *ASTROMER: A transformer-based embedding for the representation of light curves.* Astronomy & Astrophysics. — **VERIFY BEFORE CITING**
8. Cádiz-Leyton, M., Cabrera-Vives, G., et al. *ATAT: Astronomical Transformer for time series And Tabular data.* (Astronomy & Astrophysics, around 2024–2025). — **VERIFY BEFORE CITING** (year and venue uncertain)
9. Allam, T. Jr., McEwen, J. D. *Paying attention to astronomical transients: introducing the time-series transformer for photometric classification.* (RAS Techniques and Instruments, around 2023–2024). — **VERIFY BEFORE CITING** (year and venue uncertain)
10. Shallue, C. J., Vanderburg, A. (2018). *Identifying Exoplanets with Deep Learning: A Five-planet Resonant Chain around Kepler-80 and an Eighth Planet around Kepler-90.* The Astronomical Journal (1D CNN on phase-folded light curves). — **VERIFY BEFORE CITING**
11. Charnock, T., Moss, A. (2017). *Deep Recurrent Neural Networks for Supernovae Classification.* The Astrophysical Journal Letters. — **VERIFY BEFORE CITING**
12. Naul, B., Bloom, J. S., Pérez, F., van der Walt, S. (2018). *A recurrent neural network for classification of unevenly sampled variable stars.* Nature Astronomy. — **VERIFY BEFORE CITING**
13. Dawid, A. P., Skene, A. M. (1979). *Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm.* Journal of the Royal Statistical Society, Series C (Applied Statistics). — **VERIFY BEFORE CITING**
14. Raykar, V. C., Yu, S., Zhao, L. H., Valadez, G. H., Florin, C., Bogoni, L., Moy, L. (2010). *Learning From Crowds.* Journal of Machine Learning Research. — **VERIFY BEFORE CITING**
15. Rodrigues, F., Pereira, F. C. (2018). *Deep Learning from Crowds.* AAAI 2018. — **VERIFY BEFORE CITING**
16. Lomb, N. R. (1976), *Least-squares frequency analysis of unequally spaced data*, Astrophysics and Space Science; Scargle, J. D. (1982), *Studies in astronomical time series analysis II*, The Astrophysical Journal (the Lomb–Scargle periodogram used for the ML features). — **VERIFY BEFORE CITING**
