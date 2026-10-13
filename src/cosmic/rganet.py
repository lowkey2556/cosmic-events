"""Step 8 — RGA-Net (Reliability-Gated Attention Network), the proposed deep learning model. PyTorch.

What it does:
  A small Transformer-style encoder over the irregularly sampled observations of one window
  (the token sequences Spark exported to work/windows_seq.npz):
  1. Token embedding: a linear layer on the numeric token features (standardised with the
     TRAINING windows' mean and std) plus learned embeddings of the band and observation type.
  2. Continuous-time encoding: sin and cos of 2π·t/P for periods P log-spaced from 0.1 to
     1000 days, where t is the actual number of days since the window start (no position indices).
  3. Reliability-gated self-attention (2 layers, 4 heads, width 64):
         score(i, j) = (q_i · k_j) / √d_head + λ · log(A_j + ε),   λ = softplus(ρ) ≥ 0,
     where A_j is the agreement score of the observer who made observation j. Padding is masked.
  4. Reliability-gated attention pooling to one window vector, with its own learnable λ.
  5. MLP head with dropout → one logit (anomalous vs normal).
  Training: class-weighted binary cross-entropy (pos_weight = negatives / positives in the
  training windows), AdamW, gradient clipping, early stopping on validation PR-AUC (the best
  epoch's weights are kept), fixed seeds. The decision threshold is chosen later on the
  validation windows by the comparison step, never on test.
  Ablations, under identical conditions (same splits, seeds, settings):
    full            the model above
    no_gate         λ fixed at 0 everywhere; A_k is still one of the token features
    observer_blind  no gate, and A_k and the observer offset are removed from the tokens
Outputs (work/ and outputs/review2/): predictions_rganet.csv, rganet_history.csv (loss and
validation curves per epoch), rganet_runs.csv (best epoch, training time, parameters, learned λ),
and every trained model in work/models/rganet_<variant>_seed<seed>.pt (weights + the training
statistics needed to feed it), which Step 8b (infer.py) applies to all windows with Spark.

Why: crowdsourced observations differ in quality. The gate lets the model learn how much
more attention reliable observers' measurements should get, directly inside attention.
"""
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from .evaluation import average_precision, roc_auc

OBSERVER_TOKEN_FEATURES = ("a_k", "obs_offset")


# ------------------------------------------------------------------ model
class TimeEncoding(nn.Module):
    """sin/cos(2π t / P) for log-spaced periods P, projected to d_model."""

    def __init__(self, d_model, n_freqs, p_min, p_max):
        super().__init__()
        periods = torch.logspace(math.log10(p_min), math.log10(p_max), n_freqs)
        self.register_buffer("omega", 2 * math.pi / periods)
        self.proj = nn.Linear(2 * n_freqs, d_model)

    def forward(self, t):  # t: (B, L) days
        x = t.unsqueeze(-1) * self.omega
        return self.proj(torch.cat([torch.sin(x), torch.cos(x)], dim=-1))


class GatedSelfAttention(nn.Module):
    """Multi-head self-attention with the additive reliability term λ·log(A_j + ε) on every key j."""

    def __init__(self, d, heads, dropout, gate, lambda_init):
        super().__init__()
        self.h, self.dh, self.gate = heads, d // heads, gate
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        if gate:
            self.rho = nn.Parameter(torch.tensor(float(lambda_init)))

    def lam(self):
        return F.softplus(self.rho) if self.gate else torch.zeros(())

    def forward(self, x, mask, log_a):
        B, L, d = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        s = q @ k.transpose(-1, -2) / math.sqrt(self.dh)          # (B, h, L, L)
        if self.gate:
            s = s + self.lam() * log_a[:, None, None, :]
        s = s.masked_fill(~mask[:, None, None, :], float("-inf"))
        a = self.drop(torch.softmax(s, dim=-1))
        return self.out((a @ v).transpose(1, 2).reshape(B, L, d))


class Block(nn.Module):
    def __init__(self, d, heads, ff_mult, dropout, gate, lambda_init):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = GatedSelfAttention(d, heads, dropout, gate, lambda_init)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask, log_a):
        x = x + self.drop(self.att(self.n1(x), mask, log_a))
        return x + self.drop(self.ff(self.n2(x)))


class GatedPool(nn.Module):
    """Attention pooling with a learned query; score_j = w·h_j/√d + λ_pool·log(A_j + ε)."""

    def __init__(self, d, gate, lambda_init):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.gate = gate
        if gate:
            self.rho = nn.Parameter(torch.tensor(float(lambda_init)))

    def lam(self):
        return F.softplus(self.rho) if self.gate else torch.zeros(())

    def forward(self, h, mask, log_a):
        s = h @ self.w / math.sqrt(h.shape[-1])
        if self.gate:
            s = s + self.lam() * log_a
        a = torch.softmax(s.masked_fill(~mask, float("-inf")), dim=-1)
        return (a.unsqueeze(-1) * h).sum(1)


class RGANet(nn.Module):
    def __init__(self, n_num, n_band, n_type, p, gate=True):
        super().__init__()
        d = int(p["d_model"])
        self.eps = float(p["eps"])
        self.num = nn.Linear(n_num, d)
        self.band = nn.Embedding(n_band, int(p["band_emb"]), padding_idx=0)
        self.otype = nn.Embedding(n_type, int(p["type_emb"]), padding_idx=0)
        self.cat = nn.Linear(int(p["band_emb"]) + int(p["type_emb"]), d)
        self.time = TimeEncoding(d, int(p["n_time_freqs"]), float(p["time_min_period_days"]),
                                 float(p["time_max_period_days"]))
        self.blocks = nn.ModuleList([Block(d, int(p["heads"]), int(p["ff_mult"]), float(p["dropout"]), gate,
                                           p["lambda_init"]) for _ in range(int(p["layers"]))])
        self.norm = nn.LayerNorm(d)
        self.pool = GatedPool(d, gate, p["lambda_init"])
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(float(p["dropout"])), nn.Linear(d, 1))

    def forward(self, num, band, otype, t, a, mask):
        log_a = torch.log(a.clamp(min=0) + self.eps)
        x = self.num(num) + self.cat(torch.cat([self.band(band), self.otype(otype)], -1)) + self.time(t)
        for blk in self.blocks:
            x = blk(x, mask, log_a)
        return self.head(self.pool(self.norm(x), mask, log_a)).squeeze(-1)

    def lambdas(self):
        out = {f"lambda_layer{i + 1}": float(b.att.lam().detach()) for i, b in enumerate(self.blocks)}
        out["lambda_pool"] = float(self.pool.lam().detach())
        return out


# ------------------------------------------------------------------ data
def load_windows(work_dir):
    z = np.load(os.path.join(work_dir, "windows_seq.npz"), allow_pickle=False)
    data = {k: z[k] for k in z.files}
    with open(os.path.join(work_dir, "vocab.json")) as fh:
        data["vocab"] = json.load(fh)
    return data


def make_inputs(num_raw, band, otype, mask, names, variant, mean, std):
    """Model inputs from raw token arrays (N x L x features). Used for training AND for Spark inference,
    so both prepare windows identically. Numeric features are standardised with TRAIN statistics;
    the observer-blind variant drops A_k and the offset."""
    names = list(names)
    mean, std = np.asarray(mean, np.float64), np.asarray(std, np.float64)
    keep = [i for i, n in enumerate(names) if not (variant == "observer_blind" and n in OBSERVER_TOKEN_FEATURES)]
    num = (num_raw[..., keep] - mean[keep]) / std[keep]
    num = np.where(mask[..., None], num, 0.0).astype(np.float32)
    t = num_raw[..., names.index("t_days")]
    a = num_raw[..., names.index("a_k")]
    return dict(num=torch.from_numpy(num), band=torch.from_numpy(np.asarray(band, np.int64)),
                otype=torch.from_numpy(np.asarray(otype, np.int64)), t=torch.from_numpy(t.astype(np.float32)),
                a=torch.from_numpy(a.astype(np.float32)), mask=torch.from_numpy(np.asarray(mask, bool)))


def tensors(data, idx, variant, mean, std):
    """Model inputs for windows idx of the exported data."""
    return make_inputs(data["num"][idx], data["band"][idx], data["otype"][idx], data["mask"][idx],
                       data["feature_names"], variant, mean, std)


def save_model(path, model, meta):
    """Weights + everything needed to rebuild and feed the model (sizes, settings, train statistics)."""
    torch.save(dict(state={k: v.detach().cpu() for k, v in model.state_dict().items()}, meta=meta), path)


def load_model(source):
    """(model on CPU in eval mode, meta) from a file path or the file's bytes."""
    import io
    blob = torch.load(io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source,
                      map_location="cpu", weights_only=False)
    m = blob["meta"]
    model = RGANet(m["n_num"], m["n_band"], m["n_type"], m["p"], gate=(m["variant"] == "full"))
    model.load_state_dict(blob["state"])
    model.eval()
    return model, m


def predict_frame(pdf, models):
    """Spark inference on one batch of windows. pdf: window_sequences rows (array columns + seq_len);
    models: list of (model, meta). Returns window_id, seed, variant, prob for every model."""
    if len(pdf) == 0:
        return pd.DataFrame(columns=["window_id", "seed", "variant", "prob"])
    names = models[0][1]["feature_names"]
    num = np.stack([np.stack([np.asarray(v, np.float64) for v in pdf[c]]) for c in names], axis=-1)
    L = num.shape[1]
    mask = np.arange(L)[None, :] < pdf["seq_len"].to_numpy()[:, None]
    band = np.stack([np.asarray(v, np.int64) for v in pdf["band_idx"]])
    otype = np.stack([np.asarray(v, np.int64) for v in pdf["type_idx"]])
    out = []
    with torch.no_grad():
        for model, m in models:
            X = make_inputs(num, band, otype, mask, names, m["variant"], m["mean"], m["std"])
            prob = torch.sigmoid(model(**X)).numpy().astype(float)
            out.append(pd.DataFrame(dict(window_id=pdf["window_id"].to_numpy(), seed=int(m["seed"]),
                                         variant=m["variant"], prob=prob)))
    return pd.concat(out, ignore_index=True)


def train_stats(data, idx):
    """Per-feature mean and std over the valid (non-padding) tokens of the training windows."""
    x = data["num"][idx][data["mask"][idx]]
    mean, std = x.mean(0), x.std(0)
    return mean, np.where(std > 1e-8, std, 1.0)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _batches(n, bs, shuffle, gen):
    order = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        yield order[i:i + bs]


def predict(model, X, device, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for b in _batches(len(X["num"]), bs, False, None):
            out.append(torch.sigmoid(model(**{k: v[b].to(device) for k, v in X.items()})).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0)


def train_one(data, split, variant, seed, p, device, log=print, save_path=None):
    """Train one variant on one split. Returns (val/test predictions, history, run info)."""
    set_seed(seed)
    sets = {s: np.flatnonzero(np.isin(data["auid"], split.loc[split["set"] == s, "auid"])) for s in ("train", "val", "test")}
    y = data["label"].astype(np.float32)
    mean, std = train_stats(data, sets["train"])
    X = {s: tensors(data, i, variant, mean, std) for s, i in sets.items()}
    Y = {s: torch.from_numpy(y[i]) for s, i in sets.items()}
    n_num = X["train"]["num"].shape[-1]
    model = RGANet(n_num, data["vocab"]["n_band"], data["vocab"]["n_type"], p, gate=(variant == "full")).to(device)
    n_pos = float(Y["train"].sum()); n_neg = float(len(Y["train"]) - n_pos)
    pos_weight = torch.tensor(n_neg / max(n_pos, 1.0), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(p["lr"]), weight_decay=float(p["weight_decay"]))
    gen = torch.Generator().manual_seed(seed)
    best, best_state, wait, hist = None, None, 0, []
    t0 = time.time()
    for epoch in range(1, int(p["max_epochs"]) + 1):
        model.train()
        tot, n = 0.0, 0
        for b in _batches(len(Y["train"]), int(p["batch_size"]), True, gen):
            xb = {k: v[b].to(device) for k, v in X["train"].items()}
            loss = F.binary_cross_entropy_with_logits(model(**xb), Y["train"][b].to(device), pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(p["grad_clip"]))
            opt.step()
            tot += loss.item() * len(b); n += len(b)
        pv = predict(model, X["val"], device)
        pv_t = torch.from_numpy(pv).clamp(1e-7, 1 - 1e-7)
        val_loss = float(F.binary_cross_entropy(pv_t, Y["val"], weight=torch.where(Y["val"] == 1, pos_weight.cpu(), 1.0)))
        ap, auc = average_precision(Y["val"].numpy(), pv), roc_auc(Y["val"].numpy(), pv)
        hist.append(dict(epoch=epoch, train_loss=tot / max(n, 1), val_loss=val_loss, val_pr_auc=ap, val_roc_auc=auc,
                         **model.lambdas()))
        key = (ap, -val_loss)
        if best is None or key > best:
            best, wait = key, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        else:
            wait += 1
            if wait >= int(p["patience"]):
                break
    train_s = time.time() - t0
    model.load_state_dict(best_state)
    if save_path:
        save_model(save_path, model, dict(n_num=n_num, n_band=data["vocab"]["n_band"], n_type=data["vocab"]["n_type"],
                                          p=dict(p), variant=variant, seed=int(seed), mean=mean.tolist(),
                                          std=std.tolist(), feature_names=[str(f) for f in data["feature_names"]]))
    preds = []
    for s in ("val", "test"):
        idx = sets[s]
        preds.append(pd.DataFrame(dict(set=s, window_id=data["window_id"][idx], auid=data["auid"][idx],
                                       label=data["label"][idx].astype(int), prob=predict(model, X[s], device))))
    info = dict(best_epoch=best_epoch, epochs_run=len(hist), best_val_pr_auc=best[0], train_seconds=round(train_s, 1),
                parameters=sum(p_.numel() for p_ in model.parameters()), n_train=len(sets["train"]),
                n_val=len(sets["val"]), n_test=len(sets["test"]), pos_weight=float(pos_weight), **model.lambdas())
    log(f"  {variant:15s} seed {seed}: best epoch {best_epoch}/{len(hist)}, val PR-AUC {best[0]:.3f}, "
        f"{train_s:.0f} s" + (f", λ = {[round(v, 3) for v in model.lambdas().values()]}" if variant == "full" else ""))
    return pd.concat(preds, ignore_index=True), pd.DataFrame(hist), info


def device_info():
    if torch.cuda.is_available():
        return torch.device("cuda"), f"GPU: {torch.cuda.get_device_name(0)}"
    return torch.device("cpu"), "no GPU found - training on CPU (slower; same results up to floating-point noise)"


def run(cfg, variants=None, seeds=None):
    p = cfg.section("rganet")
    data = load_windows(cfg.work_dir)
    splits = pd.read_csv(os.path.join(cfg.work_dir, "splits.csv"))
    variants = variants or list(p["variants"])
    seeds = seeds or [int(s) for s in cfg.section("split")["seeds"]]
    device, msg = device_info()
    print(f"RGA-Net on {msg}; {len(data['label'])} windows; variants {variants}; seeds {seeds}")
    preds, hists, runs = [], [], []
    model_dir = os.path.join(cfg.work_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    for seed in seeds:
        split = splits[splits["seed"] == seed]
        for v in variants:
            pr, h, info = train_one(data, split, v, seed, p, device,
                                    save_path=os.path.join(model_dir, f"rganet_{v}_seed{seed}.pt"))
            preds.append(pr.assign(model="RGA-Net", variant=v, seed=seed))
            hists.append(h.assign(variant=v, seed=seed))
            runs.append(dict(variant=v, seed=seed, device=str(device), **info))
    pred = pd.concat(preds, ignore_index=True)[["model", "variant", "seed", "set", "window_id", "auid", "label", "prob"]]
    hist = pd.concat(hists, ignore_index=True)
    runs = pd.DataFrame(runs)
    for df, name in ((pred, "predictions_rganet.csv"), (hist, "rganet_history.csv"), (runs, "rganet_runs.csv")):
        df.to_csv(os.path.join(cfg.work_dir, name), index=False)
        df.to_csv(os.path.join(cfg.output_dir, name), index=False)
        print(f"saved {os.path.join(cfg.output_dir, name)}")
    lam_cols = [c for c in runs.columns if c.startswith("lambda_")]
    full = runs[runs["variant"] == "full"]
    if len(full):
        print("\nLearned λ (full model), per seed:\n" + full[["seed"] + lam_cols].to_string(index=False))
    return pred, hist, runs
