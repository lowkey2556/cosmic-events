"""Step 8 — RGA-Net (Reliability-Gated Attention Network), the proposed deep learning model. PyTorch.

Model: a small Transformer-style encoder over the irregularly sampled observations of one window
  1. Token embedding: a linear layer on the numeric token features (standardised) plus learned embeddings of
     the band and observation type.
  2. Continuous-time encoding: sin and cos of 2*pi*t/P for periods P log-spaced from 0.1 to 1000 days, where t
     is the number of days since the window start (no position indices).
  3. Reliability-gated self-attention (2 layers, 4 heads, width 64):
         score(i, j) = (q_i . k_j) / sqrt(d_head) + λ · log(A_j + ε),   λ = softplus(ρ) >= 0,
     where A_j is the agreement score A_k of the observer who made observation j. Padding is masked.
  4. Reliability-gated attention pooling to one window vector, with its own λ.
  5. MLP head -> one logit. Optional hybrid head: the window's summary features (imputed and standardised on
     the training windows) are joined to the pooled vector before the MLP (chosen on validation).
Training:
  1. Pre-training ONCE on the labelled ZTF windows (ztf_windows step; no AAVSO star among them). ZTF is one
     professional survey, so A = 1 on every token: the gate is inactive (no λ parameters). A second encoder
     without the observer token features is pre-trained the same way for the observer-blind ablation.
  2. Fine-tuning per seed on the AAVSO training stars: all layers, a lower learning rate, the gate active.
     Training-only augmentation: observation dropout, a random time shift, magnitude noise scaled by each
     observation's uncertainty. Class-weighted BCE, AdamW, gradient clipping, early stopping on the
     validation AUPRC (best epoch kept).
  3. A small grid (learning rate x dropout x hybrid head) is judged separately on EACH split's validation
     stars for the full model; the chosen setting is shared by the variants within that split only.
Ablations (same splits, seeds and settings):
    full            pre-trained + gate
    no_pretrain     gate, trained from scratch on AAVSO only
    no_gate         pre-trained, λ fixed at 0 (A_k is still a token feature)
    observer_blind  pre-trained blind encoder, no gate, no A_k and no offset anywhere
Outputs: predictions_rganet.csv, rganet_history.csv, rganet_runs.csv (incl. learned λ), rganet_grid.csv,
rganet_pretraining.csv; models in work/models/ (applied to all windows by infer.py). A disconnected run resumes:
finished (variant, seed) runs and grid points are reused.
"""
import hashlib
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
    def __init__(self, n_num, n_band, n_type, p, gate=True, n_summary=0):
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
        self.n_summary = int(n_summary)  # hybrid head: window summary features joined to the pooled vector
        self.head = nn.Sequential(nn.Linear(d + self.n_summary, d), nn.GELU(), nn.Dropout(float(p["dropout"])),
                                  nn.Linear(d, 1))

    def forward(self, num, band, otype, t, a, mask, summ=None):
        log_a = torch.log(a.clamp(min=0) + self.eps)
        x = self.num(num) + self.cat(torch.cat([self.band(band), self.otype(otype)], -1)) + self.time(t)
        for blk in self.blocks:
            x = blk(x, mask, log_a)
        h = self.pool(self.norm(x), mask, log_a)
        if self.n_summary:
            h = torch.cat([h, summ], -1)
        return self.head(h).squeeze(-1)

    def lambdas(self):
        out = {f"lambda_layer{i + 1}": float(b.att.lam().detach()) for i, b in enumerate(self.blocks)}
        out["lambda_pool"] = float(self.pool.lam().detach())
        return out


VARIANT = {  # pre-trained encoder used, gate active, observer token features removed
    "full": dict(pretrained="full", gate=True, blind=False),
    "no_pretrain": dict(pretrained=None, gate=True, blind=False),
    "no_gate": dict(pretrained="full", gate=False, blind=False),
    "observer_blind": dict(pretrained="blind", gate=False, blind=True),
}


# ------------------------------------------------------------------ data
def load_windows(work_dir, fname="windows_seq.npz"):
    # allow_pickle=True: our own file (strings are fixed-width; older files stored object arrays)
    z = np.load(os.path.join(work_dir, fname), allow_pickle=True)
    data = {k: z[k] for k in z.files}
    with open(os.path.join(work_dir, "vocab.json")) as fh:
        data["vocab"] = json.load(fh)
    return data


def kept_features(names, blind):
    return [i for i, n in enumerate(list(names)) if not (blind and n in OBSERVER_TOKEN_FEATURES)]


def summary_cols(names, blind):
    """Summary features fed to the hybrid head (the observer-blind model gets the light-curve ones only)."""
    from .windows import OBSERVER_FEATURES
    return [i for i, n in enumerate(list(names)) if not (blind and n in OBSERVER_FEATURES)]


def summary_inputs(summary, cols, med, mean, std):
    x = np.asarray(summary, np.float64)[:, cols]
    x = np.where(np.isfinite(x), x, np.asarray(med)[None, :])
    return torch.from_numpy(((x - np.asarray(mean)) / np.asarray(std)).astype(np.float32))


def make_inputs(num_raw, band, otype, mask, names, variant, mean, std, summary=None, meta=None):
    """Model inputs from raw token arrays (N x L x features), standardised with the saved statistics. Used by the
    Spark inference (predict_frame); training standardises the same way, after augmentation (standardise())."""
    names = list(names)
    keep = kept_features(names, VARIANT[variant]["blind"])
    mean, std = np.asarray(mean, np.float64), np.asarray(std, np.float64)
    num = (num_raw[..., keep] - mean) / std
    num = np.where(mask[..., None], num, 0.0).astype(np.float32)
    out = dict(num=torch.from_numpy(num), band=torch.from_numpy(np.asarray(band, np.int64)),
               otype=torch.from_numpy(np.asarray(otype, np.int64)),
               t=torch.from_numpy(num_raw[..., names.index("t_days")].astype(np.float32)),
               a=torch.from_numpy(num_raw[..., names.index("a_k")].astype(np.float32)),
               mask=torch.from_numpy(np.asarray(mask, bool)))
    if meta and meta.get("n_summary"):
        out["summ"] = summary_inputs(summary, list(range(len(meta["summary_names"]))), meta["summary_median"],
                                     meta["summary_mean"], meta["summary_std"])
    return out


def raw_inputs(data, idx, blind):
    """Unstandardised token tensors of windows idx (standardised per batch, after augmentation)."""
    names = list(data["feature_names"])
    raw = data["num"][idx]
    return dict(num=torch.from_numpy(raw[..., kept_features(names, blind)].astype(np.float32)),
                band=torch.from_numpy(data["band"][idx].astype(np.int64)),
                otype=torch.from_numpy(data["otype"][idx].astype(np.int64)),
                t=torch.from_numpy(raw[..., names.index("t_days")].astype(np.float32)),
                a=torch.from_numpy(raw[..., names.index("a_k")].astype(np.float32)),
                mask=torch.from_numpy(data["mask"][idx].astype(bool)))


def token_stats(data, idx, blind):
    """Per-feature mean and std over the valid (non-padding) tokens of the given windows."""
    x = data["num"][idx][..., kept_features(data["feature_names"], blind)][data["mask"][idx]].astype(np.float64)
    mean, std = x.mean(0), x.std(0)
    return mean, np.where(std > 1e-8, std, 1.0)


def augment(xb, names, aug, gen):
    """Training-only augmentation on a batch of raw tokens: observation dropout, a random time shift of the whole
    window, and magnitude noise ~ N(0, scale * uncertainty) (a default uncertainty where none was reported)."""
    num, mask, t = xb["num"].clone(), xb["mask"].clone(), xb["t"].clone()
    B, L = mask.shape
    p = float(aug.get("obs_dropout", 0))
    if p > 0:
        keep = mask & (torch.rand(B, L, generator=gen) >= p)
        empty = keep.sum(1) == 0
        keep[empty] = mask[empty]
        mask = keep
    ts = float(aug.get("time_shift_days", 0))
    if ts > 0:
        shift = (torch.rand(B, 1, generator=gen) * 2 - 1) * ts
        t = t + shift
        if "t_days" in names:
            num[..., names.index("t_days")] += shift
    sc = float(aug.get("mag_noise_scale", 0))
    if sc > 0 and "mag_norm" in names:
        unc = num[..., names.index("unc")] if "unc" in names else torch.zeros_like(t)
        miss = (num[..., names.index("unc_missing")] > 0.5) if "unc_missing" in names else (unc <= 0)
        default = torch.full_like(unc, float(aug.get("default_uncertainty", 0.05)))
        sigma = torch.where(miss | (unc <= 0), default, unc)
        num[..., names.index("mag_norm")] += torch.randn(B, L, generator=gen) * sigma * sc
    return dict(xb, num=num, mask=mask, t=t)


def standardise(xb, mean, std):
    return dict(xb, num=(xb["num"] - mean) / std * xb["mask"].unsqueeze(-1))


def save_model(path, model, meta):
    """Weights + everything needed to rebuild and feed the model (sizes, settings, statistics)."""
    torch.save(dict(state={k: v.detach().cpu() for k, v in model.state_dict().items()}, meta=meta), path)


def build(meta):
    return RGANet(meta["n_num"], meta["n_band"], meta["n_type"], meta["p"], gate=VARIANT[meta["variant"]]["gate"],
                  n_summary=meta.get("n_summary", 0))


def load_model(source):
    """(model on CPU in eval mode, meta) from a file path or the file's bytes."""
    import io
    blob = torch.load(io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source,
                      map_location="cpu", weights_only=False)
    model = build(blob["meta"])
    model.load_state_dict(blob["state"])
    model.eval()
    return model, blob["meta"]


def predict_frame(pdf, models):
    """Spark inference on one batch of windows. pdf: window_sequences rows (array columns + seq_len) plus the
    window summary features; models: list of (model, meta). Returns window_id, seed, variant, prob."""
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
            summ = pdf[m["summary_names"]].astype(float).to_numpy() if m.get("n_summary") else None
            X = make_inputs(num, band, otype, mask, names, m["variant"], m["mean"], m["std"], summ, m)
            prob = torch.sigmoid(model(**X)).numpy().astype(float)
            out.append(pd.DataFrame(dict(window_id=pdf["window_id"].to_numpy(), seed=int(m["seed"]),
                                         variant=m["variant"], prob=prob)))
    return pd.concat(out, ignore_index=True)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _batches(n, bs, shuffle, gen):
    order = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        yield order[i:i + bs]


def predict(model, X, mean, std, device, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for b in _batches(len(X["num"]), bs, False, None):
            xb = standardise({k: v[b] for k, v in X.items()}, mean, std)
            out.append(torch.sigmoid(model(**{k: v.to(device) for k, v in xb.items()})).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0)


def fit(model, X, Y, mean, std, names, p, device, lr, epochs, patience, bs, seed, aug, log_every=None):
    """Class-weighted BCE + AdamW + clipping + training-only augmentation; early stopping on the validation AUPRC
    (best epoch's weights kept). X, Y: {'train', 'val'}. Returns (history, best epoch, best validation AUPRC)."""
    mean_t, std_t = torch.as_tensor(mean, dtype=torch.float32), torch.as_tensor(std, dtype=torch.float32)
    n_pos = float(Y["train"].sum()); n_neg = float(len(Y["train"]) - n_pos)
    pos_weight = torch.tensor(n_neg / max(n_pos, 1.0), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(p["weight_decay"]))
    gen = torch.Generator().manual_seed(int(seed))
    best, best_state, best_epoch, wait, hist = None, None, 0, 0, []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        tot, n = 0.0, 0
        for b in _batches(len(Y["train"]), int(bs), True, gen):
            xb = {k: v[b] for k, v in X["train"].items()}
            if aug:
                xb = augment(xb, names, aug, gen)
            xb = {k: v.to(device) for k, v in standardise(xb, mean_t, std_t).items()}
            loss = F.binary_cross_entropy_with_logits(model(**xb), Y["train"][b].to(device), pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(p["grad_clip"]))
            opt.step()
            tot += loss.item() * len(b); n += len(b)
        pv = predict(model, X["val"], mean_t, std_t, device)
        pv_t = torch.from_numpy(pv).clamp(1e-7, 1 - 1e-7)
        val_loss = float(F.binary_cross_entropy(pv_t, Y["val"], weight=torch.where(Y["val"] == 1, pos_weight.cpu(), 1.0)))
        ap, auc = average_precision(Y["val"].numpy(), pv), roc_auc(Y["val"].numpy(), pv)
        hist.append(dict(epoch=epoch, train_loss=tot / max(n, 1), val_loss=val_loss, val_pr_auc=ap, val_roc_auc=auc,
                         **model.lambdas()))
        if log_every and epoch % log_every == 0:
            print(f"    epoch {epoch}: train loss {hist[-1]['train_loss']:.4f}, val AUPRC {ap:.3f}", flush=True)
        key = (float(np.nan_to_num(ap, nan=-1.0)), -val_loss)
        if best is None or key > best:
            best, wait, best_epoch = key, 0, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= int(patience):
                break
    model.load_state_dict(best_state)
    return pd.DataFrame(hist), best_epoch, best[0]


# ------------------------------------------------------------------ 1. pre-training on ZTF
def pretrain(cfg, p, device, kind):
    """Pre-train the encoder ONCE on the labelled ZTF windows (gate inactive: no λ parameters; A = 1 on every ZTF
    token). kind: 'full' (all token features) or 'blind' (A_k and offset removed). Cached in work/models/."""
    path = os.path.join(cfg.work_dir, "models", f"pretrained_{kind}.pt")
    src = os.path.join(cfg.work_dir, "ztf_windows_seq.npz")
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} is missing: run the ztf_windows step first (it is needed for pre-training)")
    keys = ("d_model", "heads", "layers", "ff_mult", "dropout", "n_time_freqs", "time_min_period_days",
            "time_max_period_days", "band_emb", "type_emb", "eps", "weight_decay", "grad_clip", "pretrain_epochs",
            "pretrain_lr", "pretrain_batch_size", "pretrain_patience", "augment")
    stamp = json.dumps([file_digest(src), file_digest(os.path.join(cfg.work_dir, "vocab.json")),
                        file_digest(__file__), {k: p.get(k) for k in keys}], sort_keys=True, default=str)
    if os.path.exists(path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob["meta"].get("stamp") == stamp:
            print(f"  pre-trained '{kind}' encoder reused from {path}")
            return blob
    data = load_windows(cfg.work_dir, "ztf_windows_seq.npz")
    blind = kind == "blind"
    names_all = list(data["feature_names"])
    names = [names_all[i] for i in kept_features(names_all, blind)]
    lab = data["label"] >= 0
    h = np.array([int(hashlib.md5(a.encode()).hexdigest(), 16) % 10 for a in data["auid"].astype(str)])
    sets = {"train": np.flatnonzero(lab & (h != 0)), "val": np.flatnonzero(lab & (h == 0))}  # 90/10 by light curve
    mean, std = token_stats(data, sets["train"], blind)
    X = {s: raw_inputs(data, i, blind) for s, i in sets.items()}
    Y = {s: torch.from_numpy(data["label"][i].astype(np.float32)) for s, i in sets.items()}
    set_seed(0)
    model = RGANet(len(names), data["vocab"]["n_band"], data["vocab"]["n_type"], p, gate=False).to(device)
    print(f"  pre-training the '{kind}' encoder on {len(sets['train']):,} ZTF windows "
          f"({len(sets['val']):,} held out by light curve) ...", flush=True)
    t0 = time.time()
    hist, ep, ap = fit(model, X, Y, mean, std, names, p, device, p["pretrain_lr"], p["pretrain_epochs"],
                       p["pretrain_patience"], p["pretrain_batch_size"], 0, p.get("augment"), log_every=1)
    meta = dict(kind=kind, stamp=stamp, mean=mean.tolist(), std=std.tolist(), best_epoch=ep, holdout_pr_auc=ap,
                seconds=round(time.time() - t0, 1), n_train=len(sets["train"]), n_holdout=len(sets["val"]))
    blob = dict(state={k: v.detach().cpu() for k, v in model.state_dict().items()}, meta=meta, history=hist)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(blob, path)
    print(f"  done in {meta['seconds']:.0f} s: best epoch {ep}, hold-out AUPRC {ap:.3f}")
    return blob


def load_matching(model, state):
    """Copy every pre-trained weight whose name and shape match (the gate's λ and a hybrid head start fresh)."""
    own = model.state_dict()
    ok = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(ok, strict=False)
    return len(ok), len(own)


# ------------------------------------------------------------------ 2. fine-tuning on AAVSO
def train_one(data, split, variant, seed, p, device, pre=None, hybrid=False, log=print, save_path=None,
              evaluate_test=True):
    """Fine-tune (or, for no_pretrain, train from scratch) one variant on one split. All layers are trained; the
    gate is active for 'full' and 'no_pretrain'. Returns (val/test predictions, history, run info)."""
    v = VARIANT[variant]
    set_seed(seed)
    names_all = list(data["feature_names"])
    names = [names_all[i] for i in kept_features(names_all, v["blind"])]
    set_names = ("train", "val", "test") if evaluate_test else ("train", "val")
    sets = {s: np.flatnonzero(np.isin(data["auid"], split.loc[split["set"] == s, "auid"])) for s in set_names}
    y = data["label"].astype(np.float32)
    if pre is not None:  # the pre-training token statistics: the encoder sees inputs on the scale it learned
        mean, std = np.asarray(pre["meta"]["mean"]), np.asarray(pre["meta"]["std"])
    else:
        mean, std = token_stats(data, sets["train"], v["blind"])
    X = {s: raw_inputs(data, i, v["blind"]) for s, i in sets.items()}
    Y = {s: torch.from_numpy(y[i]) for s, i in sets.items()}
    smeta = dict(n_summary=0)
    if hybrid:
        snames = list(data["summary_names"])
        cols = summary_cols(snames, v["blind"])
        tr = np.asarray(data["summary"][sets["train"]], np.float64)[:, cols]
        with np.errstate(all="ignore"):
            med = np.nanmedian(np.where(np.isfinite(tr), tr, np.nan), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        filled = np.where(np.isfinite(tr), tr, med)
        smean, sstd = filled.mean(0), filled.std(0)
        sstd = np.where(sstd > 1e-8, sstd, 1.0)
        smeta = dict(n_summary=len(cols), summary_names=[str(snames[c]) for c in cols], summary_median=med.tolist(),
                     summary_mean=smean.tolist(), summary_std=sstd.tolist())
        for s, i in sets.items():
            X[s]["summ"] = summary_inputs(data["summary"][i], cols, med, smean, sstd)
    model = RGANet(len(names), data["vocab"]["n_band"], data["vocab"]["n_type"], p, gate=v["gate"],
                   n_summary=smeta["n_summary"]).to(device)
    loaded = load_matching(model, pre["state"]) if pre is not None else (0, 0)
    t0 = time.time()
    hist, best_epoch, best_ap = fit(model, X, Y, mean, std, names, p, device, p["lr"], p["max_epochs"], p["patience"],
                                    p["batch_size"], seed, p.get("augment"))
    train_s = time.time() - t0
    meta = dict(n_num=len(names), n_band=data["vocab"]["n_band"], n_type=data["vocab"]["n_type"], p=dict(p),
                variant=variant, seed=int(seed), mean=np.asarray(mean).tolist(), std=np.asarray(std).tolist(),
                feature_names=[str(f) for f in names_all], pretrained=pre is not None, **smeta)
    if save_path:
        save_model(save_path, model, meta)
    mean_t, std_t = torch.as_tensor(mean, dtype=torch.float32), torch.as_tensor(std, dtype=torch.float32)
    preds = []
    for s in (("val", "test") if evaluate_test else ("val",)):
        idx = sets[s]
        preds.append(pd.DataFrame(dict(set=s, window_id=data["window_id"][idx], auid=data["auid"][idx],
                                       label=data["label"][idx].astype(int),
                                       prob=predict(model, X[s], mean_t, std_t, device))))
    info = dict(best_epoch=best_epoch, epochs_run=len(hist), best_val_pr_auc=best_ap, train_seconds=round(train_s, 1),
                parameters=sum(p_.numel() for p_ in model.parameters()), n_train=len(sets["train"]),
                n_val=len(sets["val"]), n_test=len(sets.get("test", [])), pretrained=pre is not None,
                pretrained_tensors_loaded=loaded[0], hybrid_head=bool(hybrid), lr=float(p["lr"]),
                dropout=float(p["dropout"]), **model.lambdas())
    log(f"  {variant:15s} seed {seed}: best epoch {best_epoch}/{len(hist)}, val AUPRC {best_ap:.3f}, {train_s:.0f} s"
        + (f", λ = {[round(x, 3) for x in model.lambdas().values()]}" if v["gate"] else ""))
    return pd.concat(preds, ignore_index=True), hist, info


def device_info():
    if torch.cuda.is_available():
        return torch.device("cuda"), f"GPU: {torch.cuda.get_device_name(0)}"
    return torch.device("cpu"), "no GPU found - training on CPU (slower; same results up to floating-point noise)"


def file_digest(path):
    """Content identity for cached training inputs; size/mtime alone can reuse stale models."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def training_signature(cfg):
    """Invalidate completed runs and tuning when data, splits, vocabulary or training code changes."""
    files = [os.path.join(cfg.work_dir, f) for f in
             ("windows_seq.npz", "ztf_windows_seq.npz", "splits.csv", "vocab.json")]
    files.append(__file__)
    return hashlib.sha256(json.dumps([file_digest(p) for p in files]).encode()).hexdigest()


def choose_config(cfg, data, split, p, device, pres, seed, signature):
    """Tune on this seed's train/validation stars only; other splits must never select its settings."""
    g, gseed = p["grid"], int(seed)
    split = split[split["seed"] == gseed]
    tune_split = split[split["set"] != "test"].copy()
    stamp = hashlib.sha256(json.dumps([signature, p, gseed], sort_keys=True, default=str).encode()).hexdigest()[:16]
    grid_dir = os.path.join(cfg.work_dir, "rganet_grids", stamp)
    os.makedirs(grid_dir, exist_ok=True)
    path = os.path.join(grid_dir, f"seed{gseed}.csv")
    combos = [dict(lr=float(lr), dropout=float(dr), hybrid_head=bool(hh))
              for lr in g["lr"] for dr in g["dropout"] for hh in g["hybrid_head"]]
    old = pd.read_csv(path) if os.path.exists(path) else None
    rows = []
    for c in combos:
        hit = None if old is None else old[(old["lr"] == c["lr"]) & (old["dropout"] == c["dropout"])
                                           & (old["hybrid_head"] == c["hybrid_head"])]
        if hit is not None and len(hit):
            rows.append({k: v for k, v in hit.iloc[0].to_dict().items() if k != "chosen"})
            continue
        pc = dict(p, lr=c["lr"], dropout=c["dropout"])
        _, _, info = train_one(data, tune_split, "full", gseed, pc, device, pre=pres["full"],
                               hybrid=c["hybrid_head"], log=lambda m: print("  grid" + m, flush=True),
                               evaluate_test=False)
        rows.append(dict(c, val_pr_auc=info["best_val_pr_auc"], best_epoch=info["best_epoch"],
                         train_seconds=info["train_seconds"], grid_seed=gseed))
        pd.DataFrame(rows).to_csv(path, index=False)
    grid = pd.DataFrame(rows)
    grid["chosen"] = False
    best = grid["val_pr_auc"].idxmax()
    grid.loc[best, "chosen"] = True
    grid.to_csv(path, index=False)
    b = grid.loc[best]
    print(f"chosen on validation (seed {gseed}): lr {b['lr']:g}, dropout {b['dropout']:g}, "
          f"hybrid head {bool(b['hybrid_head'])} (val AUPRC {b['val_pr_auc']:.3f})")
    return dict(lr=float(b["lr"]), dropout=float(b["dropout"])), bool(b["hybrid_head"]), grid


def run(cfg, variants=None, seeds=None):
    """Pre-train once on ZTF, choose the fine-tuning setting on validation, then train every variant on every
    split. Each finished (variant, seed) is saved in work/rganet_runs/, so a disconnected Colab resumes."""
    p = dict(cfg.section("rganet"))
    data = load_windows(cfg.work_dir)
    splits = pd.read_csv(os.path.join(cfg.work_dir, "splits.csv"))
    variants = variants or list(p["variants"])
    seeds = seeds or [int(s) for s in cfg.section("split")["seeds"]]
    device, msg = device_info()
    print(f"RGA-Net on {msg}; {len(data['label'])} AAVSO windows; variants {variants}; seeds {seeds}")
    kinds = sorted({VARIANT[v]["pretrained"] for v in variants + ["full"] if VARIANT[v]["pretrained"]})
    pres = {k: pretrain(cfg, p, device, k) for k in kinds}
    signature = training_signature(cfg)
    pd.DataFrame([dict(encoder=k, **{m: pres[k]["meta"][m] for m in ("best_epoch", "holdout_pr_auc", "seconds",
                                                                       "n_train", "n_holdout")}) for k in kinds]) \
        .to_csv(os.path.join(cfg.output_dir, "rganet_pretraining.csv"), index=False)
    model_dir = os.path.join(cfg.work_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    preds, hists, runs = [], [], []
    grids = []
    for seed in seeds:
        split = splits[splits["seed"] == seed]
        best, hybrid, grid = choose_config(cfg, data, split, p, device, pres, seed, signature)
        pc = dict(p, **best)
        grids.append(grid)
        grid = pd.concat(grids, ignore_index=True)
        grid.to_csv(os.path.join(cfg.work_dir, "rganet_grid.csv"), index=False)
        grid.to_csv(os.path.join(cfg.output_dir, "rganet_grid.csv"), index=False)
        tag = hashlib.sha256(json.dumps([signature, pc, hybrid, int(seed)], sort_keys=True,
                                        default=str).encode()).hexdigest()[:16]
        run_dir = os.path.join(cfg.work_dir, "rganet_runs", tag)
        os.makedirs(run_dir, exist_ok=True)
        for v in variants:
            base = os.path.join(run_dir, f"{v}_seed{seed}")
            mpath = os.path.join(model_dir, f"rganet_{v}_seed{seed}.pt")
            reusable = False
            if all(os.path.exists(base + e) for e in ("_pred.csv", "_hist.csv", "_info.json")) and os.path.exists(mpath):
                with open(base + "_info.json") as fh:
                    info = json.load(fh)
                reusable = info.get("checkpoint_sha256") == file_digest(mpath)
            if reusable:
                pr, h = pd.read_csv(base + "_pred.csv"), pd.read_csv(base + "_hist.csv")
                print(f"  {v:15s} seed {seed}: reused (finished earlier)")
            else:
                pre = pres[VARIANT[v]["pretrained"]] if VARIANT[v]["pretrained"] else None
                pr, h, info = train_one(data, split, v, seed, pc, device, pre=pre, hybrid=hybrid, save_path=mpath)
                info = dict(variant=v, seed=seed, device=str(device), checkpoint_sha256=file_digest(mpath), **info)
                pr.to_csv(base + "_pred.csv", index=False)
                h.to_csv(base + "_hist.csv", index=False)
                with open(base + "_info.json", "w") as fh:
                    json.dump(info, fh, default=float)
            preds.append(pr.assign(model="RGA-Net", variant=v, seed=seed))
            hists.append(h.assign(variant=v, seed=seed))
            runs.append(info)
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
    from . import dataflow
    files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".pt")]
    dataflow.record(cfg, "rganet_checkpoints", "PyTorch pre-trains the encoder on ZTF windows once, then fine-tunes "
                    "every variant on every split's AAVSO training stars; best-epoch weights are saved.", model_dir,
                    "PyTorch .pt (weights + statistics)", f"PyTorch on {device}",
                    size_bytes=sum(os.path.getsize(f) for f in files), rows=len(pred), files=len(files),
                    changes=f"{len(runs)} fine-tuned models; settings chosen separately on each split's validation stars",
                    seconds=float(runs["train_seconds"].sum()))
    return pred, hist, runs
