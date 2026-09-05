"""
Idea H — trained pairwise similarity head on frozen WavLM features.

Not cosine: a small trained model that learns the perceptual similarity metric.
Design choices baked in against the system-overfitting we kept hitting:
  - frozen WavLM features (cached), only a tiny head is trained
  - per-task LEARNABLE softmax layer weights (SVSNet+ finding: weighted-sum > last)
  - low-dim projection (128) then pair interactions [proj_a, proj_b, |diff|, prod]
  - MSE + margin-ranking loss on within-batch pairs (matches SRCC), batches drawn
    ACROSS systems so ranking can't be satisfied by system identity
  - deterministic 7-fold-by-system grouped CV, early stop on held-out systems
  - compared head-to-head vs the ridge-on-layer-cosines baseline on identical folds

Runs on the VM (H100) reading cached wavlm_layers.pt + train.csv/dev.csv.

Usage:
    python train_head.py --features_dir features --data_dir data --out idea-h/answer.txt
"""

import argparse, csv, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

DATASET_DIR = "vmc2026_track3_train_phase_distro_v3_syn"
N_FOLDS = 7
ALPHA = {"spk": 0.5, "acc": 0.6}


def emb_key(p):
    q = p.strip("/").split("/"); return f"{DATASET_DIR}@{q[-2]}@{q[-1]}"


def load(path):
    d = torch.load(path, weights_only=False)
    return {k: (v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)) for k, v in d.items()}


def srcc(a, b): return spearmanr(a, b).statistic


def det_folds(systems):
    """Deterministic system->fold map (sorted round-robin), version-independent."""
    uniq = sorted(set(systems))
    return {s: i % N_FOLDS for i, s in enumerate(uniq)}


class PairHead(nn.Module):
    def __init__(self, n_layers, dim, proj=128, trunk=64, dropout=0.3):
        super().__init__()
        # per-task learnable layer weights
        self.lw_spk = nn.Parameter(torch.zeros(n_layers))
        self.lw_acc = nn.Parameter(torch.zeros(n_layers))
        self.proj = nn.Linear(dim, proj)
        self.trunk = nn.Sequential(
            nn.Linear(proj * 4, trunk), nn.ReLU(), nn.Dropout(dropout),
        )
        self.spk = nn.Linear(trunk, 1)
        self.acc = nn.Linear(trunk, 1)

    def weighted(self, x, lw):                       # x: (B, L, D)
        w = torch.softmax(lw, 0).view(1, -1, 1)
        e = (x * w).sum(1)                           # (B, D)
        return self.proj(e)

    def head(self, pa, pb, which):
        feat = torch.cat([pa, pb, (pa - pb).abs(), pa * pb], -1)
        t = self.trunk(feat)
        return (self.spk if which == "spk" else self.acc)(t).squeeze(-1)

    def forward(self, xa, xb):
        pa_s, pb_s = self.weighted(xa, self.lw_spk), self.weighted(xb, self.lw_spk)
        pa_a, pb_a = self.weighted(xa, self.lw_acc), self.weighted(xb, self.lw_acc)
        return self.head(pa_s, pb_s, "spk"), self.head(pa_a, pb_a, "acc")


def make_tensors(df, feats):
    Xa = torch.tensor(np.stack([feats[emb_key(p)] for p in df.wav_a_path]), dtype=torch.float32)
    Xb = torch.tensor(np.stack([feats[emb_key(p)] for p in df.wav_b_path]), dtype=torch.float32)
    return Xa, Xb


def train_fold(Xa, Xb, y, groups, tr_mask, va_mask, device, epochs=400, lr=1e-3, lam=0.3):
    n_layers, dim = Xa.shape[1], Xa.shape[2]
    model = PairHead(n_layers, dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    mse = nn.MSELoss(); rank = nn.MarginRankingLoss(margin=0.3)
    Xa, Xb, y = Xa.to(device), Xb.to(device), y.to(device)
    tr = torch.tensor(np.where(tr_mask)[0], device=device)
    g = torch.tensor(groups, device=device)
    best, best_state = -1, None
    rng = np.random.default_rng(np.random.randint(2**31))  # seed-dependent (global np seeded)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        ps, pa = model(Xa[tr], Xb[tr])
        loss = mse(ps, y[tr, 0]) + mse(pa, y[tr, 1])
        # ranking on random cross-system pairs
        idx = rng.integers(0, len(tr), size=(2, min(512, len(tr))))
        i, j = tr[idx[0]], tr[idx[1]]
        diffmask = (g[i] != g[j])
        if diffmask.sum() > 0:
            for k, pred in enumerate([ps, pa]):
                pi, pj = model(Xa[i], Xb[i])[k], model(Xa[j], Xb[j])[k]
                tgt = torch.sign(y[i, k] - y[j, k])
                m = (tgt != 0) & diffmask
                if m.sum() > 0:
                    loss = loss + lam * rank(pi[m], pj[m], tgt[m])
        loss.backward(); opt.step()
        if ep % 20 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                vs, va = model(Xa[va_mask], Xb[va_mask])
            s = (srcc(y[va_mask, 0].cpu(), vs.cpu()) + srcc(y[va_mask, 1].cpu(), va.cpu())) / 2
            if s > best:
                best, best_state = s, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_oof", default=None, help="npz path: OOF train preds + raw dev preds")
    ap.add_argument("--feature_file", default="wavlm_layers.pt", help="which SSL layer .pt in features_dir")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    fd, dd = os.path.expanduser(args.features_dir), os.path.expanduser(args.data_dir)
    wavlm = load(os.path.join(fd, args.feature_file))
    for k in wavlm:  # L2-normalize each layer up front
        e = wavlm[k].astype(np.float32)
        wavlm[k] = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)

    train = pd.read_csv(os.path.join(dd, "train_avg.csv")).reset_index(drop=True)
    dev = pd.read_csv(os.path.join(dd, "sets", "dev.csv"))
    fmap = det_folds(train.system_id)
    folds = train.system_id.map(fmap).values
    groups = pd.factorize(train.system_id)[0]

    Xa, Xb = make_tensors(train, wavlm)
    y = torch.tensor(train[["spk_sim", "acc_sim"]].values, dtype=torch.float32)

    # ---- ridge-on-WavLM-cosines baseline on the SAME deterministic folds (apples-to-apples) ----
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    Xa_np, Xb_np = Xa.numpy(), Xb.numpy()
    lcos = np.einsum("nld,nld->nl", Xa_np, Xb_np)   # (N, 25) per-layer cosines (already L2-normed)
    rp = np.zeros((len(train), 2))
    for f in range(N_FOLDS):
        te = folds == f; tr = ~te
        for j, t in enumerate(["spk_sim", "acc_sim"]):
            mdl = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            mdl.fit(lcos[tr], train[t].values[tr]); rp[te, j] = mdl.predict(lcos[te])
    print(f"RIDGE WavLM-cosines grouped-CV (SAME det folds): "
          f"spk {srcc(train.spk_sim, rp[:,0]):.4f}  acc {srcc(train.acc_sim, rp[:,1]):.4f}")

    # NESTED grouped CV (no leak): test = fold f, early-stop on inner-val = fold (f+1),
    # train on the remaining folds. Predict the untouched test fold.
    print(f"device={device}  layers={Xa.shape[1]}  lam={args.lam}")
    preds = np.zeros((len(train), 2))
    fold_models = []
    for f in range(N_FOLDS):
        test = folds == f
        inner = folds == (f + 1) % N_FOLDS
        tr = ~test & ~inner
        m = train_fold(Xa, Xb, y, groups, tr, inner, device, lam=args.lam)
        fold_models.append(m)
        m.eval()
        with torch.no_grad():
            ps, pa = m(Xa[test].to(device), Xb[test].to(device))
        preds[test, 0] = ps.cpu().numpy(); preds[test, 1] = pa.cpu().numpy()
    spk = srcc(train.spk_sim, preds[:, 0]); acc = srcc(train.acc_sim, preds[:, 1])
    print(f"\nTRAINED HEAD grouped-CV (nested, leak-free):  spk {spk:.4f}  acc {acc:.4f}")
    print("(ref: ridge-on-WavLM-cosines grouped-CV spk 0.500 acc 0.488)")

    # dev = ENSEMBLE of the 7 fold models (each trained on held-out systems)
    Da, Db = make_tensors(dev, wavlm)
    ds_l, da_l = [], []
    for m in fold_models:
        m.eval()
        with torch.no_grad():
            s, a = m(Da.to(device), Db.to(device))
        ds_l.append(s.cpu().numpy()); da_l.append(a.cpu().numpy())
    ds = torch.tensor(np.mean(ds_l, 0)); da = torch.tensor(np.mean(da_l, 0))
    if args.save_oof:
        np.savez(os.path.expanduser(args.save_oof),
                 oof_spk=preds[:, 0], oof_acc=preds[:, 1],
                 dev_spk=ds.cpu().numpy(), dev_acc=da.cpu().numpy())
        print(f"Saved OOF+dev raw preds -> {args.save_oof}")
    from scipy.stats import zscore
    def blend(pred, target, alpha):
        prior = dev.system_id.map(train.groupby("system_id")[target].mean()).values.astype(float)
        seen = ~np.isnan(prior); zc = zscore(pred); zp = np.empty(len(dev))
        zp[seen] = zscore(prior[seen]); zp[~seen] = zc[~seen]
        v = alpha * zc + (1 - alpha) * zp
        return 1 + 4 * (v - v.min()) / (v.max() - v.min() + 1e-8)
    spk_out = blend(ds.cpu().numpy(), "spk_sim", ALPHA["spk"])
    acc_out = blend(da.cpu().numpy(), "acc_sim", ALPHA["acc"])

    os.makedirs(os.path.dirname(os.path.expanduser(args.out)), exist_ok=True)
    with open(os.path.expanduser(args.out), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["system_id", "utterance_id", "wav_a_path",
                                           "wav_b_path", "pred_acc_sim", "pred_spk_sim"])
        w.writeheader()
        for i, row in dev.iterrows():
            w.writerow({"system_id": row.system_id, "utterance_id": row.utterance_id,
                        "wav_a_path": row.wav_a_path, "wav_b_path": row.wav_b_path,
                        "pred_acc_sim": float(acc_out[i]), "pred_spk_sim": float(spk_out[i])})
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
