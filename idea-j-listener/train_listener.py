"""
Idea J — listener-dependent trained head (UTMOS/LDNet-style denoising).

Trains on all 13,687 listener-wise rows (not the 2,800 means) with a learned
per-listener embedding, so the model separates each rater's bias from the true
similarity. At inference we use the "mean listener" (average of learned listener
embeddings). Same frozen-WavLM + learnable-layer-weight + pair-head backbone as
Idea H, plus the listener embedding.

Value: (a) denoising may make it individually competitive; (b) it is a genuinely
DIFFERENT model -> strong fresh member for the diversity ensemble that's winning.

Runs on the VM (H100). Outputs a dev answer.txt (mean-listener + system prior).

Usage:
    python train_listener.py --features_dir features --data_dir data --out idea-j/answer.txt
"""

import argparse, csv, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr, zscore

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
    uniq = sorted(set(systems)); return {s: i % N_FOLDS for i, s in enumerate(uniq)}


class ListenerHead(nn.Module):
    def __init__(self, n_layers, dim, n_listeners, proj=128, trunk=64, lemb=16, dropout=0.3):
        super().__init__()
        self.lw_spk = nn.Parameter(torch.zeros(n_layers))
        self.lw_acc = nn.Parameter(torch.zeros(n_layers))
        self.proj = nn.Linear(dim, proj)
        self.listener = nn.Embedding(n_listeners, lemb)
        self.trunk = nn.Sequential(nn.Linear(proj * 4 + lemb, trunk), nn.ReLU(), nn.Dropout(dropout))
        self.spk = nn.Linear(trunk, 1); self.acc = nn.Linear(trunk, 1)

    def w(self, x, lw):
        return self.proj((x * torch.softmax(lw, 0).view(1, -1, 1)).sum(1))

    def head(self, pa, pb, le, which):
        feat = torch.cat([pa, pb, (pa - pb).abs(), pa * pb, le], -1)
        t = self.trunk(feat)
        return (self.spk if which == "spk" else self.acc)(t).squeeze(-1)

    def _le(self, lid, B, device):
        if lid is None:  # mean listener = average of trained embeddings
            return self.listener.weight.mean(0, keepdim=True).expand(B, -1)
        return self.listener(lid)

    def forward(self, xa, xb, lid=None):
        le = self._le(lid, xa.shape[0], xa.device)
        return (self.head(self.w(xa, self.lw_spk), self.w(xb, self.lw_spk), le, "spk"),
                self.head(self.w(xa, self.lw_acc), self.w(xb, self.lw_acc), le, "acc"))


def tens(df, feats):
    Xa = torch.tensor(np.stack([feats[emb_key(p)] for p in df.wav_a_path]), dtype=torch.float32)
    Xb = torch.tensor(np.stack([feats[emb_key(p)] for p in df.wav_b_path]), dtype=torch.float32)
    return Xa, Xb


def train_model(Xa, Xb, lid, y, groups, tr_rows, va_pairs, device, epochs=200, lr=1e-3, lam=0.3):
    n_layers, dim = Xa.shape[1], Xa.shape[2]
    n_listeners = int(lid.max().item()) + 1
    model = ListenerHead(n_layers, dim, n_listeners).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    mse = nn.MSELoss(); rank = nn.MarginRankingLoss(margin=0.3)
    Xa, Xb, lid, y = Xa.to(device), Xb.to(device), lid.to(device), y.to(device)
    g = torch.tensor(groups, device=device)
    tr = torch.tensor(tr_rows, device=device)
    best, best_state = -1, None
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        ps, pa = model(Xa[tr], Xb[tr], lid[tr])
        loss = mse(ps, y[tr, 0]) + mse(pa, y[tr, 1])
        idx = np.random.randint(0, len(tr), size=(2, min(512, len(tr))))
        i, j = tr[idx[0]], tr[idx[1]]
        dm = g[i] != g[j]
        if dm.sum() > 0:
            for k in range(2):
                pi = model(Xa[i], Xb[i], lid[i])[k]; pj = model(Xa[j], Xb[j], lid[j])[k]
                tgt = torch.sign(y[i, k] - y[j, k]); m = (tgt != 0) & dm
                if m.sum() > 0: loss = loss + lam * rank(pi[m], pj[m], tgt[m])
        loss.backward(); opt.step()
        if ep % 20 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                va_idx = va_pairs["rows"].to(device)
                vs, va = model(Xa[va_idx], Xb[va_idx], None)  # mean listener
            s = (srcc(va_pairs["spk"], vs.cpu()) + srcc(va_pairs["acc"], va.cpu())) / 2
            if s > best:
                best, best_state = s, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, n_listeners


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", required=True); ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fd, dd = os.path.expanduser(args.features_dir), os.path.expanduser(args.data_dir)

    wavlm = load(os.path.join(fd, "wavlm_layers.pt"))
    for k in wavlm:
        e = wavlm[k].astype(np.float32); wavlm[k] = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)

    lw = pd.read_csv(os.path.join(dd, "sets", "train.csv"))
    lmap = {l: i for i, l in enumerate(sorted(lw.listener_id.unique()))}
    lw["lid"] = lw.listener_id.map(lmap)
    avg = pd.read_csv(os.path.join(dd, "train_avg.csv"))
    fmap = det_folds(avg.system_id)

    Xa, Xb = tens(lw, wavlm)
    lid = torch.tensor(lw.lid.values, dtype=torch.long)
    y = torch.tensor(lw[["spk_sim", "acc_sim"]].values, dtype=torch.float32)
    groups = pd.factorize(lw.system_id)[0]

    # per-pair mean-label eval set built from train_avg, with a row index into lw
    # (use the FIRST lw row of each pair to fetch its wavlm features)
    first_row = {(r.system_id, r.utterance_id): i for i, r in enumerate(lw.itertuples()) }
    avg["row"] = [first_row[(s, u)] for s, u in zip(avg.system_id, avg.utterance_id)]
    avg["fold"] = avg.system_id.map(fmap)

    print(f"device={device} listeners={len(lmap)} rows={len(lw)} pairs={len(avg)} seed={args.seed}")
    preds = np.zeros((len(avg), 2))
    dev = pd.read_csv(os.path.join(dd, "sets", "dev.csv"))
    Da, Db = tens(dev, wavlm)
    dev_acc, dev_spk = [], []

    for f in range(N_FOLDS):
        test_pairs = avg[avg.fold == f]; inner = (f + 1) % N_FOLDS
        tr_rows = np.where((lw.system_id.map(fmap) != f) & (lw.system_id.map(fmap) != inner))[0]
        va = avg[avg.fold == inner]
        va_pairs = {"rows": torch.tensor(va.row.values), "spk": va.spk_sim.values, "acc": va.acc_sim.values}
        model, nl = train_model(Xa, Xb, lid, y, groups, tr_rows, va_pairs, device, lam=0.3)
        model.eval()
        with torch.no_grad():
            tr_idx = torch.tensor(test_pairs.row.values)  # CPU index for CPU Xa
            ps, pa = model(Xa[tr_idx].to(device), Xb[tr_idx].to(device), None)  # mean listener
            preds[test_pairs.index, 0] = ps.cpu().numpy(); preds[test_pairs.index, 1] = pa.cpu().numpy()
            ds, da = model(Da.to(device), Db.to(device), None)  # mean listener
            dev_spk.append(ds.cpu().numpy()); dev_acc.append(da.cpu().numpy())

    print(f"\nLISTENER HEAD grouped-CV:  spk {srcc(avg.spk_sim, preds[:,0]):.4f}  acc {srcc(avg.acc_sim, preds[:,1]):.4f}")
    print("(ref: Idea-H head 0.557/0.570 ; ridge WavLM 0.589/0.564 on same det folds)")

    ds = np.mean(dev_spk, 0); da = np.mean(dev_acc, 0)
    def blend(pred, tgt, a):
        prior = dev.system_id.map(avg.groupby("system_id")[tgt].mean()).values.astype(float)
        seen = ~np.isnan(prior); zc = zscore(pred); zp = np.empty(len(dev))
        zp[seen] = zscore(prior[seen]); zp[~seen] = zc[~seen]
        v = a * zc + (1 - a) * zp; return 1 + 4 * (v - v.min()) / (v.max() - v.min() + 1e-8)
    so = blend(ds, "spk_sim", ALPHA["spk"]); ao = blend(da, "acc_sim", ALPHA["acc"])
    os.makedirs(os.path.dirname(os.path.expanduser(args.out)), exist_ok=True)
    with open(os.path.expanduser(args.out), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["system_id","utterance_id","wav_a_path","wav_b_path","pred_acc_sim","pred_spk_sim"])
        w.writeheader()
        for i, row in dev.iterrows():
            w.writerow({"system_id": row.system_id, "utterance_id": row.utterance_id,
                        "wav_a_path": row.wav_a_path, "wav_b_path": row.wav_b_path,
                        "pred_acc_sim": float(ao[i]), "pred_spk_sim": float(so[i])})
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
