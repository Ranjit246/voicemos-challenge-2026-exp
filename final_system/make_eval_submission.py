"""
Idea K eval submission — train on train+dev (3400 labeled pairs), predict test.csv.

Pipeline (= our best dev system, submission-2 richer-RAMP):
  features    : base cosines (ecapa,titanet,wespeaker,wav2vec2 hidden/prob,commonaccent emb/prob) + utmos + WavLM 25-layer cosines
  ridge       : parametric fusion over those
  RAMP        : kNN retrieval in SSL embedding-diff space (ECAPA,CommonAccent,WavLM-L3,WavLM-L12) PCA-128
  fuse        : 0.8*z(ridge) + 0.2*z(retrieval)
  head        : trained WavLM pairwise head (6 seeds), ensembled 1:1
  prior       : train+dev system-label mean; unseen eval systems (sys004,sys021) -> fallback to prediction

Runs on the L4. trainfeat/ = train+dev features; evalfeat/ = eval features.

Usage:
    python make_eval_submission.py --trainfeat ~/vmc_eval/trainfeat --evalfeat ~/vmc_eval/evalfeat \
        --data ~/vmc_eval/data --out ~/vmc_eval/answer.txt
"""

import argparse, csv, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr, zscore
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATASET_DIR = "vmc2026_track3_train_phase_distro_v3_syn"
K = 30; PCA_DIM = 128; ALPHA = {"spk_sim": 0.5, "acc_sim": 0.6}
BASE = ["ecapa", "titanet", "wespeaker", "wav2vec2_hidden", "wav2vec2_prob", "commonaccent_emb", "commonaccent_prob"]
RAMP_SRC = ["ecapa", "commonaccent_emb", ("wavlm", 3), ("wavlm", 12)]
N_SEEDS = 6


def emb_key(rel): p = rel.strip("/").split("/"); return f"{DATASET_DIR}@{p[-2]}@{p[-1]}"
def norm(v): return v.astype(np.float32) / (np.linalg.norm(v) + 1e-8)
def srcc(a, b): return spearmanr(a, b).statistic


def load(path):
    d = torch.load(path, weights_only=False)
    return {k: (v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)) for k, v in d.items()}


def cos(dic, a, b):
    ea, eb = dic[emb_key(a)].astype(np.float32), dic[emb_key(b)].astype(np.float32)
    return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-8))


def wl_cos(dic, a, b, L):
    ea, eb = dic[emb_key(a)][L].astype(np.float32), dic[emb_key(b)][L].astype(np.float32)
    return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-8))


def build_feats(df, feats, wavlm, n_layers):
    for n in BASE:
        df[f"cos_{n}"] = [cos(feats[n], a, b) for a, b in zip(df.wav_a_path, df.wav_b_path)]
    df["utmos_a"] = [feats["utmos"][emb_key(a)] for a in df.wav_a_path]
    df["utmos_b"] = [feats["utmos"][emb_key(b)] for b in df.wav_b_path]
    df["utmos_diff"] = df.utmos_a - df.utmos_b
    for L in range(n_layers):
        df[f"wl_{L}"] = [wl_cos(wavlm, a, b, L) for a, b in zip(df.wav_a_path, df.wav_b_path)]
    return [f"cos_{n}" for n in BASE] + ["utmos_a", "utmos_b", "utmos_diff"] + [f"wl_{L}" for L in range(n_layers)]


def diff_matrix(df, feats, wavlm):
    rows = []
    for a, b in zip(df.wav_a_path, df.wav_b_path):
        ka, kb = emb_key(a), emb_key(b); parts = []
        for s in RAMP_SRC:
            if isinstance(s, tuple): va, vb = wavlm[ka][s[1]], wavlm[kb][s[1]]
            else: va, vb = feats[s][ka], feats[s][kb]
            parts.append(norm(va) - norm(vb))
        rows.append(np.concatenate(parts))
    return np.stack(rows)


def retrieve(Xtr, ytr, Xq, k=K):
    D = cdist(Xq, Xtr); out = np.empty(len(Xq))
    for i in range(len(Xq)):
        nn = np.argpartition(D[i], k)[:k]; d = D[i, nn]
        w = np.exp(-d / (d.mean() + 1e-8)); w /= w.sum(); out[i] = (w * ytr[nn]).sum()
    return out


class Head(nn.Module):
    def __init__(self, nl, dim, proj=128, trunk=64, dp=0.3):
        super().__init__()
        self.lw_s = nn.Parameter(torch.zeros(nl)); self.lw_a = nn.Parameter(torch.zeros(nl))
        self.proj = nn.Linear(dim, proj)
        self.trunk = nn.Sequential(nn.Linear(proj*4, trunk), nn.ReLU(), nn.Dropout(dp))
        self.s = nn.Linear(trunk, 1); self.a = nn.Linear(trunk, 1)
    def w(self, x, lw): return self.proj((x*torch.softmax(lw, 0).view(1, -1, 1)).sum(1))
    def head(self, pa, pb, which):
        t = self.trunk(torch.cat([pa, pb, (pa-pb).abs(), pa*pb], -1))
        return (self.s if which == "s" else self.a)(t).squeeze(-1)
    def forward(self, xa, xb):
        return (self.head(self.w(xa, self.lw_s), self.w(xb, self.lw_s), "s"),
                self.head(self.w(xa, self.lw_a), self.w(xb, self.lw_a), "a"))


def wl_tensor(df, wavlm):
    Xa = torch.tensor(np.stack([wavlm[emb_key(p)] for p in df.wav_a_path]), dtype=torch.float32)
    Xb = torch.tensor(np.stack([wavlm[emb_key(p)] for p in df.wav_b_path]), dtype=torch.float32)
    # L2-normalize each layer
    Xa = Xa / (Xa.norm(dim=-1, keepdim=True) + 1e-8); Xb = Xb / (Xb.norm(dim=-1, keepdim=True) + 1e-8)
    return Xa, Xb


def train_head(Xa, Xb, y, groups, dev_wl, seed, device, epochs=300, lam=0.3):
    torch.manual_seed(seed); np.random.seed(seed)
    nl, dim = Xa.shape[1], Xa.shape[2]
    m = Head(nl, dim).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-3)
    mse = nn.MSELoss(); rank = nn.MarginRankingLoss(margin=0.3)
    # 90/10 split for early stop
    idx = np.random.permutation(len(y)); va = idx[:len(y)//10]; tr = idx[len(y)//10:]
    Xa, Xb, y = Xa.to(device), Xb.to(device), torch.tensor(y, dtype=torch.float32).to(device)
    g = torch.tensor(groups, device=device); tr = torch.tensor(tr, device=device); va = torch.tensor(va, device=device)
    best, best_state = -1, None
    for ep in range(epochs):
        m.train(); opt.zero_grad()
        ps, pa = m(Xa[tr], Xb[tr]); loss = mse(ps, y[tr, 0]) + mse(pa, y[tr, 1])
        ii = np.random.randint(0, len(tr), (2, min(512, len(tr)))); i, j = tr[ii[0]], tr[ii[1]]
        dm = g[i] != g[j]
        if dm.sum() > 0:
            for k in range(2):
                pi = m(Xa[i], Xb[i])[k]; pj = m(Xa[j], Xb[j])[k]; tgt = torch.sign(y[i, k]-y[j, k]); mk = (tgt != 0) & dm
                if mk.sum() > 0: loss = loss + lam*rank(pi[mk], pj[mk], tgt[mk])
        loss.backward(); opt.step()
        if ep % 20 == 0 or ep == epochs-1:
            m.eval()
            with torch.no_grad(): vs, va_ = m(Xa[va], Xb[va])
            s = (srcc(y[va, 0].cpu(), vs.cpu()) + srcc(y[va, 1].cpu(), va_.cpu()))/2
            if s > best: best, best_state = s, {k: v.cpu().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad(): ds, da = m(dev_wl[0].to(device), dev_wl[1].to(device))
    return ds.cpu().numpy(), da.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainfeat", required=True); ap.add_argument("--evalfeat", required=True)
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tf, ef, dd = map(os.path.expanduser, [a.trainfeat, a.evalfeat, a.data])

    print("loading features...")
    trainF = {n: load(os.path.join(tf, f"{n}.pt")) for n in BASE + ["utmos"]}
    trainWL = load(os.path.join(tf, "wavlm_layers.pt"))
    evalF = {n: load(os.path.join(ef, f"{n}.pt")) for n in BASE + ["utmos"]}
    evalWL = load(os.path.join(ef, "wavlm_layers.pt"))
    n_layers = next(iter(trainWL.values())).shape[0]

    # labels: train_avg + dev_with_labels
    tr_avg = pd.read_csv(os.path.join(dd, "train_avg.csv"))
    dev = pd.read_csv(os.path.join(dd, "dev_with_labels.csv"))
    cols = ["system_id", "utterance_id", "wav_a_path", "wav_b_path", "spk_sim", "acc_sim"]
    train = pd.concat([tr_avg[cols], dev[cols]], ignore_index=True).reset_index(drop=True)
    test = pd.read_csv(os.path.join(dd, "test.csv"))
    print(f"train+dev pairs: {len(train)}  eval pairs: {len(test)}")

    feats = build_feats(train, trainF, trainWL, n_layers)
    build_feats(test, evalF, evalWL, n_layers)
    Rtr = diff_matrix(train, trainF, trainWL); Rte = diff_matrix(test, evalF, evalWL)
    groups = pd.factorize(train.system_id)[0]

    # head (6 seeds averaged), predict eval
    print("training head (6 seeds)...")
    Xa, Xb = wl_tensor(train, trainWL); Dwl = wl_tensor(test, evalWL)
    y2 = train[["spk_sim", "acc_sim"]].values
    hs = np.mean([np.stack(train_head(Xa, Xb, y2, groups, Dwl, s, device), 1) for s in range(N_SEEDS)], 0)

    # richer-RAMP + head + prior
    rs = StandardScaler().fit(Rtr); pca = PCA(PCA_DIM, random_state=0).fit(rs.transform(Rtr))
    Ztr, Zte = pca.transform(rs.transform(Rtr)), pca.transform(rs.transform(Rte))
    Xtr_np, Xte_np = train[feats].values, test[feats].values
    unseen = sorted(set(test.system_id) - set(train.system_id))
    print(f"unseen eval systems (no prior): {unseen}")

    out_cols = {}
    for j, t in enumerate(["spk_sim", "acc_sim"]):
        sc = StandardScaler().fit(Xtr_np); y = train[t].values
        rid = Ridge(alpha=1.0).fit(sc.transform(Xtr_np), y).predict(sc.transform(Xte_np))
        ret = retrieve(Ztr, y, Zte)
        ramp = 0.8*zscore(rid) + 0.2*zscore(ret)                      # beta=0.8 (spk) / 0.7 (acc)? use 0.8/0.7
        if t == "acc_sim": ramp = 0.7*zscore(rid) + 0.3*zscore(ret)
        param = np.mean([zscore(ramp), zscore(hs[:, j])], 0)          # ensemble with head
        prior = test.system_id.map(train.groupby("system_id")[t].mean()).values.astype(float)
        seen = ~np.isnan(prior); zp = np.empty(len(test))
        zp[seen] = zscore(prior[seen]); zp[~seen] = zscore(param)[~seen]
        al = ALPHA[t]; v = al*zscore(param) + (1-al)*zp
        out_cols[t] = 1 + 4*(v - v.min())/(v.max()-v.min()+1e-8)

    os.makedirs(os.path.dirname(os.path.expanduser(a.out)), exist_ok=True)
    with open(os.path.expanduser(a.out), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["system_id","utterance_id","wav_a_path","wav_b_path","pred_acc_sim","pred_spk_sim"])
        for i, r in test.iterrows():
            w.writerow([r.system_id, r.utterance_id, r.wav_a_path, r.wav_b_path, float(out_cols["acc_sim"][i]), float(out_cols["spk_sim"][i])])
    print(f"Saved -> {a.out} ({len(test)} rows)")


if __name__ == "__main__":
    main()
