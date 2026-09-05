"""
Idea L+K — richer-space RAMP + PPG accent features (our best pipeline + PPG).

Parametric ridge features = base(10) + WavLM cosines(25) + PPG distances(4).
Retrieval = richer RAMP (4-source SSL embedding-diff space, PCA-128, kNN).
Fuse (beta) -> ensemble trained head -> system prior -> [1,5].

Usage:
    python ppg_ramp.py [--submit]
"""

import argparse, csv, glob, os, sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, zscore
from scipy.spatial.distance import cdist
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(EXP, "idea-e-feature-fusion"))
from features import build, train_table, emb_key  # noqa: E402

DEV_CSV = "/Users/ranjitpatro/Home/Research/VoiceMOS/dataset/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv"
N_SPLITS = 7; K = 30; ALPHA = {"spk_sim": 0.5, "acc_sim": 0.6}; PCA_DIM = 128
OOF = os.path.join(EXP, "idea-i-ensemble/oof")


def srcc(a, b): return spearmanr(a, b).statistic
def norm(v): return v.astype(np.float32) / (np.linalg.norm(v) + 1e-8)


def load(p):
    d = torch.load(p, weights_only=False)
    return {k: (v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)) for k, v in d.items()}


ECAPA = load(os.path.join(EXP, "ecapa/embeddings.pt"))
CACC  = load(os.path.join(EXP, "idea-e-feature-fusion/commonaccent_emb.pt"))
WAVLM = load(os.path.join(EXP, "idea-f-wavlm/wavlm_layers.pt"))
PPG   = load(os.path.join(BASE, "ppg_dist.pt"))
SRC = [("ecapa", lambda k: ECAPA[k]), ("cacc", lambda k: CACC[k]),
       ("wl3", lambda k: WAVLM[k][3]), ("wl12", lambda k: WAVLM[k][12])]


def diff_matrix(df):
    return np.stack([np.concatenate([norm(f(emb_key(a))) - norm(f(emb_key(b))) for _, f in SRC])
                     for a, b in zip(df.wav_a_path, df.wav_b_path)])


def wl_cos(df):
    n = next(iter(WAVLM.values())).shape[0]
    for L in range(n):
        df[f"wl_{L}"] = [float(np.dot(a := WAVLM[emb_key(p)][L].astype(np.float32),
                                      b := WAVLM[emb_key(q)][L].astype(np.float32)) /
                              (np.linalg.norm(a)*np.linalg.norm(b)+1e-8))
                        for p, q in zip(df.wav_a_path, df.wav_b_path)]
    return [f"wl_{L}" for L in range(n)]


def ppg_feats(df):
    cos, js, bhat, l2 = [], [], [], []
    for a, b in zip(df.wav_a_path, df.wav_b_path):
        p = PPG[emb_key(a)].astype(np.float64); q = PPG[emb_key(b)].astype(np.float64)
        p /= p.sum()+1e-12; q /= q.sum()+1e-12; m = 0.5*(p+q)
        kl = lambda x, y: np.sum(x*np.log((x+1e-12)/(y+1e-12)))
        cos.append(float(np.dot(p, q)/(np.linalg.norm(p)*np.linalg.norm(q)+1e-12)))
        js.append(0.5*kl(p, m)+0.5*kl(q, m)); bhat.append(float(np.sum(np.sqrt(p*q)))); l2.append(float(np.linalg.norm(p-q)))
    df["ppg_cos"]=cos; df["ppg_js"]=js; df["ppg_bhat"]=bhat; df["ppg_l2"]=l2
    return ["ppg_cos", "ppg_js", "ppg_bhat", "ppg_l2"]


def retrieve(Xtr, ytr, Xq, k=K):
    D = cdist(Xq, Xtr); out = np.empty(len(Xq))
    for i in range(len(Xq)):
        nn = np.argpartition(D[i], k)[:k]; d = D[i, nn]
        w = np.exp(-d/(d.mean()+1e-8)); w /= w.sum(); out[i] = (w*ytr[nn]).sum()
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--submit", action="store_true"); args = ap.parse_args()
    train, base = train_table(); train = train.reset_index(drop=True)
    dev, _ = build(pd.read_csv(DEV_CSV))
    feats = base + wl_cos(train) + ppg_feats(train); wl_cos(dev); ppg_feats(dev)
    g = train.system_id.values
    Rtr, Rdev = diff_matrix(train), diff_matrix(dev)

    betas = np.linspace(0, 1, 11)
    print(f"grouped-CV. K={K}  features={len(feats)} (incl PPG)")
    print(f"{'target':>8} {'ridge':>7} {'retr':>7} {'fuse':>7} {'beta*':>6}")
    bb = {}
    for t in ["spk_sim", "acc_sim"]:
        gkf = GroupKFold(n_splits=N_SPLITS); rid = np.zeros(len(train)); ret = np.zeros(len(train))
        X = train[feats].values; y = train[t].values
        for tr, te in gkf.split(train, groups=g):
            sc = StandardScaler().fit(X[tr]); rid[te] = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
            rs = StandardScaler().fit(Rtr[tr]); pca = PCA(PCA_DIM, random_state=0).fit(rs.transform(Rtr[tr]))
            ret[te] = retrieve(pca.transform(rs.transform(Rtr[tr])), y[tr], pca.transform(rs.transform(Rtr[te])))
        fuse = [srcc(y, b*zscore(rid)+(1-b)*zscore(ret)) for b in betas]; bi = int(np.argmax(fuse)); bb[t]=betas[bi]
        print(f"{t:>8} {srcc(y,rid):>7.4f} {srcc(y,ret):>7.4f} {fuse[bi]:>7.4f} {betas[bi]:>6.1f}")
    print("\n(richer-RAMP without PPG was: fuse spk 0.6652 / acc 0.6297)")
    if not args.submit: return

    head_dev = None; fs = sorted(glob.glob(os.path.join(OOF, "head_s*.npz")))
    if fs: head_dev = np.mean([np.stack([np.load(f)["dev_spk"], np.load(f)["dev_acc"]],1) for f in fs],0)
    out = os.path.join(BASE, "submission-1"); os.makedirs(out, exist_ok=True); cols = {}
    X = train[feats].values; Xd = dev[feats].values
    rs = StandardScaler().fit(Rtr); pca = PCA(PCA_DIM, random_state=0).fit(rs.transform(Rtr))
    Ztr, Zdev = pca.transform(rs.transform(Rtr)), pca.transform(rs.transform(Rdev))
    for j, t in enumerate(["spk_sim", "acc_sim"]):
        sc = StandardScaler().fit(X); y = train[t].values
        rid = Ridge(alpha=1.0).fit(sc.transform(X), y).predict(sc.transform(Xd)); ret = retrieve(Ztr, y, Zdev)
        b = bb[t]; ramp = b*zscore(rid)+(1-b)*zscore(ret)
        parts = [zscore(ramp)]
        if head_dev is not None: parts.append(zscore(head_dev[:, j]))
        param = np.mean(parts, 0)
        prior = dev.system_id.map(train.groupby("system_id")[t].mean()).values.astype(float)
        seen = ~np.isnan(prior); zp = np.empty(len(dev)); zp[seen]=zscore(prior[seen]); zp[~seen]=zscore(param)[~seen]
        a = ALPHA[t]; v = a*zscore(param)+(1-a)*zp; cols[t] = 1+4*(v-v.min())/(v.max()-v.min()+1e-8)
    with open(os.path.join(out, "answer.txt"), "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["system_id","utterance_id","wav_a_path","wav_b_path","pred_acc_sim","pred_spk_sim"])
        for i, r in dev.iterrows():
            wr.writerow([r.system_id, r.utterance_id, r.wav_a_path, r.wav_b_path, float(cols["acc_sim"][i]), float(cols["spk_sim"][i])])
    print(f"Saved -> {out}/answer.txt")


if __name__ == "__main__":
    main()
