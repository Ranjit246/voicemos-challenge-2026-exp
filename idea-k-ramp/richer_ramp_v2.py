"""
Idea K (v3) — per-target RAMP retrieval spaces.

richer_ramp used ONE shared retrieval space for both targets. This uses a
TARGET-SPECIFIC embedding-difference space:
  spk retrieval: ECAPA + TitaNet + WeSpeaker + WavLM early layers (speaker info)
  acc retrieval: CommonAccent + wav2vec2-accent + WavLM mid layers (accent info)
so each score is retrieved by the representation that actually carries it.

Validation: sklearn GroupKFold-by-system. --submit builds submission-3/answer.txt.

Usage:
    python richer_ramp_v2.py [--submit]
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


# ── embedding sources ──
ECAPA = load(os.path.join(EXP, "ecapa/embeddings.pt"))
TITA  = load(os.path.join(EXP, "titanet_large/infer/embeddings/embeddings/manifest_embeddings.pt"))
WESP  = load(os.path.join(EXP, "pyannote/embeddings.pt"))
CACC  = load(os.path.join(EXP, "idea-e-feature-fusion/commonaccent_emb.pt"))
W2VA  = load(os.path.join(EXP, "wav2vec2/accent_hidden_embeddings.pt"))
WAVLM = load(os.path.join(EXP, "idea-f-wavlm/wavlm_layers.pt"))

def wl(k, L): return WAVLM[k][L]

SPK_SRC = [("ecapa", lambda k: ECAPA[k]), ("tita", lambda k: TITA[k]), ("wesp", lambda k: WESP[k]),
           ("wl0", lambda k: wl(k, 0)), ("wl3", lambda k: wl(k, 3))]
ACC_SRC = [("cacc", lambda k: CACC[k]), ("w2va", lambda k: W2VA[k]),
           ("wl11", lambda k: wl(k, 11)), ("wl12", lambda k: wl(k, 12)), ("wl13", lambda k: wl(k, 13))]


def diff_matrix(df, srcs):
    rows = []
    for a, b in zip(df.wav_a_path, df.wav_b_path):
        ka, kb = emb_key(a), emb_key(b)
        rows.append(np.concatenate([norm(f(ka)) - norm(f(kb)) for _, f in srcs]))
    return np.stack(rows)


def wl_cosfeats(df):
    n = next(iter(WAVLM.values())).shape[0]
    for L in range(n):
        df[f"wl_{L}"] = [float(np.dot(a := WAVLM[emb_key(p)][L].astype(np.float32),
                                      b := WAVLM[emb_key(q)][L].astype(np.float32)) /
                              (np.linalg.norm(a)*np.linalg.norm(b)+1e-8))
                        for p, q in zip(df.wav_a_path, df.wav_b_path)]
    return [f"wl_{L}" for L in range(n)]


def retrieve(Xtr, ytr, Xq, k=K):
    D = cdist(Xq, Xtr); out = np.empty(len(Xq))
    for i in range(len(Xq)):
        nn = np.argpartition(D[i], k)[:k]; d = D[i, nn]
        w = np.exp(-d / (d.mean() + 1e-8)); w /= w.sum(); out[i] = (w * ytr[nn]).sum()
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--submit", action="store_true"); args = ap.parse_args()
    train, base = train_table(); train = train.reset_index(drop=True)
    dev, _ = build(pd.read_csv(DEV_CSV))
    feats = base + wl_cosfeats(train); wl_cosfeats(dev)
    groups = train.system_id.values
    print("building per-target retrieval spaces...")
    R = {"spk_sim": (diff_matrix(train, SPK_SRC), diff_matrix(dev, SPK_SRC)),
         "acc_sim": (diff_matrix(train, ACC_SRC), diff_matrix(dev, ACC_SRC))}

    betas = np.linspace(0, 1, 11)
    print(f"\ngrouped-CV (calibrated). K={K}")
    print(f"{'target':>8} {'ridge':>7} {'retr':>7} {'fuse':>7} {'beta*':>6}")
    best_beta = {}
    for t in ["spk_sim", "acc_sim"]:
        Rtr, _ = R[t]
        gkf = GroupKFold(n_splits=N_SPLITS); rid = np.zeros(len(train)); ret = np.zeros(len(train))
        X = train[feats].values; y = train[t].values
        for tr, te in gkf.split(train, groups=groups):
            sc = StandardScaler().fit(X[tr]); m = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
            rid[te] = m.predict(sc.transform(X[te]))
            rs = StandardScaler().fit(Rtr[tr]); pca = PCA(PCA_DIM, random_state=0).fit(rs.transform(Rtr[tr]))
            ret[te] = retrieve(pca.transform(rs.transform(Rtr[tr])), y[tr], pca.transform(rs.transform(Rtr[te])))
        fuse = [srcc(y, b*zscore(rid)+(1-b)*zscore(ret)) for b in betas]; bi = int(np.argmax(fuse)); best_beta[t]=betas[bi]
        print(f"{t:>8} {srcc(y,rid):>7.4f} {srcc(y,ret):>7.4f} {fuse[bi]:>7.4f} {betas[bi]:>6.1f}")
    print("\n(v2 shared-space fuse was spk 0.6652 / acc 0.6297)")
    if not args.submit: return

    head_dev = None; fs = sorted(glob.glob(os.path.join(OOF, "head_s*.npz")))
    if fs: head_dev = np.mean([np.stack([np.load(f)["dev_spk"], np.load(f)["dev_acc"]],1) for f in fs],0)
    out = os.path.join(BASE, "submission-3"); os.makedirs(out, exist_ok=True); cols = {}
    X = train[feats].values; Xd = dev[feats].values
    for j, t in enumerate(["spk_sim", "acc_sim"]):
        Rtr, Rdev = R[t]; sc = StandardScaler().fit(X); y = train[t].values
        rid = Ridge(alpha=1.0).fit(sc.transform(X), y).predict(sc.transform(Xd))
        rs = StandardScaler().fit(Rtr); pca = PCA(PCA_DIM, random_state=0).fit(rs.transform(Rtr))
        ret = retrieve(pca.transform(rs.transform(Rtr)), y, pca.transform(rs.transform(Rdev)))
        b = best_beta[t]; ramp = b*zscore(rid)+(1-b)*zscore(ret)
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
