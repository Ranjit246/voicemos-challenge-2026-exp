"""
Idea K — RAMP-style retrieval augmentation (kNN over training pairs).

Parametric predictor (ridge over base+WavLM cosine features) + a non-parametric
kNN retrieval head, fused. For an unseen-system query, kNN over the training
datastore in feature space bypasses the parametric representation->score mapping
that breaks under distribution shift (RAMP, arXiv:2308.16488, won VMC'24).

Validation: sklearn GroupKFold-by-system (the calibrated harness). Reports ridge
vs retrieval vs fusion on held-out systems, sweeps the fusion weight, then builds
the dev submission (fusion -> ensemble with the trained head -> system prior).

Usage:
    python ramp.py            # grouped-CV report
    python ramp.py --submit   # also write submission-1/answer.txt
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
from sklearn.pipeline import make_pipeline

BASE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(EXP, "idea-e-feature-fusion"))
from features import build, train_table, emb_key  # noqa: E402

DEV_CSV = "/Users/ranjitpatro/Home/Research/VoiceMOS/dataset/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv"
N_SPLITS = 7; K = 30; ALPHA = {"spk_sim": 0.5, "acc_sim": 0.6}
OOF = os.path.join(EXP, "idea-i-ensemble/oof")


def srcc(a, b): return spearmanr(a, b).statistic


def load_wavlm():
    d = torch.load(os.path.join(EXP, "idea-f-wavlm/wavlm_layers.pt"), weights_only=False)
    return {k: (v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)) for k, v in d.items()}


def add_wavlm(df, wavlm, n):
    for L in range(n):
        df[f"wl_{L}"] = [float(np.dot(a := wavlm[emb_key(p)][L].astype(np.float32),
                                      b := wavlm[emb_key(q)][L].astype(np.float32)) /
                              (np.linalg.norm(a)*np.linalg.norm(b)+1e-8))
                        for p, q in zip(df.wav_a_path, df.wav_b_path)]
    return [f"wl_{L}" for L in range(n)]


def retrieve(Xtr, ytr, Xq, k=K):
    """kNN weighted-average prediction; weights = softmax(-dist / mean-dist)."""
    D = cdist(Xq, Xtr)
    out = np.empty(len(Xq))
    for i in range(len(Xq)):
        nn = np.argpartition(D[i], k)[:k]; d = D[i, nn]
        w = np.exp(-d / (d.mean() + 1e-8)); w /= w.sum()
        out[i] = (w * ytr[nn]).sum()
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    train, base = train_table(); train = train.reset_index(drop=True)
    dev, _ = build(pd.read_csv(DEV_CSV))
    wavlm = load_wavlm(); n = next(iter(wavlm.values())).shape[0]
    wl = add_wavlm(train, wavlm, n); add_wavlm(dev, wavlm, n)
    feats = base + wl
    groups = train.system_id.values

    betas = np.linspace(0, 1, 11)   # beta = weight on parametric
    print(f"grouped-CV (calibrated). K={K}  features={len(feats)}")
    print(f"{'target':>8} {'ridge':>7} {'retr':>7} {'best_fuse':>9} {'beta*':>6}")
    best_beta = {}
    for t in ["spk_sim", "acc_sim"]:
        gkf = GroupKFold(n_splits=N_SPLITS)
        rid = np.zeros(len(train)); ret = np.zeros(len(train))
        Xall = train[feats].values; y = train[t].values
        for tr, te in gkf.split(train, groups=groups):
            sc = StandardScaler().fit(Xall[tr])
            Xtr, Xte = sc.transform(Xall[tr]), sc.transform(Xall[te])
            m = Ridge(alpha=1.0).fit(Xtr, y[tr]); rid[te] = m.predict(Xte)
            ret[te] = retrieve(Xtr, y[tr], Xte)
        r_s, t_s = srcc(y, rid), srcc(y, ret)
        fuse = [srcc(y, b*zscore(rid) + (1-b)*zscore(ret)) for b in betas]
        bi = int(np.argmax(fuse)); best_beta[t] = betas[bi]
        print(f"{t:>8} {r_s:>7.4f} {t_s:>7.4f} {fuse[bi]:>9.4f} {betas[bi]:>6.1f}")

    if not args.submit:
        return

    # dev: fuse ridge(all-train) + retrieval(datastore=all-train), ensemble w/ head, prior
    head_dev = None
    fs = sorted(glob.glob(os.path.join(OOF, "head_s*.npz")))
    if fs:
        head_dev = np.mean([np.stack([np.load(f)["dev_spk"], np.load(f)["dev_acc"]], 1) for f in fs], 0)

    out = os.path.join(BASE, "submission-1"); os.makedirs(out, exist_ok=True)
    cols = {}
    Xall = train[feats].values; Xdev = dev[feats].values
    for j, t in enumerate(["spk_sim", "acc_sim"]):
        sc = StandardScaler().fit(Xall)
        Xtr, Xdv = sc.transform(Xall), sc.transform(Xdev)
        y = train[t].values
        rid = Ridge(alpha=1.0).fit(Xtr, y).predict(Xdv)
        ret = retrieve(Xtr, y, Xdv)
        b = best_beta[t]
        ramp = b*zscore(rid) + (1-b)*zscore(ret)          # RAMP fusion
        parts = [zscore(ramp)]
        if head_dev is not None:
            parts.append(zscore(head_dev[:, j]))           # ensemble with trained head
        param = np.mean(parts, 0)
        prior = dev.system_id.map(train.groupby("system_id")[t].mean()).values.astype(float)
        seen = ~np.isnan(prior); zp = np.empty(len(dev))
        zp[seen] = zscore(prior[seen]); zp[~seen] = zscore(param)[~seen]
        a = ALPHA[t]; v = a*zscore(param) + (1-a)*zp
        cols[t] = 1 + 4*(v - v.min())/(v.max()-v.min()+1e-8)

    with open(os.path.join(out, "answer.txt"), "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["system_id","utterance_id","wav_a_path","wav_b_path","pred_acc_sim","pred_spk_sim"])
        for i, r in dev.iterrows():
            wr.writerow([r.system_id, r.utterance_id, r.wav_a_path, r.wav_b_path,
                         float(cols["acc_sim"][i]), float(cols["spk_sim"][i])])
    print(f"\nSaved -> {out}/answer.txt")


if __name__ == "__main__":
    main()
