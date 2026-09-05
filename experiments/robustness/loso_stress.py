"""
LOSO stress test — does the system-label prior survive as more systems are UNSEEN?

The July-31 eval set may hold out more systems than dev's 49%. The literature warns
learned system/domain priors fail OOD (VMC'24 winner T05; UTMOS listener module).
This sweeps the number of fully-unseen systems and measures, on a dev-like mixed
test set, the SRCC of:
  cosine      = Idea-F ridge (base + WavLM) prediction, no prior
  prior-blend = cosine blended with training-system-mean prior (unseen -> cosine fallback)
If (prior - cosine) stays >= 0 as unseen count grows, the prior is safe for eval.
If it goes negative, we must gate/down-weight it before Aug 7.

Usage:
    python loso_stress.py
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, zscore
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BASE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(EXP, "idea-e-feature-fusion"))
from features import train_table, emb_key  # noqa: E402

SEEN_PER_SYS = 14; N_REP = 60; ALPHA = {"spk_sim": 0.5, "acc_sim": 0.6}
UNSEEN_LEVELS = [2, 4, 6, 8, 10]


def srcc(a, b): return spearmanr(a, b).statistic


def load_wavlm():
    d = torch.load(os.path.join(EXP, "idea-f-wavlm/wavlm_layers.pt"), weights_only=False)
    return {k: (v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)) for k, v in d.items()}


def main():
    train, base = train_table(); train = train.reset_index(drop=True)
    wavlm = load_wavlm(); n = next(iter(wavlm.values())).shape[0]
    for L in range(n):
        train[f"wl_{L}"] = [float(np.dot(a := wavlm[emb_key(p)][L].astype(np.float32),
                                         b := wavlm[emb_key(q)][L].astype(np.float32)) /
                                  (np.linalg.norm(a)*np.linalg.norm(b)+1e-8))
                           for p, q in zip(train.wav_a_path, train.wav_b_path)]
    feats = base + [f"wl_{L}" for L in range(n)]
    systems = train.system_id.unique()
    rng = np.random.default_rng(0)

    print(f"{'unseen':>7} {'target':>8} {'cosine':>8} {'prior':>8} {'delta':>8}  {'frac_uns':>8}")
    for nun in UNSEEN_LEVELS:
        for t in ["spk_sim", "acc_sim"]:
            cos_s, pri_s, fr = [], [], []
            for _ in range(N_REP):
                unseen = set(rng.choice(systems, nun, replace=False))
                test_idx = list(train.index[train.system_id.isin(unseen)])
                for s in systems:
                    if s in unseen: continue
                    idx = train.index[train.system_id == s].to_numpy()
                    test_idx += list(rng.choice(idx, min(SEEN_PER_SYS, len(idx)), replace=False))
                te = train.index.isin(test_idx); tr = ~te
                m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                m.fit(train[feats].values[tr], train[t].values[tr])
                cos = m.predict(train[feats].values[te])
                sub = train[te]
                prior = sub.system_id.map(train[tr].groupby("system_id")[t].mean()).values.astype(float)
                seen = ~np.isnan(prior)
                zc = zscore(cos); zp = np.empty(len(sub))
                zp[seen] = zscore(prior[seen]); zp[~seen] = zc[~seen]
                a = ALPHA[t]
                y = sub[t].values
                cos_s.append(srcc(y, cos)); pri_s.append(srcc(y, a*zc + (1-a)*zp))
                fr.append((~seen).mean())
            c, p = np.mean(cos_s), np.mean(pri_s)
            print(f"{nun:>7} {t:>8} {c:>8.4f} {p:>8.4f} {p-c:>+8.4f}  {np.mean(fr):>8.2f}")
    print("\ndev is ~49% unseen (2 of 23 systems, 296/600 pairs). Read the row nearest that frac.")


if __name__ == "__main__":
    main()
