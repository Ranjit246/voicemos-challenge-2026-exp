"""
Idea L — evaluate PPG (phonetic-posteriorgram) accent features on the harness.

Pair features from the per-utterance mean phone distributions p (wav_a), q (wav_b):
  ppg_cos   = cosine(p, q)
  ppg_js    = Jensen-Shannon divergence
  ppg_bhat  = Bhattacharyya coefficient  sum(sqrt(p*q))
  ppg_l2    = ||p - q||
Tests: single-feature grouped-CV, and F (base+WavLM) with vs without PPG.

Usage:
    python ppg_eval.py
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BASE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(EXP, "idea-e-feature-fusion"))
from features import train_table, emb_key  # noqa: E402
N_SPLITS = 7


def srcc(a, b): return spearmanr(a, b).statistic


def load(p):
    d = torch.load(p, weights_only=False)
    return {k: (v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)) for k, v in d.items()}


def ppg_feats(df, ppg):
    cos, js, bhat, l2 = [], [], [], []
    for a, b in zip(df.wav_a_path, df.wav_b_path):
        p = ppg[emb_key(a)].astype(np.float64); q = ppg[emb_key(b)].astype(np.float64)
        p = p/ (p.sum()+1e-12); q = q/(q.sum()+1e-12)
        cos.append(float(np.dot(p, q)/(np.linalg.norm(p)*np.linalg.norm(q)+1e-12)))
        m = 0.5*(p+q)
        kl = lambda x, y: np.sum(x*np.log((x+1e-12)/(y+1e-12)))
        js.append(0.5*kl(p, m)+0.5*kl(q, m))
        bhat.append(float(np.sum(np.sqrt(p*q))))
        l2.append(float(np.linalg.norm(p-q)))
    df["ppg_cos"] = cos; df["ppg_js"] = js; df["ppg_bhat"] = bhat; df["ppg_l2"] = l2
    return ["ppg_cos", "ppg_js", "ppg_bhat", "ppg_l2"]


def wl_cos(df, wavlm):
    n = next(iter(wavlm.values())).shape[0]
    for L in range(n):
        df[f"wl_{L}"] = [float(np.dot(a := wavlm[emb_key(p)][L].astype(np.float32),
                                      b := wavlm[emb_key(q)][L].astype(np.float32)) /
                              (np.linalg.norm(a)*np.linalg.norm(b)+1e-8))
                        for p, q in zip(df.wav_a_path, df.wav_b_path)]
    return [f"wl_{L}" for L in range(n)]


def grouped(df, feats, t, g):
    gkf = GroupKFold(n_splits=N_SPLITS); X, y = df[feats].values, df[t].values; s = []
    for tr, te in gkf.split(df, groups=g):
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)); m.fit(X[tr], y[tr]); s.append(srcc(y[te], m.predict(X[te])))
    return np.mean(s)


def main():
    train, base = train_table(); train = train.reset_index(drop=True)
    ppg = load(os.path.join(BASE, "ppg_dist.pt"))
    wavlm = load(os.path.join(EXP, "idea-f-wavlm/wavlm_layers.pt"))
    pf = ppg_feats(train, ppg); wl = wl_cos(train, wavlm)
    g = train.system_id.values
    F = base + wl

    print("single PPG features (grouped-CV SRCC):")
    for f in pf:
        print(f"  {f:>10}: spk {grouped(train, [f], 'spk_sim', g):.4f}  acc {grouped(train, [f], 'acc_sim', g):.4f}")
    print("\nfusion:")
    for label, feats in [("F (base+WavLM)", F), ("F + PPG", F + pf)]:
        print(f"  {label:>16}: spk {grouped(train, feats, 'spk_sim', g):.4f}  acc {grouped(train, feats, 'acc_sim', g):.4f}")


if __name__ == "__main__":
    main()
