"""
WavLM layer probe on the CALIBRATED Mac harness (sklearn 1.9.0).

The VM's sklearn 1.7.2 gives different GroupKFold folds (higher absolute SRCC).
The Mac harness is the one calibrated to real dev, so we confirm the WavLM
conclusion here.

Usage:
    python mac_probe.py
"""

import os
import sys
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(BASE), "idea-e-feature-fusion"))
from features import train_table, emb_key  # noqa: E402

N_SPLITS = 7


def srcc(a, b):
    return spearmanr(a, b).statistic


def layer_cos(wavlm, a, b, L):
    ea = wavlm[emb_key(a)][L].astype(np.float32); eb = wavlm[emb_key(b)][L].astype(np.float32)
    return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-8))


def grouped(df, feats, target, groups):
    gkf = GroupKFold(n_splits=N_SPLITS)
    X, y = df[feats].values, df[target].values
    s = []
    for tr, te in gkf.split(df, groups=groups):
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)); m.fit(X[tr], y[tr])
        s.append(srcc(y[te], m.predict(X[te])))
    return np.mean(s)


def main():
    df, base_feats = train_table()
    df = df.reset_index(drop=True)
    groups = df.system_id.values
    wavlm = torch.load(os.path.join(BASE, "wavlm_layers.pt"), weights_only=False)
    wavlm = {k: v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v) for k, v in wavlm.items()}
    n_layers = next(iter(wavlm.values())).shape[0]

    print(f"Calibrated (Mac sklearn) grouped-CV. base features: {len(base_feats)}\n")
    print(f"{'layer':>5} {'spk':>8} {'acc':>8}")
    lsr = {}
    for L in range(n_layers):
        c = np.array([layer_cos(wavlm, a, b, L) for a, b in zip(df.wav_a_path, df.wav_b_path)])
        df[f"wavlm_L{L}"] = c
        gkf = GroupKFold(n_splits=N_SPLITS)
        sp = np.mean([srcc(df.spk_sim.values[te], c[te]) for _, te in gkf.split(df, groups=groups)])
        ac = np.mean([srcc(df.acc_sim.values[te], c[te]) for _, te in gkf.split(df, groups=groups)])
        lsr[L] = (sp, ac)
        print(f"{L:>5} {sp:>8.4f} {ac:>8.4f}")

    bs = max(lsr, key=lambda L: lsr[L][0]); ba = max(lsr, key=lambda L: lsr[L][1])
    print(f"\nbest spk layer {bs} ({lsr[bs][0]:.4f})   best acc layer {ba} ({lsr[ba][1]:.4f})")
    print(f"(ref specialized: titanet spk 0.439, ecapa 0.433 | commonaccent acc 0.407)\n")

    wl = sorted({f"wavlm_L{bs}", f"wavlm_L{ba}"})
    all_layers = [f"wavlm_L{L}" for L in range(n_layers)]
    for label, feats in [("base (Idea E)", base_feats),
                         ("base + best 2 WavLM", base_feats + wl),
                         ("base + ALL WavLM layers", base_feats + all_layers)]:
        sp = grouped(df, feats, "spk_sim", groups); ac = grouped(df, feats, "acc_sim", groups)
        print(f"  {label:>26}: spk {sp:.4f}  acc {ac:.4f}")


if __name__ == "__main__":
    main()
