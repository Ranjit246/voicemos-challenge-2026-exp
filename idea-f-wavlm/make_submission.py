"""
Idea F — submission: Idea-E fusion + ALL 25 WavLM-Large layer cosines + system prior.

Calibrated Mac grouped-CV: base 0.474/0.455 -> base+all-WavLM 0.501/0.489.
Individual WavLM layer cosines are weak, but ridge combines all 25 into
complementary signal. Same system-label-prior blend as Idea D/E.

Usage:
    python make_submission.py
"""

import csv
import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import zscore
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(BASE), "idea-e-feature-fusion"))
from features import build, train_table, emb_key  # noqa: E402

DEV_CSV = "/Users/ranjitpatro/Home/Research/VoiceMOS/dataset/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv"
OUT     = os.path.join(BASE, "answer.txt")
ALPHA   = {"spk_sim": 0.5, "acc_sim": 0.6}


def load_wavlm():
    d = torch.load(os.path.join(BASE, "wavlm_layers.pt"), weights_only=False)
    return {k: (v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)) for k, v in d.items()}


def layer_cos(wavlm, a, b, L):
    ea = wavlm[emb_key(a)][L].astype(np.float32); eb = wavlm[emb_key(b)][L].astype(np.float32)
    return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-8))


def add_wavlm(df, wavlm, n_layers):
    for L in range(n_layers):
        df[f"wavlm_L{L}"] = [layer_cos(wavlm, a, b, L)
                             for a, b in zip(df.wav_a_path, df.wav_b_path)]
    return [f"wavlm_L{L}" for L in range(n_layers)]


def scale_1_5(x):
    x = np.asarray(x); return 1.0 + 4.0 * (x - x.min()) / (x.max() - x.min() + 1e-8)


def main():
    wavlm = load_wavlm()
    n_layers = next(iter(wavlm.values())).shape[0]

    train, base_feats = train_table()
    wl = add_wavlm(train, wavlm, n_layers)
    feats = base_feats + wl

    dev, _ = build(pd.read_csv(DEV_CSV))
    add_wavlm(dev, wavlm, n_layers)

    unseen = sorted(set(dev.system_id) - set(train.system_id))
    print(f"Dev pairs: {len(dev)}  features: {len(feats)} ({len(base_feats)} base + {n_layers} WavLM)  unseen: {unseen}")

    cols = {}
    for target in ["spk_sim", "acc_sim"]:
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        m.fit(train[feats].values, train[target].values)
        cpred = m.predict(dev[feats].values)
        prior = dev.system_id.map(train.groupby("system_id")[target].mean()).values.astype(float)
        seen = ~np.isnan(prior)
        zc = zscore(cpred); zp = np.empty(len(dev))
        zp[seen] = zscore(prior[seen]); zp[~seen] = zc[~seen]
        a = ALPHA[target]
        cols[target] = scale_1_5(a * zc + (1 - a) * zp)

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system_id", "utterance_id", "wav_a_path",
                                          "wav_b_path", "pred_acc_sim", "pred_spk_sim"])
        w.writeheader()
        for i, row in dev.iterrows():
            w.writerow({"system_id": row.system_id, "utterance_id": row.utterance_id,
                        "wav_a_path": row.wav_a_path, "wav_b_path": row.wav_b_path,
                        "pred_acc_sim": float(cols["acc_sim"][i]),
                        "pred_spk_sim": float(cols["spk_sim"][i])})
    print(f"Saved -> {OUT} ({len(dev)} rows)")
    print(f"To submit:  zip -j submission_wavlm_fusion.zip {OUT}")


if __name__ == "__main__":
    main()
