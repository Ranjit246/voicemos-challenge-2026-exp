"""
Multi-SSL layer fusion (calibrated Mac harness) + submission.

Extends Idea F (WavLM) by stacking layer cosines from several SSL backbones:
WavLM-Large, HuBERT-Large, wav2vec2-XLSR-53, WavLM-base-plus. Ridge combines
all layer cosines; system-label prior blend at inference.

Usage:
    python mac_fusion.py            # grouped-CV report only
    python mac_fusion.py --submit   # also write answer.txt
"""

import argparse
import csv
import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, zscore
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BASE = os.path.dirname(os.path.abspath(__file__))
EXP  = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(EXP, "idea-e-feature-fusion"))
from features import build, train_table, emb_key  # noqa: E402

DEV_CSV = "/Users/ranjitpatro/Home/Research/VoiceMOS/dataset/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv"
OUT     = os.path.join(BASE, "answer.txt")
ALPHA   = {"spk_sim": 0.5, "acc_sim": 0.6}
N_SPLITS = 7
RIDGE_ALPHA = 5.0   # more features -> a touch more L2

SSL_PATHS = {
    "wavlm":     os.path.join(EXP, "idea-f-wavlm/wavlm_layers.pt"),
    "hubert":    os.path.join(BASE, "hubert_layers.pt"),
    "xlsr":      os.path.join(BASE, "xlsr_layers.pt"),
    "wavlmbase": os.path.join(BASE, "wavlmbase_layers.pt"),
}


def load(path):
    d = torch.load(path, weights_only=False)
    return {k: (v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)) for k, v in d.items()}


def lcos(dic, a, b, L):
    ea = dic[emb_key(a)][L].astype(np.float32); eb = dic[emb_key(b)][L].astype(np.float32)
    return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-8))


def add_ssl(df, name, dic):
    n = next(iter(dic.values())).shape[0]
    feats = []
    for L in range(n):
        col = f"{name}_L{L}"
        df[col] = [lcos(dic, a, b, L) for a, b in zip(df.wav_a_path, df.wav_b_path)]
        feats.append(col)
    return feats


def srcc(a, b):
    return spearmanr(a, b).statistic


def grouped(df, feats, target, groups):
    gkf = GroupKFold(n_splits=N_SPLITS)
    X, y = df[feats].values, df[target].values
    s = []
    for tr, te in gkf.split(df, groups=groups):
        m = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA)); m.fit(X[tr], y[tr])
        s.append(srcc(y[te], m.predict(X[te])))
    return np.mean(s)


def scale_1_5(x):
    x = np.asarray(x); return 1.0 + 4.0 * (x - x.min()) / (x.max() - x.min() + 1e-8)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    ssl = {n: load(p) for n, p in SSL_PATHS.items() if os.path.exists(p)}
    print(f"SSL models loaded: {list(ssl)}")

    train, base_feats = train_table()
    train = train.reset_index(drop=True)
    groups = train.system_id.values

    ssl_feats = {}
    for n, dic in ssl.items():
        ssl_feats[n] = add_ssl(train, n, dic)
        print(f"  {n}: {len(ssl_feats[n])} layers")

    # incremental grouped-CV
    print(f"\nGrouped-CV (calibrated, ridge alpha={RIDGE_ALPHA}):")
    cum = list(base_feats)
    combos = [("base", base_feats)]
    for n in ["wavlm", "hubert", "xlsr", "wavlmbase"]:
        if n in ssl_feats:
            cum = cum + ssl_feats[n]
            combos.append((f"+{n}", list(cum)))
    for label, feats in combos:
        sp = grouped(train, feats, "spk_sim", groups)
        ac = grouped(train, feats, "acc_sim", groups)
        print(f"  {label:>12} ({len(feats):>3}f): spk {sp:.4f}  acc {ac:.4f}")

    if not args.submit:
        return

    all_feats = combos[-1][1]
    dev, _ = build(pd.read_csv(DEV_CSV))
    for n, dic in ssl.items():
        add_ssl(dev, n, dic)

    cols = {}
    for target in ["spk_sim", "acc_sim"]:
        m = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
        m.fit(train[all_feats].values, train[target].values)
        cpred = m.predict(dev[all_feats].values)
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
    print(f"\nSaved -> {OUT} ({len(dev)} rows, {len(all_feats)} features)")


if __name__ == "__main__":
    main()
