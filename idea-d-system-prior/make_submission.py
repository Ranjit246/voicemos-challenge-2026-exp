"""
Idea D — submission: cosine fusion blended with training-label system prior.

For each dev pair:
  cosine_pred = ridge fusion over cosines (trained on all train pairs)
  prior       = training-label mean score of that system   (seen systems)
              = cosine_pred itself                          (unseen systems: sys003, sys015)
  final       = alpha * z(cosine_pred) + (1-alpha) * z(prior)   -> scaled to [1,5]

alpha from dev-like sweep (dev_like_eval.py): 0.7 for both targets.

Usage:
    python make_submission.py
"""

import csv
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BASE     = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.dirname(BASE)
sys.path.insert(0, EXP_ROOT)
from eval_harness import build_feature_table, EMBEDDING_DICTS, emb_key  # noqa: E402

import torch

DEV_CSV   = "/Users/ranjitpatro/Home/Research/VoiceMOS/dataset/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv"
TRAIN_AVG = os.path.join(EXP_ROOT, "titanet_large", "train_avg.csv")
OUT       = os.path.join(BASE, "answer.txt")
ALPHA     = {"spk_sim": 0.5, "acc_sim": 0.5}  # from dev_like_eval (realistic 49%-unseen)


def load(path):
    d = torch.load(path, weights_only=False)
    return {k: (v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)) for k, v in d.items()}


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def scale_1_5(x):
    x = np.asarray(x); return 1.0 + 4.0 * (x - x.min()) / (x.max() - x.min() + 1e-8)


def main():
    train, feats = build_feature_table()
    embs = {n: load(p) for n, p in EMBEDDING_DICTS.items() if os.path.exists(p)}
    avail = [f for f in feats]  # cos_<name> for available dicts

    dev = pd.read_csv(DEV_CSV)
    # dev cosine features
    for name in embs:
        col = f"cos_{name}"
        if col in avail:
            dev[col] = [cosine(embs[name][emb_key(a)], embs[name][emb_key(b)])
                        for a, b in zip(dev.wav_a_path, dev.wav_b_path)]

    train_systems = set(train.system_id)
    unseen = sorted(set(dev.system_id) - train_systems)
    print(f"Dev pairs: {len(dev)}  |  unseen systems (no prior): {unseen}")

    out_cols = {}
    for target in ["spk_sim", "acc_sim"]:
        # cosine fusion trained on all train
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(train[avail].values, train[target].values)
        cos_pred = model.predict(dev[avail].values)

        # training-label system mean prior
        sysmean = train.groupby("system_id")[target].mean()
        prior = dev.system_id.map(sysmean).values.astype(float)

        # unseen systems: fall back to cosine prediction (so blend reduces to cosine there)
        zc = zscore(cos_pred)
        zp = np.where(np.isnan(prior), np.nan, prior)
        # z-score prior over the seen entries, put cosine-z where unseen
        seen_mask = ~np.isnan(prior)
        zp_full = np.empty(len(dev))
        zp_full[seen_mask] = zscore(prior[seen_mask])
        zp_full[~seen_mask] = zc[~seen_mask]

        a = ALPHA[target]
        final = a * zc + (1 - a) * zp_full
        out_cols[target] = scale_1_5(final)
        print(f"{target}: alpha={a}  seen={seen_mask.sum()}  unseen={(~seen_mask).sum()}")

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system_id", "utterance_id", "wav_a_path",
                                          "wav_b_path", "pred_acc_sim", "pred_spk_sim"])
        w.writeheader()
        for i, row in dev.iterrows():
            w.writerow({
                "system_id": row.system_id, "utterance_id": row.utterance_id,
                "wav_a_path": row.wav_a_path, "wav_b_path": row.wav_b_path,
                "pred_acc_sim": float(out_cols["acc_sim"][i]),
                "pred_spk_sim": float(out_cols["spk_sim"][i]),
            })
    print(f"\nSaved -> {OUT}  ({len(dev)} rows)")
    print(f"To submit:  zip -j submission_system_prior.zip {OUT}")


if __name__ == "__main__":
    main()
