"""
Idea E — submission: extended fusion (all features) + system-label prior.

Same blend as Idea D but with the richer feature set (adds WeSpeaker,
CommonAccent embedding+posterior cosines, and UTMOS naturalness scalars).

Usage:
    python make_submission.py [--alpha 0.5]
"""

import argparse
import csv
import os
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from features import build, train_table

DEV_CSV = "/Users/ranjitpatro/Home/Research/VoiceMOS/dataset/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv"
BASE    = os.path.dirname(os.path.abspath(__file__))
OUT     = os.path.join(BASE, "answer.txt")


def scale_1_5(x):
    x = np.asarray(x); return 1.0 + 4.0 * (x - x.min()) / (x.max() - x.min() + 1e-8)


# per-target blend weight (from fusion_eval dev-like optima)
ALPHA = {"spk_sim": 0.5, "acc_sim": 0.6}


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()

    train, feats = train_table()
    dev, _ = build(pd.read_csv(DEV_CSV))
    unseen = sorted(set(dev.system_id) - set(train.system_id))
    print(f"Dev pairs: {len(dev)}  features: {len(feats)}  unseen systems: {unseen}")

    cols = {}
    for target in ["spk_sim", "acc_sim"]:
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        m.fit(train[feats].values, train[target].values)
        cos = m.predict(dev[feats].values)
        prior = dev.system_id.map(train.groupby("system_id")[target].mean()).values.astype(float)
        seen = ~np.isnan(prior)
        zc = zscore(cos)
        zp = np.empty(len(dev)); zp[seen] = zscore(prior[seen]); zp[~seen] = zc[~seen]
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
    print(f"Saved -> {OUT} ({len(dev)} rows, alpha={ALPHA})")
    print(f"To submit:  zip -j submission_feature_fusion.zip {OUT}")


if __name__ == "__main__":
    main()
