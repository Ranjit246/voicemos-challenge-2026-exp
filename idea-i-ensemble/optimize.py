"""
Offline ensemble optimizer (calibrated Mac harness, deterministic system folds).

Finds the best per-target blend weight between:
  F    = Idea-F ridge (base 10 features + all 25 WavLM layer cosines)
  head = trained-head OOF (average of 6 seeds)
  G    = multi-SSL ridge (base + WavLM+HuBERT+XLSR+WavLMbase)  [optional]

Uses out-of-fold TRAIN predictions on the SAME deterministic 7-fold system split
as the head, so ensemble weights are chosen against the mean labels — then the
chosen weighting is applied to the DEV raw predictions, blended with the system
prior, and written to submission-4/answer.txt.

Usage:
    python optimize.py
"""

import csv
import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, zscore
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BASE = os.path.dirname(os.path.abspath(__file__))
EXP  = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(EXP, "idea-e-feature-fusion"))
from features import build, train_table, emb_key  # noqa: E402

DEV_CSV = "/Users/ranjitpatro/Home/Research/VoiceMOS/dataset/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv"
N_FOLDS = 7
ALPHA = {"spk_sim": 0.5, "acc_sim": 0.6}
OOF_DIR = os.path.join(BASE, "oof")   # head_s*.npz synced from VM


def det_folds(systems):
    uniq = sorted(set(systems)); m = {s: i % N_FOLDS for i, s in enumerate(uniq)}
    return np.array([m[s] for s in systems])


def srcc(a, b): return spearmanr(a, b).statistic


def load_wavlm():
    d = torch.load(os.path.join(EXP, "idea-f-wavlm/wavlm_layers.pt"), weights_only=False)
    return {k: (v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)) for k, v in d.items()}


def add_wavlm_cos(df, wavlm, n):
    for L in range(n):
        df[f"wl_{L}"] = [float(np.dot(a := wavlm[emb_key(p)][L].astype(np.float32),
                                      b := wavlm[emb_key(q)][L].astype(np.float32)) /
                              (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
                         for p, q in zip(df.wav_a_path, df.wav_b_path)]
    return [f"wl_{L}" for L in range(n)]


def ridge_oof_dev(train, dev, feats, folds):
    oof = np.zeros((len(train), 2)); devp = np.zeros((len(dev), 2))
    for j, t in enumerate(["spk_sim", "acc_sim"]):
        for f in range(N_FOLDS):
            te = folds == f
            m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            m.fit(train[feats].values[~te], train[t].values[~te])
            oof[te, j] = m.predict(train[feats].values[te])
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        m.fit(train[feats].values, train[t].values); devp[:, j] = m.predict(dev[feats].values)
    return oof, devp


def main():
    wavlm = load_wavlm(); n = next(iter(wavlm.values())).shape[0]
    train, base_feats = train_table(); train = train.reset_index(drop=True)
    dev, _ = build(pd.read_csv(DEV_CSV))
    wl_tr = add_wavlm_cos(train, wavlm, n); add_wavlm_cos(dev, wavlm, n)
    folds = det_folds(train.system_id.values)

    # F member (Idea F: base + WavLM cosines)
    F_oof, F_dev = ridge_oof_dev(train, dev, base_feats + wl_tr, folds)

    # head member (avg of 6 seed OOFs + dev raw)
    files = sorted(glob.glob(os.path.join(OOF_DIR, "head_s*.npz")))
    print(f"head OOF seeds: {len(files)}")
    H_oof = np.mean([np.stack([np.load(f)["oof_spk"], np.load(f)["oof_acc"]], 1) for f in files], 0)
    H_dev = np.mean([np.stack([np.load(f)["dev_spk"], np.load(f)["dev_acc"]], 1) for f in files], 0)

    y = train[["spk_sim", "acc_sim"]].values

    def zc(x): return zscore(x)
    print(f"\n{'target':>8} {'F_only':>8} {'H_only':>8} {'best_w(F)':>10} {'best_srcc':>10}")
    best_w = {}
    for j, t in enumerate(["spk_sim", "acc_sim"]):
        f_s = srcc(y[:, j], F_oof[:, j]); h_s = srcc(y[:, j], H_oof[:, j])
        grid = np.linspace(0, 1, 21)
        scores = [srcc(y[:, j], w * zc(F_oof[:, j]) + (1 - w) * zc(H_oof[:, j])) for w in grid]
        bi = int(np.argmax(scores)); best_w[t] = grid[bi]
        print(f"{t:>8} {f_s:>8.4f} {h_s:>8.4f} {grid[bi]:>10.2f} {scores[bi]:>10.4f}")

    # apply best weights to DEV raw, blend prior
    def blend(pred, t, alpha):
        prior = dev.system_id.map(train.groupby("system_id")[t].mean()).values.astype(float)
        seen = ~np.isnan(prior); zp = np.empty(len(dev))
        zp[seen] = zscore(prior[seen]); zp[~seen] = zscore(pred)[~seen]
        v = alpha * zscore(pred) + (1 - alpha) * zp
        return 1 + 4 * (v - v.min()) / (v.max() - v.min() + 1e-8)

    out = os.path.join(BASE, "submission-4"); os.makedirs(out, exist_ok=True)
    cols = {}
    for j, t in enumerate(["spk_sim", "acc_sim"]):
        w = best_w[t]
        dev_pred = w * zscore(F_dev[:, j]) + (1 - w) * zscore(H_dev[:, j])
        cols[t] = blend(dev_pred, t, ALPHA[t])
    with open(os.path.join(out, "answer.txt"), "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["system_id","utterance_id","wav_a_path","wav_b_path","pred_acc_sim","pred_spk_sim"])
        for i, r in dev.iterrows():
            wr.writerow([r.system_id, r.utterance_id, r.wav_a_path, r.wav_b_path,
                         float(cols["acc_sim"][i]), float(cols["spk_sim"][i])])
    print(f"\nSaved optimized -> {out}/answer.txt  (weights F: spk={best_w['spk_sim']:.2f} acc={best_w['acc_sim']:.2f})")


if __name__ == "__main__":
    main()
