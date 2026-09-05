"""
Phase 0 — Honest validation harness (the gate for everything).

Grouped cross-validation BY SYSTEM: we hold out whole systems in each fold,
mimicking the dev/eval sets which contain systems unseen in training. This is
the fix for the 0.63-internal / 0.36-real gap that came from random splits
leaking system-level score patterns.

Provides:
  - build_feature_table(): per-pair scalar features (cosines from every
    available embedding dict). Doubles as the Phase-2 fusion feature builder.
  - single_feature_srcc(): grouped-CV SRCC of each raw cosine (no training).
  - fusion_srcc(estimator): grouped-CV SRCC of a trained model over all features.
  - random_split_srcc(): the memorization gauge (compare vs grouped-CV).

Run directly to print the single-feature leaderboard + a ridge-fusion number
for whatever embeddings are currently extracted.

Usage:
    python eval_harness.py
"""

import os
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BASE        = os.path.dirname(os.path.abspath(__file__))
TRAIN_AVG   = os.path.join(BASE, "experiments", "titanet_large", "train_avg.csv")
DATASET_DIR = "vmc2026_track3_train_phase_distro_v3_syn"
N_SPLITS    = 7
SEED        = 42

# Every embedding dict we know how to load. Missing files are skipped.
EMBEDDING_DICTS = {
    "ecapa":           os.path.join(BASE, "experiments/ecapa/embeddings.pt"),
    "titanet":         os.path.join(BASE, "experiments/titanet_large/infer/embeddings/embeddings/manifest_embeddings.pt"),
    "wespeaker":       os.path.join(BASE, "experiments/pyannote/embeddings.pt"),
    "wav2vec2_hidden": os.path.join(BASE, "experiments/wav2vec2/accent_hidden_embeddings.pt"),
    "wav2vec2_prob":   os.path.join(BASE, "experiments/wav2vec2/accent_prob_embeddings.pt"),
}


def emb_key(rel_path):
    parts = rel_path.strip("/").split("/")
    return f"{DATASET_DIR}@{parts[-2]}@{parts[-1]}"


def _load(path):
    embs = torch.load(path, weights_only=False)
    return {k: (v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
            for k, v in embs.items()}


def _cosine_col(df, embs):
    out = np.empty(len(df))
    for i, (a, b) in enumerate(zip(df.wav_a_path, df.wav_b_path)):
        ea, eb = embs[emb_key(a)], embs[emb_key(b)]
        out[i] = np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-8)
    return out


def build_feature_table():
    """Return (df, feature_names). df has one cosine column per available embedding dict."""
    df = pd.read_csv(TRAIN_AVG)
    feats = []
    for name, path in EMBEDDING_DICTS.items():
        if not os.path.exists(path):
            print(f"[skip] {name}: not extracted yet")
            continue
        df[f"cos_{name}"] = _cosine_col(df, _load(path))
        feats.append(f"cos_{name}")
    return df, feats


def _srcc(a, b):
    return spearmanr(a, b).statistic


def single_feature_srcc(df, feature, target, groups):
    """Grouped-CV SRCC of one raw feature (no training) — averaged over test folds."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    scores = [
        _srcc(df[target].values[te], df[feature].values[te])
        for _, te in gkf.split(df, groups=groups)
    ]
    return np.mean(scores), np.std(scores)


def fusion_srcc(df, features, target, groups, estimator_factory):
    """Grouped-CV SRCC of a trained estimator over all features."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    X, y = df[features].values, df[target].values
    scores = []
    for tr, te in gkf.split(df, groups=groups):
        model = estimator_factory()
        model.fit(X[tr], y[tr])
        scores.append(_srcc(y[te], model.predict(X[te])))
    return np.mean(scores), np.std(scores)


def random_split_srcc(df, features, target, estimator_factory):
    """Random-split (NOT grouped) — the memorization gauge."""
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    X, y = df[features].values, df[target].values
    scores = []
    for tr, te in kf.split(X):
        model = estimator_factory()
        model.fit(X[tr], y[tr])
        scores.append(_srcc(y[te], model.predict(X[te])))
    return np.mean(scores), np.std(scores)


def ridge_factory():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def main():
    df, feats = build_feature_table()
    groups = df.system_id.values
    print(f"\nPairs: {len(df)}  |  systems: {df.system_id.nunique()}  |  features: {feats}")
    print(f"Grouped {N_SPLITS}-fold CV by system (holds out ~3 unseen systems/fold)\n")

    for target in ["spk_sim", "acc_sim"]:
        print(f"===== {target} =====")
        print(f"{'feature':>20}  {'grouped-CV SRCC':>16}")
        ranked = []
        for f in feats:
            m, s = single_feature_srcc(df, f, target, groups)
            ranked.append((f, m, s))
        for f, m, s in sorted(ranked, key=lambda x: -x[1]):
            print(f"{f:>20}  {m:>8.4f} ± {s:.3f}")

        if len(feats) >= 2:
            gm, gs = fusion_srcc(df, feats, target, groups, ridge_factory)
            rm, rs = random_split_srcc(df, feats, target, ridge_factory)
            print(f"{'RIDGE FUSION':>20}  {gm:>8.4f} ± {gs:.3f}   (grouped)")
            print(f"{'  random-split gauge':>20}  {rm:>8.4f} ± {rs:.3f}   (gap = memorization)")
        print()


if __name__ == "__main__":
    main()
