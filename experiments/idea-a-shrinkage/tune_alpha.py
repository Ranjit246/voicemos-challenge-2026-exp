"""
Idea A — System-mean shrinkage: tune alpha on the train set.

For a zero-shot predictor (embedding cosine), predictions never see labels,
so shrinkage toward the per-system mean prediction is label-free:

    final(pair) = alpha * score(pair) + (1 - alpha) * mean(score of same-system pairs)

alpha = 1.0  -> raw per-utterance score (no shrinkage)
alpha = 0.0  -> pure system-level score (every pair gets its system's mean)

We sweep alpha and measure UTT-SRCC against the human mean labels on:
  1. the full train set (2800 pairs, ~133 pairs/system)
  2. a "dev-like" simulation: 15 pairs/system subsampled, averaged over
     many random draws (dev has ~12-16 pairs/system, so system means are
     noisier there — this checks the benefit survives)

Usage:
    python tune_alpha.py
"""

import os
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

BASE        = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT    = os.path.dirname(BASE)
TRAIN_AVG   = os.path.join(EXP_ROOT, "titanet_large", "train_avg.csv")
DATASET_DIR = "vmc2026_track3_train_phase_distro_v3_syn"

EMBEDDING_SETS = {
    "titanet":        os.path.join(EXP_ROOT, "titanet_large/infer/embeddings/embeddings/manifest_embeddings.pt"),
    "wav2vec2_hidden": os.path.join(EXP_ROOT, "wav2vec2/accent_hidden_embeddings.pt"),
    "wav2vec2_prob":   os.path.join(EXP_ROOT, "wav2vec2/accent_prob_embeddings.pt"),
}

ALPHAS = np.round(np.arange(0.0, 1.01, 0.1), 2)
DEV_LIKE_PAIRS_PER_SYSTEM = 15
DEV_LIKE_DRAWS = 50
SEED = 42


def emb_key(rel_path):
    parts = rel_path.strip("/").split("/")
    return f"{DATASET_DIR}@{parts[-2]}@{parts[-1]}"


def load_embeddings(path):
    embs = torch.load(path, weights_only=False)
    return {k: (v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
            for k, v in embs.items()}


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def shrink(scores, systems, alpha):
    """Blend each score toward its system's mean score."""
    s = pd.Series(scores)
    sys_mean = s.groupby(systems).transform("mean")
    return alpha * s + (1 - alpha) * sys_mean


def srcc(a, b):
    return spearmanr(a, b).statistic


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(TRAIN_AVG)

    print(f"Train pairs: {len(df)}  |  systems: {df.system_id.nunique()}")
    print(f"Sweep: alpha in {list(ALPHAS)}")
    print(f"Dev-like sim: {DEV_LIKE_PAIRS_PER_SYSTEM} pairs/system x {DEV_LIKE_DRAWS} draws\n")

    results = []

    for name, path in EMBEDDING_SETS.items():
        if not os.path.exists(path):
            print(f"[skip] {name}: {path} not found")
            continue

        embs = load_embeddings(path)
        df["score"] = [
            cosine(embs[emb_key(a)], embs[emb_key(b)])
            for a, b in zip(df.wav_a_path, df.wav_b_path)
        ]

        print(f"=== {name} ===")
        print(f"{'alpha':>6}  {'spk_full':>9}  {'acc_full':>9}  {'spk_devlike':>11}  {'acc_devlike':>11}")

        for alpha in ALPHAS:
            blended = shrink(df.score.values, df.system_id.values, alpha)
            spk_full = srcc(df.spk_sim, blended)
            acc_full = srcc(df.acc_sim, blended)

            # dev-like: subsample pairs per system, shrink within the subsample
            spk_sub, acc_sub = [], []
            for _ in range(DEV_LIKE_DRAWS):
                sub = (df.groupby("system_id", group_keys=False)
                         .apply(lambda g: g.sample(min(DEV_LIKE_PAIRS_PER_SYSTEM, len(g)),
                                                   random_state=rng.integers(1 << 31)),
                                include_groups=False)
                         .assign(system_id=lambda d: d.index.map(
                             dict(zip(df.index, df.system_id)))))
                b = shrink(sub.score.values, sub.system_id.values, alpha)
                spk_sub.append(srcc(sub.spk_sim, b))
                acc_sub.append(srcc(sub.acc_sim, b))

            row = dict(predictor=name, alpha=alpha,
                       spk_full=spk_full, acc_full=acc_full,
                       spk_devlike=np.mean(spk_sub), acc_devlike=np.mean(acc_sub))
            results.append(row)
            print(f"{alpha:>6.1f}  {spk_full:>9.4f}  {acc_full:>9.4f}  "
                  f"{np.mean(spk_sub):>11.4f}  {np.mean(acc_sub):>11.4f}")
        print()

    res = pd.DataFrame(results)
    out_csv = os.path.join(BASE, "alpha_sweep_results.csv")
    res.to_csv(out_csv, index=False)

    print("=== Best configs (dev-like) ===")
    for target in ["spk_devlike", "acc_devlike"]:
        best = res.loc[res[target].idxmax()]
        print(f"{target}: predictor={best.predictor}  alpha={best.alpha}  SRCC={best[target]:.4f}")
    print(f"\nSaved -> {out_csv}")


if __name__ == "__main__":
    main()
