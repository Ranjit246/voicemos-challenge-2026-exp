"""
Idea B — Counterfactual pseudo-pairs.

Problem: in real training pairs, spk_sim and acc_sim correlate 0.86, so the
acc head can never learn "same accent, DIFFERENT speaker" — the case that
decouples the two. We manufacture exactly those pairs from natural VCTK speech.

Method:
  1. Cluster natural wavs (sys019 + sys008) by ECAPA embedding -> ~32 speaker clusters.
  2. Label each cluster's accent via mean wav2vec2 accent posterior.
  3. Manufacture three pair types with pseudo-labels:
       same-speaker            -> spk high, acc high     (the GT case)
       diff-speaker SAME accent -> spk LOW,  acc HIGH     <- the decorrelating case
       diff-speaker diff accent -> spk low,  acc low
  4. Compute the same cosine features used by the fusion harness.

Output: counterfactual_features.csv  (cosine features + pseudo-labels + pair_type)

Usage:
    python build_pairs.py
"""

import os
import numpy as np
import pandas as pd
import torch
from collections import Counter
from sklearn.cluster import AgglomerativeClustering

BASE     = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.dirname(BASE)
OUT      = os.path.join(BASE, "counterfactual_features.csv")
SEED     = 42

EMBEDDING_DICTS = {
    "ecapa":           os.path.join(EXP_ROOT, "ecapa/embeddings.pt"),
    "titanet":         os.path.join(EXP_ROOT, "titanet_large/infer/embeddings/embeddings/manifest_embeddings.pt"),
    "wav2vec2_hidden": os.path.join(EXP_ROOT, "wav2vec2/accent_hidden_embeddings.pt"),
    "wav2vec2_prob":   os.path.join(EXP_ROOT, "wav2vec2/accent_prob_embeddings.pt"),
}
ACC_LABELS = ['American','Australian','British','Canadian','English','Indian',
              'Irish','NewZealand','NorthernIrish','Scottish','SouthAfrican','Unknown','Welsh']

N_CLUSTERS = 32
# pseudo-labels (rank order is what matters for SRCC, not exact values)
PSEUDO = {
    "same_speaker":      dict(spk_sim=4.8, acc_sim=4.7),
    "diff_spk_same_acc": dict(spk_sim=1.5, acc_sim=4.0),   # the key decorrelating case
    "diff_spk_diff_acc": dict(spk_sim=1.3, acc_sim=1.8),
}
N_PER_TYPE = 350


def load(path):
    d = torch.load(path, weights_only=False)
    return {k: (v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)) for k, v in d.items()}


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    rng = np.random.default_rng(SEED)
    embs = {name: load(path) for name, path in EMBEDDING_DICTS.items()}
    ecapa, prob = embs["ecapa"], embs["wav2vec2_prob"]

    nat_keys = [k for k in ecapa if ("sys019" in k or "sys008" in k)]
    print(f"Natural wavs: {len(nat_keys)}")

    # cluster by speaker
    X = np.stack([ecapa[k] / np.linalg.norm(ecapa[k]) for k in nat_keys])
    labels = AgglomerativeClustering(n_clusters=N_CLUSTERS, metric="cosine",
                                     linkage="average").fit_predict(X)
    clusters = {c: [nat_keys[i] for i in np.where(labels == c)[0]] for c in set(labels)}

    # accent per cluster
    cluster_accent = {}
    for c, keys in clusters.items():
        meanp = np.mean([prob[k] for k in keys], axis=0)
        cluster_accent[c] = ACC_LABELS[int(np.argmax(meanp))]
    print(f"Clusters: {N_CLUSTERS}  accents: {dict(Counter(cluster_accent.values()))}")

    usable = [c for c, ks in clusters.items() if len(ks) >= 2]
    accent_to_clusters = {}
    for c in usable:
        accent_to_clusters.setdefault(cluster_accent[c], []).append(c)

    def feats(ka, kb):
        return {f"cos_{n}": cosine(embs[n][ka], embs[n][kb]) for n in EMBEDDING_DICTS}

    rows = []

    # same speaker
    for _ in range(N_PER_TYPE):
        c = usable[rng.integers(len(usable))]
        ka, kb = rng.choice(clusters[c], size=2, replace=False)
        rows.append({**feats(ka, kb), **PSEUDO["same_speaker"], "pair_type": "same_speaker"})

    # different speaker, SAME accent (needs an accent with >=2 clusters)
    multi = {a: cs for a, cs in accent_to_clusters.items() if len(cs) >= 2}
    for _ in range(N_PER_TYPE):
        a = list(multi)[rng.integers(len(multi))]
        c1, c2 = rng.choice(multi[a], size=2, replace=False)
        ka = clusters[c1][rng.integers(len(clusters[c1]))]
        kb = clusters[c2][rng.integers(len(clusters[c2]))]
        rows.append({**feats(ka, kb), **PSEUDO["diff_spk_same_acc"], "pair_type": "diff_spk_same_acc"})

    # different speaker, different accent
    for _ in range(N_PER_TYPE):
        c1, c2 = rng.choice(usable, size=2, replace=False)
        if cluster_accent[c1] == cluster_accent[c2]:
            continue
        ka = clusters[c1][rng.integers(len(clusters[c1]))]
        kb = clusters[c2][rng.integers(len(clusters[c2]))]
        rows.append({**feats(ka, kb), **PSEUDO["diff_spk_diff_acc"], "pair_type": "diff_spk_diff_acc"})

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nManufactured {len(df)} pairs: {dict(df.pair_type.value_counts())}")
    print("Mean cosines by pair type:")
    print(df.groupby("pair_type")[[f"cos_{n}" for n in EMBEDDING_DICTS]].mean().round(3))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
