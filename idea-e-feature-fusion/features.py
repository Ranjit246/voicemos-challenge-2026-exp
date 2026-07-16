"""
Extended feature builder for the fusion (Idea E).

Cosine features from every embedding dict + UTMOS naturalness scalars.
Reused by fusion_eval.py and make_submission.py.
"""

import os
import numpy as np
import pandas as pd
import torch

BASE     = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.dirname(BASE)
TRAIN_AVG = os.path.join(EXP_ROOT, "titanet_large", "train_avg.csv")
DATASET_DIR = "vmc2026_track3_train_phase_distro_v3_syn"

# cosine-feature sources (embedding dicts)
EMBEDDING_DICTS = {
    "ecapa":            os.path.join(EXP_ROOT, "ecapa/embeddings.pt"),
    "titanet":          os.path.join(EXP_ROOT, "titanet_large/infer/embeddings/embeddings/manifest_embeddings.pt"),
    "wespeaker":        os.path.join(EXP_ROOT, "pyannote/embeddings.pt"),
    "wav2vec2_hidden":  os.path.join(EXP_ROOT, "wav2vec2/accent_hidden_embeddings.pt"),
    "wav2vec2_prob":    os.path.join(EXP_ROOT, "wav2vec2/accent_prob_embeddings.pt"),
    "commonaccent_emb": os.path.join(BASE, "commonaccent_emb.pt"),
    "commonaccent_prob":os.path.join(BASE, "commonaccent_prob.pt"),
}
# per-wav scalar sources (naturalness)
SCALAR_DICTS = {
    "utmos": os.path.join(BASE, "utmos_scores.pt"),
}


def emb_key(rel_path):
    parts = rel_path.strip("/").split("/")
    return f"{DATASET_DIR}@{parts[-2]}@{parts[-1]}"


def _load(path):
    d = torch.load(path, weights_only=False)
    return {k: (v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
            if not np.isscalar(v) else float(v) for k, v in d.items()}


def _cos(embs, a, b):
    ea, eb = embs[emb_key(a)], embs[emb_key(b)]
    return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-8))


def build(df):
    """Add cosine + scalar features to df in place. Returns (df, feature_names)."""
    feats = []
    for name, path in EMBEDDING_DICTS.items():
        if not os.path.exists(path):
            print(f"[skip] {name}: not extracted"); continue
        embs = _load(path)
        df[f"cos_{name}"] = [_cos(embs, a, b) for a, b in zip(df.wav_a_path, df.wav_b_path)]
        feats.append(f"cos_{name}")
    for name, path in SCALAR_DICTS.items():
        if not os.path.exists(path):
            print(f"[skip] {name}: not extracted"); continue
        sc = _load(path)
        df[f"{name}_a"] = [sc[emb_key(a)] for a in df.wav_a_path]
        df[f"{name}_b"] = [sc[emb_key(b)] for b in df.wav_b_path]
        df[f"{name}_diff"] = df[f"{name}_a"] - df[f"{name}_b"]
        feats += [f"{name}_a", f"{name}_b", f"{name}_diff"]
    return df, feats


def train_table():
    return build(pd.read_csv(TRAIN_AVG))
