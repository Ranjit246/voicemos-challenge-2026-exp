"""
Idea A — build answer.txt with system-mean shrinkage applied.

Best configs from tune_alpha.py (full-train sweep, alpha_sweep_results.csv):
  pred_spk_sim: TitaNet cosine,        alpha = 0.4   (train SRCC 0.626 vs 0.607 raw)
  pred_acc_sim: wav2vec2-hidden cosine, alpha = 0.2  (train SRCC 0.561 vs 0.425 raw)

Shrinkage on dev uses per-system mean of the *predicted* scores computed on
dev itself — label-free, transductive.

Usage:
    python make_submission.py
"""

import csv
import os
import numpy as np
import pandas as pd
import torch

BASE        = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT    = os.path.dirname(BASE)
DEV_CSV     = "/Users/ranjitpatro/Home/Research/VoiceMOS/dataset/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv"
OUT         = os.path.join(BASE, "answer.txt")
DATASET_DIR = "vmc2026_track3_train_phase_distro_v3_syn"

SPK_EMB_PATH = os.path.join(EXP_ROOT, "titanet_large/infer/embeddings/embeddings/manifest_embeddings.pt")
ACC_EMB_PATH = os.path.join(EXP_ROOT, "wav2vec2/accent_hidden_embeddings.pt")
SPK_ALPHA    = 0.4
ACC_ALPHA    = 0.2


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
    s = pd.Series(scores)
    sys_mean = s.groupby(list(systems)).transform("mean")
    return (alpha * s + (1 - alpha) * sys_mean).values


def scale_to_1_5(scores):
    scores = np.asarray(scores)
    mn, mx = scores.min(), scores.max()
    return 1.0 + 4.0 * (scores - mn) / (mx - mn + 1e-8)


df = pd.read_csv(DEV_CSV)
print(f"Dev pairs: {len(df)}  |  systems: {df.system_id.nunique()}")

print("Scoring spk (TitaNet cosine)...")
spk_embs = load_embeddings(SPK_EMB_PATH)
spk_raw = [cosine(spk_embs[emb_key(a)], spk_embs[emb_key(b)])
           for a, b in zip(df.wav_a_path, df.wav_b_path)]

print("Scoring acc (wav2vec2-hidden cosine)...")
acc_embs = load_embeddings(ACC_EMB_PATH)
acc_raw = [cosine(acc_embs[emb_key(a)], acc_embs[emb_key(b)])
           for a, b in zip(df.wav_a_path, df.wav_b_path)]

spk_shrunk = shrink(spk_raw, df.system_id, SPK_ALPHA)
acc_shrunk = shrink(acc_raw, df.system_id, ACC_ALPHA)

spk_final = scale_to_1_5(spk_shrunk)
acc_final = scale_to_1_5(acc_shrunk)

print(f"spk: raw range [{min(spk_raw):.4f}, {max(spk_raw):.4f}]  alpha={SPK_ALPHA}  -> [1,5] scaled")
print(f"acc: raw range [{min(acc_raw):.4f}, {max(acc_raw):.4f}]  alpha={ACC_ALPHA}  -> [1,5] scaled")

with open(OUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "system_id", "utterance_id", "wav_a_path", "wav_b_path",
        "pred_acc_sim", "pred_spk_sim"
    ])
    writer.writeheader()
    for i, row in df.iterrows():
        writer.writerow({
            "system_id":    row.system_id,
            "utterance_id": row.utterance_id,
            "wav_a_path":   row.wav_a_path,
            "wav_b_path":   row.wav_b_path,
            "pred_acc_sim": float(acc_final[i]),
            "pred_spk_sim": float(spk_final[i]),
        })

print(f"\nSaved -> {OUT}  ({len(df)} rows)")
print(f"To submit:  zip -j submission_shrinkage.zip {OUT}")
