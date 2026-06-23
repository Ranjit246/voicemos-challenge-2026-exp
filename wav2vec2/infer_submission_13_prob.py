"""
Submission using wav2vec2 accent prob embeddings for acc_sim only.

pred_acc_sim = cosine(prob_vec_a, prob_vec_b)   [13-dim softmax over accent classes]

Note: CodaBench may require both columns — if scores return empty, add pred_spk_sim.

Requires: accent_prob_embeddings.pt  (run infer_accent_embed.py first)

Usage:
    python infer_submission.py
"""

import csv
import os
import numpy as np
import torch

BASE        = os.path.dirname(os.path.abspath(__file__))
PROB_PT     = os.path.join(BASE, "accent_prob_embeddings.pt")
DEV_CSV     = "/Users/ranjitpatro/Home/Research/VoiceMOS/dataset/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv"
OUT         = os.path.join(BASE, "answer.txt")
DATASET_DIR = "vmc2026_track3_train_phase_distro_v3_syn"


def emb_key(rel_path):
    parts = rel_path.strip("/").split("/")
    return f"{DATASET_DIR}@{parts[-2]}@{parts[-1]}"


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def scale_to_1_5(scores):
    scores = np.array(scores)
    mn, mx = scores.min(), scores.max()
    return 1.0 + 4.0 * (scores - mn) / (mx - mn + 1e-8)


print("Loading wav2vec2 prob embeddings (13-dim)...")
prob_embs = torch.load(PROB_PT, weights_only=False)
prob_embs = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in prob_embs.items()}
print(f"  {len(prob_embs)} entries  dim={next(iter(prob_embs.values())).shape}")

rows = list(csv.DictReader(open(DEV_CSV)))

acc_scores, valid_rows = [], []
skipped = 0
for row in rows:
    ka = emb_key(row["wav_a_path"])
    kb = emb_key(row["wav_b_path"])
    if ka not in prob_embs or kb not in prob_embs:
        skipped += 1
        continue
    acc_scores.append(cosine(prob_embs[ka], prob_embs[kb]))
    valid_rows.append(row)

if skipped:
    print(f"  Warning: skipped {skipped} pairs (missing embeddings)")

acc_scaled = scale_to_1_5(acc_scores)

print(f"\n{len(valid_rows)} predictions")
print(f"  acc_sim  cosine [{min(acc_scores):.4f}, {max(acc_scores):.4f}]  → scaled [{acc_scaled.min():.2f}, {acc_scaled.max():.2f}]  mean={acc_scaled.mean():.4f}")

with open(OUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "system_id", "utterance_id", "wav_a_path", "wav_b_path", "pred_acc_sim"
    ])
    writer.writeheader()
    for row, acc in zip(valid_rows, acc_scaled):
        writer.writerow({
            "system_id":    row["system_id"],
            "utterance_id": row["utterance_id"],
            "wav_a_path":   row["wav_a_path"],
            "wav_b_path":   row["wav_b_path"],
            "pred_acc_sim": float(acc),
        })

print(f"\nSaved → {OUT}")
print(f"To submit:  zip -j submission_wav2vec2_acc.zip {OUT}")
