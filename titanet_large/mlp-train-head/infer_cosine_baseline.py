"""
Cosine similarity baseline — no training.
Uses cosine(emb_a, emb_b) directly as both pred_spk_sim and pred_acc_sim.
SRCC is rank-based so no scaling needed, but we map to [1,5] to be safe.
"""

import csv
import os
import numpy as np
import torch

BASE          = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PT = os.path.join(BASE, "../infer/embeddings/embeddings/manifest_embeddings.pt")
DEV_CSV       = "/Users/ranjitpatro/Home/Research/VoiceMOS/dataset/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv"
OUT           = os.path.join(BASE, "answer_cosine.txt")

DATASET_DIR_NAME = "vmc2026_track3_train_phase_distro_v3_syn"


def emb_key(rel_path):
    parts = rel_path.strip("/").split("/")
    return f"{DATASET_DIR_NAME}@{parts[-2]}@{parts[-1]}"


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


print("Loading embeddings...")
embeddings = torch.load(EMBEDDINGS_PT, weights_only=False)
embeddings = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in embeddings.items()}
print(f"  {len(embeddings)} embeddings loaded")

rows = list(csv.DictReader(open(DEV_CSV)))

# collect cosine scores first to fit linear scale to [1, 5]
cos_scores = []
for row in rows:
    ka, kb = emb_key(row["wav_a_path"]), emb_key(row["wav_b_path"])
    cos_scores.append(cosine(embeddings[ka], embeddings[kb]))

cos = np.array(cos_scores)
# linear map: [min, max] → [1, 5]
scaled = 1.0 + 4.0 * (cos - cos.min()) / (cos.max() - cos.min() + 1e-8)

print(f"  cosine range: [{cos.min():.4f}, {cos.max():.4f}]")
print(f"  scaled range: [{scaled.min():.4f}, {scaled.max():.4f}]  mean={scaled.mean():.4f}")

with open(OUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "system_id", "utterance_id", "wav_a_path", "wav_b_path",
        "pred_acc_sim", "pred_spk_sim"
    ])
    writer.writeheader()
    for row, score in zip(rows, scaled):
        writer.writerow({
            "system_id":    row["system_id"],
            "utterance_id": row["utterance_id"],
            "wav_a_path":   row["wav_a_path"],
            "wav_b_path":   row["wav_b_path"],
            "pred_acc_sim": float(score),
            "pred_spk_sim": float(score),
        })

print(f"Saved → {OUT}")
print(f"\nTo submit:")
print(f"  zip -j submission_cosine.zip {OUT}")
