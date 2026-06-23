"""
Cosine similarity submission using pyannote WeSpeakerResNet34 embeddings.

pred_spk_sim = cosine(emb_a, emb_b)   [pyannote 256-dim, primary target]
pred_acc_sim = cosine(emb_a, emb_b)   [same score reused as proxy]

Both scaled linearly to [1, 5] — SRCC is rank-based so scaling doesn't affect score.

Requires: embeddings.pt (run infer_embed.py first)

Usage:
    python infer_submission.py
"""

import csv
import os
import numpy as np
import torch

BASE             = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PT    = os.path.join(BASE, "embeddings.pt")
DEV_CSV          = "/Users/ranjitpatro/Home/Research/VoiceMOS/dataset/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv"
OUT              = os.path.join(BASE, "answer.txt")
DATASET_DIR_NAME = "vmc2026_track3_train_phase_distro_v3_syn"


def emb_key(rel_path):
    parts = rel_path.strip("/").split("/")
    return f"{DATASET_DIR_NAME}@{parts[-2]}@{parts[-1]}"


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def scale_to_1_5(scores):
    scores = np.array(scores)
    mn, mx = scores.min(), scores.max()
    return 1.0 + 4.0 * (scores - mn) / (mx - mn + 1e-8)


print("Loading pyannote embeddings...")
embeddings = torch.load(EMBEDDINGS_PT, weights_only=False)
embeddings = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in embeddings.items()}
print(f"  {len(embeddings)} entries  |  dim={next(iter(embeddings.values())).shape}")

rows = list(csv.DictReader(open(DEV_CSV)))

cos_scores, valid_rows = [], []
skipped = 0
for row in rows:
    ka = emb_key(row["wav_a_path"])
    kb = emb_key(row["wav_b_path"])
    if ka not in embeddings or kb not in embeddings:
        skipped += 1
        continue
    cos_scores.append(cosine(embeddings[ka], embeddings[kb]))
    valid_rows.append(row)

if skipped:
    print(f"  Warning: skipped {skipped} pairs (missing embeddings)")

scaled = scale_to_1_5(cos_scores)

print(f"\n{len(valid_rows)} predictions")
print(f"  cosine range [{min(cos_scores):.4f}, {max(cos_scores):.4f}]")
print(f"  scaled range [{scaled.min():.4f}, {scaled.max():.4f}]  mean={scaled.mean():.4f}")

with open(OUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "system_id", "utterance_id", "wav_a_path", "wav_b_path",
        "pred_acc_sim", "pred_spk_sim"
    ])
    writer.writeheader()
    for row, score in zip(valid_rows, scaled):
        writer.writerow({
            "system_id":    row["system_id"],
            "utterance_id": row["utterance_id"],
            "wav_a_path":   row["wav_a_path"],
            "wav_b_path":   row["wav_b_path"],
            "pred_acc_sim": float(score),
            "pred_spk_sim": float(score),
        })

print(f"\nSaved → {OUT}")
print(f"To submit:  zip -j submission_pyannote.zip {OUT}")
