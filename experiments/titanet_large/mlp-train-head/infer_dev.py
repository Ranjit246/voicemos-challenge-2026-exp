"""
Run inference on dev.csv and write answer.txt for CodaBench submission.

Usage:
    python infer_dev.py
    python infer_dev.py --out /path/to/answer.txt

Output format:
    system_id,utterance_id,wav_a_path,wav_b_path,pred_acc_sim,pred_spk_sim
"""

import argparse
import os
import csv
import numpy as np
import torch
import torch.nn as nn


# ── paths ──────────────────────────────────────────────────────────────────────
BASE          = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PT = os.path.join(BASE, "../infer/embeddings/embeddings/manifest_embeddings.pt")
DEV_CSV       = "/Users/ranjitpatro/Home/Research/VoiceMOS/dataset/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv"
CKPT          = os.path.join(BASE, "best_model.pt")
DEFAULT_OUT   = os.path.join(BASE, "answer.txt")

DATASET_DIR_NAME = "vmc2026_track3_train_phase_distro_v3_syn"


# ── helpers (must match train.py exactly) ─────────────────────────────────────
def emb_key(rel_path):
    parts = rel_path.strip("/").split("/")
    return f"{DATASET_DIR_NAME}@{parts[-2]}@{parts[-1]}"


def pair_feature(ea, eb):
    return np.concatenate([ea, eb, ea - eb, ea * eb]).astype(np.float32)


class MLPHead(nn.Module):
    def __init__(self, in_dim=768, hidden=(128,), dropout=0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",     type=str, default=DEFAULT_OUT)
    parser.add_argument("--hidden",  type=int, nargs="+", default=[128])
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()

    print("Loading embeddings...")
    embeddings = torch.load(EMBEDDINGS_PT, weights_only=False)
    embeddings = {k: (v.numpy() if isinstance(v, torch.Tensor) else v)
                  for k, v in embeddings.items()}
    print(f"  {len(embeddings)} embeddings loaded")

    print("Loading model...")
    model = MLPHead(in_dim=768, hidden=tuple(args.hidden), dropout=args.dropout)
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    model.eval()

    print("Running inference on dev.csv...")
    rows = list(csv.DictReader(open(DEV_CSV)))

    results = []
    skipped = 0
    with torch.no_grad():
        for row in rows:
            ka = emb_key(row["wav_a_path"])
            kb = emb_key(row["wav_b_path"])
            if ka not in embeddings or kb not in embeddings:
                skipped += 1
                continue
            feat = torch.tensor(pair_feature(embeddings[ka], embeddings[kb])).unsqueeze(0)
            pred = model(feat).squeeze().numpy()
            pred = np.clip(pred, 1.0, 5.0)
            results.append({
                "system_id":    row["system_id"],
                "utterance_id": row["utterance_id"],
                "wav_a_path":   row["wav_a_path"],
                "wav_b_path":   row["wav_b_path"],
                "pred_acc_sim": float(pred[1]),
                "pred_spk_sim": float(pred[0]),
            })

    if skipped:
        print(f"  Warning: skipped {skipped} pairs (missing embeddings)")

    print(f"  {len(results)} predictions generated")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "system_id", "utterance_id", "wav_a_path", "wav_b_path",
            "pred_acc_sim", "pred_spk_sim"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved → {args.out}")
    print(f"\nTo submit:")
    print(f"  zip -j submission.zip {args.out}")


if __name__ == "__main__":
    main()
