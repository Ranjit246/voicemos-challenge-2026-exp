"""
Extract speaker embeddings using WeSpeakerResNet34 from the pyannote
speaker-diarization-community-1 pipeline's embedding sub-model.

Model:  pyannote/speaker-diarization-community-1  (subfolder: embedding)
Type:   WeSpeakerResNet34
Dim:    256
Token:  not required (community model)

Output:
  embeddings.pt  — dict: key → np.array (256,)
  Key format (matches TitaNet):
    vmc2026_track3_train_phase_distro_v3_syn@wav@<filename>.wav

Usage:
    python infer_embed.py
"""

import csv
import os
import numpy as np
import torch
from pyannote.audio import Model, Inference
from tqdm import tqdm

MODEL_ID         = "pyannote/speaker-diarization-community-1"
WAV_ROOT         = "/Users/ranjitpatro/Home/Research/VoiceMOS/dataset/vmc2026_track3_train_phase_distro_v3_syn"
TRAIN_CSV        = f"{WAV_ROOT}/sets/train.csv"
DEV_CSV          = f"{WAV_ROOT}/sets/dev.csv"
DATASET_DIR_NAME = "vmc2026_track3_train_phase_distro_v3_syn"
BASE             = os.path.dirname(os.path.abspath(__file__))
OUT              = os.path.join(BASE, "embeddings.pt")


def emb_key(rel_path):
    parts = rel_path.strip("/").split("/")
    return f"{DATASET_DIR_NAME}@{parts[-2]}@{parts[-1]}"


def collect_wav_paths():
    paths = set()
    for csv_path in [TRAIN_CSV, DEV_CSV]:
        for row in csv.DictReader(open(csv_path)):
            paths.add(row["wav_a_path"])
            paths.add(row["wav_b_path"])
    return sorted(paths)


print(f"Loading model from {MODEL_ID} (subfolder: embedding)...")
model = Model.from_pretrained(MODEL_ID, subfolder="embedding")
inference = Inference(model, window="whole")
print(f"  Model: {type(model).__name__}  |  Embedding dim: 256")

rel_paths = collect_wav_paths()
print(f"\nExtracting embeddings for {len(rel_paths)} wav files...")

embeddings = {}
for rel in tqdm(rel_paths):
    abs_path = os.path.join(WAV_ROOT, rel)
    emb = inference(abs_path)          # returns np.array (256,)
    embeddings[emb_key(rel)] = emb

torch.save(embeddings, OUT)

sample = next(iter(embeddings.values()))
print(f"\nDone.")
print(f"  {len(embeddings)} embeddings  |  dim={sample.shape}  |  dtype={sample.dtype}")
print(f"  Saved → {OUT}")
