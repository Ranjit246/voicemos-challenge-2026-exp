"""
Extract ECAPA-TDNN speaker embeddings (speechbrain/spkrec-ecapa-voxceleb).

This is the OFFICIAL baseline backbone (paper's O-SPK-SIM; VMC baseline B1/B2).
192-dim. Same key format as every other extractor in this repo.

Output: ecapa/embeddings.pt  — dict: key -> np.array (192,)

Usage:
    python infer_embed.py
"""

import csv
import os
import numpy as np
import torch
import librosa
from tqdm import tqdm
from speechbrain.inference.speaker import EncoderClassifier

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


print("Loading ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)...")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=os.path.join(BASE, "pretrained_ecapa"),
    run_opts={"device": "cpu"},
)

rel_paths = collect_wav_paths()
print(f"Extracting embeddings for {len(rel_paths)} wav files...")

embeddings = {}
with torch.no_grad():
    for rel in tqdm(rel_paths):
        audio, _ = librosa.load(os.path.join(WAV_ROOT, rel), sr=16000, mono=True)
        signal = torch.from_numpy(audio).unsqueeze(0).float()
        emb = classifier.encode_batch(signal).squeeze().cpu().numpy()
        embeddings[emb_key(rel)] = emb

torch.save(embeddings, OUT)
sample = next(iter(embeddings.values()))
print(f"\nDone. {len(embeddings)} embeddings, dim={sample.shape} -> {OUT}")
