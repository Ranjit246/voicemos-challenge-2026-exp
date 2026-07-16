"""
Extract UTMOS naturalness score (per wav) — tarepan/SpeechMOS utmos22_strong.

UTMOS is the VMC2022 winner; the paper shows it correlates with S-SPK-SIM (0.31)
and S-ACC-SIM (0.15) at utterance level — an independent quality signal that
generalizes to unseen systems (unlike a system-label prior).

Output: utmos_scores.pt  — dict: key -> float (predicted MOS)

Usage:
    python extract_utmos.py
"""

import csv
import os
import numpy as np
import torch
import librosa
from tqdm import tqdm

WAV_ROOT         = "/Users/ranjitpatro/Home/Research/VoiceMOS/dataset/vmc2026_track3_train_phase_distro_v3_syn"
TRAIN_CSV        = f"{WAV_ROOT}/sets/train.csv"
DEV_CSV          = f"{WAV_ROOT}/sets/dev.csv"
DATASET_DIR_NAME = "vmc2026_track3_train_phase_distro_v3_syn"
BASE             = os.path.dirname(os.path.abspath(__file__))
OUT              = os.path.join(BASE, "utmos_scores.pt")


def emb_key(rel_path):
    parts = rel_path.strip("/").split("/")
    return f"{DATASET_DIR_NAME}@{parts[-2]}@{parts[-1]}"


def collect_wav_paths():
    paths = set()
    for csv_path in [TRAIN_CSV, DEV_CSV]:
        for row in csv.DictReader(open(csv_path)):
            paths.add(row["wav_a_path"]); paths.add(row["wav_b_path"])
    return sorted(paths)


print("Loading UTMOS (tarepan/SpeechMOS:v1.2.0 utmos22_strong)...")
model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
model.eval()

rel_paths = collect_wav_paths()
print(f"Scoring {len(rel_paths)} wav files...")

scores = {}
with torch.no_grad():
    for rel in tqdm(rel_paths):
        audio, _ = librosa.load(os.path.join(WAV_ROOT, rel), sr=16000, mono=True)
        wav = torch.from_numpy(audio).unsqueeze(0).float()
        scores[emb_key(rel)] = float(model(wav, 16000))

torch.save(scores, OUT)
vals = np.array(list(scores.values()))
print(f"\nDone. {len(scores)} scores  range [{vals.min():.2f}, {vals.max():.2f}]  mean {vals.mean():.2f}")
print(f"Saved -> {OUT}")
