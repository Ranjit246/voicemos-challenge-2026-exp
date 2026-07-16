"""
Extract CommonAccent-ECAPA features (Jzuluaga/accent-id-commonaccent_ecapa).

This is the paper's O-ACC-SIM backbone. Outputs both:
  commonaccent_emb.pt   -> dict key -> np.array (192,)   accent embedding (cosine = O-ACC-SIM)
  commonaccent_prob.pt  -> dict key -> np.array (16,)    accent-class posterior

Usage:
    python extract_commonaccent.py
"""

import csv
import os
import numpy as np
import torch
import librosa
from tqdm import tqdm
from speechbrain.inference.classifiers import EncoderClassifier

WAV_ROOT         = "/Users/ranjitpatro/Home/Research/VoiceMOS/dataset/vmc2026_track3_train_phase_distro_v3_syn"
TRAIN_CSV        = f"{WAV_ROOT}/sets/train.csv"
DEV_CSV          = f"{WAV_ROOT}/sets/dev.csv"
DATASET_DIR_NAME = "vmc2026_track3_train_phase_distro_v3_syn"
BASE             = os.path.dirname(os.path.abspath(__file__))
OUT_EMB          = os.path.join(BASE, "commonaccent_emb.pt")
OUT_PROB         = os.path.join(BASE, "commonaccent_prob.pt")


def emb_key(rel_path):
    parts = rel_path.strip("/").split("/")
    return f"{DATASET_DIR_NAME}@{parts[-2]}@{parts[-1]}"


def collect_wav_paths():
    paths = set()
    for csv_path in [TRAIN_CSV, DEV_CSV]:
        for row in csv.DictReader(open(csv_path)):
            paths.add(row["wav_a_path"]); paths.add(row["wav_b_path"])
    return sorted(paths)


print("Loading CommonAccent-ECAPA (Jzuluaga/accent-id-commonaccent_ecapa)...")
clf = EncoderClassifier.from_hparams(
    source="Jzuluaga/accent-id-commonaccent_ecapa",
    savedir=os.path.join(BASE, "pretrained_commonaccent"),
    run_opts={"device": "cpu"},
)

rel_paths = collect_wav_paths()
print(f"Extracting for {len(rel_paths)} wav files...")

emb_d, prob_d = {}, {}
with torch.no_grad():
    for rel in tqdm(rel_paths):
        audio, _ = librosa.load(os.path.join(WAV_ROOT, rel), sr=16000, mono=True)
        sig = torch.from_numpy(audio).unsqueeze(0).float()
        k = emb_key(rel)
        emb_d[k]  = clf.encode_batch(sig).squeeze().cpu().numpy()
        prob_d[k] = clf.classify_batch(sig)[0].squeeze().exp().cpu().numpy()  # logprob -> prob

torch.save(emb_d, OUT_EMB)
torch.save(prob_d, OUT_PROB)
print(f"\nDone. emb dim={next(iter(emb_d.values())).shape}  prob dim={next(iter(prob_d.values())).shape}")
print(f"Saved -> {OUT_EMB}\n         {OUT_PROB}")
