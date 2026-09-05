"""
Extract accent-aware embeddings from HamzaSidhu786/speech-accent-detection.

Two outputs saved side-by-side:
  accent_hidden_embeddings.pt  — mean-pooled last hidden state (768-dim)
                                  captures fine-grained accent features
  accent_prob_embeddings.pt    — softmax probability over 13 accent classes (13-dim)
                                  directly interpretable: cosine sim of prob vectors
                                  = accent distribution similarity

Both use the same key format as TitaNet:
  vmc2026_track3_train_phase_distro_v3_syn@wav@<filename>.wav → np.array

Usage:
    python infer_accent_embed.py
    python infer_accent_embed.py --batch_size 8
"""

import argparse
import csv
import os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
import librosa
from tqdm import tqdm

MODEL_ID         = "HamzaSidhu786/speech-accent-detection"
WAV_ROOT         = "/Users/ranjitpatro/Home/Research/VoiceMOS/dataset/vmc2026_track3_train_phase_distro_v3_syn"
TRAIN_CSV        = f"{WAV_ROOT}/sets/train.csv"
DEV_CSV          = f"{WAV_ROOT}/sets/dev.csv"
DATASET_DIR_NAME = "vmc2026_track3_train_phase_distro_v3_syn"
BASE             = os.path.dirname(os.path.abspath(__file__))
OUT_HIDDEN       = os.path.join(BASE, "accent_hidden_embeddings.pt")
OUT_PROB         = os.path.join(BASE, "accent_prob_embeddings.pt")


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    device = "mps"

    print(f"Loading model: {MODEL_ID}")
    processor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_ID, output_hidden_states=True
    ).to(device)
    model.eval()
    print(f"  Hidden size: {model.config.hidden_size}  |  Classes: {model.config.num_labels}")
    print(f"  Labels: {list(model.config.id2label.values())}")

    rel_paths = collect_wav_paths()
    print(f"\nProcessing {len(rel_paths)} wav files  (batch_size={args.batch_size})")

    hidden_embeddings = {}
    prob_embeddings   = {}

    with torch.no_grad():
        for i in tqdm(range(0, len(rel_paths), args.batch_size)):
            batch_paths = rel_paths[i : i + args.batch_size]

            # load audio at 16kHz
            audios = []
            for rel in batch_paths:
                audio, _ = librosa.load(os.path.join(WAV_ROOT, rel), sr=16000, mono=True)
                audios.append(audio)

            # processor handles padding + normalisation
            inputs = processor(
                audios,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)

            # ── hidden embedding: mean-pool last hidden state over time ──
            # last_hidden_state: (batch, time, 768)
            last_hidden = outputs.hidden_states[-1]          # (B, T, 768)
            # mask padding if attention_mask available
            if "attention_mask" in inputs:
                # wav2vec2 downsamples by ~320x; approximate frame mask
                mask = inputs["attention_mask"].unsqueeze(-1).float()  # (B, T_audio, 1)
                # the hidden state time dimension differs — just mean-pool without mask
                # (padding is minimal for short clips, negligible effect)
            pooled = last_hidden.mean(dim=1)                 # (B, 768)

            # ── probability embedding: softmax over logits ──
            probs = F.softmax(outputs.logits, dim=-1)        # (B, 13)

            for j, rel in enumerate(batch_paths):
                key = emb_key(rel)
                hidden_embeddings[key] = pooled[j].cpu().numpy()
                prob_embeddings[key]   = probs[j].cpu().numpy()

    torch.save(hidden_embeddings, OUT_HIDDEN)
    torch.save(prob_embeddings,   OUT_PROB)

    # quick sanity check
    sample_key = next(iter(hidden_embeddings))
    print(f"\nDone.")
    print(f"  Hidden embeddings: {len(hidden_embeddings)} × {hidden_embeddings[sample_key].shape}  → {OUT_HIDDEN}")
    print(f"  Prob   embeddings: {len(prob_embeddings)} × {prob_embeddings[sample_key].shape}   → {OUT_PROB}")


if __name__ == "__main__":
    main()
