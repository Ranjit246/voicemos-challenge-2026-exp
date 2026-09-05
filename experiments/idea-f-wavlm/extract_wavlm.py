"""
Extract WavLM-Large layer-wise features (microsoft/wavlm-large) on GPU.

Runs on the H100 VM. For each wav, mean-pools every transformer layer's hidden
states over time -> (n_layers, 1024). Enables layer probing: speaker info lives
in early layers, accent/phonetic info around the middle (~layer 9).

Output: wavlm_layers.pt  -> dict key -> np.float16 array (n_layers, 1024)
  key format matches the rest of the repo:
    vmc2026_track3_train_phase_distro_v3_syn@wav@<filename>.wav

Usage (on VM):
    python extract_wavlm.py --wav_root ~/voicemos/data --out ~/voicemos/features/wavlm_layers.pt
"""

import argparse
import csv
import os
import numpy as np
import torch
import librosa
from tqdm import tqdm
from transformers import AutoFeatureExtractor, WavLMModel

DATASET_DIR_NAME = "vmc2026_track3_train_phase_distro_v3_syn"
MODEL_ID = "microsoft/wavlm-large"


def emb_key(rel_path):
    parts = rel_path.strip("/").split("/")
    return f"{DATASET_DIR_NAME}@{parts[-2]}@{parts[-1]}"


def collect_wavs(wav_root):
    paths = set()
    for name in ["train.csv", "dev.csv"]:
        p = os.path.join(wav_root, "sets", name)
        for row in csv.DictReader(open(p)):
            paths.add(row["wav_a_path"]); paths.add(row["wav_b_path"])
    return sorted(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_root", required=True, help="dir containing wav/ and sets/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} on {device}...")
    fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = WavLMModel.from_pretrained(MODEL_ID, output_hidden_states=True).to(device).eval().half()

    rel_paths = collect_wavs(args.wav_root)
    print(f"Extracting {len(rel_paths)} wavs, batch_size={args.batch_size}")

    out = {}
    with torch.no_grad():
        for i in tqdm(range(0, len(rel_paths), args.batch_size)):
            batch = rel_paths[i:i + args.batch_size]
            audios = [librosa.load(os.path.join(args.wav_root, r), sr=16000, mono=True)[0]
                      for r in batch]
            inp = fe(audios, sampling_rate=16000, return_tensors="pt", padding=True)
            inp = {k: v.to(device) for k, v in inp.items()}
            if "input_values" in inp:
                inp["input_values"] = inp["input_values"].half()   # match .half() weights
            hs = model(**inp).hidden_states           # tuple(n_layers+1) each (B, T, 1024)
            # build a frame mask from attention_mask downsampled to feature length
            T = hs[0].shape[1]
            if "attention_mask" in inp:
                am = inp["attention_mask"]
                out_len = model._get_feat_extract_output_lengths(am.sum(-1)).long()
            else:
                out_len = torch.full((len(batch),), T, device=device)
            layers = torch.stack(hs, dim=1)            # (B, L, T, 1024)
            for b, r in enumerate(batch):
                n = int(out_len[b].item()); n = max(1, min(n, T))
                pooled = layers[b, :, :n, :].mean(dim=1)   # (L, 1024)
                out[emb_key(r)] = pooled.float().cpu().numpy().astype(np.float16)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(out, args.out)
    sample = next(iter(out.values()))
    print(f"\nDone. {len(out)} wavs, per-wav shape {sample.shape} (n_layers, 1024) -> {args.out}")


if __name__ == "__main__":
    main()
