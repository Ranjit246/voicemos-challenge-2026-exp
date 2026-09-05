"""
Idea L — phonetic-posteriorgram (PPG) accent features.

Extract per-frame phoneme posteriors from a wav2vec2 phoneme-CTC model
(facebook/wav2vec2-lv-60-espeak-cv-ft, eSpeak IPA vocab). Per utterance we save:
  - mean phone distribution over frames (V-dim)  -> pronunciation/accent fingerprint
Pair distances between wav_a and wav_b of these distributions (cosine, JS, L2)
become accent-similarity features that are largely INDEPENDENT of timbre — the
pronunciation-level signal the organizers' CommonAccent cosine misses.

Runs on the VM (H100). Output: ppg_dist.pt -> dict key -> np.float16 (V,)

Usage (VM):
    python extract_ppg.py --wav_root data --out features/ppg_dist.pt --batch_size 16
"""

import argparse, csv, os
import numpy as np
import torch
import torch.nn.functional as F
import librosa
from tqdm import tqdm
from transformers import AutoFeatureExtractor, Wav2Vec2ForCTC  # feature extractor only (no phoneme tokenizer)

DATASET_DIR = "vmc2026_track3_train_phase_distro_v3_syn"
MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"


def emb_key(rel):
    p = rel.strip("/").split("/"); return f"{DATASET_DIR}@{p[-2]}@{p[-1]}"


def collect(wav_root):
    s = set()
    for n in ["train.csv", "dev.csv"]:
        for r in csv.DictReader(open(os.path.join(wav_root, "sets", n))):
            s.add(r["wav_a_path"]); s.add(r["wav_b_path"])
    return sorted(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} on {device}...")
    proc = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(device).eval().half()
    V = model.config.vocab_size
    print(f"  vocab (phonemes) = {V}")

    rels = collect(args.wav_root)
    print(f"Extracting PPG for {len(rels)} wavs, batch={args.batch_size}")
    out = {}
    with torch.no_grad():
        for i in tqdm(range(0, len(rels), args.batch_size)):
            batch = rels[i:i+args.batch_size]
            audios = [librosa.load(os.path.join(args.wav_root, r), sr=16000, mono=True)[0] for r in batch]
            inp = proc(audios, sampling_rate=16000, return_tensors="pt", padding=True)
            iv = inp.input_values.to(device).half()
            am = inp.get("attention_mask", None)
            logits = model(iv).logits                      # (B, T, V)
            probs = F.softmax(logits.float(), dim=-1)       # (B, T, V)
            for b, r in enumerate(batch):
                out[emb_key(r)] = probs[b].mean(0).cpu().numpy().astype(np.float16)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(out, args.out)
    s = next(iter(out.values()))
    print(f"\nDone. {len(out)} wavs, dist dim {s.shape} -> {args.out}")


if __name__ == "__main__":
    main()
