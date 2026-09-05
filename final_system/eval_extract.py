"""
Unified eval feature extractor (L4 GPU) — 6 models (TitaNet done separately on Mac).

Extracts, for every unique wav referenced by test.csv, the feature sets Idea K needs,
keyed identically to the train features (constant DATASET_DIR prefix):
  ecapa.pt(192) · wespeaker.pt(256) · commonaccent_emb.pt(192)/prob.pt(16)
  · wav2vec2_hidden.pt(768)/prob.pt(13) · wavlm_layers.pt(25,1024) · utmos.pt(scalar)

Usage:
    python eval_extract.py --wav_root ~/vmc_eval/evalwav --csv ~/vmc_eval/data/test.csv --out ~/vmc_eval/evalfeat
"""

import argparse, csv, os
import numpy as np
import torch
import torch.nn.functional as F
import librosa
from tqdm import tqdm

DATASET_DIR = "vmc2026_track3_train_phase_distro_v3_syn"


def emb_key(rel):
    p = rel.strip("/").split("/"); return f"{DATASET_DIR}@{p[-2]}@{p[-1]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_root", required=True); ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    root = os.path.expanduser(a.wav_root)

    rels = set()
    for r in csv.DictReader(open(os.path.expanduser(a.csv))):
        rels.add(r["wav_a_path"]); rels.add(r["wav_b_path"])
    rels = sorted(x for x in rels if os.path.exists(os.path.join(root, x)))
    print(f"{len(rels)} eval wavs on {dev}")
    def wav(r): return librosa.load(os.path.join(root, r), sr=16000, mono=True)[0]
    def save(d, n): torch.save(d, os.path.join(a.out, n)); print(f"  saved {n} ({len(d)})")

    # ── ECAPA ──
    from speechbrain.inference.speaker import EncoderClassifier
    m = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb",
                                       savedir=os.path.join(a.out, "_ecapa"), run_opts={"device": dev})
    d = {}
    with torch.no_grad():
        for r in tqdm(rels, desc="ecapa"):
            sig = torch.from_numpy(wav(r)).unsqueeze(0).float().to(dev)
            d[emb_key(r)] = m.encode_batch(sig).squeeze().cpu().numpy()
    save(d, "ecapa.pt"); del m; torch.cuda.empty_cache()

    # ── CommonAccent (emb + posterior) ──
    from speechbrain.inference.classifiers import EncoderClassifier as EC
    m = EC.from_hparams("Jzuluaga/accent-id-commonaccent_ecapa",
                        savedir=os.path.join(a.out, "_ca"), run_opts={"device": dev})
    emb, prob = {}, {}
    with torch.no_grad():
        for r in tqdm(rels, desc="commonaccent"):
            sig = torch.from_numpy(wav(r)).unsqueeze(0).float().to(dev)
            emb[emb_key(r)] = m.encode_batch(sig).squeeze().cpu().numpy()
            prob[emb_key(r)] = m.classify_batch(sig)[0].squeeze().exp().cpu().numpy()
    save(emb, "commonaccent_emb.pt"); save(prob, "commonaccent_prob.pt"); del m; torch.cuda.empty_cache()

    # ── WeSpeaker (pyannote) ──
    from pyannote.audio import Model, Inference
    wm = Model.from_pretrained("pyannote/speaker-diarization-community-1", subfolder="embedding")
    inf = Inference(wm, window="whole", device=torch.device(dev))
    d = {}
    for r in tqdm(rels, desc="wespeaker"):
        d[emb_key(r)] = np.asarray(inf(os.path.join(root, r)))
    save(d, "wespeaker.pt"); del wm, inf; torch.cuda.empty_cache()

    # ── wav2vec2 accent (hidden + posterior) ──
    from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
    fe = AutoFeatureExtractor.from_pretrained("HamzaSidhu786/speech-accent-detection")
    m = Wav2Vec2ForSequenceClassification.from_pretrained(
        "HamzaSidhu786/speech-accent-detection", output_hidden_states=True).to(dev).eval()
    hid, prob = {}, {}
    with torch.no_grad():
        for i in tqdm(range(0, len(rels), a.bs), desc="wav2vec2acc"):
            b = rels[i:i+a.bs]; au = [wav(r) for r in b]
            inp = fe(au, sampling_rate=16000, return_tensors="pt", padding=True).to(dev)
            o = m(**inp); h = o.hidden_states[-1].mean(1); p = F.softmax(o.logits, -1)
            for j, r in enumerate(b):
                hid[emb_key(r)] = h[j].cpu().numpy(); prob[emb_key(r)] = p[j].cpu().numpy()
    save(hid, "wav2vec2_hidden.pt"); save(prob, "wav2vec2_prob.pt"); del m; torch.cuda.empty_cache()

    # ── WavLM-Large (25 layers mean-pooled) ──
    from transformers import WavLMModel
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
    m = WavLMModel.from_pretrained("microsoft/wavlm-large", output_hidden_states=True).to(dev).eval().half()
    d = {}
    with torch.no_grad():
        for i in tqdm(range(0, len(rels), a.bs), desc="wavlm"):
            b = rels[i:i+a.bs]; au = [wav(r) for r in b]
            inp = fe(au, sampling_rate=16000, return_tensors="pt", padding=True)
            iv = inp.input_values.to(dev).half()
            hs = m(input_values=iv).hidden_states           # tuple(25) (B,T,1024)
            layers = torch.stack(hs, 1)                     # (B,25,T,1024)
            for j, r in enumerate(b):
                d[emb_key(r)] = layers[j].mean(1).float().cpu().numpy().astype(np.float16)
    save(d, "wavlm_layers.pt"); del m; torch.cuda.empty_cache()

    # ── UTMOS ──
    um = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(dev).eval()
    d = {}
    with torch.no_grad():
        for r in tqdm(rels, desc="utmos"):
            w = torch.from_numpy(wav(r)).unsqueeze(0).float().to(dev)
            d[emb_key(r)] = float(um(w, 16000))
    save(d, "utmos.pt")
    print("ALL EVAL FEATURES DONE")


if __name__ == "__main__":
    main()
