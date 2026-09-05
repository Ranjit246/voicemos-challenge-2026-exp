"""Extract WeSpeaker (pyannote) embeddings for eval wavs on the Mac. Reads HF token from env HF_TOKEN."""
import csv, os, numpy as np, torch
from tqdm import tqdm
from pyannote.audio import Model, Inference

DATASET_DIR = "vmc2026_track3_train_phase_distro_v3_syn"
WAV_ROOT = "/Users/ranjitpatro/Home/Research/VoiceMOS/eval_set/vmc2026_track3_eval_phase_distro_v3_syn"
CSV = os.path.join(WAV_ROOT, "sets/test.csv")
OUT = "/Users/ranjitpatro/Home/Research/VoiceMOS/voicemos-challenge-2026-exp/final_system/evalfeat/wespeaker.pt"


def emb_key(rel):
    p = rel.strip("/").split("/"); return f"{DATASET_DIR}@{p[-2]}@{p[-1]}"


rels = set()
for r in csv.DictReader(open(CSV)):
    rels.add(r["wav_a_path"]); rels.add(r["wav_b_path"])
rels = sorted(x for x in rels if os.path.exists(os.path.join(WAV_ROOT, x)))
print(f"{len(rels)} eval wavs")

tok = os.environ.get("HF_TOKEN")
m = Model.from_pretrained("pyannote/speaker-diarization-community-1", subfolder="embedding", use_auth_token=tok)
inf = Inference(m, window="whole")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
d = {}
for r in tqdm(rels):
    d[emb_key(r)] = np.asarray(inf(os.path.join(WAV_ROOT, r)))
torch.save(d, OUT)
print(f"saved {len(d)} -> {OUT}")
