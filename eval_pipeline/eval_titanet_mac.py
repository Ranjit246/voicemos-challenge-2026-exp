"""Extract TitaNet embeddings for eval wavs on the Mac (NeMo already installed here)."""
import csv, os, numpy as np, torch
from tqdm import tqdm
import nemo.collections.asr as nemo_asr

DATASET_DIR = "vmc2026_track3_train_phase_distro_v3_syn"
WAV_ROOT = "/Users/ranjitpatro/Home/Research/VoiceMOS/eval_set/vmc2026_track3_eval_phase_distro_v3_syn"
CSV = os.path.join(WAV_ROOT, "sets/test.csv")
OUT = "/Users/ranjitpatro/Home/Research/VoiceMOS/voicemos-challenge-2026-exp/eval_pipeline/evalfeat/titanet.pt"


def emb_key(rel):
    p = rel.strip("/").split("/"); return f"{DATASET_DIR}@{p[-2]}@{p[-1]}"


rels = set()
for r in csv.DictReader(open(CSV)):
    rels.add(r["wav_a_path"]); rels.add(r["wav_b_path"])
rels = sorted(x for x in rels if os.path.exists(os.path.join(WAV_ROOT, x)))
print(f"{len(rels)} eval wavs")

m = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("titanet_large").cpu().eval()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
d = {}
with torch.no_grad():
    for r in tqdm(rels):
        emb = m.get_embedding(os.path.join(WAV_ROOT, r))
        d[emb_key(r)] = emb.squeeze().cpu().numpy()
torch.save(d, OUT)
print(f"saved {len(d)} -> {OUT}")
