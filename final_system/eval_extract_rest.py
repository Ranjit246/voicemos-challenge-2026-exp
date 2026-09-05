"""Remaining L4 eval features: wav2vec2-accent, WavLM-Large, UTMOS (ecapa/commonaccent already done, wespeaker on Mac)."""
import argparse, csv, os
import numpy as np, torch, torch.nn.functional as F, librosa
from tqdm import tqdm
DATASET_DIR = "vmc2026_track3_train_phase_distro_v3_syn"
def emb_key(rel): p=rel.strip("/").split("/"); return f"{DATASET_DIR}@{p[-2]}@{p[-1]}"

ap=argparse.ArgumentParser(); ap.add_argument("--wav_root",required=True); ap.add_argument("--csv",required=True)
ap.add_argument("--out",required=True); ap.add_argument("--bs",type=int,default=16); a=ap.parse_args()
dev="cuda" if torch.cuda.is_available() else "cpu"; root=os.path.expanduser(a.wav_root)
rels=set()
for r in csv.DictReader(open(os.path.expanduser(a.csv))): rels.add(r["wav_a_path"]); rels.add(r["wav_b_path"])
rels=sorted(x for x in rels if os.path.exists(os.path.join(root,x)))
print(f"{len(rels)} wavs on {dev}")
def wav(r): return librosa.load(os.path.join(root,r),sr=16000,mono=True)[0]
def save(d,n): torch.save(d,os.path.join(a.out,n)); print(f"saved {n} ({len(d)})")
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification, WavLMModel

# wav2vec2 accent
fe=AutoFeatureExtractor.from_pretrained("HamzaSidhu786/speech-accent-detection")
m=Wav2Vec2ForSequenceClassification.from_pretrained("HamzaSidhu786/speech-accent-detection",output_hidden_states=True).to(dev).eval()
hid,prob={},{}
with torch.no_grad():
    for i in tqdm(range(0,len(rels),a.bs),desc="wav2vec2acc"):
        b=rels[i:i+a.bs]; inp=fe([wav(r) for r in b],sampling_rate=16000,return_tensors="pt",padding=True).to(dev)
        o=m(**inp); h=o.hidden_states[-1].mean(1); p=F.softmax(o.logits,-1)
        for j,r in enumerate(b): hid[emb_key(r)]=h[j].cpu().numpy(); prob[emb_key(r)]=p[j].cpu().numpy()
save(hid,"wav2vec2_hidden.pt"); save(prob,"wav2vec2_prob.pt"); del m; torch.cuda.empty_cache()

# WavLM-Large
fe=AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
m=WavLMModel.from_pretrained("microsoft/wavlm-large",output_hidden_states=True).to(dev).eval().half()
d={}
with torch.no_grad():
    for i in tqdm(range(0,len(rels),a.bs),desc="wavlm"):
        b=rels[i:i+a.bs]; inp=fe([wav(r) for r in b],sampling_rate=16000,return_tensors="pt",padding=True)
        hs=m(input_values=inp.input_values.to(dev).half()).hidden_states; layers=torch.stack(hs,1)
        for j,r in enumerate(b): d[emb_key(r)]=layers[j].mean(1).float().cpu().numpy().astype(np.float16)
save(d,"wavlm_layers.pt"); del m; torch.cuda.empty_cache()

# UTMOS
um=torch.hub.load("tarepan/SpeechMOS:v1.2.0","utmos22_strong",trust_repo=True).to(dev).eval()
d={}
with torch.no_grad():
    for r in tqdm(rels,desc="utmos"):
        d[emb_key(r)]=float(um(torch.from_numpy(wav(r)).unsqueeze(0).float().to(dev),16000))
save(d,"utmos.pt"); print("REST DONE")
