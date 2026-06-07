# TitaNet-Large Baseline — VoiceMOS 2026 Track 3

Predicting speaker similarity (`spk_sim`) and accent similarity (`acc_sim`) using NVIDIA TitaNet-Large speaker embeddings + MLP regression head.

---

## Task

Given a pair of speech files — `wav_a` (synthesized) and `wav_b` (VCTK natural reference, always `sys019`) — predict both scores on a 1–5 scale. Evaluation metric: utterance-level Spearman Rank Correlation Coefficient (UTT-SRCC).

---

## Directory structure

```
titanet_large/
├── infer/
│   ├── gen_manifest.py              # builds manifest.json from train+dev CSVs
│   ├── manifest.json                # 3548 unique wavs (train + dev, wav_a + wav_b)
│   ├── speaker_embed_train_gen.sh   # runs gen_manifest.py then NeMo extraction
│   └── embeddings/embeddings/
│       └── manifest_embeddings.pt   # dict: filename_key → np.array (192-dim)
├── avg_scores.py                    # averages listener scores → train_avg.csv
├── train_avg.csv                    # 2800 pairs, mean spk_sim + acc_sim
├── mlp-train-head/
│   ├── train.py                     # MLP trainer + cosine baseline
│   ├── train.log                    # training output
│   └── best_model.pt                # best checkpoint by mean SRCC
└── baseline_speaker_embed.log       # raw NeMo extraction log
```

---

## Pipeline

### 1. Extract embeddings

```bash
cd infer
bash speaker_embed_train_gen.sh
```

- Downloads `titanet_large` from NGC (~100MB, cached after first run)
- Runs on CPU (MPS not supported by NeMo's filterbank op)
- Outputs `infer/embeddings/embeddings/manifest_embeddings.pt`
- Embedding key format: `vmc2026_track3_train_phase_distro_v3_syn@wav@<filename>.wav`
- Embedding dim: 192

### 2. Average listener scores

```bash
/path/to/venv/bin/python3 avg_scores.py
```

Aggregates `train.csv` (13,687 listener-wise rows) → `train_avg.csv` (2,800 unique pairs).

| Stat | spk_sim | acc_sim |
|---|---|---|
| Mean | 4.039 | 4.024 |
| Std | 0.817 | 0.773 |
| Range | 1.0 – 5.0 | 1.0 – 5.0 |

### 3. Train MLP head

```bash
cd mlp-train-head
/path/to/venv/bin/python3 train.py
/path/to/venv/bin/python3 train.py --epochs 200 --hidden 256 128  # larger
```

Pair feature: `[emb_a | emb_b | emb_a − emb_b | emb_a × emb_b]` → 768-dim  
Architecture: `768 → 128 → 2` (default), jointly predicts spk_sim and acc_sim

---

## Results (val split, 10%, seed=42)

| Model | SRCC spk_sim | SRCC acc_sim |
|---|---|---|
| Cosine similarity baseline | 0.5991 | 0.4944 |
| MLP 768→128→2 (50 epochs) | 0.6264 | 0.5919 |

- MLP improves `acc_sim` by +0.10 SRCC over cosine baseline — the two output heads learn different weightings of the embedding for each target
- `spk_sim` modest gain (+0.027) — cosine sim already a strong signal for speaker identity
- Training loss still decreasing at epoch 50 — not fully converged

---

## What next

### Short-term (same embedding, better training)
- [ ] Train longer (`--epochs 200`) — loss was still falling at epoch 50
- [ ] Try `--hidden 256 128` and `--hidden 512 256 128` to find the right capacity for 2800 samples
- [ ] Tune dropout and learning rate
- [ ] Use the full dataset for training, evaluate on official `dev.csv` (no labels, submit to CodaBench)

### Better embeddings
- [ ] **WavLM-large** (`microsoft/wavlm-large`) — stronger general-purpose SSL model, likely better on both targets; replace TitaNet 192-dim with WavLM 1024-dim frame-level features (mean-pooled)
- [ ] **ECAPA-TDNN** (SpeechBrain) — already available in `exp-asv/infer_all.py`
- [ ] **Ensemble** cosine scores from multiple speaker models as input features

### Better accent modeling
- [ ] TitaNet captures speaker identity but not accent explicitly — the `acc_sim` gap vs `spk_sim` reflects this
- [ ] Consider using a model trained with accent/dialect supervision (e.g. wav2vec2 fine-tuned on accent classification) as a second embedding source
- [ ] Concatenate TitaNet (speaker) + accent-aware embeddings for the pair feature

### Architecture
- [ ] Separate MLP heads for `spk_sim` and `acc_sim` — test if decoupling helps `acc_sim`
- [ ] Add a cosine similarity scalar as an explicit input feature alongside the 768-dim pair vector

### Submission
- [ ] Write `infer_dev.py` — load best checkpoint, run on `dev.csv`, write `answer.txt`
- [ ] Zip and submit to CodaBench: `zip -j submission.zip answer.txt`
