# VoiceMOS Challenge 2026 — Track 3 Experiments

Predicting speaker and accent similarity scores for neural audio codec (NAC)-based speech synthesis systems.

---

## Task Overview

**Track 3** of the VoiceMOS Challenge 2026 requires predicting two subjective similarity scores for pairs of speech audio:

| Score | Description | Range |
|---|---|---|
| `spk_sim` | Speaker similarity — how similar the speaker identity sounds | 1–5 |
| `acc_sim` | Accent similarity — how similar the accent/dialect sounds | 1–5 |

Given `wav_a` (a synthesized speech sample) and `wav_b` (a natural VCTK reference), the model must output predicted scores for both targets. The primary leaderboard metric is **utterance-level Spearman Rank Correlation Coefficient (UTT-SRCC)**.

Participants may submit predictions for one or both targets independently.

---

## Dataset — CodecMOS-Accent

Based on the **VCTK** corpus. Contains resynthesis and voice-cloned TTS samples from 24 open-source NAC-based systems, covering **32 speakers** and **10 English accents**. Subjective ratings collected from **25 listeners**, totalling **19,600 annotations**.

Reference: *"CodecMOS-Accent: A MOS Benchmark of Resynthesized and TTS Speech from Neural Codecs Across English Accents"* — Huang et al., INTERSPEECH 2026. [arXiv:2603.14328](https://arxiv.org/abs/2603.14328)

### Splits

| Split | Pairs | File |
|---|---|---|
| Train | 2,800 | `sets/train.csv` |
| Dev | 600 | `sets/dev.csv` |
| Eval | 600 | TBA |

### train.csv structure

```
system_id, utterance_id, listener_id, wav_a_path, wav_b_path, spk_sim, acc_sim
sys017,     utt026,       15,          wav/...,     wav/...,    4,       5
```

Each row is one **listener's rating** for one pair. Multiple rows exist per pair (3–5 listeners). `spk_sim` and `acc_sim` are integer scores 1–5.

| Fact | Value |
|---|---|
| Total listener ratings | 13,687 |
| Unique pairs | 2,800 |
| Listeners per pair | 3–5 (mostly 5) |
| Unique listeners | 25 |
| Synthesis systems in train | 21 (`sys001`–`sys026`, excl. `sys019`) |
| `wav_b` always from | `sys019` (VCTK natural speech) |
| Audio sample rate | 16 kHz |

### dev.csv structure

```
system_id, utterance_id, wav_a_path, wav_b_path
```

No labels — generate predictions and submit to CodaBench.

### Score distribution (training set)

Both targets are **skewed toward 4–5** — most systems reproduce speaker and accent well:

| Score | spk_sim | acc_sim |
|---|---|---|
| 1 | 690 | 761 |
| 2 | 1,126 | 1,128 |
| 3 | 1,859 | 1,806 |
| 4 | 3,177 | 3,234 |
| 5 | 6,835 | 6,758 |
| **Mean** | **4.04** | **4.02** |

### Data distributions (two separate downloads)

| Distribution | Contains | How to obtain |
|---|---|---|
| SYN | Synthesized samples (sys001–sys026, excl. sys008/sys019) | License agreement required — email ecooper@nict.go.jp |
| VCTK | Natural VCTK samples (sys008, sys019) | [Google Drive link](https://drive.google.com/file/d/1_l5Aj74vhiEfCX6_dh6B4gvUXii8iinR/view?usp=share_link) |

After downloading both, copy all VCTK `wav/` files into the SYN `wav/` directory. Total: **3,548 unique wav files**.

---

## Evaluation Metrics

From `metrics_voicemos.py` — all computed against utterance-level mean scores (averaged over listeners):

| Metric | Description | Primary? |
|---|---|---|
| **UTT-SRCC** | Utterance-level Spearman Rank Correlation | **Yes (leaderboard)** |
| UTT-LCC | Linear Correlation Coefficient | Summary paper |
| UTT-MSE | Mean Squared Error | Summary paper |
| UTT-KTAU | Kendall Tau Rank Correlation | Summary paper |
| SYS-* | Same metrics at system level | Summary paper |

Getting the **ordering** of pairs right matters more than absolute score accuracy.

### Submission format

```
system_id,utterance_id,wav_a_path,wav_b_path,pred_acc_sim,pred_spk_sim
sys003,utt010,wav/...,wav/...,3.76,4.12
```

Submit as: `zip -j <any_name>.zip answer.txt`

---

## Repository Structure

```
voicemos-challenge-2026-exp/
├── README.md                        ← this file
├── metrics_voicemos.py              ← official evaluation metrics (snippet)
├── NeMo/                            ← NVIDIA NeMo toolkit (submodule)
│   └── examples/speaker_tasks/recognition/
│       └── extract_speaker_embeddings.py   ← patched for CPU/MPS device detection
└── titanet_large/                   ← Experiment 1: TitaNet-Large baseline
    ├── README.md                    ← experiment-specific notes
    ├── avg_scores.py                ← aggregate listener ratings → mean per pair
    ├── train_avg.csv                ← 2800 pairs with mean spk_sim + acc_sim
    ├── baseline_speaker_embed.log   ← NeMo embedding extraction log
    ├── infer/
    │   ├── gen_manifest.py          ← builds NeMo manifest from train+dev CSVs
    │   ├── manifest.json            ← 3548 unique wavs
    │   ├── speaker_embed_train_gen.sh  ← end-to-end extraction script
    │   └── embeddings/embeddings/
    │       └── manifest_embeddings.pt  ← extracted embeddings (192-dim each)
    ├── mlp-train-head/
    │   ├── train.py                 ← MLP trainer with cosine baseline
    │   ├── train.log                ← training results
    │   └── best_model.pt            ← best checkpoint by mean SRCC
    └── train/
        └── Speaker_Identification_Verification.ipynb
```

---

## Experiment 1 — TitaNet-Large + MLP Head

### Approach

1. Extract **192-dim speaker embeddings** for every wav using NVIDIA TitaNet-Large (pretrained on VoxCeleb + LibriSpeech + Fisher + Switchboard)
2. Build a **768-dim pair feature vector** per pair:
   ```
   [emb_a | emb_b | emb_a − emb_b | emb_a × emb_b]
   ```
3. Train a small **MLP regressor** with two output heads: `pred_spk_sim` and `pred_acc_sim`
4. Compare against a **cosine similarity baseline** (no training required)

### Cosine similarity baseline

Computes `cosine(emb_a, emb_b)` directly as the predictor:
- For `spk_sim`: natural fit — TitaNet is trained for speaker identity
- For `acc_sim`: weak proxy — same score reused, TitaNet not trained for accent

### MLP architecture

```
Input: 768-dim pair feature
  → Linear(768, 128) → ReLU → Dropout(0.2)
  → Linear(128, 2)
Output: [pred_spk_sim, pred_acc_sim]
```

Training: MSE loss, Adam optimizer, cosine LR annealing, 90/10 train/val split.

### Results (val split, seed=42)

| Model | SRCC spk_sim | SRCC acc_sim | Mean SRCC |
|---|---|---|---|
| Cosine similarity baseline | 0.5991 | 0.4944 | 0.5468 |
| MLP 768→128→2 (50 epochs) | **0.6264** | **0.5919** | **0.6091** |

**Key observations:**
- MLP improves `acc_sim` by **+0.097** — the two output heads learn different weightings of embedding dimensions for each target
- `spk_sim` improvement is modest (**+0.027**) — cosine similarity was already a strong signal for speaker identity
- Training loss still decreasing at epoch 50 — model not fully converged

### Training curve

```
Epoch      loss   SRCC_spk   SRCC_acc
    1   13.5684    -0.1192    -0.1444
    5    0.5354     0.4516     0.4468
   10    0.3880     0.5530     0.5320
   20    0.3226     0.6019     0.5749
   35    0.2860     0.6254     0.5914  ← best
   50    0.2734     0.6264     0.5919
```

### How to reproduce

```bash
# 1. Extract embeddings (~20 min on CPU)
cd titanet_large/infer
bash speaker_embed_train_gen.sh

# 2. Average listener scores
cd ..
/path/to/venv/bin/python3 avg_scores.py

# 3. Train MLP head
cd mlp-train-head
/path/to/venv/bin/python3 train.py
```

---

## Planned / Next Experiments

### Improve current TitaNet baseline
- [ ] Train longer (`--epochs 200`) — loss was still falling at epoch 50
- [ ] Search over hidden sizes: `[256]`, `[256 128]`, `[512 256]`
- [ ] Add cosine scalar as explicit input alongside the 768-dim pair vector
- [ ] Separate MLP heads for `spk_sim` and `acc_sim` — may help accent modeling

### Stronger base embeddings
- [ ] **WavLM-large** (`microsoft/wavlm-large`) — 1024-dim SSL features, top-performing on SUPERB speaker tasks; likely biggest single improvement
- [ ] **ECAPA-TDNN** (SpeechBrain `spkrec-ecapa-voxceleb`) — already available in `exp-asv/infer_all.py`; compare against TitaNet
- [ ] **Qwen3 speaker encoder** — ONNX model already available locally (see `exp-asv/`)
- [ ] **Ensemble** — combine cosine scores from multiple speaker models as input features

### Better accent modeling
- [ ] TitaNet captures speaker identity but not accent explicitly — the persistent gap between `spk_sim` and `acc_sim` SRCC reflects this
- [ ] Explore wav2vec2 / HuBERT middle layers (frame-level features capture phonetics better than speaker embeddings)
- [ ] Consider a model with explicit accent supervision

### End-to-end fine-tuning
- [ ] Fine-tune TitaNet encoder itself on the similarity regression task (not just the MLP head)
- [ ] Use listener-wise scores (not just mean) as training signal — introduces uncertainty modeling

### Submission
- [ ] Write `infer_dev.py` — load best checkpoint, run on `dev.csv`, write `answer.txt`
- [ ] Zip and submit: `zip -j submission.zip answer.txt`

---

## Notes

- NeMo's filterbank op does not support Apple MPS — extraction runs on CPU only. `PYTORCH_ENABLE_MPS_FALLBACK=1` does not catch this op.
- NeMo embedding key format: `<dataset_dir_name>@wav@<filename>.wav` (last 3 path components joined with `@`)
- `wav_b` is always from `sys019` (VCTK natural speech) in both train and dev sets
- Paper reference for system-level analysis: [arXiv:2603.14328](https://arxiv.org/abs/2603.14328) — Table 1 shows per-system similarity scores useful for sanity-checking predictions
