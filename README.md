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

Participants may submit predictions for one or both targets, but **CodaBench scoring requires both columns to be present** in the submission file — even if both contain the same values.

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

### Train vs Dev system overlap

| | Systems |
|---|---|
| Train only | — |
| Dev only (unseen during training) | `sys003`, `sys015` |
| Both | `sys001`, `sys002`, `sys005`–`sys018`, `sys020`, `sys022`–`sys026` |

Dev contains **2 unseen systems** — a key generalization challenge.

### Score distribution (training set, listener-wise)

Both targets are **skewed toward 4–5** — most systems reproduce speaker and accent well:

| Score | spk_sim | acc_sim |
|---|---|---|
| 1 | 690 | 761 |
| 2 | 1,126 | 1,128 |
| 3 | 1,859 | 1,806 |
| 4 | 3,177 | 3,234 |
| 5 | 6,835 | 6,758 |
| **Mean** | **4.04** | **4.02** |

### Per-system mean scores (training set)

| System | spk_sim | acc_sim | n pairs |
|---|---|---|---|
| sys008 | 4.764 | 4.702 | 136 |
| sys006 | 4.665 | 4.603 | 135 |
| sys014 | 4.587 | 4.529 | 133 |
| sys011 | 4.577 | 4.467 | 135 |
| sys009 | 4.529 | 4.229 | 135 |
| sys023 | 4.472 | 4.345 | 134 |
| sys024 | 4.393 | 4.284 | 134 |
| sys002 | 4.347 | 4.107 | 135 |
| sys013 | 4.333 | 4.311 | 132 |
| sys001 | 4.296 | 4.292 | 134 |
| sys010 | 4.143 | 4.051 | 134 |
| sys017 | 4.169 | 4.166 | 129 |
| sys012 | 4.036 | 4.046 | 135 |
| sys025 | 3.928 | 3.978 | 134 |
| sys022 | 3.835 | 3.778 | 130 |
| sys026 | 3.726 | 3.769 | 134 |
| sys005 | 3.290 | 3.443 | 135 |
| sys007 | 3.425 | 3.405 | 132 |
| sys016 | 3.323 | 3.562 | 131 |
| sys018 | 3.162 | 3.345 | 130 |
| sys020 | 2.754 | 3.036 | 133 |

System mean std = 0.56 — large spread, which makes system-level memorization a risk.

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

- Both `pred_acc_sim` and `pred_spk_sim` columns must be present (CodaBench requirement)
- Scores are floats, no range restriction enforced by scorer — but clamp to [1, 5] to be safe
- Submit as: `zip -j <any_name>.zip answer.txt`

---

## Repository Structure

```
voicemos-challenge-2026-exp/
├── README.md                        ← this file
├── metrics_voicemos.py              ← official evaluation metrics (snippet)
├── NeMo/                            ← NVIDIA NeMo toolkit (submodule)
│   └── examples/speaker_tasks/recognition/
│       └── extract_speaker_embeddings.py   ← patched for CPU device detection
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
    └── mlp-train-head/
        ├── train.py                 ← MLP trainer with cosine baseline printout
        ├── train.log                ← training results
        ├── best_model.pt            ← best checkpoint by mean SRCC
        ├── infer_dev.py             ← run best_model.pt on dev.csv → answer.txt
        ├── infer_cosine_baseline.py ← cosine-only inference → answer_cosine.txt
        ├── answer.txt               ← MLP predictions (submitted)
        └── answer_cosine.txt        ← cosine baseline predictions (submitted)
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

Computes `cosine(emb_a, emb_b)` directly as the predictor. Linearly scaled to [1, 5] before submission:
- For `spk_sim`: natural fit — TitaNet is trained for speaker identity
- For `acc_sim`: weak proxy — same score reused, TitaNet not trained for accent

### MLP architecture

```
Input: 768-dim pair feature
  → Linear(768, 128) → ReLU → Dropout(0.2)
  → Linear(128, 2)
Output: [pred_spk_sim, pred_acc_sim]
```

Training: MSE loss, Adam optimizer, cosine LR annealing, 90/10 random train/val split.

### Internal validation results (val split, seed=42)

| Model | SRCC spk_sim | SRCC acc_sim | Mean SRCC |
|---|---|---|---|
| Cosine similarity baseline | 0.5991 | 0.4944 | 0.5468 |
| MLP 768→128→2 (50 epochs) | **0.6264** | **0.5919** | **0.6091** |

### CodaBench (dev set) results

| Submission | SRCC spk_sim | SRCC acc_sim |
|---|---|---|
| MLP 768→128→2 | 0.3579 | 0.3750 |
| Cosine baseline | TBD | TBD |

### Analysis — why the gap?

The internal val SRCC (~0.63) is much higher than CodaBench dev SRCC (~0.36). Root cause: **system-level memorization**.

- The MLP learned to predict largely based on *which system* the embedding comes from (system means range from 2.75 to 4.76, std=0.56) rather than from acoustic pair features
- The random 90/10 val split included the same 21 systems as training, so val looked artificially good
- Dev has 2 **unseen systems** (`sys003`, `sys015`) — the model has no prior for these
- A correct evaluation strategy is **leave-one-system-out** or split by utterance ID, not randomly

### Training curve

```
Epoch      loss   SRCC_spk   SRCC_acc
    1   13.5684    -0.1192    -0.1444
    5    0.5354     0.4516     0.4468
   10    0.3880     0.5530     0.5320
   20    0.3226     0.6019     0.5749
   35    0.2860     0.6254     0.5914  ← best checkpoint
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

# 4. Run inference → answer.txt
/path/to/venv/bin/python3 infer_dev.py

# 5. Run cosine baseline → answer_cosine.txt
/path/to/venv/bin/python3 infer_cosine_baseline.py

# 6. Submit
zip -j submission.zip answer.txt
```

---

## Planned / Next Experiments

### Fix evaluation strategy
- [ ] **Leave-one-system-out cross-validation** — train on 20 systems, validate on 1; honest measure of generalization to unseen systems
- [ ] Split val by utterance ID (not randomly) to avoid system-level leakage

### Improve generalization of MLP
- [ ] Higher dropout (0.3–0.5), weight decay, early stopping on system-agnostic val
- [ ] Add cosine scalar as an explicit input feature — it's a purer acoustic signal
- [ ] Separate MLP heads per target — decouple `spk_sim` and `acc_sim` learning

### Stronger base embeddings
- [ ] **WavLM-large** (`microsoft/wavlm-large`) — 1024-dim SSL features, top on SUPERB speaker tasks; likely biggest single improvement
- [ ] **ECAPA-TDNN** (SpeechBrain `spkrec-ecapa-voxceleb`) — already in `exp-asv/infer_all.py`
- [ ] **Qwen3 speaker encoder** — ONNX model already available locally
- [ ] **Ensemble** — combine cosine scores from multiple speaker models as features

### Better accent modeling
- [ ] TitaNet captures speaker identity, not accent — persistent `acc_sim` gap reflects this
- [ ] Explore wav2vec2 / HuBERT middle layers (phonetic features)
- [ ] Consider a model with explicit accent/dialect supervision

### End-to-end fine-tuning
- [ ] Fine-tune TitaNet encoder on similarity regression (not just the MLP head)
- [ ] Train with listener-wise scores rather than mean — captures annotation uncertainty

---

## Implementation Notes

- **CodaBench requires both `pred_acc_sim` and `pred_spk_sim` columns** — submitting only one returns `{}` (empty scores), even though the challenge description says single-target submission is allowed
- NeMo's filterbank op does not support Apple MPS — extraction runs on CPU only; `PYTORCH_ENABLE_MPS_FALLBACK=1` does not catch this op
- NeMo embedding key format: `<dataset_dir_name>@wav@<filename>.wav` (last 3 path components joined with `@`)
- `wav_b` is always `sys019` (VCTK natural speech) in both train and dev
- Python venv: `/Users/ranjitpatro/Home/Research/VoiceMOS/venv`
