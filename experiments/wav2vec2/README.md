# wav2vec2 Accent Model — VoiceMOS 2026 Track 3

Accent similarity prediction using a wav2vec2-based accent classification model.

---

## Motivation

The TitaNet baseline reuses speaker cosine similarity as a proxy for `acc_sim`. This experiment uses a model explicitly trained for **accent classification** to produce better `acc_sim` predictions — both from the probability output (13-dim) and the encoder's internal representations (768-dim).

---

## Model

**`HamzaSidhu786/speech-accent-detection`** (Hugging Face)

| Detail | Value |
|---|---|
| Architecture | `Wav2Vec2ForSequenceClassification` (wav2vec2-base fine-tuned) |
| Hidden size | 768-dim |
| Output classes | 13 accent labels |
| Token required | No |

**Accent labels** — all directly relevant to VCTK speakers:

| ID | Label | ID | Label |
|---|---|---|---|
| 0 | American | 7 | NewZealand |
| 1 | Australian | 8 | NorthernIrish |
| 2 | British | 9 | Scottish |
| 3 | Canadian | 10 | SouthAfrican |
| 4 | English | 11 | Unknown |
| 5 | Indian | 12 | Welsh |
| 6 | Irish | | |

---

## Directory Structure

```
wav2vec2/
├── README.md                      ← this file
├── app_accent.py                  ← quick demo: classify a single wav
├── infer_accent_embed.py          ← extract both embedding types for all 3548 wavs
├── infer_submission_13_prob.py    ← submission using 13-dim prob cosine (pred_acc_sim only)
├── infer_submission_hidden.py     ← submission using 768-dim hidden cosine (pred_acc_sim only)
├── accent_prob_embeddings.pt      ← 13-dim softmax prob vectors [generated]
├── accent_hidden_embeddings.pt    ← 768-dim mean-pooled hidden states [generated]
├── submission-1/
│   ├── answer.txt                 ← 13-dim prob cosine predictions
│   └── scoring_result.zip         ← CodaBench result
└── submission-2/
    └── answer_hidden.txt          ← 768-dim hidden cosine predictions (pending result)
```

---

## Two Embedding Types

### 1. Prob embeddings — `accent_prob_embeddings.pt`
- **Dim:** 13
- **How:** Softmax of classification logits
- **Meaning:** Probability distribution over accent classes
- **Use for acc_sim:** `cosine(prob_a, prob_b)` — are the accent distributions similar?
- **Script:** `infer_submission_13_prob.py`

### 2. Hidden embeddings — `accent_hidden_embeddings.pt`
- **Dim:** 768
- **How:** Mean-pool the last transformer hidden state across all time frames
- **Meaning:** Fine-grained accent-phonetic features from the encoder
- **Use for acc_sim:** `cosine(hidden_a, hidden_b)` — richer than prob vectors
- **Script:** `infer_submission_hidden.py`

---

## Pipeline

### Step 1 — Extract embeddings

```bash
/path/to/venv/bin/python3 infer_accent_embed.py
```

Runs on MPS. Produces both `accent_prob_embeddings.pt` and `accent_hidden_embeddings.pt` for all 3,548 wavs.

**Note:** Uses `AutoFeatureExtractor` (not `Wav2Vec2Processor`) — this model has no tokenizer.

### Step 2 — Generate submission

```bash
# 13-dim prob cosine
/path/to/venv/bin/python3 infer_submission_13_prob.py
zip -j submission_wav2vec2_prob.zip submission-1/answer.txt

# 768-dim hidden cosine
/path/to/venv/bin/python3 infer_submission_hidden.py
zip -j submission_wav2vec2_hidden.zip submission-2/answer_hidden.txt
```

Both scripts output `pred_acc_sim` only. CodaBench accepts single-column submissions for `acc_sim`.

---

## CodaBench Results

| Submission | Method | TRACK3_ACC_UTT_SRCC |
|---|---|---|
| submission-1 | 13-dim prob cosine | **0.3186** |
| submission-2 | 768-dim hidden cosine | TBD |

### Analysis — submission-1 (13-dim prob)

SRCC = 0.319 is **lower** than the TitaNet MLP baseline (0.375 for acc_sim). The 13-dim prob vector is too coarse:
- Both `wav_a` and `wav_b` are from the same VCTK speaker with the same accent
- Most pairs will have very similar prob vectors regardless of synthesis quality
- Very little discriminative signal remains for ranking

The 768-dim hidden embeddings (submission-2) capture finer-grained phonetic features and should rank pairs more precisely.

---

## All Submissions Summary (across experiments)

| # | Model | Method | SPK_SRCC | ACC_SRCC |
|---|---|---|---|---|
| 1 | TitaNet MLP | 768-dim pair feature → MLP | 0.3579 | 0.3750 |
| 2 | TitaNet cosine | 192-dim cosine | TBD | TBD |
| 3 | wav2vec2 prob | 13-dim prob cosine | — | 0.3186 |
| 4 | wav2vec2 hidden | 768-dim hidden cosine | — | TBD |

---

## Next Steps

- [ ] Submit submission-2 (768-dim hidden) and record result
- [ ] Train MLP on `accent_hidden_embeddings.pt` pair features for `acc_sim`
- [ ] Combine pyannote WeSpeaker (256-dim, spk) + wav2vec2 hidden (768-dim, acc) in a joint MLP
- [ ] Leave-one-system-out CV before training any MLP to get honest generalization estimate
