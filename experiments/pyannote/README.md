# pyannote WeSpeaker — VoiceMOS 2026 Track 3

Speaker embedding extraction and similarity baseline using the embedding sub-model from `pyannote/speaker-diarization-community-1`.

---

## Model

**`pyannote/speaker-diarization-community-1`** — embedding subfolder

| Detail | Value |
|---|---|
| Architecture | WeSpeakerResNet34 |
| Embedding dim | 256 |
| HuggingFace token | Not required (community model) |
| Trained on | VoxCeleb (speaker verification) |
| API | `pyannote.audio` — `Model` + `Inference` |

The diarization pipeline bundles a speaker embedding model internally. We extract just that sub-model directly using `subfolder="embedding"`, bypassing the full diarization pipeline entirely since we only need per-utterance speaker embeddings for single-speaker audio.

---

## Directory Structure

```
pyannote/
├── README.md              ← this file
├── infer_embed.py         ← extract 256-dim embeddings for all 3548 wavs
├── infer_submission.py    ← build answer.txt for CodaBench
├── embeddings.pt          ← dict: key → np.array (256,) [generated]
└── answer.txt             ← submission predictions [generated]
```

---

## Embedding Key Format

Matches TitaNet and wav2vec2 — compatible with the same downstream MLP pipeline:

```
vmc2026_track3_train_phase_distro_v3_syn@wav@<filename>.wav → np.array (256,)
```

---

## Pipeline

### Step 1 — Extract embeddings

```bash
cd pyannote
/path/to/venv/bin/python3 infer_embed.py
```

Uses `pyannote.audio`'s `Inference(model, window='whole')` API — takes a file path directly, handles resampling internally. No manifest file or batching setup required.

Processes all **3,548 unique wav files** (train + dev, both `wav_a` and `wav_b`). Output saved to `embeddings.pt`.

### Step 2 — Generate submission

```bash
/path/to/venv/bin/python3 infer_submission.py
```

Computes `cosine(emb_a, emb_b)` for each dev pair, scales linearly to [1, 5], writes `answer.txt`.

### Step 3 — Submit

```bash
zip -j submission_pyannote.zip answer.txt
```

---

## Submission Strategy

| Column | Value | Rationale |
|---|---|---|
| `pred_spk_sim` | `cosine(emb_a, emb_b)` | WeSpeakerResNet34 is trained for speaker verification — direct fit |
| `pred_acc_sim` | same cosine score | Proxy only — WeSpeaker is not accent-aware |

Both columns must be present in the submission file. CodaBench returns empty scores `{}` if only one column is submitted (confirmed from prior attempt).

---

## Comparison vs Other Approaches

| Model | Embedding dim | Trained for | spk_sim fit | acc_sim fit |
|---|---|---|---|---|
| TitaNet-Large (NeMo) | 192 | Speaker verification (VoxCeleb + LibriSpeech) | Strong | Weak (proxy) |
| **WeSpeakerResNet34 (pyannote)** | **256** | **Speaker verification (VoxCeleb)** | **Strong** | **Weak (proxy)** |
| wav2vec2 accent (HamzaSidhu786) | 768 hidden / 13 prob | Accent classification | Moderate | Strong |

TitaNet and WeSpeaker serve the same purpose — this submission lets us compare which speaker verification backbone produces better SRCC on the dev set.

---

## CodaBench Results

| Submission | TRACK3_SPK_UTT_SRCC | TRACK3_ACC_UTT_SRCC | Notes |
|---|---|---|---|
| TitaNet MLP | 0.3579 | 0.3750 | MLP overfit to system patterns |
| TitaNet cosine | TBD | TBD | Cosine-only baseline |
| wav2vec2 accent | TBD | TBD | Accent prob cosine for acc_sim |
| **pyannote WeSpeaker** | **TBD** | **TBD** | **256-dim cosine baseline** |

---

## Implementation Notes

- `pyannote.audio` must be installed: `pip install pyannote.audio`
- No HuggingFace token needed — `pyannote/speaker-diarization-community-1` is a public community model
- `Inference(model, window='whole')` returns a `np.ndarray` of shape `(256,)` directly — no manual audio loading or feature extraction needed
- Embeddings are L2-normalised internally by pyannote before output

---

## Next Steps

- [ ] Submit and compare SRCC vs TitaNet cosine baseline
- [ ] Train MLP head on pyannote pair features (1024-dim: `[emb_a | emb_b | diff | prod]`)
- [ ] Combine pyannote (spk) + wav2vec2 accent (acc) embeddings in a joint MLP — best-of-both approach
- [ ] Leave-one-system-out CV before submitting trained models to avoid overfitting
