# Idea L — Phonetic-Posteriorgram (PPG) Accent Features

Pronunciation-level accent-similarity features, motivated by [2505.14410](https://arxiv.org/abs/2505.14410) which
argues speaker-embedding cosine (the organizers' O-ACC-SIM) is inadequate for accent and that
pronunciation features (PPG / vowel-formant distances) are the right signal.

## Method

Per-frame phoneme posteriors from `facebook/wav2vec2-lv-60-espeak-cv-ft` (eSpeak IPA, 392 phones),
mean-pooled to a per-utterance phone distribution. Pair features between wav_a and wav_b:
cosine, Jensen-Shannon divergence, Bhattacharyya coefficient, L2. (Feature-extractor only —
no phoneme tokenizer needed, avoids the `phonemizer` dependency.)

## Result — novel, helps a weak baseline, but REDUNDANT with our best (a documented finding)

Single PPG features are weak alone (acc SRCC ≤ 0.074) — expected, since the pairs are **different
sentences**, so the mean phone distribution is content-confounded. But in fusion the ridge extracts
the content-invariant accent component:

| | spk | acc |
|---|---|---|
| F = base + WavLM | 0.5007 | 0.4886 |
| **F + PPG** | 0.5037 | **0.5019 (+0.013)** |

However, added on top of the **richer-RAMP** best pipeline it's a **wash**:

| | spk | acc |
|---|---|---|
| richer-RAMP | 0.6652 | 0.6297 |
| richer-RAMP + PPG | 0.6643 | 0.6308 |

**Finding:** pronunciation-distance (PPG) features carry real accent-similarity signal that helps a
plain speaker-embedding baseline (+0.013 acc), but that signal is **already captured by WavLM
mid-layers + CommonAccent** in our best system — so PPG is redundant once those are present. This
corroborates the organizers' surprising result that SSL/speaker embeddings already encode accent
similarity well (system-level O-SPK-SIM predicts accent better than O-ACC-SIM).

## Files
```
idea-l-ppg/
├── extract_ppg.py   # phoneme-posterior extraction (VM/H100)
├── ppg_dist.pt      # per-utterance mean phone distributions (392-d) [gitignored]
├── ppg_eval.py      # PPG features on the harness (helps weak baseline)
└── ppg_ramp.py      # richer-RAMP + PPG (wash vs richer-RAMP alone)
```
