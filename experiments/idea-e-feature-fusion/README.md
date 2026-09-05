# Idea E — Extended Feature Fusion

Adds three feature sources to the Idea-D fusion+prior pipeline: WeSpeaker (2nd speaker view), CommonAccent-ECAPA (the paper's O-ACC-SIM backbone), and UTMOS naturalness. Same system-label-prior blend as Idea D.

## Motivation

Idea D (spk 0.478 / acc 0.416) beat both baselines on speaker but trailed B2 (0.440) on accent. The 4-feature fusion lacked a dedicated accent representation. This batch adds one (CommonAccent) plus complementary signal.

## New features

| Source | Dim | What | Extractor |
|---|---|---|---|
| WeSpeaker (pyannote) | 256 | 2nd speaker embedding | `../pyannote/infer_embed.py` |
| CommonAccent-ECAPA emb | 192 | paper's O-ACC-SIM accent embedding | `extract_commonaccent.py` |
| CommonAccent-ECAPA prob | 16 | accent-class posterior | `extract_commonaccent.py` |
| UTMOS | scalar/wav | naturalness (VMC2022 winner) → utmos_a, utmos_b, utmos_diff | `extract_utmos.py` |

10 features total (7 cosines + 3 UTMOS scalars).

## Grouped-CV results (calibrated to real dev)

Single-feature leaderboard (top per target):

| spk_sim | | acc_sim | |
|---|---|---|---|
| cos_titanet | 0.439 | **cos_commonaccent_emb** | **0.407** |
| cos_ecapa | 0.433 | cos_ecapa | 0.377 |
| cos_wespeaker | 0.399 | cos_titanet | 0.357 |

**CommonAccent is the new #1 accent feature** — exactly the gap-filler we needed.

Ridge fusion (grouped-CV):

| | 4-feature (Idea D) | 10-feature (Idea E) | gain |
|---|---|---|---|
| spk | 0.459 | 0.474 | +0.015 |
| acc | 0.408 | **0.455** | **+0.047** |

Dev-like + system prior: **spk 0.622 (α=0.5) / acc 0.571 (α=0.6)**.

## Result (submitted)

| | Idea D | **Idea E** | prediction |
|---|---|---|---|
| spk | 0.478 | **0.5008** | ~0.49 ✓ |
| acc | 0.416 | **0.4769** | ~0.46 ✓ |

**Best submission — beats every official baseline on both targets.** spk crossed 0.50; acc 0.477 beats B2 (0.440) for the first time. CommonAccent-ECAPA delivered the accent unlock (+0.061 acc over Idea D). Harness prediction (0.49/0.46) matched the result (0.501/0.477).

## Files

```
idea-e-feature-fusion/
├── extract_utmos.py           # UTMOS naturalness per wav
├── extract_commonaccent.py    # CommonAccent-ECAPA emb + posterior
├── features.py                # extended feature table builder (cosines + UTMOS scalars)
├── fusion_eval.py             # grouped-CV + dev-like eval with prior
├── make_submission.py         # answer.txt (per-target alpha: spk 0.5, acc 0.6)
├── *.pt                       # extracted features
└── submission_feature_fusion.zip
```

## Submit

```bash
python make_submission.py
zip -j submission_feature_fusion.zip answer.txt   # already done
```
