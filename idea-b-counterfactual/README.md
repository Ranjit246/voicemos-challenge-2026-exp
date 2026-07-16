# Idea B — Counterfactual Pseudo-Pairs (NEGATIVE result)

An elegant idea that the leaderboard metric structurally cannot reward. Documented so we don't retry it.

## The idea

Real training pairs have spk_sim ↔ acc_sim correlated 0.86, so the accent head can never learn "same accent, DIFFERENT speaker." We manufactured exactly those pairs from natural VCTK speech:

1. Cluster natural wavs (sys019 + sys008, 296 files) by ECAPA embedding → 32 clean speaker clusters (within-cluster cosine 0.78), spanning 9 accents.
2. Label each cluster's accent via mean wav2vec2 accent posterior.
3. Manufacture pairs with pseudo-labels:

| Pair type | ECAPA cos (speaker) | wav2vec2-prob cos (accent) | spk label | acc label |
|---|---|---|---|---|
| same speaker | 0.790 | 1.0 | 4.8 | 4.7 |
| **diff speaker, same accent** | **0.151** | **1.0** | 1.5 | 4.0 |
| diff speaker, diff accent | 0.125 | 0.0 | 1.3 | 1.8 |

The middle row is the decorrelating case real data never contains. The cosine separation is exactly as intended.

## Why it failed

Grouped-CV fusion, real-only vs real+counterfactual:

| target | baseline | + counterfactual | delta |
|---|---|---|---|
| spk_sim | 0.4589 | 0.4524 | −0.0065 |
| acc_sim | 0.4079 | 0.4072 | −0.0007 |

**No improvement.** Root cause, measured directly:

- **Within-system, spk and acc still correlate 0.79.** The confound isn't an artifact of cross-system variation — it's intrinsic.
- Every dev/eval pair is a *same-speaker attempt* (synthesized speech trying to match the reference speaker). There are no "different speaker, same accent" cases in the test set to reward.
- The organizers' paper confirms it: utterance-level, the speaker-embedding cosine predicts accent similarity *better* than the accent-embedding cosine (0.34 vs 0.31). The best accent predictor is a good speaker predictor.

Teaching accent ≠ speaker is real, but the challenge never measures that axis.

## Files

```
idea-b-counterfactual/
├── build_pairs.py                 # clustering + counterfactual pair manufacture
├── counterfactual_features.csv    # 1016 manufactured pairs with cosine features + pseudo-labels
└── train_fusion.py                # real vs real+counterfactual grouped-CV comparison
```

## Salvage value

The **speaker clustering + accent labeling** of natural VCTK (build_pairs.py) is reusable — Idea C (degradation ladders) and any accent-aware work can use these clean speaker/accent groupings.
