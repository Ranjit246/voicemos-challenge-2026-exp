# Idea K — RAMP Retrieval Augmentation

A non-parametric kNN retrieval head fused with the parametric ridge, targeting the
unseen-system OOD axis (RAMP, [arXiv:2308.16488](https://arxiv.org/abs/2308.16488), a VMC'24 winning technique).

## Result — new best, broke the plateau

| Variant | dev spk | dev acc |
|---|---|---|
| Previous plateau (balanced ensemble) | 0.568 | 0.588 |
| RAMP in **cosine-feature space** (`ramp.py`, submission-1) | 0.5661 | 0.5890 (tie) |
| **★ RAMP in SSL embedding-diff space** (`richer_ramp.py`, submission-2) | **0.5745** | **0.5986** |
| Leader | 0.629 | 0.608 |

**Accent is now 0.009 from the leader.** The rich-space retrieval is what mattered.

## Why the space matters (the key finding)

- **Cosine-feature space → tie.** Retrieving in the 35 compact cosine scalars where the
  ridge already generalizes gives kNN nothing to add (grouped-CV fusion +0.004).
- **SSL embedding-difference space → real gain.** Retrieving in the 2432-d space of
  L2-normalized embedding differences (ECAPA + CommonAccent + WavLM-L3 + WavLM-L12), PCA→128,
  gives grouped-CV fusion **+0.021 spk / +0.032 acc** over ridge. kNN captures non-linear
  structure in the raw representation that the linear ridge misses — RAMP's actual mechanism.
- Fusion weight β≈0.8 (parametric-heavy); retrieval alone is weaker but complementary.

## Ablations (retrieval-space design — good paper content)

| Retrieval space | grouped-CV fuse spk | acc |
|---|---|---|
| cosine scalars (35-d) | 0.6483 | 0.6017 |
| **SSL diff, 4 sources (shared)** | **0.6652** | **0.6297** |
| SSL diff, per-target spaces (`_v2`) | 0.6525 | 0.6226 |
| SSL diff, 9 sources (enriched, `_v3`) | 0.6587 | 0.6209 |

Two clean negatives: **per-target retrieval spaces hurt** (spk↔acc correlate 0.86, so accent
info aids speaker retrieval and vice-versa — the shared space wins), and **more embedding
sources hurt** (redundant speaker embeddings dilute the kNN distances). The 4-source shared
space is the sweet spot.

## Files

```
idea-k-ramp/
├── ramp.py            # cosine-space RAMP (tie) -> submission-1
├── richer_ramp.py     # SSL embedding-diff space (BEST) -> submission-2
├── richer_ramp_v2.py  # per-target spaces (ablation, worse)
├── richer_ramp_v3.py  # 9-source enriched (ablation, worse)
└── submission-2/      # best: answer.txt + submission.zip
```

## Method (best, submission-2)

```
retrieval = kNN(k=30) over PCA-128 of L2-normed embedding diffs
            [ECAPA, CommonAccent, WavLM-L3, WavLM-L12], weights = softmax(-dist/mean-dist)
RAMP      = 0.8*z(ridge) + 0.2*z(retrieval)
final     = ensemble(RAMP, trained-head) -> system prior (alpha 0.5 spk / 0.6 acc) -> [1,5]
```
