# Idea D — System-Label Prior (best submission so far)

Blend a generalizable cosine-fusion prediction with the **training-label mean score of each system**, exploiting that dev.csv exposes `system_id` and 21 of 23 dev systems appear in train.

## Result

| Submission | spk UTT-SRCC | acc UTT-SRCC |
|---|---|---|
| Our previous best (shrinkage) | 0.426 | 0.373 |
| Official B1 (ECAPA cosine) | 0.432 | 0.369 |
| Official B2 (ECAPA + trained head) | 0.451 | 0.440 |
| **Idea D (α=0.7)** | **0.478** | 0.416 |
| Leaderboard leader | 0.629 | 0.608 |

**spk 0.478 beats both official baselines and our previous best.** acc 0.416 beats B1 and our previous best; still below B2.

## Why it works

- 33–45% of the score variance is **between-system**. A system's human-rated quality is stable, so its training-label mean is a strong prior for its dev pairs.
- 21/23 dev systems are seen in train → the prior applies to ~51% of dev pairs (304/600).
- The 2 unseen systems (sys003, sys015) are 49% of dev (148 pairs each) → they get no prior and fall back to the cosine prediction.

## Method

```
cosine_pred = ridge fusion over cosines (ecapa, titanet, wav2vec2 hidden/prob), trained on all train
prior       = training-label mean of the pair's system   (seen)
            = cosine_pred                                 (unseen: sys003, sys015)
final       = alpha * z(cosine_pred) + (1-alpha) * z(prior)  ->  scaled to [1,5]
```

## Files

```
idea-d-system-prior/
├── dev_like_eval.py     # validation: mimics dev (2 whole unseen systems = 49% + seen pairs)
├── make_submission.py   # builds answer.txt (alpha currently 0.5)
├── answer.txt           # predictions
└── submission-1/        # submitted answer (alpha=0.7) + scoring_result.zip -> 0.478/0.416
```

## Validation vs reality (calibration lesson)

Two validators, and which one told the truth:

| Validator | spk cosine-only | interpretation |
|---|---|---|
| dev-like eval (realistic 49% unseen) | 0.593 | **optimistic** — SRCC over many systems has more spread than dev's 2-big-block structure |
| grouped-CV harness | 0.459 | **calibrated** — matches real dev |
| **real dev (submitted)** | **0.478 (with prior)** | grouped-CV was right; dev-like overstated |

The honest pre-submission call ("0.47–0.55, not 0.61") matched the 0.478 result. Trust grouped-CV; treat dev-like as an upper bound.

## alpha note

Submitted run used α=0.7 → 0.478/0.416. Re-analysis with the correct 49%-unseen composition put the optimum at **α=0.5** (dev-like: 0.613 vs 0.609 — marginal). `make_submission.py` now defaults to 0.5; not worth a scarce re-submission for the ~0.004 predicted difference.

## Where the remaining gap is

The prior is largely tapped out (helps only the seen 51%). The 49% unseen half is ranked purely by cosine generalization — that's the cap. Closing it needs **better generalizable features** (UTMOS naturalness, WeSpeaker, WavLM SSL) and **Idea C degradation ladders** (synthetic unseen systems to train monotonic codec-damage sensitivity). See `../PLAN.md`.
