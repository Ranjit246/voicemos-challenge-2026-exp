# Official results — VoiceMOS Challenge 2026, Track 3 (Team T15)

Source: organisers' released results (2026-08-31) and `vmc2026_track3_test_with_labels.csv`.
Track 3 had **7 participating teams** plus 2 official baselines. Team T13 submitted
accent scores only.

![results](figures/results.png)

## Our scores

| | Utterance-level (primary) | | System-level | |
|---|---|---|---|---|
| | **SPK** | **ACC** | **SPK** | **ACC** |
| SRCC | **0.549** | **0.474** | **0.942** | **0.902** |
| LCC | 0.541 | 0.469 | 0.903 | 0.841 |
| KTAU | 0.393 | 0.336 | 0.800 | 0.747 |
| MSE | 0.702 | 0.773 | 0.312 | 0.374 |
| **rank (of 7 teams)** | **4th** | **5th** | **3rd** | **3rd** |

## Full standing — utterance-level SRCC

| Entry | SPK | ACC |
|---|---|---|
| T04 | 0.644 | 0.565 |
| T16 | 0.606 | 0.523 |
| T07 | 0.585 | 0.530 |
| **T15 (ours)** | **0.549** | **0.474** |
| T06 | 0.484 | 0.505 |
| T17 | 0.481 | 0.386 |
| B1 — ECAPA cosine (baseline) | 0.414 | 0.386 |
| B2 — ECAPA + head (baseline) | 0.403 | 0.316 |
| T13 (accent only) | — | 0.371 |

We beat the stronger official baseline by **+0.135 SPK** and **+0.088 ACC**.

## What mattered most

Transductive system-mean smoothing, validated at +0.008 under grouped CV,
delivered far more on the real evaluation set:

| System | SPK | ACC | mean |
|---|---|---|---|
| Base (retrieval + head + prior) | 0.512 | 0.458 | 0.485 |
| **+ transductive smoothing (w=0.7)** | **0.549** | **0.474** | **0.512** |
| Δ | **+0.037** | **+0.016** | **+0.026** |

The evaluation set contains two unseen systems of 160 pairs each, which give very
stable predicted system means — exactly the regime the smoothing exploits.

## Known weakness

**MSE ranked 7th.** We optimised rank correlation and min-max rescaled to [1,5],
so absolute calibration was sacrificed. The challenge metric (SRCC) did not
penalise this, but the predictor is not usable as an absolute MOS estimator as-is.

## Reproducing the figures

```
python results/make_figures.py        # -> figures/results.png
python results/make_architecture.py   # -> figures/architecture.png
```
