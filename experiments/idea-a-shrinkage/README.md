# Idea A — System-Mean Shrinkage

Label-free, transductive smoothing of per-utterance predictions toward their system's mean prediction. See `../PLAN.md` (Next-level ideas, Idea A).

## The idea in one line

Per-utterance similarity scores are mostly noise (listener std ±0.9); per-system averages are far more reliable (paper: SYS-SRCC 0.86 vs UTT 0.39 for plain cosine) — so blend every prediction toward its own system's mean prediction, computed on the prediction set itself with **zero labels**:

```
final(pair) = alpha * score(pair) + (1 - alpha) * mean(score of same-system pairs)
```

`dev.csv` exposes `system_id`, and dev has ~12–16 pairs per system — enough for stable means.

## Files

```
idea-a-shrinkage/
├── tune_alpha.py            # alpha sweep on train (full + dev-like subsample simulation)
├── alpha_sweep_results.csv  # sweep output
├── make_submission.py       # applies best configs to dev.csv -> answer.txt
└── answer.txt               # submission (600 rows)
```

## Sweep results (train set, UTT-SRCC vs human mean labels)

Raw cosine (α=1.0) vs best shrinkage per predictor:

| Predictor | Target | Raw (α=1.0) | Best shrunk | Best α | Gain |
|---|---|---|---|---|---|
| TitaNet cosine | spk | 0.607 | **0.626** | 0.4–0.5 | +0.02 |
| TitaNet cosine | acc | 0.510 | **0.526** | 0.4–0.5 | +0.02 |
| wav2vec2-hidden cosine | spk | 0.431 | **0.604** | 0.1 | **+0.17** |
| wav2vec2-hidden cosine | acc | 0.425 | **0.561** | 0.2 | **+0.14** |
| wav2vec2-prob cosine | spk | 0.370 | **0.599** | 0.1 | **+0.23** |
| wav2vec2-prob cosine | acc | 0.373 | **0.555** | 0.1 | **+0.18** |

Key findings:
- **Shrinkage never hurts** — every predictor improves at some α < 1.
- **Noisy predictors gain massively**: wav2vec2 cosines are weak per-utterance but their system means carry strong signal. The wav2vec2-prob predictor that scored **0.319 on CodaBench raw** reaches 0.555–0.599 on train when shrunk.
- Benefit survives the dev-like simulation (15 pairs/system, 50 draws) — system means from ~15 pairs are still ~4× less noisy than single utterances.
- Even α=0 (pure system mean) beats most raw predictors — system-level signal dominates UTT-SRCC, as the organizers' paper implied.

## Submission config

| Column | Predictor | α |
|---|---|---|
| `pred_spk_sim` | TitaNet cosine | 0.4 |
| `pred_acc_sim` | wav2vec2-hidden cosine | 0.2 |

```bash
python make_submission.py
zip -j submission_shrinkage.zip answer.txt
```

## Caveats

- α tuned on train labels — but it's a single flat-peaked parameter; overfitting risk is minimal.
- Train-set SRCC absolute values run higher than CodaBench dev; the *relative* gains are the evidence.
- If a system's quality is genuinely bimodal across utterances, shrinkage blurs it. Nothing in the sweep suggests this happens here.
- The same trick applies unchanged to the final eval set (it will also expose `system_id`).
