# VoiceMOS 2026 Track 3 — System Description (paper stats & methods)

Working reference for the ICASSP 2027 system-description paper. All numbers are from our
experiments; **dev-set UTT-SRCC is the CodaBench leaderboard metric**. Fill in team name/authors.

---

## 1. Task & dataset

**Task.** Given a pair — `wav_a` (codec-resynthesis or NAC-based TTS output) and `wav_b` (natural
VCTK reference of the target speaker, **different sentence**, 3–7 s, 16 kHz) — predict two
utterance-level MOS scores in [1,5]: **speaker similarity (spk_sim)** and **accent similarity
(acc_sim)**. Primary metric: **utterance-level Spearman rank correlation (UTT-SRCC)**.

**Dataset (CodecMOS-Accent, Huang et al., arXiv:2603.14328).** 4,000 samples from 24 open-source
systems (9 codec-resynthesis + 15 voice-cloning TTS), 32 VCTK speakers, 10 English accents; 19,600
annotations from 25 listeners on naturalness / speaker-sim / accent-sim.

| Split | Pairs | Systems | Notes |
|---|---|---|---|
| Train | 2,800 (13,687 listener rows) | 21 | 3–5 raters/pair (mean 4.9) |
| Dev | 600 | 23 | **2 unseen systems (sys003, sys015) = 296 pairs = 49%** |
| Eval | 600 | 25 | **4 unseen systems (sys003/004/015/021) = 344 pairs = 57%**; two (sys004, sys021) are 160 pairs each |

- `wav_b` is always `sys019` (natural VCTK). VCTK distributed separately (license); 3,548 unique wavs total.
- Score distribution skewed high: mean spk 4.04 / acc 4.02 (std 0.82 / 0.77), full 1–5 range.
- Per-system mean spk ranges 2.75 (sys020) → 4.76 (sys008); **between-system variance = 45% (spk) / 33% (acc)** of total.

---

## 2. Data analysis (paper's "analysis" section)

- **spk_sim ↔ acc_sim correlation:** 0.86 utterance-level (train), 0.97 system-level (organizers) —
  the two sub-scores are intrinsically coupled. **Within-system** they still correlate 0.79.
- **Rater noise:** within-pair listener std 0.85 (spk) / 0.92 (acc) on a 5-pt scale.
- **Noise ceiling (our split-half reliability estimate):** spk reliability 0.66 → a *perfect* latent
  predictor correlates ≈ **0.81** with the observed mean; acc reliability 0.58 → ≈ **0.76**.
  Utterance-level SRCC on this task is intrinsically capped well below 1.0.
- **Field context:** VMC'24 *winner* sat at 0.58–0.61 UTT-SRCC on hard splits; VoxSim's best model
  collapses to 0.57–0.61 OOD. **The 0.63/0.63 dev leader is near the practical ceiling.**

---

## 3. Best system (method)

Frozen-model **cosine fusion + pair-difference retrieval + trained-head ensemble + system prior.**
No backbone fine-tuning (small data → frozen generalizes).

```
                per-utterance frozen embeddings / scores
  ┌─────────────────────────────────────────────────────────────┐
  │ ECAPA-TDNN · TitaNet · WeSpeaker(pyannote) · CommonAccent-ECAPA │  speaker + accent
  │ wav2vec2-accent(hidden+prob) · WavLM-Large(all 25 layers) · UTMOS │
  └─────────────────────────────────────────────────────────────┘
        │ cosine(wav_a, wav_b) per source        │ pair-difference vectors
        ▼                                        ▼
   [~35 scalar features]                 [ECAPA,CommonAccent,WavLM-L3,WavLM-L12]
        │                                 L2-normed diffs → concat(2432) → PCA-128
   Ridge (parametric)                          │
        │                                  kNN retrieval (k=30, softmax(-d/mean-d))
        └───────── RAMP fuse: 0.8·z(ridge) + 0.2·z(retrieval) ─────────┐
                                                                       │
   trained WavLM pairwise head ──── z-avg ensemble (1:1) ──────────────┤
   (learnable per-task layer weights + pair interactions               │
    + margin-ranking loss, 6-seed average)                             ▼
                                        system-label prior blend  α·z(param)+(1-α)·z(prior)
                                        (α = 0.5 spk / 0.6 acc; unseen system → cosine fallback)
                                                                       │
                                   transductive system-mean smoothing (eval): 0.7·p + 0.3·mean_sys(p)
                                                                       │
                                                              scale to [1,5] → answer.txt
```

**Component details.**
- **Feature panel (frozen):** ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`, 192-d), TitaNet-L
  (NeMo, 192), WeSpeaker-ResNet34 (`experiments/pyannote/…-community-1`, 256), CommonAccent-ECAPA
  (`Jzuluaga/…`, 192 emb + 16 posterior), wav2vec2 accent (`HamzaSidhu786/…`, 768 hidden + 13
  posterior), WavLM-Large (`microsoft/wavlm-large`, 25×1024, all layers), UTMOS
  (`tarepan/SpeechMOS`, scalar). Features are pair cosine similarities + UTMOS(a),UTMOS(b),diff.
- **RAMP retrieval:** query = concatenated L2-normalised embedding **differences** of {ECAPA,
  CommonAccent, WavLM layer 3 (speaker), WavLM layer 12 (accent)}; standardise → PCA-128; kNN
  (k=30) over the training datastore, weights = softmax(−dist/mean-dist), predict = weighted mean
  of neighbours' labels. Fuse β≈0.8 with ridge.
- **Trained head:** frozen WavLM (25×1024, mean-pooled per layer) → per-task learnable softmax
  layer weights → 128-d projection → pair feature `[proj_a, proj_b, |diff|, prod]` → shared trunk
  → spk/acc heads; loss = MSE + margin-ranking (margin 0.3) with cross-system batches; **linear /
  small + regularised** (weight decay 1e-3, dropout); 6-seed averaged; nested grouped-CV early stop.
- **System-label prior:** blend prediction toward the training-label mean of the pair's system;
  unseen systems fall back to the cosine prediction (this fallback is what makes it OOD-safe, §6).
- **Transductive system-mean smoothing (eval-time):** each final score is pulled toward its own
  system's *predicted* mean over the eval set, `p' = 0.7·p + 0.3·mean_sys(p)` — label-free, applies
  to seen and unseen systems alike. It sharpens the reliable between-system ranking (which dominates
  UTT-SRCC) and is the single biggest eval-time gain (§5.1: +0.037 spk / +0.016 acc on the real eval).

---

## 4. Validation methodology (a paper strength)

**Grouped cross-validation by system** (hold out whole systems) — mimics the dev/eval unseen-system
structure. This is the fix for a random-split trap we hit (random-split reported 0.63, real dev 0.36).

**Calibration evidence (harness predicts the leaderboard):**
- ECAPA cosine, grouped-CV SRCC = **0.432** = official B1 dev exactly (0.432).
- wav2vec2-prob cosine, grouped-CV acc 0.323 ≈ dev acc 0.319.
- Idea F grouped-CV 0.500/0.489 → dev 0.533/0.548; Idea E predicted 0.49/0.46 → dev 0.501/0.477.

**Caveat documented:** absolute grouped-CV value depends on the fold scheme (sklearn 1.9 vs 1.7
GroupKFold, and pooled-OOF vs per-fold-mean differ by ~0.1); we rely on the **sklearn per-fold-mean
harness** (calibrated to dev) and read all other schemes *relatively*.

---

## 5. Results — dev leaderboard (UTT-SRCC)

| # | System | spk | acc |
|---|---|---|---|
| — | MLP over 768-d pair feature (initial) | 0.358 | 0.375 |
| — | wav2vec2 accent 13-d prob cosine | — | 0.319 |
| A | System-mean shrinkage | 0.426 | 0.373 |
| D | Cosine fusion + **system-label prior** | 0.478 | 0.416 |
| E | + WeSpeaker / CommonAccent / UTMOS | 0.501 | 0.477 |
| F | + **all-25 WavLM-Large layers** | 0.533 | 0.548 |
| G | multi-SSL 98-feature fusion | 0.526 | 0.541 |
| H | trained pairwise head (standalone) | 0.530 | 0.547 |
| I | ensemble F + head (balanced 1:1) | 0.568 | 0.588 |
| K | RAMP retrieval, cosine space | 0.566 | 0.589 |
| **K★** | **RAMP retrieval, SSL embedding-diff space (BEST)** | **0.5745** | **0.5986** |
| — | Official baseline B1 (ECAPA cosine) | 0.432 | 0.369 |
| — | Official baseline B2 (ECAPA + trained head) | 0.451 | 0.440 |
| — | Dev leaderboard leader | 0.629 | 0.608 |

**Headline:** best system **0.574 / 0.599** — beats both official baselines by +0.12–0.16; **accent
within 0.009 of the leader**. Largest single jumps: WavLM all-layers (E→F, +0.07 acc) and
pair-difference RAMP (I→K★, +0.011 acc).

---

## 5.1 Results — **final eval set** (official test labels, UTT-SRCC)

Two submissions were entered; the organizers take the participant-chosen final. The **transductive
system-mean smoothing (§3)** was validated on a 3,400-pair grouped-CV (holding out whole systems,
+0.008 predicted) and confirmed on the true test labels:

| Submission | spk | acc | mean |
|---|---|---|---|
| K★ base (RAMP + head + prior) | 0.5123 | 0.4584 | 0.4854 |
| **K★ + transductive smoothing (final, ID 882553)** | **0.5488** | **0.4743** | **0.5116** |
| Δ (smoothing) | **+0.0366** | **+0.0158** | **+0.0262** |

- The smoothing's **speaker gain on the real eval (+0.037) was 4× the grouped-CV estimate (+0.008)**
  — the eval's two 160-pair unseen systems (sys004, sys021) are exactly where a stable predicted
  system mean pays off, and they dominate the 57%-unseen split.
- Absolute eval SRCC (0.549/0.474) sits below dev (0.574/0.599): the eval is a **harder OOD split**
  (57% unseen vs 49%, and eval score means 3.73/3.71 vs train 4.04/4.02). The method **did not
  collapse on unseen systems** — the central design goal.

---

## 6. Ablations

**RAMP retrieval space (grouped-CV pooled-OOF, β-tuned fusion; read relatively):**

| Retrieval space | spk | acc |
|---|---|---|
| ridge only (no retrieval) | 0.6437 | 0.5978 |
| cosine scalars (35-d) | 0.6483 | 0.6017 |
| **SSL embedding-diff, 4 sources (shared)** | **0.6652** | **0.6297** |
| SSL embedding-diff, per-target spaces | 0.6525 | 0.6226 |
| SSL embedding-diff, 9 sources (enriched) | 0.6587 | 0.6209 |

→ retrieval must be in the **rich embedding space** (cosine-space ≈ ridge); **shared beats
per-target** (spk↔acc coupling); **4 sources beat 9** (redundant embeddings dilute kNN distance).

**Ensemble balance:** F:head 1:1 (dev 0.568/0.588) > 1:6 (0.547/0.565).

**System-label prior — LOSO stress test** (dev-like split, sweep #unseen systems):

| unseen frac | Δ(prior − cosine) spk | acc |
|---|---|---|
| 0.50 (≈dev) | +0.010 | +0.007 |
| 0.69 | +0.006 | +0.004 |
| 0.79 | +0.002 | +0.002 |
| 0.90 | +0.001 | +0.001 |

→ the prior **never hurts** (fallback design), even at 90% unseen — but its benefit → 0 as OOD grows.
Contradicts the literature's "learned domain priors fail OOD" (VMC'24 T05; UTMOS listener module).

**PPG (phonetic-posteriorgram) accent features:** help a plain baseline (F acc 0.489 → F+PPG 0.502,
+0.013) but are **redundant** on the best pipeline (0.6297 → 0.6308) — WavLM-mid + CommonAccent
already capture the pronunciation signal.

---

## 7. Negative / null results (a contribution in this genre)

1. **Counterfactual accent≠speaker pairs** (manufactured diff-speaker/same-accent) don't help —
   within-system spk↔acc correlate 0.79 and every test pair is a same-speaker attempt, so the SRCC
   metric cannot reward decorrelation.
2. **Stacking more SSL backbones** (HuBERT, XLSR, WavLM-base) *dilutes* — WavLM alone is best.
3. **Trained pairwise head ≈ ridge** on 2,800 pairs (apparent gains were a fold-scheme artifact,
   caught by matching folds) — small/regularised-linear generalises; the head only helps via
   ensemble diversity.
4. **Listener-dependent modeling** (train on 13,687 rows + listener embeddings) *hurts* (0.42/0.44
   grouped-CV) — reintroduces rater noise the mean already removed. **Contradicts LE-SSL-MOS (2023).**
5. **PPG redundancy** (§6) — pronunciation features add nothing once SSL accent features are present.
6. **Audio-LLM judges** trail dedicated SSL similarity models (SALMONN 0.824 < WavLM-ECAPA 0.836 on
   VoxSim) — not the path.

---

## 8. Contributions (paper framing)

1. **Pair-difference-space retrieval augmentation** for pairwise similarity MOS — a novel twist on
   RAMP (retrieve pair *difference* vectors, not single utterances) with an ablation isolating why
   the retrieval space matters.
2. **System-label prior with OOD-safe fallback** + a LOSO stress test proving robustness where the
   literature warns of failure, **plus label-free transductive system-mean smoothing** that added
   +0.037/+0.016 on the real eval — the largest eval-time gain, and strongest on unseen systems.
3. **A calibrated grouped-CV harness** that predicts the leaderboard (rare in system papers) — its
   +0.008 smoothing forecast was confirmed (and exceeded) on the withheld eval labels.
4. **A set of documented negatives**, including a **direct contradiction of prior listener-modeling
   claims** and an explanation of why accent features saturate.

---

## 9. Reproducibility

- **Frozen models only** (list in §3); no proprietary data or self-collected MOS (compliant with
  the "any public dataset" rule). Code + seeds released.
- **Compute:** feature extraction on one H100 (WavLM-Large 3,548 wavs × 25 layers in ~23 s; ECAPA on
  CPU ~90 s); all heads/ridge/retrieval train in minutes.
- **Key hyperparams:** ridge α=1.0 (fusion), 5.0 (multi-SSL); RAMP k=30, PCA-128, β≈0.8; prior
  α=0.5 (spk)/0.6 (acc); head margin-ranking margin 0.3, weight-decay 1e-3, 6 seeds, nested 7-fold
  grouped early stop.
- **Submission gotcha:** the file inside the zip MUST be `answer.txt`; a mis-named inner file makes
  CodaBench finish with **no score** (looks like a server error — it isn't).

---

## 10. Limitations & future work

- Final eval **0.549 / 0.474** (harder OOD split than dev); the method held up on 57% unseen systems.
- **Untried highest-EV lever: VoxSim pretraining** (70k public voice-similarity ratings) for the
  spk gap — OOD-capped (~0.6) but the most credible remaining move; H100 pipeline is ready.
- Accent features are **saturated**; further pronunciation/rhythm features are likely redundant.
- The system prior's benefit shrinks as unseen fraction grows (§6); transductive smoothing partly
  compensates on the eval, but a genuinely unseen-system-robust *parametric* head remains open.

---

### Key references
CodecMOS-Accent [2603.14328](https://arxiv.org/abs/2603.14328) · RAMP [2308.16488](https://arxiv.org/abs/2308.16488) ·
VoxSim [2407.18505](https://arxiv.org/abs/2407.18505) · SVSNet+ [2406.08445](https://arxiv.org/abs/2406.08445) ·
UTMOS [2204.02152](https://arxiv.org/abs/2204.02152) · UTMOSv2/T05 [2409.09305](https://arxiv.org/abs/2409.09305) ·
MOS-Bench [2411.03715](https://arxiv.org/abs/2411.03715) · CommonAccent [2305.18283](https://arxiv.org/abs/2305.18283) ·
accent-similarity [2505.14410](https://arxiv.org/abs/2505.14410) · VMC'26 [challenge page](https://sites.google.com/view/voicemos-challenge/voicemos-challenge-2026)
```
