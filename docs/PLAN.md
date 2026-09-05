# VoiceMOS 2026 Track 3 — Research Plan

## Context

We predict utterance-level speaker similarity (`spk_sim`) and accent similarity (`acc_sim`) for pairs (synthesized `wav_a`, natural VCTK reference `wav_b` — a *different sentence* by the target speaker, 3–7 s, 16 kHz). Leaderboard metric: UTT-SRCC on a 600-pair dev set containing **2 systems unseen in training** (`sys003`, `sys015`). Compute: Mac (MPS/CPU) + **CUDA GPU available** for heavier training. CodaBench submissions treated as **scarce**.

### Where we stand vs. what we must beat

| System | spk UTT-SRCC | acc UTT-SRCC |
|---|---|---|
| Our TitaNet MLP (768-d pair feature) | 0.358 | 0.375 |
| Our wav2vec2 accent 13-d prob cosine | — | 0.319 |
| **Official B1**: zero-shot ECAPA cosine (`speechbrain/spkrec-ecapa-voxceleb`) | 0.432 | 0.369 |
| **Official B2**: ECAPA + trained projection head | **0.451** | **0.440** |

Official baselines: https://github.com/voicemos-challenge/vmc2026-baselines (`track3/`). B2 recipe: AdamW lr 1e-3, bs 16, 20k steps, range clipping, repetitive padding. **We are below the zero-shot baseline** — the MLP learned system identity, not similarity. Target: beat 0.451 / 0.440.

### Key facts from research (organizers' paper arXiv:2603.14328, read in full + literature survey)

1. **spk_sim ↔ acc_sim correlate 0.75–0.86 at utterance level** (0.97 system-level) → one joint multi-task model is the right shape; features carry both targets.
2. **Speaker-embedding cosine predicts accent similarity better than accent-embedding cosine** (paper Table 2: 0.34 vs 0.31 utt-level Pearson) → strong speaker features are the foundation; accent features are a complement, not a replacement.
3. **Individual ratings are very noisy**: within-pair listener std ≈ 0.85–0.92 (5-pt scale), ~4.9 ratings/pair; inter-listener agreement lowest for Welsh/Irish/NZ accents → UTT-SRCC ceiling is well below 1.0; gains of +0.05 may decide the track. De-noising (listener modeling) > model capacity.
4. **Listener same-accent bias is statistically significant** (p<0.001) → listener-dependent training (we have `listener_id`; each listener rated 255–573 pairs) is justified — the UTMOS "mean listener" trick.
5. **Small/frozen generalizes; big/fine-tuned overfits** on this data scale: VMC'23 saw a 768-parameter SVM place top-3; SVSNet+ found fine-tuning WavLM *hurt* vs frozen + learnable layer-weighted sum.
6. **Layer probing**: speaker cues live in early SSL layers (~1–7), accent-phonetic cues around layer ~9 → per-task layer-weighted sums over one frozen backbone.
7. **VoxSim** (public, 70k human voice-similarity ratings on VoxCeleb pairs, different-content, https://mm.kaist.ac.kr/projects/voxsim): trained similarity heads beat raw embedding cosine (SRCC 0.836 vs ~0.75), and VoxSim-pretraining transfers → the strongest lead for spk_sim.
8. **UTMOS contrastive loss** (margin 0.5, weight 0.5 vs MSE 1.0) trains rank order directly — matches the SRCC metric. Batches must be sampled **across systems** so ranking can't be satisfied by system ID.
9. **RAMP** (retrieval-augmented kNN head, VMC'24 Track-1 winner, code: https://github.com/NKU-HLT/RAMP_MOS) is designed for data-scarce OOD robustness — cheap blend option.
10. **Vox-Profile Whisper accent classifiers** (`tiantiaf/whisper-large-v3-narrow-accent`) have a near-exact VCTK accent inventory (Scottish, Irish, NI, Welsh, English, NA, Oceania, SA) — best accent posterior source; CommonAccent ECAPA (`Jzuluaga/accent-id-commonaccent_ecapa`, the paper's own O-ACC-SIM) is the 16 kHz-native lighter option.

---

## Plan (phased; each phase gated by the validation harness)

### Phase 0 — Honest validation harness (build first; everything is gated on it)

New `voicemos-challenge-2026-exp/eval_harness.py`:
- **Grouped CV by system**: ~7 folds × 3 held-out systems (129–136 pairs each → ~400-pair folds, close to dev's 600). Report mean±std UTT-SRCC via the official formula (`metrics_voicemos.py`).
- Report random-split SRCC alongside — the gap is the "system-memorization gauge" for every model.
- Reuse: `experiments/titanet_large/train_avg.csv` (mean labels), existing embedding dicts + shared key format.
- **Rule: nothing is submitted unless grouped-CV SRCC beats the best previous submission.** (Submissions are scarce.)

### Phase 1 — Replicate official baselines + honest cosine leaderboard (~1 day, no training)

1. Clone `vmc2026-baselines`, run B1 (ECAPA cosine) locally; verify our harness's grouped-CV number is consistent with their reported dev 0.432/0.369 → calibrates harness↔leaderboard trust.
2. Reproduce **B2** (ECAPA + projection head) exactly — it's the number to beat and its training details (range clipping, repetitive padding, small head) are the anti-overfit recipe our MLP lacked.
3. Extract remaining embedding sets (same key format, all 3,548 wavs):
   - ECAPA `speechbrain/spkrec-ecapa-voxceleb` (pattern exists in `exp-asv/infer_all.py`)
   - CommonAccent ECAPA embeddings + 16-class posteriors
   - UTMOS scalar per wav: `torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong")`
   - (GPU, one pass) Vox-Profile Whisper narrow-accent posteriors + penultimate embeddings
4. Score **every cosine individually** on the harness → single-feature leaderboard (ECAPA, TitaNet, WeSpeaker, wav2vec2-hidden, CommonAccent, Whisper-accent).

### Phase 2 — Scalar-feature fusion (quick win; likely first submission)

~15 scalar features per pair — scalars physically can't memorize system identity:
- Cosines: ECAPA, TitaNet, WeSpeaker, wav2vec2-hidden, CommonAccent-emb, Whisper-accent-emb
- Posterior features: accent-posterior cosine + KL + top-class-match + entropy(wav_a)
- Quality: UTMOS(wav_a), UTMOS(wav_b), difference
- Duration difference

Models in order: **ridge regression** (rank-normalized features) → HistGradientBoosting (depth≤3) → tiny MLP (15→16→2) only if it wins on grouped CV. Two heads (spk, acc) + also test one shared score (corr 0.86 may make it equivalent). **Ensemble fold models** for the dev prediction.

### Phase 3 — SVSNet+-style learned model on GPU (main bet for the win)

Architecture (per SVSNet+ findings, reimplement — no official code):
- **Frozen WavLM-Large** → two **per-task learnable layer-weighted sums** (spk head → early layers; acc head → mid layers) → linear to 256-d
- **Bidirectional co-attention** between wav_a/wav_b frame sequences (handles different sentence content) → mean-pool → small regression + classification heads
- **Loss**: MSE + UTMOS contrastive (margin 0.5, weight 0.5) + optional soft-rank SROCC loss; batches sampled across systems
- **Listener-dependent training** on all 13,687 listener-wise rows with listener embedding; "mean listener" at inference
- **Pretraining**: first train the spk head on **VoxSim** (70k pairs), then fine-tune (head-only) on challenge data — the single most promising transfer per the literature
- Cache frozen WavLM features once (GPU); head training is then cheap and repeatable
- Early stopping on grouped-CV only; fold-ensemble for submission

Optional robustness add-on: **RAMP-style kNN blend** — retrieve k nearest training pairs in feature space, blend parametric + retrieved scores with confidence weighting.

### Phase 4 — Accent-specific extras (only if acc_sim lags after Phase 3)

- Layer-wise cosine probe on frozen WavLM (which single layer's cosine best predicts acc_sim on the harness) → add as fusion feature
- Whisper accent posterior distances already in Phase 1/2; try posterior-CE-style features

### Submission strategy (scarce budget)

1. Phase 2 fusion (only if grouped-CV clearly beats 0.451/0.440-equivalent locally… or beats our own 0.358/0.375 with harness-calibrated margin)
2. Phase 3 model after fold-ensemble
3. Final: best-of ensemble (Phase 2 + Phase 3 average)
- Keep per-submission `answer.txt` + generating script in `submissions/`; log all results in root README table.

### Verification

- Every change: `eval_harness.py` → grouped-CV UTT-SRCC (mean±std) + random-split gauge
- Phase 1 calibration: harness number for B1 should track its official dev 0.432/0.369
- Pre-zip checks: 600 rows, exact header, scores clamped to [1,5]

## Key files

| File | Role |
|---|---|
| `voicemos-challenge-2026-exp/eval_harness.py` | NEW — grouped-CV gate |
| `voicemos-challenge-2026-exp/baselines/` | NEW — clone + run official B1/B2 |
| `voicemos-challenge-2026-exp/ecapa/`, `commonaccent/`, `utmos/`, `whisper_accent/` | NEW extractors (mirror `experiments/pyannote/infer_embed.py` pattern) |
| `voicemos-challenge-2026-exp/fusion/` | NEW — Phase 2 feature table + ridge/GBM + submission writer |
| `voicemos-challenge-2026-exp/svsnetplus/` | NEW — Phase 3 model (GPU) |
| `experiments/titanet_large/train_avg.csv`, existing `.pt` dicts, `metrics_voicemos.py` | reuse |

---

## Next-level ideas (v3) — beyond the phased plan: one hacky, two innovative

These are NOT from the literature — they exploit structural quirks of *this specific challenge* that nobody's baseline uses.

### Idea A (the hack) — System-mean shrinkage: smooth predictions using dev's own structure

**The insight, simply:** `dev.csv` shows us the `system_id` of every row. We know from the paper that system-level signal is far easier than utterance-level (ECAPA cosine: SYS-SRCC 0.86 vs UTT 0.39). Individual utterance scores are mostly noise around the system mean (listener std ±0.9!). So: don't trust any single utterance prediction — pull it toward the average prediction of its own system, computed **on the dev set itself, no labels needed**.

```
final_score(pair) = α · model_score(pair) + (1−α) · mean(model_score of all dev pairs from the same system)
```

- Dev has ~12–16 pairs per system → system means are ~4× less noisy than single predictions.
- Tune α on the grouped-CV harness (likely α ≈ 0.3–0.5 wins).
- **Works on top of ANY predictor we already have — even the plain ECAPA cosine — ~30 lines of code, could improve a submission today.**
- Why it's legit: it's transductive smoothing, uses zero labels; the same trick applies to the final eval set.
- Risk: if a system has genuinely bimodal quality, shrinkage blurs it — the harness will tell us.

### Idea B (innovative) — Counterfactual pseudo-pairs: manufacture the labels that don't exist

**The problem it solves:** spk_sim and acc_sim correlate 0.86 in training data — the model can never learn what "same accent but DIFFERENT speaker" looks like, because no training pair shows it. That's exactly where acc_sim predictions have no signal of their own.

**The insight, simply:** we have ~148 natural VCTK reference wavs (sys019) spanning 32 speakers and 10 accents. We can build **new pairs with near-free labels**:

| Manufactured pair | spk_sim label | acc_sim label |
|---|---|---|
| Same speaker, two different natural utterances | ~4.8 (the GT rating from the paper) | ~4.7 |
| **Different speaker, SAME accent** | low (~1.5) | **high (~4)** ← the decorrelating case! |
| Different speaker, different accent | low (~1.5) | low (~2) |

- Group the sys019 wavs by speaker: cluster their WeSpeaker embeddings (`experiments/pyannote/embeddings.pt` — speaker clustering is exactly what these are for). Get accent labels per cluster via CommonAccent/Whisper-accent posteriors (majority vote per cluster).
- Add a few hundred such pairs to training with soft labels (or use them only in a ranking loss: "this pair must score below that pair on spk but above it on acc").
- **This is the only way the acc head can learn accent ≠ speaker** — the real data never shows it. It should specifically raise acc_sim SRCC, our weakest number.

### Idea C (innovative) — Degradation ladders: synthesize unlimited "unseen systems" with known rank order

**The problem it solves:** we fail on unseen systems. Two dev systems weren't in training, and the eval set will bring more.

**The insight, simply:** a "system" is just a codec/TTS that damages the reference speaker's identity by some amount. We can create brand-new fake systems by re-encoding natural VCTK wavs through public codecs at descending bitrates (Encodec 24→12→6→3→1.5 kbps is `pip install encodec`; plus Opus/GSM/mp3 via ffmpeg). We don't know what MOS a human would give — **but we know the ORDER**: more compression ⇒ lower similarity to the reference. And SRCC only cares about order.

- Build "ladders": (reference, degraded@high-bitrate) > (reference, degraded@mid) > (reference, degraded@low) …
- Train with a margin-ranking loss on ladder pairs (no absolute labels needed) mixed into the Phase 3 loss.
- Effect: the model learns a *smooth, monotonic sensitivity to codec damage* covering system types it has never seen — exactly the generalization axis where the dev/eval sets hurt us.
- Bonus: each ladder is a fresh "system", so grouped-CV stops being data-starved.

### Suggested order

1. **Idea A first** — it's an afternoon, it stacks on everything, and it can improve even the current cosine baseline submission.
2. **Idea B** next — a day of work (clustering + label table + rank constraints), directly targets acc_sim.
3. **Idea C** with Phase 3 — it's a training-loss addition, best combined with the SVSNet+-style model on GPU.

## References

Official baselines: github.com/voicemos-challenge/vmc2026-baselines · CodecMOS-Accent arXiv:2603.14328 · SVSNet+ arXiv:2406.08445 · VoxSim arXiv:2407.18505 (data: mm.kaist.ac.kr/projects/voxsim) · UTMOS arXiv:2204.02152 · RAMP arXiv:2308.16488 · CommonAccent arXiv:2305.18283 · Vox-Profile arXiv:2505.14648 · VMC'23 OOD lessons arXiv:2310.02640 · SHEET toolkit github.com/unilight/sheet