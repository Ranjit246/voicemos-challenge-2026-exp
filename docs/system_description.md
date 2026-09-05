# The VoiceMOS Challenge 2026 — System Description and Participant Questionnaire
**Team T15**

---

## About your team

**Team ID**
T15

**Affiliation, country, academia or industry or other (individual, etc.)?**
Independent researcher (no institutional affiliation), India — other (individual).

**How many people were involved in your team?**
1 (solo participant).

**What is your team's background?**
Independent researcher working on speech processing — speaker and accent representation learning, speaker verification, and self-supervised speech models. This was my first project on MOS / perceptual-similarity prediction; I came to it from the speaker-embedding side rather than from a TTS or speech-enhancement background.

**Did you participate in the past VoiceMOS Challenges?**
No — this was my first VoiceMOS Challenge.

---

## About your system and components

**If any, please specify the name of your system**
PD-RAMP (Pair-Difference Retrieval-Augmented MOS Prediction).

**Is your system available commercially or as open source?**
Yes, open source: https://github.com/Ranjit246/voicemos-challenge-2026-exp

**If any, please specify components from the common system(s) or toolkit(s) used in your system.**
All backbones are public pretrained checkpoints used **frozen** (no fine-tuning):
- SpeechBrain — ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`), 192-d
- NVIDIA NeMo — TitaNet-Large, 192-d
- pyannote.audio — WeSpeaker ResNet34 (`pyannote/speaker-diarization-community-1`, embedding subfolder), 256-d
- CommonAccent-ECAPA (`Jzuluaga/accent-id-commonaccent_ecapa`) — 192-d embedding + 16-class posterior
- wav2vec2 accent classifier (`HamzaSidhu786/speech-accent-detection`) — 768-d hidden + 13-class posterior
- WavLM-Large (`microsoft/wavlm-large`, HuggingFace transformers) — all 25 hidden layers
- UTMOS (`tarepan/SpeechMOS`, utmos22_strong) — scalar quality score
- scikit-learn (ridge regression, PCA, GroupKFold), PyTorch

The retrieval component is inspired by **RAMP** (retrieval-augmented MOS prediction), but adapted to operate on *pair-difference* vectors rather than single utterances.

---

## Tracks

**Which tracks did you participate in?**
Track 3 only (speaker similarity + accent similarity).

**If you did not participate in certain tracks, why not?**
As a solo participant with limited time and self-funded compute, I chose to go deep on one track rather than spread thinly across three. Track 3 was also the closest match to my background in speaker/accent representations.

---

## About the use of the data

**What data did you use to train your system?**
Only the challenge-provided Track 3 data — no external MOS or similarity datasets.
- **Training set:** 2,800 pairs from 21 systems (13,687 listener-wise ratings, mean 4.9 raters/pair). I used the **utterance-averaged** scores.
- **Development set:** used first for validation; for the final evaluation submission I retrained on **train + dev = 3,400 labeled pairs** (23 systems) to maximize the datastore and training signal.
- **Pre-training:** none of my own — the public checkpoints listed above provide all pretrained representation learning, and they were kept **entirely frozen**. Only a small head, a ridge regressor, and a kNN datastore were fit on challenge data.

**Please specify any pre-processing... Also, describe the input features.**
Audio was already 16 kHz; I loaded all waveforms as 16 kHz mono (librosa) and otherwise applied each model's own feature extractor. No additional volume normalization, trimming, or augmentation.

Input features are **frozen-model embeddings and SSL features**, not raw waveform into a trained network:
- WavLM-Large: all 25 hidden layers, mean-pooled over time (25 x 1024)
- ECAPA-TDNN (192), TitaNet-L (192), WeSpeaker (256), CommonAccent-ECAPA (192 + 16-class posterior)
- wav2vec2-accent (768 last-hidden mean + 13-class posterior)
- UTMOS scalar for wav_a and wav_b
These are turned into (a) ~35 **pair cosine-similarity scalars** and (b) **L2-normalized embedding-difference vectors** for retrieval.

**During training, did you use averaged scores per utterance, or individual per-listener scores? Did you use any listener info?**
I used **utterance-averaged** scores, and **no listener information** in the final system.

I did explicitly try the per-listener approach: training on all 13,687 listener-wise rows with learned listener embeddings and a "mean listener" at inference (the UTMOS-style trick). It **consistently hurt** — roughly 0.42 / 0.44 grouped-CV SRCC versus 0.50+ for the averaged-label model. My interpretation is that with only ~4.9 ratings per pair, per-listener training reintroduces exactly the rater noise that averaging removes. This is a negative result that contradicts prior listener-modeling claims (e.g., LE-SSL-MOS), and I think it is worth reporting.

---

## System architecture and training/inference process

**Please describe the architecture of your system.**

```
                per-utterance frozen embeddings / scores
  +-------------------------------------------------------------+
  | ECAPA-TDNN . TitaNet . WeSpeaker . CommonAccent-ECAPA        |  speaker + accent
  | wav2vec2-accent (hidden+prob) . WavLM-Large (25 layers) . UTMOS |
  +-------------------------------------------------------------+
        | cosine(wav_a, wav_b) per source     | pair-difference vectors
        v                                     v
   [~35 scalar features]              [ECAPA, CommonAccent, WavLM-L3, WavLM-L12]
        |                              L2-normed diffs -> concat(2432) -> PCA-128
   Ridge (parametric)                        |
        |                               kNN retrieval (k=30, softmax(-d / mean-d))
        +------ RAMP fuse: 0.8*z(ridge) + 0.2*z(retrieval) ------+
                                                                 |
   trained WavLM pairwise head ---- z-average ensemble (1:1) ----+
   (learnable per-task layer weights + pair interactions          |
    + margin-ranking loss, 6-seed average)                        v
                                    system-label prior blend: a*z(param) + (1-a)*z(prior)
                                    (a = 0.5 spk / 0.6 acc; unseen system -> cosine fallback)
                                                                 |
                            transductive system-mean smoothing: 0.7*p + 0.3*mean_sys(p)
                                                                 |
                                                    scale to [1,5] -> answer.txt
```

The three ideas that make it work:
1. **Pair-difference-space retrieval.** Instead of retrieving similar *utterances*, I retrieve similar *pairs* by their embedding-**difference** vectors (ECAPA, CommonAccent, WavLM layer 3 for speaker, WavLM layer 12 for accent), standardized and reduced to PCA-128, with k=30 neighbours weighted by softmax(-distance/mean-distance). Retrieval in this rich space clearly beat retrieval over the compact cosine scalars.
2. **System-label prior with an OOD-safe fallback.** Predictions are blended toward the training-label mean of the pair's system; for systems unseen in training the model falls back to the plain cosine-based prediction, so the prior can never hurt on OOD systems (verified with a leave-one-system-out stress test up to 90% unseen).
3. **Transductive system-mean smoothing at inference.** Each final score is pulled toward its own system's *predicted* mean over the evaluation set (label-free). This was validated on grouped CV (+0.008) and turned out to be the single largest gain on the real evaluation set: **+0.037 SRCC (speaker) and +0.016 (accent)**.

**Please describe any specific training strategies you would like to share.**
- **Grouped cross-validation by system** (7 folds, holding out whole systems) was my only model-selection criterion, since both dev and eval contain systems unseen in training. This was essential: a naive random split reported 0.63 while the true dev score was 0.36. The harness was calibrated against the leaderboard — e.g., ECAPA cosine scored 0.432 grouped-CV, exactly matching the official B1 dev score of 0.432.
- Ridge regression with alpha = 1.0 on standardized features.
- Pairwise head: Adam, lr 1e-3, weight decay 1e-3, dropout 0.3, loss = MSE + MarginRankingLoss (margin 0.3) with **batches sampled across systems** so the ranking objective cannot be satisfied by system identity alone; **6 seeds averaged**; nested grouped-CV early stopping.
- Retrieval: k = 30, PCA-128, fusion weight beta = 0.8 (speaker) / 0.7 (accent).
- No data augmentation, no backbone fine-tuning (small data — frozen generalizes better).

**Please describe the training and inference processes.**
Given a pair (wav_a = codec/TTS output, wav_b = natural VCTK reference of the target speaker, different sentence):
1. Extract frozen embeddings/scores for both waveforms from all seven models (one pass, cached).
2. Build (a) ~35 pair cosine features plus UTMOS(a), UTMOS(b) and their difference, and (b) the concatenated L2-normalized embedding-difference vector.
3. Predict with ridge regression on (a); separately retrieve k=30 nearest training pairs in the PCA-128 difference space of (b) and take their distance-weighted mean label.
4. Fuse the two in z-space (beta = 0.8 / 0.7), then z-average 1:1 with the trained 6-seed WavLM pairwise head.
5. Blend toward the system-label prior (alpha = 0.5 speaker / 0.6 accent); unseen systems fall back to the cosine prediction.
6. Apply transductive system-mean smoothing (0.7 * p + 0.3 * system mean) across the evaluation set.
7. Rescale to [1, 5] to produce the final `pred_spk_sim` and `pred_acc_sim`.

---

## About the computational cost

**Number of trainable parameters**
**164,212** trainable parameters per head model (~0.99 M across the 6-seed ensemble). All backbones are **frozen** (~1.2 B frozen parameters in total across the seven models). The ridge regressor and the kNN datastore are essentially non-parametric.

**How many CPU/GPU hours did it take to build/train your model?**
Roughly **1-2 GPU-hours in total** for the one-pass feature extraction over all waveforms (WavLM-Large over 3,548 training/dev wavs takes about 23 s on an H100), plus about 10 minutes to train the 6-seed head. The ridge regression and kNN retrieval fit in seconds on CPU. Total build cost was well under 5 GPU-hours — most of my time went into validation methodology, not compute.

**How many GPUs did you use?**
One at a time: a single NVIDIA H100 for the training-phase extraction and a single NVIDIA L4 for the evaluation-phase extraction (both rented), plus an Apple Silicon Mac (MPS/CPU) for the TitaNet and WeSpeaker extractions.

**What is the memory footprint of the system at runtime?**
About **6-8 GB of GPU memory** at peak during WavLM-Large fp16 feature extraction (batch size 4). Once features are cached, the predictor itself (ridge + kNN datastore + six small heads) runs in **under 1 GB and needs no GPU**.

---

## Paper writing

**Are you planning to submit a paper to the challenge's associated conference?**
No.

**If not, what are the reasons... Will you submit one elsewhere?**
Since the VoiceMOS special-session proposal was not accepted, I am no longer targeting that venue. As an independent participant I would prefer a venue where this kind of system-plus-negative-results paper fits well, so I plan to submit to **another conference (e.g., Interspeech 2027) and/or release an arXiv preprint** summarizing the system, the ablations, and the negative results. *[CONFIRM your preferred venue.]*

---

## Your opinions

**Please share your general impression of the challenge.**
Very positive. The task was clearly specified, the data and baselines were well prepared, and the dev leaderboard made it easy to check that my local validation was trustworthy. As a solo participant with no institutional backing, I appreciated that the challenge was genuinely accessible — no fees, public data, and a level playing field.

**What do you consider to be the positive points of participating?**
- The **unseen-system structure** of the dev and eval sets is an excellent design choice: it forces honest out-of-distribution validation and punishes system-identity memorization. It is what made the problem scientifically interesting rather than a leaderboard-fitting exercise.
- Having **two correlated sub-scores** (speaker and accent similarity) is a genuinely interesting modelling problem.
- Strong official baselines with public code gave a clear, fair reference point.

**What do you consider to be the negative points?**
- The window between the evaluation-set release (July 31) and the submission deadline (August 7) was **very tight** — about one week to extract features and produce a submission. For participants without a standing GPU cluster this was the hardest constraint of the challenge.
- A **CodaBench packaging gotcha** cost me several submissions: the file inside the zip must be named exactly `answer.txt`. A mis-named inner file makes the submission "finish with no score", with no error message explaining why. This is invisible to the participant and looks like a platform failure.
- It was also easy to submit to the **wrong phase** (development vs evaluation), and the resulting failure message was equally opaque.
- During the evaluation phase all scores display as 0.00, which is understandable (withheld scores) but initially indistinguishable from a genuine scoring failure. A short note in the instructions would have saved a lot of anxiety.

**Please provide your impression about the characteristics and quality of the tracks/datasets.**
The dataset is well constructed and the annotation documentation is good. Two intrinsic characteristics dominate the difficulty:
- **Rater noise is high** — within-pair listener standard deviation is about 0.85-0.92 on a 5-point scale with only ~4.9 ratings per pair. My split-half estimate puts the utterance-level ceiling around 0.81 (speaker) / 0.76 (accent), so utterance-level SRCC is intrinsically capped well below 1.0. This is worth stating explicitly to participants.
- **spk_sim and acc_sim correlate ~0.86 at utterance level** (0.79 even within a system), so the two sub-scores are hard to separate. Every training pair is a same-speaker attempt, so the data never shows "different speaker, same accent" — the accent head has no way to learn that accent is independent of speaker identity. I tried manufacturing such counterfactual pairs and it did not help, because the metric cannot reward decorrelation.
- 2,800 training pairs is small; more ratings per pair would raise the ceiling more than more pairs would.

**Which component or contribution do you consider to be your strongest?**
Two things:
1. **Retrieval augmentation in pair-difference space** — a twist on RAMP where the retrieval key is the *difference* between the two utterances' embeddings rather than a single utterance. An ablation showed this matters a lot: retrieval over compact cosine scalars performed the same as ridge alone (0.648 vs 0.644), while retrieval in the SSL embedding-difference space reached 0.665 / 0.630 grouped-CV.
2. **Transductive system-mean smoothing** — label-free, applied at inference, validated at +0.008 on grouped CV and worth **+0.037 speaker / +0.016 accent** on the real evaluation set. It is what lifted my system-level SRCC to 0.942 / 0.902 (3rd place at system level in both sub-scores).

I would also highlight the **calibrated grouped-CV harness** as a methodological contribution: it predicted the leaderboard closely enough that I could make decisions without spending submissions.

**Which is the weaker component or task of your system?**
- **Absolute calibration.** MSE was by far my worst metric (0.702 speaker / 0.773 accent, ranked 7th) because I optimized purely for rank correlation and min-max rescaled the output to [1,5]. The ranking metric did not penalize this, but the system is not usable as an absolute MOS predictor as-is.
- **Accent similarity at utterance level** (0.474) lagged speaker similarity (0.549). My accent features saturated early — adding phonetic-posteriorgram features gave no gain on the full pipeline, suggesting WavLM mid-layers plus CommonAccent already capture the available pronunciation signal.
- The trained pairwise head added little over ridge on 2,800 pairs; its value was almost entirely ensemble diversity.

**Please tell us about your future direction.**
- **VoxSim pretraining** (70k public human voice-similarity ratings) for the speaker sub-score — the most credible remaining lever for the utterance-level gap, which I did not have time to attempt.
- Producing **well-calibrated absolute scores** alongside good rankings, so the model is useful outside a rank-correlation benchmark.
- Better handling of unseen systems in the *parametric* model, rather than relying on priors and transductive smoothing to compensate.

**Please share your experience with CodaBench.**
Overall workable, and I had no trouble with account setup or the general submission flow. Two concrete pain points, both about **silent failures**:
1. The inner zip file must be named exactly `answer.txt`; any other name produces "finished with no score" and no diagnostic. I lost several submissions before identifying this.
2. Submitting to the wrong phase produces an equally opaque "child task failed / non-zero return code" message.
A clearer error message (or a format pre-check on upload) would fix both. I do not have a better platform to recommend — CodaBench itself was fine once these two traps were known, and I would suggest simply documenting them prominently in the participant instructions.

---

## Considering the next AudioMOS Challenge

**When should the next AudioMOS Challenge be?**
Next year — an annual cadence works well, and it keeps momentum in the community.

**Do you have any suggestions on new tasks to consider?**
- **Calibrated MOS prediction**: score systems on both rank correlation *and* absolute error, so participants cannot ignore calibration entirely (as I did).
- **Explainable / diagnostic quality assessment**: predict *why* a sample is rated low (e.g., speaker-identity loss vs. artefacts vs. prosody), not just a scalar.
- **Decorrelating attributes**: a track that explicitly includes different-speaker/same-accent pairs, so accent similarity can be evaluated independently of speaker identity. In the current data these two are inseparable.
- Cross-lingual or code-switched speaker/accent similarity.

**Do you have suggestions regarding the rules?**
Nothing major — the rules were clear and fair. Two small suggestions: (1) document the submission packaging requirements (the `answer.txt` naming) very prominently, and (2) if possible allow a slightly longer window between the evaluation-set release and the deadline, which would help participants without dedicated compute clusters.

**Would you be interested in organizing it?**
I would be glad to help in a supporting role (for example with baselines, evaluation tooling, or documentation). *[CONFIRM — adjust to your preference.]*

**Do you have any large-scale listening test data that you might be able to share?**
No, I do not have listening-test data to share.

---

*Contact: Ranjit Patro — ranjitpatro100@gmail.com — https://github.com/Ranjit246/voicemos-challenge-2026-exp*
