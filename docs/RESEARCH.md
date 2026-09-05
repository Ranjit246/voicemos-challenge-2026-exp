# VoiceMOS 2026 Track 3 — Deep Research & Next-Lever Analysis

Consolidated from a 5-agent literature sweep (VoxSim/similarity data, SVSNet+/co-attention,
data augmentation, 2024–26 SSQA SOTA, OOD generalization) + our own experiments and a LOSO
stress test. All claims cited.

## TL;DR

- **0.63/0.63 (the dev leader) is near the practical ceiling.** Utterance-level SRCC on hard/OOD
  speech tops out ~0.6: the VMC'24 winner UTMOSv2 sat at 0.58–0.61 on hard Track-1; VoxSim's best
  model collapses to 0.57–0.61 OOD. We're at **0.568/0.588 — genuinely close.**
- **The leader is almost certainly the same family as us** (WavLM/ECAPA + trained head). The
  headroom is OOD-robustness, embedding choice, and tuning — not a paradigm we're missing.
- **Our system-prior is SAFE** (LOSO stress test: never negative, even at 90% unseen) — but its help
  → 0 as OOD grows, so eval rests on generalizable features.

## Ranked next-big-levers (evidence-weighted)

| # | Lever | Targets | Evidence | Realistic gain | Effort |
|---|---|---|---|---|---|
| 1 | **RAMP retrieval head** (kNN datastore + confidence fusion) | unseen systems = the eval axis | +0.23 SRCC OOD (weak baseline); won VMC'24; public code | +0.01–0.05 on our strong base | cheap (H100) |
| 2 | **VoxSim pretrain + WavLM-ECAPA backbone** | spk mapping calibration | same task shape (pair→sim); +2.5–5% LCC transfer | +0.03–0.06, but OOD-capped ~0.6 | 1–2 days |
| 3 | **Codec-degradation ladders** (Encodec/Opus bitrate → known-order pairs, ranking loss) | spk, unseen-codec generalization | SESQA cut error 36% under label scarcity; matches codec-based Track 3 | +0.02–0.04 | cheap |
| 4 | **Cheap feature adds**: attentive-stats (mean+**std**) pooling on WavLM; SpeechBERTScore frame-matching; **PPG/formant distances for accent** | both; accent esp. | std is where ASV pooling gains come from; generic SSL weak for accent | +0.01–0.03 each | cheap |
| 5 | **Pseudo-label dev/eval** (BApMOS balanced selection) | OOD | zero-shot BC2019 SRCC 0.559→0.686 | +0.01–0.03 | medium |

### Avoid (proven weak for THIS metric)
- **Frame-level co-attention** — lost to WavLM-ECAPA cosine on VoxSim (0.742 vs 0.836); SVSNet+ moved
  system-SRCC a lot but utterance-SRCC 0.572→0.577. Won't move our metric.
- **Audio-LLM judges** — SALMONN 0.824 < WavLM-ECAPA 0.836 on VoxSim similarity; novelty angle only.
- **Monotonic score calibration / z-scoring** — mathematically cannot change utterance-SRCC (rank metric).
- **Speed/pitch augmentation** — alters identity/accent → counterproductive for *similarity* (use only as
  paired consistency regularizer).
- **Test-time adaptation (TENT/BN)** — no MOS track record; ill-posed for regression.

## What the leader is plausibly doing
A well-tuned SV-pretrained WavLM/ECAPA embedding + light trained head, likely with **retrieval
augmentation** and **training-data diversity** for OOD — not a new paradigm. Reachable via levers 1–4.

## Recommended plan for the Aug 7 eval set
1. **RAMP retrieval head** over our existing features — highest-EV, targets the OOD axis that decides the ranking.
2. **Codec-degradation ladders + std-pooling + SpeechBERTScore** — cheap diverse members / OOD robustness.
3. **VoxSim pretrain** (parallel bigger bet for spk).
4. Keep the system-prior (safe per LOSO) but don't rely on it for OOD.

## Competition / paper context
- Eval set released **Jul 31**, submit by **Aug 7**, results Aug 31. Paper deadline **mid-Sep**.
- Venue: a special session / satellite workshop was planned but not accepted; system-description paper **mandatory**.
- **Any public dataset allowed** (VoxSim, MOS-Bench, etc.); no proprietary/self-collected MOS.
- **Non-winning teams routinely publish** (DDOS, Nguyen, UWB-NTIS, Chinen analysis paper). Bar = beat
  baselines + document learnable content. We beat B2 by +0.12–0.15 → clear the bar decisively.
- Our publishable assets: the **system-label prior** (+ its LOSO robustness analysis), the **calibrated
  grouped-CV harness that predicts the leaderboard**, and the **negative results** (counterfactuals,
  multi-SSL stacking, trained-head-vs-ridge, listener modeling — the last contradicts LE-SSL-MOS 2023).

## Key sources
RAMP [2308.16488](https://arxiv.org/abs/2308.16488) · MOS-Bench [2411.03715](https://arxiv.org/html/2411.03715v2) ·
VMC'24 [2409.07001](https://arxiv.org/html/2409.07001v1) · UTMOSv2/T05 [2409.09305](https://arxiv.org/html/2409.09305v1) ·
VoxSim [2407.18505](https://arxiv.org/html/2407.18505v1) · SVSNet+ [2406.08445](https://arxiv.org/html/2406.08445v1) ·
SpeechBERTScore [2401.16812](https://arxiv.org/html/2401.16812v2) · SESQA [2010.00368](https://ar5iv.labs.arxiv.org/abs/2010.00368) ·
Partial-Rank pseudo-label [2310.05078](https://ar5iv.labs.arxiv.org/abs/2310.05078) ·
accent-similarity [2505.14410](https://arxiv.org/abs/2505.14410) · UTMOS [2204.02152](https://arxiv.org/abs/2204.02152) ·
DDOS [2204.03219](https://arxiv.org/abs/2204.03219) · Kunesova OOD lessons [ISCA 2024](https://www.isca-archive.org/interspeech_2024/kunesova24_interspeech.html) ·
CodecMOS-Accent [2603.14328](https://arxiv.org/html/2603.14328v1) · VMC'26 [challenge page](https://sites.google.com/view/voicemos-challenge/voicemos-challenge-2026)
