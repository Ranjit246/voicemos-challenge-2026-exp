# Idea F — WavLM-Large Layer Fusion (H100)

Adds all 25 WavLM-Large transformer-layer cosine similarities to the Idea-E fusion. First GPU experiment (feature extraction on an H100).

## Result (calibrated Mac grouped-CV)

| Fusion | spk | acc |
|---|---|---|
| base (Idea E) | 0.474 | 0.455 |
| base + best-2 WavLM layers | 0.481 | 0.448 |
| **base + ALL 25 WavLM layers** | **0.501** | **0.489** |

Expected dev ~0.52 / ~0.51 (base's 0.474/0.455 grouped-CV → 0.501/0.477 real dev with prior). **Submitted: `submission_wavlm_fusion.zip`.**

## Key findings

1. **Layer probe confirms the SSL literature exactly.** Best speaker layer = **early (0–3)**; best accent layer = **middle (~12)**. Speaker identity lives in early WavLM layers, accent/phonetic info in the middle.
2. **Individual WavLM cosines are weak** — best spk layer 0.386, best acc layer 0.349, both below the specialized embeddings (TitaNet 0.439, CommonAccent 0.407). WavLM-Large is a general SSL model; purpose-trained speaker/accent embeddings beat its raw per-layer cosine.
3. **But all 25 layers together help.** Ridge combines many individually-weak layer views into complementary signal: +0.026 spk / +0.034 acc. Using only the "best 2" layers does *not* work — the gain comes from the full layer stack.
4. Grouped-CV (holds out whole systems) validated the improvement → it generalizes, not overfitting.

## Compute

- WavLM-Large extraction (3548 wavs × 25 layers, fp16): **23 seconds on H100**.
- Everything else (probe, fusion, submission) runs on the Mac calibrated harness.

## sklearn caveat

The VM's sklearn 1.7.2 gives different GroupKFold folds than the Mac's 1.9.0 — same features scored 0.543 (VM) vs 0.474 (Mac). The **Mac harness is the calibrated one** (it predicted B1 and Idea-E dev correctly), so all validation is done there; the VM is used only for GPU feature extraction.

## Files

```
idea-f-wavlm/
├── extract_wavlm.py        # WavLM-Large layer extraction (runs on VM/H100)
├── layer_probe_fusion.py   # VM-side probe+fusion (uncalibrated sklearn)
├── mac_probe.py            # CALIBRATED probe on Mac harness
├── make_submission.py      # base + all-25-WavLM + prior -> answer.txt
├── wavlm_layers.pt         # cached features (25, 1024) per wav [gitignored]
└── submission_wavlm_fusion.zip
```

## VM workflow

```bash
# on H100 VM (~/voicemos):
python3 code/extract_wavlm.py --wav_root data --out features/wavlm_layers.pt --batch_size 16
# sync wavlm_layers.pt back to Mac, then:
python3 make_submission.py       # calibrated, Mac-side
```
