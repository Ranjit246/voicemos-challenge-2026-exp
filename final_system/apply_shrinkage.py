"""Apply transductive system-mean smoothing (w=0.7) to the eval answer.txt (validated +0.007/+0.008)."""
import pandas as pd
W = 0.7
IN = "/Users/ranjitpatro/Home/Research/VoiceMOS/voicemos-challenge-2026-exp/final_system/submission/answer.txt"
OUT = "/Users/ranjitpatro/Home/Research/VoiceMOS/voicemos-challenge-2026-exp/final_system/submission_v2/answer.txt"
import os; os.makedirs(os.path.dirname(OUT), exist_ok=True)

d = pd.read_csv(IN)
for c in ["pred_spk_sim", "pred_acc_sim"]:
    m = d.groupby("system_id")[c].transform("mean")
    v = W * d[c] + (1 - W) * m
    d[c] = 1 + 4 * (v - v.min()) / (v.max() - v.min() + 1e-8)   # rescale to [1,5]
d = d[["system_id", "utterance_id", "wav_a_path", "wav_b_path", "pred_acc_sim", "pred_spk_sim"]]
d.to_csv(OUT, index=False)
print(f"Saved -> {OUT} ({len(d)} rows)")
print("spk uniq", d.pred_spk_sim.nunique(), "acc uniq", d.pred_acc_sim.nunique())
