"""
Compute utterance-level mean MOS scores from listener-wise train.csv.
Output: train_avg.csv with columns:
  system_id, utterance_id, wav_a_path, wav_b_path, spk_sim, acc_sim, n_listeners
"""

import pandas as pd

TRAIN_CSV = "/Users/ranjitpatro/Home/Research/VoiceMOS/dataset/vmc2026_track3_train_phase_distro_v3_syn/sets/train.csv"
OUT_CSV   = "/Users/ranjitpatro/Home/Research/VoiceMOS/voicemos-challenge-2026-exp/titanet_large/train_avg.csv"

df = pd.read_csv(TRAIN_CSV)

agg = (
    df.groupby(["system_id", "utterance_id", "wav_a_path", "wav_b_path"], sort=False)
    .agg(
        spk_sim=("spk_sim", "mean"),
        acc_sim=("acc_sim", "mean"),
        n_listeners=("listener_id", "count"),
    )
    .reset_index()
)

agg.to_csv(OUT_CSV, index=False)

print(f"Pairs:           {len(agg)}")
print(f"Listeners/pair:  {agg['n_listeners'].value_counts().sort_index().to_dict()}")
print(f"spk_sim  mean={agg['spk_sim'].mean():.3f}  std={agg['spk_sim'].std():.3f}  min={agg['spk_sim'].min():.1f}  max={agg['spk_sim'].max():.1f}")
print(f"acc_sim  mean={agg['acc_sim'].mean():.3f}  std={agg['acc_sim'].std():.3f}  min={agg['acc_sim'].min():.1f}  max={agg['acc_sim'].max():.1f}")
print(f"Saved → {OUT_CSV}")
