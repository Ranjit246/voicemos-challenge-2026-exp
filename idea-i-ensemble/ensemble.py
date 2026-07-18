"""
Ensemble of multiple answer.txt files.

Default method = 'zscore': z-score each model's predictions, average, then
min-max scale to [1,5]. This yields CONTINUOUS [1,5] values byte-compatible with
the single-model submissions that score reliably on CodaBench (avoids whatever
edge case makes raw rank-average files finish without a score). Ordering (all
that SRCC cares about) is preserved.

    python ensemble.py A.txt B.txt [C.txt ...] --out ensemble.txt [--method zscore|rank]
"""

import argparse
import csv
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["zscore", "rank"], default="zscore")
    args = ap.parse_args()

    dfs = [pd.read_csv(a) for a in args.answers]
    key = ["system_id", "utterance_id", "wav_a_path", "wav_b_path"]
    base = dfs[0][key].copy()
    idx = list(zip(base.system_id, base.utterance_id))
    for d in dfs:
        d.set_index(["system_id", "utterance_id"], inplace=True)
        assert not d.index.duplicated().any(), "duplicate (system_id,utterance_id) in an input"

    def combine(col):
        parts = []
        for d in dfs:
            if col not in d.columns:
                continue
            v = d.loc[idx, col].values.astype(float)
            if args.method == "rank":
                parts.append(pd.Series(v).rank().values)
            else:
                parts.append((v - v.mean()) / (v.std() + 1e-8))
        m = np.mean(parts, axis=0)
        return 1.0 + 4.0 * (m - m.min()) / (m.max() - m.min() + 1e-8)   # -> [1,5]

    acc, spk = combine("pred_acc_sim"), combine("pred_spk_sim")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system_id", "utterance_id", "wav_a_path", "wav_b_path",
                    "pred_acc_sim", "pred_spk_sim"])
        for i in range(len(base)):
            r = base.iloc[i]
            w.writerow([r.system_id, r.utterance_id, r.wav_a_path, r.wav_b_path,
                        float(acc[i]), float(spk[i])])
    print(f"Ensembled {len(args.answers)} files ({args.method}, scaled [1,5]) -> {args.out} ({len(base)} rows)")


if __name__ == "__main__":
    main()
