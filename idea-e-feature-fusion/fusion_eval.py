"""
Idea E — evaluate the extended fusion (all features) with the system prior.

Reports:
  1. grouped-CV SRCC per single feature (calibrated to real dev)
  2. grouped-CV ridge fusion (all features)
  3. dev-like eval with system prior (realistic 49%-unseen composition)

Usage:
    python fusion_eval.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, zscore
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from features import train_table

N_SPLITS = 7
N_UNSEEN = 2
SEEN_PER_SYS = 14
N_REPEATS = 40
ALPHAS = np.round(np.arange(0.0, 1.01, 0.1), 2)
SEED = 42


def srcc(a, b):
    return spearmanr(a, b).statistic


def ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def grouped_single(df, feat, target):
    gkf = GroupKFold(n_splits=N_SPLITS)
    s = [srcc(df[target].values[te], df[feat].values[te])
         for _, te in gkf.split(df, groups=df.system_id.values)]
    return np.mean(s)


def grouped_fusion(df, feats, target):
    gkf = GroupKFold(n_splits=N_SPLITS)
    X, y = df[feats].values, df[target].values
    s = []
    for tr, te in gkf.split(df, groups=df.system_id.values):
        m = ridge(); m.fit(X[tr], y[tr]); s.append(srcc(y[te], m.predict(X[te])))
    return np.mean(s), np.std(s)


def dev_like(df, feats, target, rng):
    systems = df.system_id.unique()
    out = {a: [] for a in ALPHAS}
    for _ in range(N_REPEATS):
        unseen = set(rng.choice(systems, size=N_UNSEEN, replace=False))
        test_idx = list(df.index[df.system_id.isin(unseen)])
        for s in systems:
            if s in unseen:
                continue
            idx = df.index[df.system_id == s].to_numpy()
            test_idx += list(rng.choice(idx, size=min(SEEN_PER_SYS, len(idx)), replace=False))
        test_mask = df.index.isin(test_idx)
        tr, te = df[~test_mask], df[test_mask]
        m = ridge(); m.fit(tr[feats].values, tr[target].values)
        cos = m.predict(te[feats].values)
        prior = te.system_id.map(tr.groupby("system_id")[target].mean()).values.astype(float)
        seen = ~np.isnan(prior)
        zc = zscore(cos)
        zp = np.empty(len(te)); zp[seen] = zscore(prior[seen]); zp[~seen] = zc[~seen]
        y = te[target].values
        for a in ALPHAS:
            out[a].append(srcc(y, a * zc + (1 - a) * zp))
    return {a: np.mean(v) for a, v in out.items()}


def main():
    df, feats = train_table()
    df = df.reset_index(drop=True)
    print(f"\nPairs: {len(df)}  features ({len(feats)}): {feats}\n")

    for target in ["spk_sim", "acc_sim"]:
        print(f"===== {target} =====")
        ranked = sorted([(f, grouped_single(df, f, target)) for f in feats],
                        key=lambda x: -x[1])
        for f, m in ranked:
            print(f"  {f:>22}  grouped-CV {m:.4f}")
        gm, gs = grouped_fusion(df, feats, target)
        print(f"  {'RIDGE FUSION':>22}  grouped-CV {gm:.4f} ± {gs:.3f}")
        dl = dev_like(df, feats, target, np.random.default_rng(SEED))
        best_a = max(ALPHAS, key=lambda a: dl[a])
        print(f"  {'dev-like cosine (a=1)':>22}  {dl[1.0]:.4f}")
        print(f"  {'dev-like +prior best':>22}  {dl[best_a]:.4f}  (alpha={best_a})")
        print()


if __name__ == "__main__":
    main()
