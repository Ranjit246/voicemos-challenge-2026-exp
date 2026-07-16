"""
Idea D — System-prior blending (exploits dev being mostly in-domain).

dev.csv has 23 systems; only sys003 & sys015 are NOT in train (21/23 seen).
A seen system's TRAINING-LABEL mean is a near-perfect prior for its dev pairs
(same system = same quality). 33-45% of score variance is between-system, so
this captures the dominant signal for the 21 seen systems; cosine handles
within-system ranking + the 2 unseen systems.

We validate with a DEV-LIKE split (not pure grouped-CV):
  - hold out N_UNSEEN whole systems  (mimics sys003/sys015)
  - hold out a fraction of pairs from the remaining "seen" systems
  - test set = both, mixed (like real dev)
For seen test pairs, the system's train-fold mean label is available as a prior.

Predictors compared on the mixed test set:
  cosine       : ridge fusion over cosines (generalizable, our current best)
  prior        : train system-mean label (seen) / global mean (unseen)
  blend(alpha) : alpha*z(cosine) + (1-alpha)*z(prior)

Usage:
    python dev_like_eval.py
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, zscore
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BASE     = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.dirname(BASE)
sys.path.insert(0, EXP_ROOT)
from eval_harness import build_feature_table  # noqa: E402

# Real dev composition: 2 whole unseen systems = 296/600 pairs (49%),
# remaining 304 pairs spread over 21 seen systems (~14 pairs each).
N_UNSEEN     = 2
SEEN_PER_SYS = 14      # pairs kept per seen system in the test set
N_REPEATS    = 40
ALPHAS       = np.round(np.arange(0.0, 1.01, 0.1), 2)
SEED         = 42


def srcc(a, b):
    return spearmanr(a, b).statistic


def ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def one_split(df, feats, target, rng):
    systems = df.system_id.unique()
    unseen = set(rng.choice(systems, size=N_UNSEEN, replace=False))

    is_unseen = df.system_id.isin(unseen)
    # test = ALL pairs of the 2 unseen systems + SEEN_PER_SYS pairs per seen system
    test_idx = list(df.index[is_unseen])
    for s in systems:
        if s in unseen:
            continue
        idx = df.index[df.system_id == s].to_numpy()
        test_idx += list(rng.choice(idx, size=min(SEEN_PER_SYS, len(idx)), replace=False))

    test_mask = df.index.isin(test_idx)
    train = df[~test_mask]
    test = df[test_mask]

    m = ridge(); m.fit(train[feats].values, train[target].values)
    cos_pred = m.predict(test[feats].values)

    sysmean = train.groupby("system_id")[target].mean()
    prior = test.system_id.map(sysmean).values.astype(float)
    seen_mask = ~np.isnan(prior)

    y = test[target].values
    zc = zscore(cos_pred)
    zp = np.empty(len(test))
    zp[seen_mask] = zscore(prior[seen_mask])
    zp[~seen_mask] = zc[~seen_mask]     # unseen: prior falls back to cosine

    out = {"cosine": srcc(y, cos_pred),
           "prior_seen_only": srcc(y[seen_mask], prior[seen_mask]),
           "frac_unseen": (~seen_mask).mean()}
    for a in ALPHAS:
        out[f"blend_{a:.1f}"] = srcc(y, a * zc + (1 - a) * zp)
    return out


def main():
    df, feats = build_feature_table()
    df = df.reset_index(drop=True)
    rng = np.random.default_rng(SEED)

    print(f"Dev-like (REAL composition): {N_UNSEEN} whole unseen systems + "
          f"{SEEN_PER_SYS} pairs/seen-system, {N_REPEATS} repeats\n")

    for target in ["spk_sim", "acc_sim"]:
        rows = [one_split(df, feats, target, rng) for _ in range(N_REPEATS)]
        agg = pd.DataFrame(rows).mean()
        print(f"===== {target} =====   (frac unseen in test: {agg['frac_unseen']:.2f})")
        print(f"  cosine only                 {agg['cosine']:.4f}")
        print(f"  prior on seen pairs only    {agg['prior_seen_only']:.4f}  (info: prior strength)")
        best_a = max(ALPHAS, key=lambda a: agg[f"blend_{a:.1f}"])
        for a in ALPHAS:
            mark = "  <-- best" if a == best_a else ""
            print(f"  blend a={a:.1f}                {agg[f'blend_{a:.1f}']:.4f}{mark}")
        print()


if __name__ == "__main__":
    main()
