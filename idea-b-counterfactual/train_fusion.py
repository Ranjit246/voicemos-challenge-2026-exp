"""
Idea B — train fusion with counterfactual pairs, validate on REAL grouped-CV.

For each grouped-CV fold (hold out whole systems):
  train set = real_train_fold  [+ all counterfactual pairs]
  test set  = real_test_fold   (real pairs only, always)

Compares baseline (real only) vs augmented (real + counterfactual) so we can
see whether the manufactured accent-decorrelating pairs raise acc_sim SRCC.

Usage:
    python train_fusion.py
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BASE     = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.dirname(BASE)
sys.path.insert(0, EXP_ROOT)
from eval_harness import build_feature_table  # noqa: E402

CF_CSV   = os.path.join(BASE, "counterfactual_features.csv")
N_SPLITS = 7


def srcc(a, b):
    return spearmanr(a, b).statistic


def ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def eval_cv(df, feats, target, cf_df=None):
    """Grouped-CV SRCC; if cf_df given, append it to every training fold."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    Xr, yr = df[feats].values, df[target].values
    scores = []
    for tr, te in gkf.split(df, groups=df.system_id.values):
        Xtr, ytr = Xr[tr], yr[tr]
        if cf_df is not None:
            Xtr = np.vstack([Xtr, cf_df[feats].values])
            ytr = np.concatenate([ytr, cf_df[target].values])
        m = ridge()
        m.fit(Xtr, ytr)
        scores.append(srcc(yr[te], m.predict(Xr[te])))
    return np.mean(scores), np.std(scores)


def main():
    df, feats = build_feature_table()
    cf = pd.read_csv(CF_CSV)
    print(f"Real pairs: {len(df)}  |  counterfactual: {len(cf)}  |  features: {feats}\n")

    print(f"{'target':>10}  {'baseline (real only)':>22}  {'+ counterfactual':>18}  {'delta':>7}")
    for target in ["spk_sim", "acc_sim"]:
        bm, bs = eval_cv(df, feats, target, cf_df=None)
        am, as_ = eval_cv(df, feats, target, cf_df=cf)
        print(f"{target:>10}  {bm:>13.4f} ± {bs:.3f}  {am:>11.4f} ± {as_:.3f}  {am-bm:>+7.4f}")


if __name__ == "__main__":
    main()
