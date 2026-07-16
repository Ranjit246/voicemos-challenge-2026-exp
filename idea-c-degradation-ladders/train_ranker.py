"""
Idea C — lightweight ranker with degradation-ladder ranking loss.

A small MLP over the Idea-E fusion features (10-d), trained with:
  MSE  on real pairs (labels)
  margin-ranking on ladder pairs: for a fixed reference, a higher-bitrate
    codec reconstruction must score >= a lower-bitrate one (known order,
    no absolute label needed). SRCC only cares about order, so this teaches
    monotonic sensitivity to codec damage -> better UNSEEN-system generalization.

Validation: grouped-CV by system on REAL pairs (ladders always in train).
Run with --lam 0 to reproduce the plain fusion (sanity check, no ladders needed).

Usage:
    python train_ranker.py                 # uses ladder_features.csv if present
    python train_ranker.py --lam 0         # real-only sanity vs ridge fusion
    python train_ranker.py --lam 0.5 --epochs 300
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

BASE     = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(EXP_ROOT, "idea-e-feature-fusion"))
from features import train_table  # noqa: E402

LADDER_CSV = os.path.join(BASE, "ladder_features.csv")
N_SPLITS = 7
SEED = 42


class Ranker(nn.Module):
    """NOTE (validated on Mac, lam=0): a 32-unit MLP overfits the training
    systems and scores only ~0.21/0.14 grouped-CV vs ridge's 0.474/0.455 —
    the same small/frozen-generalizes lesson. On the VM prefer hidden=0
    (linear) or strong weight_decay so the base generalizes like ridge; the
    ladder ranking loss is the value-add, not model capacity."""

    def __init__(self, in_dim, hidden=0, dropout=0.2):
        super().__init__()
        if hidden and hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, 2),
            )
        else:
            self.net = nn.Linear(in_dim, 2)   # linear: generalizes like ridge

    def forward(self, x):
        return self.net(x)


def srcc(a, b):
    return spearmanr(a, b).statistic


def train_one(Xtr, ytr, ladder, epochs, lam, lr, device):
    """ladder: list of (Xhi, Xlo) tensor pairs (higher-bitrate, lower-bitrate)."""
    model = Ranker(Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    mse = nn.MSELoss()
    rank = nn.MarginRankingLoss(margin=0.1)
    Xtr, ytr = Xtr.to(device), ytr.to(device)
    if ladder is not None:
        Xhi, Xlo = ladder
        Xhi, Xlo = Xhi.to(device), Xlo.to(device)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss = mse(model(Xtr), ytr)
        if lam > 0 and ladder is not None and len(Xhi) > 0:
            phi, plo = model(Xhi), model(Xlo)
            tgt = torch.ones(len(Xhi), device=device)
            # higher bitrate should score >= lower bitrate, for both heads
            loss = loss + lam * (rank(phi[:, 0], plo[:, 0], tgt) +
                                 rank(phi[:, 1], plo[:, 1], tgt))
        loss.backward(); opt.step()
    return model


def build_ladder_tensors(feats):
    """Return (Xhi, Xlo) from ladder_features.csv: consecutive rungs within each ladder."""
    if not os.path.exists(LADDER_CSV):
        return None
    lad = pd.read_csv(LADDER_CSV)
    hi, lo = [], []
    for _, g in lad.sort_values(["ladder_id", "rung"]).groupby("ladder_id"):
        rows = g[feats].values
        for i in range(len(rows) - 1):
            hi.append(rows[i]); lo.append(rows[i + 1])   # rung i is higher quality than i+1
    if not hi:
        return None
    return (torch.tensor(np.array(hi), dtype=torch.float32),
            torch.tensor(np.array(lo), dtype=torch.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=0.5, help="ladder ranking-loss weight")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df, feats = train_table()
    df = df.reset_index(drop=True)
    ladder = build_ladder_tensors(feats)
    print(f"Pairs: {len(df)}  features: {len(feats)}  "
          f"ladder pairs: {0 if ladder is None else len(ladder[0])}  "
          f"lam={args.lam}  device={device}\n")

    # standardize features on train (fit per-fold below)
    from sklearn.preprocessing import StandardScaler
    gkf = GroupKFold(n_splits=N_SPLITS)
    for target_pair in [["spk_sim", "acc_sim"]]:
        spk_scores, acc_scores = [], []
        for tr, te in gkf.split(df, groups=df.system_id.values):
            sc = StandardScaler().fit(df[feats].values[tr])
            Xtr = torch.tensor(sc.transform(df[feats].values[tr]), dtype=torch.float32)
            Xte = torch.tensor(sc.transform(df[feats].values[te]), dtype=torch.float32)
            ytr = torch.tensor(df[["spk_sim", "acc_sim"]].values[tr], dtype=torch.float32)
            lad = None
            if ladder is not None:
                lad = (torch.tensor(sc.transform(ladder[0].numpy()), dtype=torch.float32),
                       torch.tensor(sc.transform(ladder[1].numpy()), dtype=torch.float32))
            model = train_one(Xtr, ytr, lad, args.epochs, args.lam, args.lr, device)
            with torch.no_grad():
                pred = model(Xte.to(device)).cpu().numpy()
            y = df[["spk_sim", "acc_sim"]].values[te]
            spk_scores.append(srcc(y[:, 0], pred[:, 0]))
            acc_scores.append(srcc(y[:, 1], pred[:, 1]))
        print(f"grouped-CV  spk {np.mean(spk_scores):.4f} ± {np.std(spk_scores):.3f}   "
              f"acc {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.3f}")
    print("\n(reference: Idea E ridge fusion grouped-CV  spk 0.474  acc 0.455)")


if __name__ == "__main__":
    main()
