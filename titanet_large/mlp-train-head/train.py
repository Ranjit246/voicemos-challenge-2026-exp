"""
MLP regressor head for spk_sim and acc_sim prediction.

Inputs:  TitaNet embeddings (192-dim per wav)
Pair feature: [emb_a | emb_b | emb_a - emb_b | emb_a * emb_b]  → 768-dim
Outputs: pred_spk_sim, pred_acc_sim  (both scalars, trained jointly)

Cosine baseline is printed before training:
  - pred_spk_sim = cosine(emb_a, emb_b)  (direct match for speaker)
  - pred_acc_sim = cosine(emb_a, emb_b)  (proxy only — same score reused)

Usage:
    python train.py
    python train.py --epochs 100 --lr 1e-3 --hidden 256 128
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from scipy.stats import spearmanr


# ── paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PT = os.path.join(BASE, "../infer/embeddings/embeddings/manifest_embeddings.pt")
TRAIN_AVG_CSV = os.path.join(BASE, "../train_avg.csv")
CKPT_OUT      = os.path.join(BASE, "best_model.pt")


# ── embedding key helper ────────────────────────────────────────────────────────
# NeMo keys embeddings by last 3 path components joined with '@'
# e.g. wav/vmc2026-track3-sys017-utt026.wav
#   → vmc2026_track3_train_phase_distro_v3_syn@wav@vmc2026-track3-sys017-utt026.wav
DATASET_DIR_NAME = "vmc2026_track3_train_phase_distro_v3_syn"

def emb_key(rel_path):
    parts = rel_path.strip("/").split("/")  # ["wav", "vmc2026-track3-...wav"]
    return f"{DATASET_DIR_NAME}@{parts[-2]}@{parts[-1]}"


# ── pair feature ───────────────────────────────────────────────────────────────
def pair_feature(ea, eb):
    return np.concatenate([ea, eb, ea - eb, ea * eb])  # (768,)


# ── dataset ────────────────────────────────────────────────────────────────────
class PairDataset(Dataset):
    def __init__(self, df, embeddings):
        self.features = []
        self.spk = []
        self.acc = []
        skipped = 0
        for _, row in df.iterrows():
            ka = emb_key(row["wav_a_path"])
            kb = emb_key(row["wav_b_path"])
            if ka not in embeddings or kb not in embeddings:
                skipped += 1
                continue
            feat = pair_feature(embeddings[ka], embeddings[kb])
            self.features.append(feat.astype(np.float32))
            self.spk.append(float(row["spk_sim"]))
            self.acc.append(float(row["acc_sim"]))
        if skipped:
            print(f"  Warning: skipped {skipped} pairs (missing embeddings)")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx]),
            torch.tensor([self.spk[idx], self.acc[idx]]),
        )


# ── model ──────────────────────────────────────────────────────────────────────
class MLPHead(nn.Module):
    def __init__(self, in_dim=768, hidden=(512, 256), dropout=0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 2))  # spk_sim, acc_sim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── cosine baseline ────────────────────────────────────────────────────────────
def cosine_baseline(dataset, indices):
    """Compute SRCC using cosine(emb_a, emb_b) as predictor for both targets."""
    cos_sims, trues_spk, trues_acc = [], [], []
    for idx in indices:
        feat, y = dataset[idx]
        ea, eb = feat[:192].numpy(), feat[192:384].numpy()
        cos = float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-8))
        cos_sims.append(cos)
        trues_spk.append(y[0].item())
        trues_acc.append(y[1].item())
    srcc_spk = spearmanr(trues_spk, cos_sims).statistic
    srcc_acc = spearmanr(trues_acc, cos_sims).statistic
    return srcc_spk, srcc_acc


# ── train / eval helpers ───────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds_spk, preds_acc = [], []
    trues_spk, trues_acc = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x).cpu().numpy()
        preds_spk.extend(out[:, 0])
        preds_acc.extend(out[:, 1])
        trues_spk.extend(y[:, 0].numpy())
        trues_acc.extend(y[:, 1].numpy())
    srcc_spk = spearmanr(trues_spk, preds_spk).statistic
    srcc_acc = spearmanr(trues_acc, preds_acc).statistic
    mse_spk  = np.mean((np.array(trues_spk) - np.array(preds_spk)) ** 2)
    mse_acc  = np.mean((np.array(trues_acc) - np.array(preds_acc)) ** 2)
    return srcc_spk, srcc_acc, mse_spk, mse_acc


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int,   default=50)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--batch",    type=int,   default=64)
    parser.add_argument("--hidden",   type=int,   nargs="+", default=[128])
    parser.add_argument("--dropout",  type=float, default=0.2)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed",     type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cpu"

    print("Loading embeddings...")
    embeddings = torch.load(EMBEDDINGS_PT, weights_only=False)
    # convert tensors to numpy if needed
    embeddings = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in embeddings.items()}
    print(f"  {len(embeddings)} embeddings loaded, dim={next(iter(embeddings.values())).shape}")

    print("Building dataset...")
    df = pd.read_csv(TRAIN_AVG_CSV)
    dataset = PairDataset(df, embeddings)
    print(f"  {len(dataset)} pairs")

    n_val = max(1, int(len(dataset) * args.val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False)

    model = MLPHead(in_dim=768, hidden=tuple(args.hidden), dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── cosine baseline ──
    val_indices = val_ds.indices
    cos_srcc_spk, cos_srcc_acc = cosine_baseline(dataset, val_indices)
    baseline_line = (
        f"\nCosine baseline (val)  SRCC_spk={cos_srcc_spk:.4f}  "
        f"SRCC_acc={cos_srcc_acc:.4f}  (acc uses same score as proxy)"
    )

    header = (
        f"\nTraining  epochs={args.epochs}  lr={args.lr}  "
        f"hidden={args.hidden}  batch={args.batch}\n"
        f"{'Epoch':>5}  {'loss':>8}  {'SRCC_spk':>9}  {'SRCC_acc':>9}  "
        f"{'MSE_spk':>8}  {'MSE_acc':>8}\n" + "-" * 60
    )

    log_path = os.path.join(BASE, "train.log")
    log_lines = [baseline_line, header]

    print(baseline_line)
    print(header)

    best_srcc = -1.0
    for epoch in range(1, args.epochs + 1):
        loss = run_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            srcc_spk, srcc_acc, mse_spk, mse_acc = evaluate(model, val_loader, device)
            mean_srcc = (srcc_spk + srcc_acc) / 2
            marker = " *" if mean_srcc > best_srcc else ""
            if mean_srcc > best_srcc:
                best_srcc = mean_srcc
                torch.save(model.state_dict(), CKPT_OUT)
            line = (f"{epoch:>5}  {loss:>8.4f}  {srcc_spk:>9.4f}  {srcc_acc:>9.4f}  "
                    f"{mse_spk:>8.4f}  {mse_acc:>8.4f}{marker}")
            print(line)
            log_lines.append(line)

    summary = f"\nBest mean SRCC: {best_srcc:.4f}\nCheckpoint saved → {CKPT_OUT}"
    print(summary)
    log_lines.append(summary)

    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"Log saved → {log_path}")


if __name__ == "__main__":
    main()
