"""
WavLM layer probing + fusion + submission (runs on the VM, self-contained).

1. Loads WavLM per-layer features + all base feature dicts + labels.
2. Layer probe: grouped-CV SRCC of each WavLM layer's cosine, per target.
3. Adds the best spk-layer and best acc-layer cosines to the base fusion.
4. Reports grouped-CV + dev-like(+prior) and writes answer.txt.

Usage (on VM):
    python layer_probe_fusion.py \
        --features_dir ~/voicemos/features \
        --data_dir ~/voicemos/data \
        --out ~/voicemos/idea-f/answer.txt
"""

import argparse
import csv
import os
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, zscore
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

DATASET_DIR = "vmc2026_track3_train_phase_distro_v3_syn"
N_SPLITS = 7
BASE_DICTS = ["ecapa", "titanet", "wespeaker", "wav2vec2_hidden", "wav2vec2_prob",
              "commonaccent_emb", "commonaccent_prob"]
ALPHA = {"spk_sim": 0.5, "acc_sim": 0.6}


def emb_key(rel_path):
    parts = rel_path.strip("/").split("/")
    return f"{DATASET_DIR}@{parts[-2]}@{parts[-1]}"


def load(path):
    d = torch.load(path, weights_only=False)
    return {k: (v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)) for k, v in d.items()}


def cos(embs, a, b):
    ea, eb = embs[emb_key(a)].astype(np.float32), embs[emb_key(b)].astype(np.float32)
    return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-8))


def layer_cos(wavlm, a, b, L):
    ea, eb = wavlm[emb_key(a)][L].astype(np.float32), wavlm[emb_key(b)][L].astype(np.float32)
    return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-8))


def srcc(a, b):
    return spearmanr(a, b).statistic


def grouped(df, feats, target, groups):
    gkf = GroupKFold(n_splits=N_SPLITS)
    X, y = df[feats].values, df[target].values
    s = []
    for tr, te in gkf.split(df, groups=groups):
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)); m.fit(X[tr], y[tr])
        s.append(srcc(y[te], m.predict(X[te])))
    return np.mean(s), np.std(s)


def scale_1_5(x):
    x = np.asarray(x); return 1.0 + 4.0 * (x - x.min()) / (x.max() - x.min() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fd = os.path.expanduser(args.features_dir)
    dd = os.path.expanduser(args.data_dir)

    print("Loading features...")
    base = {n: load(os.path.join(fd, f"{n}.pt")) for n in BASE_DICTS}
    base["utmos"] = load(os.path.join(fd, "utmos.pt"))
    wavlm = load(os.path.join(fd, "wavlm_layers.pt"))
    n_layers = next(iter(wavlm.values())).shape[0]
    print(f"  WavLM layers: {n_layers}")

    train = pd.read_csv(os.path.join(dd, "train_avg.csv")).reset_index(drop=True)
    dev = pd.read_csv(os.path.join(dd, "sets", "dev.csv"))
    groups = train.system_id.values

    # base cosine + utmos features
    base_feats = []
    for n in BASE_DICTS:
        train[f"cos_{n}"] = [cos(base[n], a, b) for a, b in zip(train.wav_a_path, train.wav_b_path)]
        base_feats.append(f"cos_{n}")
    for suff, col in [("a", "wav_a_path"), ("b", "wav_b_path")]:
        train[f"utmos_{suff}"] = [base["utmos"][emb_key(p)] for p in train[col]]
    train["utmos_diff"] = train.utmos_a - train.utmos_b
    base_feats += ["utmos_a", "utmos_b", "utmos_diff"]

    # ---- layer probe ----
    print("\nLayer probe (grouped-CV SRCC of each WavLM layer cosine):")
    print(f"{'layer':>5} {'spk':>8} {'acc':>8}")
    layer_srcc = {}
    for L in range(n_layers):
        c = np.array([layer_cos(wavlm, a, b, L) for a, b in zip(train.wav_a_path, train.wav_b_path)])
        gkf = GroupKFold(n_splits=N_SPLITS)
        sp = np.mean([srcc(train.spk_sim.values[te], c[te]) for _, te in gkf.split(train, groups=groups)])
        ac = np.mean([srcc(train.acc_sim.values[te], c[te]) for _, te in gkf.split(train, groups=groups)])
        layer_srcc[L] = (sp, ac)
        train[f"wavlm_L{L}"] = c
        print(f"{L:>5} {sp:>8.4f} {ac:>8.4f}")

    best_spk_L = max(layer_srcc, key=lambda L: layer_srcc[L][0])
    best_acc_L = max(layer_srcc, key=lambda L: layer_srcc[L][1])
    print(f"\nbest spk layer: {best_spk_L} ({layer_srcc[best_spk_L][0]:.4f})   "
          f"best acc layer: {best_acc_L} ({layer_srcc[best_acc_L][1]:.4f})")

    wavlm_feats = sorted({f"wavlm_L{best_spk_L}", f"wavlm_L{best_acc_L}"})

    # ---- fusion comparison ----
    print("\nFusion grouped-CV:")
    for label, feats in [("base (Idea E)", base_feats),
                         ("base + WavLM best layers", base_feats + wavlm_feats)]:
        sp = grouped(train, feats, "spk_sim", groups)
        ac = grouped(train, feats, "acc_sim", groups)
        print(f"  {label:>26}: spk {sp[0]:.4f}  acc {ac[0]:.4f}")

    # ---- build submission with best fusion + system prior ----
    feats = base_feats + wavlm_feats
    for n in BASE_DICTS:
        dev[f"cos_{n}"] = [cos(base[n], a, b) for a, b in zip(dev.wav_a_path, dev.wav_b_path)]
    for suff, col in [("a", "wav_a_path"), ("b", "wav_b_path")]:
        dev[f"utmos_{suff}"] = [base["utmos"][emb_key(p)] for p in dev[col]]
    dev["utmos_diff"] = dev.utmos_a - dev.utmos_b
    for L in {best_spk_L, best_acc_L}:
        dev[f"wavlm_L{L}"] = [layer_cos(wavlm, a, b, L) for a, b in zip(dev.wav_a_path, dev.wav_b_path)]

    cols = {}
    for target in ["spk_sim", "acc_sim"]:
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        m.fit(train[feats].values, train[target].values)
        cpred = m.predict(dev[feats].values)
        prior = dev.system_id.map(train.groupby("system_id")[target].mean()).values.astype(float)
        seen = ~np.isnan(prior)
        zc = zscore(cpred); zp = np.empty(len(dev))
        zp[seen] = zscore(prior[seen]); zp[~seen] = zc[~seen]
        a = ALPHA[target]
        cols[target] = scale_1_5(a * zc + (1 - a) * zp)

    os.makedirs(os.path.dirname(os.path.expanduser(args.out)), exist_ok=True)
    with open(os.path.expanduser(args.out), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system_id", "utterance_id", "wav_a_path",
                                          "wav_b_path", "pred_acc_sim", "pred_spk_sim"])
        w.writeheader()
        for i, row in dev.iterrows():
            w.writerow({"system_id": row.system_id, "utterance_id": row.utterance_id,
                        "wav_a_path": row.wav_a_path, "wav_b_path": row.wav_b_path,
                        "pred_acc_sim": float(cols["acc_sim"][i]),
                        "pred_spk_sim": float(cols["spk_sim"][i])})
    print(f"\nSaved -> {args.out} ({len(dev)} rows)")


if __name__ == "__main__":
    main()
