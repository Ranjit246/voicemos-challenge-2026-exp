"""
Validate improvement #1 on the 3400-pair train+dev with grouped-CV (hold out whole
systems, mimicking the eval's 53%-unseen structure).

Tests, on out-of-fold predictions of the richer-RAMP base (ridge + retrieval):
  1. beta re-tune (RAMP fusion weight)
  2. transductive system-mean smoothing: pred' = w*pred + (1-w)*mean(pred over same system)
     applied within each held-out fold (mimics smoothing eval predictions per system).

Uses local train+dev features. Reports baseline vs improved grouped-CV SRCC.
"""
import os, sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, zscore
from scipy.spatial.distance import cdist
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(EXP, "idea-e-feature-fusion"))
from features import build, train_table, emb_key  # noqa: E402

DEVLAB = "/Users/ranjitpatro/Home/Research/VoiceMOS/eval_set/vmc2026_track3_eval_phase_distro_v3_syn/sets/dev_with_labels.csv"
N_SPLITS = 7; K = 30; PCA_DIM = 128


def srcc(a, b): return spearmanr(a, b).statistic
def norm(v): return v.astype(np.float32) / (np.linalg.norm(v) + 1e-8)
def load(p):
    d = torch.load(p, weights_only=False)
    return {k: (v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)) for k, v in d.items()}


ECAPA = load(os.path.join(EXP, "ecapa/embeddings.pt"))
CACC = load(os.path.join(EXP, "idea-e-feature-fusion/commonaccent_emb.pt"))
WAVLM = load(os.path.join(EXP, "idea-f-wavlm/wavlm_layers.pt"))
SRC = [("ecapa", ECAPA), ("cacc", CACC), ("wl3", 3), ("wl12", 12)]


def diff_matrix(df):
    rows = []
    for a, b in zip(df.wav_a_path, df.wav_b_path):
        ka, kb = emb_key(a), emb_key(b); parts = []
        for name, s in SRC:
            if isinstance(s, int): va, vb = WAVLM[ka][s], WAVLM[kb][s]
            else: va, vb = s[ka], s[kb]
            parts.append(norm(va) - norm(vb))
        rows.append(np.concatenate(parts))
    return np.stack(rows)


def wl_cos(df):
    n = next(iter(WAVLM.values())).shape[0]
    for L in range(n):
        df[f"wl_{L}"] = [float(np.dot(a := WAVLM[emb_key(p)][L].astype(np.float32),
                                      b := WAVLM[emb_key(q)][L].astype(np.float32)) /
                              (np.linalg.norm(a)*np.linalg.norm(b)+1e-8))
                        for p, q in zip(df.wav_a_path, df.wav_b_path)]
    return [f"wl_{L}" for L in range(n)]


def retrieve(Xtr, ytr, Xq, k=K):
    D = cdist(Xq, Xtr); out = np.empty(len(Xq))
    for i in range(len(Xq)):
        nn = np.argpartition(D[i], k)[:k]; d = D[i, nn]
        w = np.exp(-d/(d.mean()+1e-8)); w /= w.sum(); out[i] = (w*ytr[nn]).sum()
    return out


def shrink(pred, systems, w):
    s = pd.Series(pred); m = s.groupby(list(systems)).transform("mean")
    return (w*s + (1-w)*m).values


def main():
    # build 3400-pair train+dev, then build features on the combined df
    tr, _ = train_table()
    dev = pd.read_csv(DEVLAB)
    cols = ["system_id", "utterance_id", "wav_a_path", "wav_b_path", "spk_sim", "acc_sim"]
    df = pd.concat([tr[cols], dev[cols]], ignore_index=True).reset_index(drop=True)
    df, base = build(df)
    feats = base + wl_cos(df)
    R = diff_matrix(df)
    groups = df.system_id.values
    print(f"train+dev pairs: {len(df)}  systems: {df.system_id.nunique()}")

    # OOF ridge + retrieval per fold
    betas = np.round(np.arange(0.5, 1.01, 0.1), 2)
    shrinks = np.round(np.arange(0.3, 1.01, 0.1), 2)
    for t in ["spk_sim", "acc_sim"]:
        gkf = GroupKFold(n_splits=N_SPLITS)
        oof_rid = np.zeros(len(df)); oof_ret = np.zeros(len(df))
        X = df[feats].values; y = df[t].values
        for tri, tei in gkf.split(df, groups=groups):
            sc = StandardScaler().fit(X[tri]); oof_rid[tei] = Ridge(alpha=1.0).fit(sc.transform(X[tri]), y[tri]).predict(sc.transform(X[tei]))
            rs = StandardScaler().fit(R[tri]); pca = PCA(PCA_DIM, random_state=0).fit(rs.transform(R[tri]))
            oof_ret[tei] = retrieve(pca.transform(rs.transform(R[tri])), y[tri], pca.transform(rs.transform(R[tei])))
        # beta tune
        best_b = max(betas, key=lambda b: srcc(y, b*zscore(oof_rid)+(1-b)*zscore(oof_ret)))
        base_fuse = b0 = srcc(y, best_b*zscore(oof_rid)+(1-best_b)*zscore(oof_ret))
        fused = best_b*zscore(oof_rid)+(1-best_b)*zscore(oof_ret)
        # transductive shrink sweep (applied per-system on OOF)
        print(f"\n===== {t} =====  best beta={best_b}  base fuse SRCC={base_fuse:.4f}")
        print(f"{'shrink_w':>9} {'SRCC':>8}")
        best_w, best_s = 1.0, base_fuse
        for w in shrinks:
            s = srcc(y, shrink(fused, groups, w))
            if s > best_s: best_s, best_w = s, w
            print(f"{w:>9.1f} {s:>8.4f}{'  <-- best' if w==best_w and s==best_s else ''}")
        print(f"  -> best shrink_w={best_w}  SRCC={best_s:.4f}  (gain {best_s-base_fuse:+.4f})")


if __name__ == "__main__":
    main()
