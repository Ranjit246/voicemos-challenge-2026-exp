"""
Multi-member ensemble optimizer with greedy selection (Caruana 2004).

Members (each has OOF train preds on the SAME deterministic 7-fold system split,
plus raw dev preds):
  F           ridge(base 10 + WavLM 25 cosines)              [our strongest linear]
  G           ridge(base + WavLM+HuBERT+XLSR+WavLMbase)       [multi-SSL linear]
  wavlm_head  trained head on WavLM       (avg of 6 seeds)
  xlsr_head   trained head on XLSR-53     (avg of 3 seeds)   [accent-diverse]
  hubert_head trained head on HuBERT      (avg of 3 seeds)   [diverse pretraining]

Per target, greedy ensemble selection builds a weighted bag maximizing grouped-CV
SRCC (mean labels), then the same weights are applied to the dev raw preds,
blended with the system prior, and written to submission-5/answer.txt.

Usage:
    python optimize2.py
"""

import csv, glob, os, sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, zscore
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BASE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(EXP, "idea-e-feature-fusion"))
from features import build, train_table, emb_key  # noqa: E402

DEV_CSV = "/Users/ranjitpatro/Home/Research/VoiceMOS/dataset/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv"
N_FOLDS = 7; ALPHA = {"spk_sim": 0.5, "acc_sim": 0.6}; STEPS = 30
OOF = os.path.join(BASE, "oof")
SSL = {"wavlm": os.path.join(EXP, "idea-f-wavlm/wavlm_layers.pt"),
       "hubert": os.path.join(EXP, "idea-g-multi-ssl/hubert_layers.pt"),
       "xlsr": os.path.join(EXP, "idea-g-multi-ssl/xlsr_layers.pt"),
       "wavlmbase": os.path.join(EXP, "idea-g-multi-ssl/wavlmbase_layers.pt")}


def load(p):
    d = torch.load(p, weights_only=False)
    return {k: (v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)) for k, v in d.items()}


def det_folds(s):
    u = sorted(set(s)); m = {x: i % N_FOLDS for i, x in enumerate(u)}; return np.array([m[x] for x in s])


def srcc(a, b): return spearmanr(a, b).statistic


def add_ssl(df, name, dic):
    n = next(iter(dic.values())).shape[0]; cols = []
    for L in range(n):
        c = f"{name}_{L}"
        df[c] = [float(np.dot(a := dic[emb_key(p)][L].astype(np.float32),
                              b := dic[emb_key(q)][L].astype(np.float32)) /
                       (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
                 for p, q in zip(df.wav_a_path, df.wav_b_path)]
        cols.append(c)
    return cols


def ridge_oof_dev(train, dev, feats, folds):
    oof = np.zeros((len(train), 2)); dv = np.zeros((len(dev), 2))
    for j, t in enumerate(["spk_sim", "acc_sim"]):
        for f in range(N_FOLDS):
            te = folds == f
            m = make_pipeline(StandardScaler(), Ridge(alpha=5.0)); m.fit(train[feats].values[~te], train[t].values[~te])
            oof[te, j] = m.predict(train[feats].values[te])
        m = make_pipeline(StandardScaler(), Ridge(alpha=5.0)); m.fit(train[feats].values, train[t].values)
        dv[:, j] = m.predict(dev[feats].values)
    return oof, dv


def head_member(prefix):
    fs = sorted(glob.glob(os.path.join(OOF, f"{prefix}_s*.npz")))
    if not fs: return None
    oof = np.mean([np.stack([np.load(f)["oof_spk"], np.load(f)["oof_acc"]], 1) for f in fs], 0)
    dv = np.mean([np.stack([np.load(f)["dev_spk"], np.load(f)["dev_acc"]], 1) for f in fs], 0)
    return oof, dv, len(fs)


def greedy(pool_oof, y, steps=STEPS):
    """Caruana greedy selection -> weights dict. pool_oof: {name: zscored oof vec}"""
    names = list(pool_oof); bag = []
    # init with single best
    bag.append(max(names, key=lambda n: srcc(y, pool_oof[n])))
    for _ in range(steps - 1):
        cur = np.mean([pool_oof[n] for n in bag], 0)
        best, bs = None, -2
        for n in names:
            cand = (cur * len(bag) + pool_oof[n]) / (len(bag) + 1)
            s = srcc(y, cand)
            if s > bs: bs, best = s, n
        bag.append(best)
    w = {n: bag.count(n) / len(bag) for n in names if bag.count(n)}
    return w


def main():
    train, base = train_table(); train = train.reset_index(drop=True)
    dev, _ = build(pd.read_csv(DEV_CSV))
    ssl = {n: load(p) for n, p in SSL.items() if os.path.exists(p)}
    cols = {}
    for n, dic in ssl.items():
        cols[n] = add_ssl(train, n, dic); add_ssl(dev, n, dic)
    folds = det_folds(train.system_id.values)
    y = train[["spk_sim", "acc_sim"]].values

    members = {}  # name -> (oof(2800,2), dev(600,2))
    members["F"] = ridge_oof_dev(train, dev, base + cols["wavlm"], folds)
    members["G"] = ridge_oof_dev(train, dev, base + sum(cols.values(), []), folds)
    for pfx, nm in [("head", "wavlm_head"), ("xlsr", "xlsr_head"), ("hubert", "hubert_head")]:
        hm = head_member(pfx)
        if hm: members[nm] = (hm[0], hm[1]); print(f"{nm}: {hm[2]} seeds")

    print("\nsingle-member grouped-CV:")
    for n, (oof, _) in members.items():
        print(f"  {n:>12}: spk {srcc(y[:,0], oof[:,0]):.4f}  acc {srcc(y[:,1], oof[:,1]):.4f}")

    out = os.path.join(BASE, "submission-5"); os.makedirs(out, exist_ok=True)
    final = {}
    for j, t in enumerate(["spk_sim", "acc_sim"]):
        pool = {n: zscore(oof[:, j]) for n, (oof, _) in members.items()}
        w = greedy(pool, y[:, j])
        ens = np.sum([wt * pool[n] for n, wt in w.items()], 0)
        print(f"\n{t}: greedy weights = { {k: round(v,2) for k,v in w.items()} }  grouped-CV {srcc(y[:,j], ens):.4f}")
        devz = np.sum([wt * zscore(members[n][1][:, j]) for n, wt in w.items()], 0)
        prior = dev.system_id.map(train.groupby("system_id")[t].mean()).values.astype(float)
        seen = ~np.isnan(prior); zp = np.empty(len(dev)); zp[seen] = zscore(prior[seen]); zp[~seen] = zscore(devz)[~seen]
        v = ALPHA[t] * zscore(devz) + (1 - ALPHA[t]) * zp
        final[t] = 1 + 4 * (v - v.min()) / (v.max() - v.min() + 1e-8)

    with open(os.path.join(out, "answer.txt"), "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["system_id","utterance_id","wav_a_path","wav_b_path","pred_acc_sim","pred_spk_sim"])
        for i, r in dev.iterrows():
            wr.writerow([r.system_id, r.utterance_id, r.wav_a_path, r.wav_b_path,
                         float(final["acc_sim"][i]), float(final["spk_sim"][i])])
    print(f"\nSaved -> {out}/answer.txt")


if __name__ == "__main__":
    main()
