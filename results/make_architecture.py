"""Render the PD-RAMP system architecture diagram.

Output: results/figures/architecture.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

RIDGE, RETR, HEAD = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
AXIS, NEUTRAL = "#c3c2b7", "#52514e"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
})

fig = plt.figure(figsize=(12.6, 9.4), facecolor=SURFACE)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def box(x, y, w, h, title, sub=None, color=NEUTRAL, tint=0.055, ts=9.6, ss=7.9):
    """Thin-stroke rounded box: near-surface fill, hairline colored border."""
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                 boxstyle="round,pad=0.006,rounding_size=0.012",
                 linewidth=1.2, edgecolor=color,
                 facecolor=color, alpha=tint, zorder=2))
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                 boxstyle="round,pad=0.006,rounding_size=0.012",
                 linewidth=1.2, edgecolor=color, facecolor="none", zorder=3))
    ty = y + (h*0.16 if sub else 0)
    ax.text(x, ty, title, ha="center", va="center", fontsize=ts,
            color=INK, fontweight="bold", zorder=4)
    if sub:
        ax.text(x, y - h*0.22, sub, ha="center", va="center", fontsize=ss,
                color=INK2, zorder=4, linespacing=1.45)


def arrow(p1, p2, color=MUTED, lw=1.4, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=11,
                 linewidth=lw, color=color, zorder=1,
                 connectionstyle=f"arc3,rad={rad}",
                 shrinkA=1.5, shrinkB=1.5))


def tag(x, y, txt, color=MUTED, size=7.6, ha="center"):
    ax.text(x, y, txt, ha=ha, va="center", fontsize=size, color=color,
            style="italic", zorder=4)


# ---------------- title ----------------
ax.text(0.035, 0.972, "PD-RAMP — system architecture", fontsize=15.5,
        fontweight="bold", color=INK, ha="left", va="top")
ax.text(0.035, 0.936,
        "VoiceMOS 2026 Track 3 · pairwise speaker- and accent-similarity prediction · "
        "no backbone fine-tuning",
        fontsize=9.2, color=INK2, ha="left", va="top")

# ---------------- input ----------------
box(0.5, 0.876, 0.30, 0.050, "input pair  (wav_a , wav_b)",
    sub="generated utterance  ·  natural reference, different sentence",
    color=NEUTRAL, tint=0.05, ts=9.8, ss=7.6)

# ---------------- frozen feature panel ----------------
box(0.5, 0.770, 0.80, 0.088, "frozen pretrained models   (no fine-tuning)",
    sub="ECAPA-TDNN · TitaNet-L · WeSpeaker · CommonAccent-ECAPA (emb + 16-class prob)\n"
        "wav2vec2-accent (hidden + 13-class prob) · WavLM-Large (all 25 layers) · UTMOS",
    color=NEUTRAL, tint=0.05, ts=10.0, ss=8.2)
arrow((0.5, 0.851), (0.5, 0.815))

# fan-out via a distribution bus (keeps arrows clear of the box text)
ax.plot([0.5, 0.5], [0.726, 0.708], color=MUTED, lw=1.4, zorder=1)
ax.plot([0.175, 0.825], [0.708, 0.708], color=MUTED, lw=1.4, zorder=1)
for xc, col in ((0.175, RIDGE), (0.5, RETR), (0.825, HEAD)):
    arrow((xc, 0.708), (xc, 0.683), color=col)

# ---------------- branch 1: parametric ridge ----------------
box(0.175, 0.650, 0.29, 0.062, "~35 pair cosine features",
    sub="cosine(a,b) per model · all 25 WavLM layers\nUTMOS(a), UTMOS(b), difference",
    color=RIDGE)
arrow((0.175, 0.617), (0.175, 0.575), color=RIDGE)
box(0.175, 0.532, 0.29, 0.052, "Ridge regression",
    sub="alpha = 1.0  ·  scalars cannot encode system identity", color=RIDGE)

# ---------------- branch 2: pair-difference retrieval ----------------
box(0.5, 0.650, 0.30, 0.062, "pair-difference vectors",
    sub="L2-normed  e(a) - e(b)  for ECAPA, CommonAccent,\nWavLM-L3 (speaker), WavLM-L12 (accent) -> 2432-d",
    color=RETR)
arrow((0.5, 0.617), (0.5, 0.575), color=RETR)
box(0.5, 0.532, 0.30, 0.052, "kNN retrieval over training pairs",
    sub="standardize -> PCA-128 · k = 30 · softmax(-d / mean-d)", color=RETR)
tag(0.483, 0.596, "the key idea: retrieve similar PAIRS, not utterances",
    color=RETR, ha="right")

# ---------------- branch 3: trained head ----------------
box(0.825, 0.650, 0.29, 0.062, "WavLM pairwise head",
    sub="learnable per-task layer weights -> 128-d proj\npair feat [p_a, p_b, |p_a-p_b|, p_a*p_b]",
    color=HEAD)
arrow((0.825, 0.617), (0.825, 0.575), color=HEAD)
box(0.825, 0.532, 0.29, 0.052, "MSE + margin-ranking loss",
    sub="margin 0.3 · cross-system batches · 6 seeds averaged", color=HEAD)

# ---------------- fuse ----------------
arrow((0.175, 0.506), (0.30, 0.452), color=RIDGE, rad=-0.15)
arrow((0.5, 0.506), (0.36, 0.452), color=RETR, rad=0.15)
box(0.33, 0.420, 0.34, 0.048, "RAMP fuse",
    sub="beta * z(ridge)  +  (1 - beta) * z(retrieval)     beta = 0.8 spk / 0.7 acc",
    color=RIDGE, tint=0.04)

arrow((0.33, 0.396), (0.45, 0.348), color=RIDGE, rad=-0.12)
arrow((0.825, 0.506), (0.60, 0.348), color=HEAD, rad=0.16)
box(0.5, 0.316, 0.40, 0.046, "ensemble   (z-average, 1 : 1)",
    sub="parametric + retrieval  <->  trained head", color=NEUTRAL, tint=0.05)

# ---------------- prior ----------------
arrow((0.5, 0.293), (0.5, 0.256))
box(0.5, 0.222, 0.56, 0.056, "system-label prior blend",
    sub="alpha * z(prediction)  +  (1 - alpha) * z(system training mean)\n"
        "alpha = 0.5 spk / 0.6 acc   ·   system unseen in training -> cosine fallback (OOD-safe)",
    color=NEUTRAL, tint=0.05)

# ---------------- transductive smoothing ----------------
arrow((0.5, 0.194), (0.5, 0.157))
box(0.5, 0.122, 0.56, 0.052, "transductive system-mean smoothing   (label-free)",
    sub="p'  =  0.7 * p  +  0.3 * mean of p over the same system in the eval set",
    color=RETR, tint=0.07)
tag(0.815, 0.122, "+0.037 spk / +0.016 acc\non the evaluation set", color="#006300",
    size=8.0, ha="left")

# ---------------- output ----------------
arrow((0.5, 0.096), (0.5, 0.062))
box(0.5, 0.033, 0.40, 0.042, "rescale to [1, 5]   ->   pred_spk_sim , pred_acc_sim",
    color=NEUTRAL, tint=0.05, ts=9.6)

# ---------------- legend of pathways ----------------
for i, (c, t) in enumerate(((RIDGE, "parametric"), (RETR, "retrieval"),
                            (HEAD, "trained head"))):
    x = 0.72 + i*0.098
    ax.add_patch(FancyBboxPatch((x, 0.952), 0.016, 0.010,
                 boxstyle="round,pad=0.002,rounding_size=0.004",
                 linewidth=0, facecolor=c, zorder=4))
    ax.text(x + 0.022, 0.957, t, fontsize=8.4, color=INK2, va="center", zorder=4)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "architecture.png")
fig.savefig(out, dpi=200, facecolor=SURFACE)
print("saved:", out)
