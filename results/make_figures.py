"""Generate the results figure for the VoiceMOS 2026 Track 3 submission (Team T15).

Output: results/figures/results.png
Data source: official challenge results (VoiceMOS Challenge 2026 raw results).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# --- palette (validated categorical slots 1-2 + chrome) ---
SPK, ACC = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, WASH = "#e1e0d9", "#c3c2b7", "#eef4fc"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "font.size": 9,
})

OURS = "T15"

# ---------------- data: official evaluation results ----------------
# entry: (label, spk_utt, acc_utt, spk_sys, acc_sys)   None = not submitted
ENTRIES = [
    ("T04",  0.644, 0.565, 0.948, 0.943),
    ("T16",  0.606, 0.523, 0.923, 0.882),
    ("T07",  0.585, 0.530, 0.948, 0.887),
    ("T15",  0.549, 0.474, 0.942, 0.902),
    ("T06",  0.484, 0.505, 0.940, 0.930),
    ("T17",  0.481, 0.386, 0.882, 0.875),
    ("B1",   0.414, 0.386, 0.874, 0.838),
    ("B2",   0.403, 0.316, 0.914, 0.786),
    ("T13",  None,  0.371, None,  0.761),
]

# development-set progression of our own system
PROG = [
    ("MLP",   0.358, 0.375),
    ("A",     0.426, 0.373),
    ("D",     0.478, 0.416),
    ("E",     0.501, 0.477),
    ("F",     0.533, 0.548),
    ("I",     0.568, 0.588),
    ("K*",    0.5745, 0.5986),
]
PROG_LBL = ["MLP\ninitial", "A\nshrinkage", "D\n+system prior",
            "E\n+model fusion", "F\n+WavLM 25L", "I\n+head ens.",
            "K*\n+pair-diff retr."]
B2_SPK, B2_ACC = 0.451, 0.440   # stronger official baseline (dev)

# transductive smoothing on the evaluation set
SMOOTH = [("Base\n(retrieval + head + prior)", 0.512, 0.458),
          ("+ transductive\nsmoothing (w=0.7)", 0.549, 0.474)]


def style(ax, ymax, ystep, ylabel):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7, linestyle="-")
    ax.xaxis.grid(False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(MultipleLocator(ystep))
    ax.set_ylabel(ylabel, fontsize=8.5, color=INK2)


def note(ax, txt, x=0.985, ha="right"):
    ax.text(x, 0.95, txt, transform=ax.transAxes, ha=ha, va="top",
            fontsize=8.5, color=INK2)


def bars(ax, labels, a, b, ymax, ystep, ylabel, highlight=OURS, ann=None):
    x = np.arange(len(labels)); w = 0.38
    va = [np.nan if v is None else v for v in a]
    vb = [np.nan if v is None else v for v in b]
    if highlight in labels:                       # emphasis wash behind our group
        i = labels.index(highlight)
        ax.axvspan(i - 0.5, i + 0.5, color=WASH, zorder=0, lw=0)
    ax.bar(x - w/2 - 0.01, va, w, color=SPK, label="Speaker (SPK)", zorder=3)
    ax.bar(x + w/2 + 0.01, vb, w, color=ACC, label="Accent (ACC)", zorder=3)
    style(ax, ymax, ystep, ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    for t, lab in zip(ax.get_xticklabels(), labels):
        if lab == highlight:
            t.set_color(INK); t.set_fontweight("bold")
        elif lab.startswith("B"):
            t.set_color(MUTED); t.set_style("italic")
    ax.set_xlim(-0.65, len(labels) - 0.35)
    if ann is not None:                            # selective direct labels: ours only
        i = labels.index(highlight)
        for xoff, val in ((-w/2 - 0.01, a[i]), (w/2 + 0.01, b[i])):
            ax.text(i + xoff, val + ymax*0.018, f"{val:.3f}", ha="center",
                    va="bottom", fontsize=8, color=INK, fontweight="bold")


fig = plt.figure(figsize=(13.2, 8.6))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.20,
                      left=0.055, right=0.985, top=0.885, bottom=0.075)

labels = [e[0] for e in ENTRIES]

# --- A: utterance-level (primary metric) ---
axA = fig.add_subplot(gs[0, 0])
bars(axA, labels, [e[1] for e in ENTRIES], [e[2] for e in ENTRIES],
     0.78, 0.2, "SRCC", ann=True)
axA.set_title("A   Utterance-level SRCC — the challenge's primary metric",
              loc="left", fontsize=10.5, fontweight="bold", pad=9)
note(axA, "ours: 4th (SPK) / 5th (ACC) of 7 teams")

# --- B: system-level ---
axB = fig.add_subplot(gs[0, 1])
bars(axB, labels, [e[3] for e in ENTRIES], [e[4] for e in ENTRIES],
     1.18, 0.25, "SRCC", ann=True)
axB.set_title("B   System-level SRCC", loc="left",
              fontsize=10.5, fontweight="bold", pad=9)
note(axB, "ours: 3rd of 7 teams in both sub-scores")

# --- C: our development progression ---
axC = fig.add_subplot(gs[1, 0])
xs = np.arange(len(PROG))
ys, ya = [p[1] for p in PROG], [p[2] for p in PROG]
axC.axhline(B2_SPK, color=AXIS, lw=0.8, zorder=1)
axC.text(len(PROG)-1.15, B2_SPK + 0.014, "official baseline B2", fontsize=7.6,
         color=MUTED, va="bottom", ha="right")
axC.plot(xs, ys, color=SPK, lw=2, marker="o", ms=6.5, mec=SURFACE, mew=1.6,
         label="Speaker (SPK)", zorder=3)
axC.plot(xs, ya, color=ACC, lw=2, marker="o", ms=6.5, mec=SURFACE, mew=1.6,
         label="Accent (ACC)", zorder=3)
style(axC, 0.68, 0.1, "dev-set SRCC")
axC.set_xticks(xs)
axC.set_xticklabels(PROG_LBL, fontsize=7.8)
axC.get_xticklabels()[-1].set_fontweight("bold")
axC.get_xticklabels()[-1].set_color(INK)
axC.set_xlim(-0.45, len(PROG) - 0.55)
axC.text(len(PROG)-1, ya[-1] + 0.020, f"{ya[-1]:.3f}", fontsize=8,
         color=INK, fontweight="bold", ha="center")          # accent on top
axC.text(len(PROG)-1, ys[-1] - 0.052, f"{ys[-1]:.3f}", fontsize=8,
         color=INK, fontweight="bold", ha="center")          # speaker below
axC.set_title("C   How the system was built — development-set progression",
              loc="left", fontsize=10.5, fontweight="bold", pad=9)

# --- D: transductive smoothing ---
axD = fig.add_subplot(gs[1, 1])
x = np.arange(len(SMOOTH)); w = 0.30
axD.bar(x - w/2 - 0.012, [s[1] for s in SMOOTH], w, color=SPK, zorder=3)
axD.bar(x + w/2 + 0.012, [s[2] for s in SMOOTH], w, color=ACC, zorder=3)
style(axD, 0.78, 0.2, "evaluation-set SRCC")
axD.set_xticks(x); axD.set_xticklabels([s[0] for s in SMOOTH], fontsize=8.5)
axD.set_xlim(-0.55, 1.55)
for i, s in enumerate(SMOOTH):
    for xoff, val in ((-w/2 - 0.012, s[1]), (w/2 + 0.012, s[2])):
        axD.text(i + xoff, val + 0.013, f"{val:.3f}", ha="center", va="bottom",
                 fontsize=8, color=INK, fontweight="bold" if i else "normal")
GAIN = "#006300"                                   # sanctioned "delta up good" ink
for xoff, d in ((-w/2 - 0.012, "+0.037"), (w/2 + 0.012, "+0.016")):
    val = SMOOTH[1][1] if xoff < 0 else SMOOTH[1][2]
    axD.text(1 + xoff, val + 0.052, d, fontsize=9.5, color=GAIN,
             fontweight="bold", ha="center")
axD.set_title("D   Label-free transductive system-mean smoothing",
              loc="left", fontsize=10.5, fontweight="bold", pad=9)
note(axD, "the largest single gain on the evaluation set")

# --- figure furniture ---
fig.suptitle("VoiceMOS Challenge 2026 — Track 3 results (Team T15)",
             x=0.055, y=0.965, ha="left", fontsize=15, fontweight="bold", color=INK)
fig.text(0.055, 0.925,
         "Speaker- and accent-similarity MOS prediction.  B1/B2 are the official "
         "baselines; T13 submitted accent scores only.",
         ha="left", fontsize=9, color=INK2)
h, l = axA.get_legend_handles_labels()
fig.legend(h, l, loc="upper right", bbox_to_anchor=(0.985, 0.975), ncol=2,
           frameon=False, fontsize=9.5, handlelength=1.1, handleheight=1.1,
           columnspacing=1.4)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "results.png")
fig.savefig(out, dpi=200, facecolor=SURFACE)
print("saved:", out)
