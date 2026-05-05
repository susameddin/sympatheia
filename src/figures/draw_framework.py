"""
Framework schematic figure for paper.
Run: conda run -n s --no-capture-output python figures/draw_framework.py
Outputs: figures/framework.pdf and figures/framework.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Color palette ──────────────────────────────────────────────────────────────
C_EEG    = "#4E79A7"
C_PHYSIO = "#59A14F"
C_AUDIO  = "#F28E2B"
C_VA     = "#E15759"
C_LLM    = "#76B7B2"
C_DEC    = "#EDC948"
C_BOX_BG = "#F7F7F7"
C_BOX_ED = "#BBBBBB"
C_TEXT   = "#222222"

# ── Emotion VA anchors ─────────────────────────────────────────────────────────
EMOTIONS = {
    "Sad":       (-0.75, -0.65),
    "Bored":     (-0.45, -0.60),
    "Disgusted": (-0.65,  0.15),
    "Angry":     (-0.55,  0.70),
    "Fearful":   (-0.30,  0.65),
    "Neutral":   ( 0.00,  0.00),
    "Surprised": ( 0.20,  0.70),
    "Contempt":  (-0.40,  0.10),
    "Happy":     ( 0.85,  0.35),
    "Excited":   ( 0.75,  0.90),
    "Calm":      ( 0.60, -0.40),
}
EMO_COLORS = [
    "#4E79A7", "#A0CBE8", "#F28E2B", "#FFBE7D", "#59A14F",
    "#8CD17D", "#B6992D", "#F1CE63", "#499894", "#86BCB6", "#E15759",
]


def rounded_box(ax, x, y, w, h, facecolor, edgecolor=C_BOX_ED,
                lw=1.2, alpha=0.92, radius=0.01):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=edgecolor, facecolor=facecolor,
        alpha=alpha, transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(box)


def label(ax, x, y, text, fontsize=8.5, color=C_TEXT,
          ha="center", va="center", bold=False):
    ax.text(x, y, text, fontsize=fontsize, color=color,
            ha=ha, va=va, fontweight="bold" if bold else "normal",
            transform=ax.transAxes)


def arrow(ax, x0, y0, x1, y1, color="#555555", lw=1.6, shrink=3):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                        shrinkA=shrink, shrinkB=shrink,
                        connectionstyle="arc3,rad=0.0"),
    )


# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 8.5), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(0.5, 0.97, "Emotion-Aware Speech Synthesis Framework",
        fontsize=14, fontweight="bold", ha="center", va="top",
        transform=ax.transAxes, color=C_TEXT)

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN A — Emotion Sensing  (x 0.01 – 0.285)
# ══════════════════════════════════════════════════════════════════════════════
CX, CW = 0.01, 0.275

label(ax, CX + CW / 2, 0.92, "Emotion Sensing", fontsize=10, bold=True)

def sensing_block(ax, y, h, edge_col, title, sub1, sub2, out):
    rounded_box(ax, CX, y, CW, h, C_BOX_BG, edgecolor=edge_col, lw=2.0)
    sx, sw = CX + 0.01, CW - 0.02
    # header
    rounded_box(ax, sx, y + h - 0.085, sw, 0.072, edge_col, alpha=0.22, radius=0.008)
    label(ax, sx + sw / 2, y + h - 0.05, title, fontsize=8, color=edge_col, bold=True)
    # middle
    rounded_box(ax, sx, y + h - 0.165, sw, 0.068, edge_col, alpha=0.10, radius=0.008)
    label(ax, sx + sw / 2, y + h - 0.132, sub1, fontsize=7.5)
    # bottom
    rounded_box(ax, sx, y + 0.005, sw, 0.085, edge_col, alpha=0.30, radius=0.008)
    label(ax, sx + sw / 2, y + 0.047, sub2, fontsize=7.5)
    # internal arrows
    arrow(ax, sx + sw / 2, y + h - 0.085,
              sx + sw / 2, y + h - 0.097, color=edge_col, lw=1.1, shrink=1)
    arrow(ax, sx + sw / 2, y + h - 0.165,
              sx + sw / 2, y + h - 0.175, color=edge_col, lw=1.1, shrink=1)
    # output label
    ax.text(CX + CW + 0.005, y + 0.045, out,
            fontsize=7, color=edge_col, ha="left", va="center",
            style="italic", transform=ax.transAxes)

sensing_block(ax, 0.60, 0.28, C_EEG,
    "EEG  (32 channels, 128 Hz)",
    "Differential Entropy\n(5 bands × 32 ch → 160-d)",
    "EEGDEModel (MLP, per-subject)\n~70% binary accuracy",
    "(v, a)")

sensing_block(ax, 0.32, 0.26, C_PHYSIO,
    "Physio  (BVP, GSR, Resp, Temp, EMG×2)",
    "6 × 1D-CNN Streams",
    "Channel Attention Fusion\n(per-subject ~70% accuracy)",
    "(v, a)")

sensing_block(ax, 0.05, 0.25, C_AUDIO,
    "Speech Audio  /  Text Description",
    "wav2vec2 (Audeering)\nor GLM-4 text → VA",
    "→ (valence, arousal)",
    "(v, a)")

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN B — VA Emotion Space  (x 0.32 – 0.53)
# ══════════════════════════════════════════════════════════════════════════════
BX, BW = 0.32, 0.205

label(ax, BX + BW / 2, 0.92, "VA Emotion Space", fontsize=10, bold=True)

rounded_box(ax, BX, 0.07, BW, 0.78, C_BOX_BG, edgecolor=C_VA, lw=2.0, radius=0.015)

ax_va = fig.add_axes([BX + 0.015, 0.13, BW - 0.03, 0.66])
ax_va.set_facecolor("#FAFAFA")
ax_va.spines[["top", "right"]].set_visible(False)
ax_va.spines[["left", "bottom"]].set_color("#AAAAAA")
ax_va.tick_params(labelsize=6.5, colors="#555555", length=3)
ax_va.set_xlim(-1.1, 1.1)
ax_va.set_ylim(-1.1, 1.1)
ax_va.set_xlabel("Valence", fontsize=7.5, color="#444444")
ax_va.set_ylabel("Arousal", fontsize=7.5, color="#444444")
ax_va.axhline(0, color="#DDDDDD", lw=0.8, zorder=0)
ax_va.axvline(0, color="#DDDDDD", lw=0.8, zorder=0)

names = list(EMOTIONS.keys())
vals  = [EMOTIONS[n][0] for n in names]
aros  = [EMOTIONS[n][1] for n in names]
ax_va.scatter(vals, aros, c=EMO_COLORS, s=55, zorder=3, edgecolors="white", linewidths=0.6)

offsets = {
    "Sad": (-0.07, -0.14), "Bored": (0.08, -0.13), "Disgusted": (0.09, 0.0),
    "Angry": (-0.13, 0.09), "Fearful": (0.09, 0.0), "Neutral": (0.08, 0.0),
    "Surprised": (0.08, 0.0), "Contempt": (0.09, 0.0), "Happy": (0.06, -0.13),
    "Excited": (0.06, 0.09), "Calm": (0.08, 0.0),
}
for n, col in zip(names, EMO_COLORS):
    dx, dy = offsets.get(n, (0.05, 0.05))
    ax_va.annotate(n, (EMOTIONS[n][0], EMOTIONS[n][1]),
                   xytext=(EMOTIONS[n][0] + dx, EMOTIONS[n][1] + dy),
                   fontsize=6, color=col, ha="left", va="center",
                   arrowprops=dict(arrowstyle="-", color=col, lw=0.5))

ax_va.set_title("11-emotion anchors", fontsize=6.5, pad=3, color="#555555")

# Arrows: sensing → VA space (all 3 streams converge to mid-right of col A)
for (ym, col) in [(0.74, C_EEG), (0.45, C_PHYSIO), (0.175, C_AUDIO)]:
    arrow(ax, CX + CW, ym, BX, 0.46, color=col, lw=1.8, shrink=2)

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN C — GLM-4-Voice + Decoder  (x 0.565 – 0.99)
# ══════════════════════════════════════════════════════════════════════════════
DX, DW = 0.565, 0.425

label(ax, DX + DW / 2, 0.92, "Emotion-Conditioned Speech Generation",
      fontsize=10, bold=True)

# ── LLM box ───────────────────────────────────────────────────────────────────
LLM_Y, LLM_H = 0.42, 0.44
rounded_box(ax, DX, LLM_Y, DW, LLM_H, C_BOX_BG, edgecolor=C_LLM, lw=2.0, radius=0.015)

# Token sequence strip
tok_x, tok_w = DX + 0.01, DW - 0.02
rounded_box(ax, tok_x, LLM_Y + 0.28, tok_w, 0.11, C_LLM, alpha=0.13, radius=0.008)

# Token boxes
tok_labels = ["[SYS]", "valence=v", "arousal=a", "T₁", "T₂", "…", "Tₙ", "A₁", "A₂", "…", "Aₘ"]
tok_colors = ["#888888", C_VA, C_VA, "#AAAAAA", "#AAAAAA", "#AAAAAA",
              "#AAAAAA", C_LLM, C_LLM, C_LLM, C_LLM]
n = len(tok_labels)
cell_w = tok_w / n
for i, (tl, tc) in enumerate(zip(tok_labels, tok_colors)):
    tx = tok_x + i * cell_w + 0.002
    tw = cell_w - 0.003
    rounded_box(ax, tx, LLM_Y + 0.295, tw, 0.075, tc, alpha=0.35, radius=0.004)
    ax.text(tx + tw / 2, LLM_Y + 0.332, tl,
            fontsize=5.2, ha="center", va="center",
            transform=ax.transAxes, color=C_TEXT)

ax.text(DX + DW / 2, LLM_Y + 0.275,
        "Input sequence — VA injected as text in system prompt",
        fontsize=6.5, ha="center", va="top", color="#555555",
        transform=ax.transAxes)

# Example system prompt box
sp_x, sp_w, sp_h = DX + 0.01, DW - 0.02, 0.075
rounded_box(ax, sp_x, LLM_Y + 0.19, sp_w, sp_h, C_VA, alpha=0.08, radius=0.007)
ax.text(DX + DW / 2, LLM_Y + 0.229,
        '<|system|>  "Please respond in English. User emotion (valence=0.75, arousal=0.90)"',
        fontsize=6.5, ha="center", va="center", color=C_VA,
        transform=ax.transAxes, style="italic",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="none", edgecolor="none"))

# Main LLM label
rounded_box(ax, sp_x, LLM_Y + 0.03, sp_w, 0.14, C_LLM, alpha=0.18, radius=0.010)
label(ax, DX + DW / 2, LLM_Y + 0.10,
      "GLM-4-Voice  LLM\n(LoRA fine-tuned, r=8, α=16)  →  audio token prediction",
      fontsize=8.5)

ax.text(DX + DW - 0.005, LLM_Y + LLM_H - 0.01, "GLM-4-Voice",
        fontsize=7, ha="right", va="top", color=C_LLM,
        fontweight="bold", transform=ax.transAxes)

# ── Decoder box ───────────────────────────────────────────────────────────────
DEC_Y, DEC_H = 0.10, 0.28
rounded_box(ax, DX, DEC_Y, DW, DEC_H, C_BOX_BG, edgecolor=C_DEC, lw=2.0, radius=0.015)

half_w = (DW - 0.03) / 2
rounded_box(ax, DX + 0.01, DEC_Y + 0.09, half_w, 0.13, C_DEC, alpha=0.28, radius=0.008)
label(ax, DX + 0.01 + half_w / 2, DEC_Y + 0.155, "Flow Matching\n(flow.pt)", fontsize=7.5)

rounded_box(ax, DX + 0.02 + half_w, DEC_Y + 0.09, half_w, 0.13, C_DEC, alpha=0.28, radius=0.008)
label(ax, DX + 0.02 + half_w + half_w / 2, DEC_Y + 0.155, "HiFi-GAN\nVocoder", fontsize=7.5)

arrow(ax, DX + 0.01 + half_w, DEC_Y + 0.155,
          DX + 0.02 + half_w, DEC_Y + 0.155,
      color=C_DEC, lw=1.2, shrink=1)

label(ax, DX + DW / 2, DEC_Y + 0.04,
      "GLM-4-Voice Decoder  (22 kHz output)", fontsize=7.5, bold=True, color=C_DEC)

# Waveform inset
wf_ax = fig.add_axes([DX + 0.02, DEC_Y - 0.075, DW - 0.04, 0.058])
t = np.linspace(0, 4 * np.pi, 500)
amp = np.exp(-0.12 * t) * np.sin(t * 3.2) * 0.85
wf_ax.plot(t, amp, color=C_DEC, lw=1.5)
wf_ax.fill_between(t, amp, alpha=0.15, color=C_DEC)
wf_ax.set_xlim(0, 4 * np.pi)
wf_ax.set_ylim(-1.1, 1.1)
wf_ax.axis("off")
wf_ax.set_title("Output: emotional speech waveform (22 kHz)",
                fontsize=7, color="#444444", pad=2)

# LLM → Decoder arrow
arrow(ax, DX + DW / 2, LLM_Y, DX + DW / 2, DEC_Y + DEC_H,
      color=C_LLM, lw=1.8, shrink=2)

# VA space → LLM arrow
arrow(ax, BX + BW, 0.46, DX, LLM_Y + LLM_H / 2,
      color=C_VA, lw=1.8, shrink=2)

# ── Legend ────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(facecolor=C_EEG,    label="EEG stream"),
    mpatches.Patch(facecolor=C_PHYSIO, label="Physio stream"),
    mpatches.Patch(facecolor=C_AUDIO,  label="Audio / Text stream"),
    mpatches.Patch(facecolor=C_VA,     label="VA text conditioning"),
    mpatches.Patch(facecolor=C_LLM,    label="GLM-4-Voice LLM"),
    mpatches.Patch(facecolor=C_DEC,    label="Vocoder / Decoder"),
]
ax.legend(handles=handles, loc="lower center", ncol=6,
          fontsize=7.5, framealpha=0.85, edgecolor="#CCCCCC",
          bbox_to_anchor=(0.5, 0.0), bbox_transform=ax.transAxes)

# ── Save ──────────────────────────────────────────────────────────────────────
import os
out_dir = os.path.dirname(__file__)
for path, dpi in [(os.path.join(out_dir, "framework.pdf"), 300),
                  (os.path.join(out_dir, "framework.png"), 200)]:
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")
