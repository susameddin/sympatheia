#!/usr/bin/env python3
"""Generate a rainbow-gradient bar-style speech waveform with transparent background.

Outputs:
    figures/speech_waveform_bars_rainbow.png  (300 DPI, transparent)
    figures/speech_waveform_bars_rainbow.pdf  (vector, transparent)

Usage:
    python figures/plot_speech_waveform_bars_rainbow.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io import wavfile

# ── Configuration ──────────────────────────────────────────────────────────
AUDIO_PATH = os.path.join(os.path.dirname(__file__),
                          "..", "new_samples", "train", "happy_query", "new_Happy_00000.wav")
OUT_DIR = os.path.dirname(__file__)
N_BARS = 100
BAR_WIDTH_FRAC = 0.75

# ── Load data ──────────────────────────────────────────────────────────────
sr, sig = wavfile.read(AUDIO_PATH)
sig = sig.astype(np.float64)
if sig.ndim > 1:
    sig = sig[:, 0]
sig /= np.max(np.abs(sig)) + 1e-8

T_START = 4.0
T_DURATION = 1.5
sig = sig[int(T_START * sr): int((T_START + T_DURATION) * sr)]

# ── Compute RMS per bar ───────────────────────────────────────────────────
window = len(sig) // N_BARS
heights = np.array([
    np.sqrt(np.mean(sig[i * window:(i + 1) * window] ** 2))
    for i in range(N_BARS)
])
heights /= heights.max() + 1e-8

# ── Plot ───────────────────────────────────────────────────────────────────
spacing = 1.0 / N_BARS
bar_w = spacing * BAR_WIDTH_FRAC
x = np.linspace(spacing / 2, 1.0 - spacing / 2, N_BARS)

cmap = cm.get_cmap("rainbow_r")
colors = [cmap(i / (N_BARS - 1)) for i in range(N_BARS)]

fig, ax = plt.subplots(figsize=(10, 2.5))
fig.patch.set_alpha(0.0)
ax.set_facecolor("none")

ax.bar(x, 2 * heights, width=bar_w, bottom=-heights, color=colors)
ax.set_xlim(0, 1)
ax.set_ylim(-1.15, 1.15)
ax.axis("off")

# ── Save ───────────────────────────────────────────────────────────────────
for ext in ("png", "pdf"):
    out = os.path.join(OUT_DIR, f"speech_waveform_bars_rainbow.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    print(f"Saved {out}")

plt.close(fig)
