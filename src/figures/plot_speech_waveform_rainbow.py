#!/usr/bin/env python3
"""Generate a rainbow-gradient speech waveform with transparent background.

Outputs:
    figures/speech_waveform_rainbow.png  (300 DPI, transparent)
    figures/speech_waveform_rainbow.pdf  (vector, transparent)

Usage:
    python figures/plot_speech_waveform_rainbow.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.io import wavfile
from scipy.signal import resample

# ── Configuration ──────────────────────────────────────────────────────────
AUDIO_PATH = os.path.join(os.path.dirname(__file__),
                          "..", "new_samples", "train", "happy_query", "new_Happy_00004.wav")
OUT_DIR = os.path.dirname(__file__)

# ── Load data ──────────────────────────────────────────────────────────────
sr, sig = wavfile.read(AUDIO_PATH)
sig = sig.astype(np.float64)
if sig.ndim > 1:
    sig = sig[:, 0]  # mono
sig /= np.max(np.abs(sig)) + 1e-8  # normalise to [-1, 1]
# Extract a portion with consistent activity, minimal silence
T_START = 3.1
T_DURATION = 0.8
sig = sig[int(T_START * sr): int((T_START + T_DURATION) * sr)]

# ── Downsample to ~500 Hz for a clean trace ───────────────────────────────
target_sr = 500
n_samples = int(len(sig) * target_sr / sr)
sig = resample(sig, n_samples)
time = np.arange(len(sig)) / target_sr

# ── Build rainbow-colored line segments ───────────────────────────────────
points = np.column_stack([time, sig]).reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Color each segment by its x-position (0→1 across the waveform)
norm = plt.Normalize(time[0], time[-1])
lc = LineCollection(segments, cmap="rainbow_r", norm=norm, linewidth=0.8)
lc.set_array(time[:-1])

# ── Plot: bare waveform, no axes ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 2.5))
fig.patch.set_alpha(0.0)
ax.set_facecolor("none")

ax.add_collection(lc)
ax.set_xlim(time[0], time[-1])
ax.set_ylim(sig.min() * 1.1, sig.max() * 1.1)
ax.axis("off")

# ── Save ───────────────────────────────────────────────────────────────────
for ext in ("png", "pdf"):
    out = os.path.join(OUT_DIR, f"speech_waveform_rainbow.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    print(f"Saved {out}")

plt.close(fig)
