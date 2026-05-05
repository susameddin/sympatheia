#!/usr/bin/env python3
"""Generate a bare speech waveform with transparent background.

Outputs:
    figures/speech_waveform.png  (300 DPI, transparent)
    figures/speech_waveform.pdf  (vector, transparent)

Usage:
    python figures/plot_speech_waveform.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample

# ── Configuration ──────────────────────────────────────────────────────────
AUDIO_PATH = os.path.join(os.path.dirname(__file__),
                          "..", "new_samples", "train", "happy_query", "new_Happy_00001.wav")
OUT_DIR = os.path.dirname(__file__)

# ── Load data ──────────────────────────────────────────────────────────────
sr, sig = wavfile.read(AUDIO_PATH)
sig = sig.astype(np.float64)
if sig.ndim > 1:
    sig = sig[:, 0]  # mono
sig /= np.max(np.abs(sig)) + 1e-8  # normalise to [-1, 1]
# Extract a portion with consistent activity, minimal silence
T_START = 0.6
T_DURATION = 0.8
sig = sig[int(T_START * sr): int((T_START + T_DURATION) * sr)]
# Compress dynamic range so quiet parts are more visible
sig = np.sign(sig) * np.abs(sig) ** 0.7
time = np.arange(len(sig)) / sr

# ── Downsample to ~1.5 kHz for a clean single-line trace ──────────────────
target_sr = 500
n_samples = int(len(sig) * target_sr / sr)
sig = resample(sig, n_samples)
time = np.arange(len(sig)) / target_sr

# ── Plot: bare waveform, no axes ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 2.5))
fig.patch.set_alpha(0.0)
ax.set_facecolor("none")

ax.plot(time, sig, color="#277be4", linewidth=0.8)
ax.axis("off")
ax.set_xlim(time[0], time[-1])

# ── Save ───────────────────────────────────────────────────────────────────
for ext in ("png", "pdf"):
    out = os.path.join(OUT_DIR, f"speech_waveform.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    print(f"Saved {out}")

plt.close(fig)
