#!/usr/bin/env python3
"""Generate a bare EEG waveform (Fp1 channel) with transparent background.

Outputs:
    figures/eeg_waveform.png  (300 DPI, transparent)
    figures/eeg_waveform.pdf  (vector, transparent)

Usage:
    python figures/plot_eeg_waveform.py
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────
SUBJECT = 1
TRIAL = 0
FS = 128  # Hz
T_START = 5.0
T_DURATION = 5.0
CH_INDEX = 0  # Fp1

DATA_DIR = os.path.join(os.path.dirname(__file__),
                        "..", "..", "Datasets", "DEAP", "data_preprocessed_python")
OUT_DIR = os.path.dirname(__file__)

# ── Load data ──────────────────────────────────────────────────────────────
dat_path = os.path.join(DATA_DIR, f"s{SUBJECT:02d}.dat")
with open(dat_path, "rb") as f:
    dat = pickle.load(f, encoding="latin1")

trial_data = dat["data"][TRIAL]  # (40, 8064)
start = int(T_START * FS)
end = int((T_START + T_DURATION) * FS)
sig = trial_data[CH_INDEX, start:end].astype(np.float64)
time = np.arange(len(sig)) / FS

# ── Plot: bare waveform, no axes ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 2.5))
fig.patch.set_alpha(0.0)
ax.set_facecolor("none")

ax.plot(time, sig, color="#386321", linewidth=2.4)
ax.axis("off")
ax.set_xlim(time[0], time[-1])

# ── Save ───────────────────────────────────────────────────────────────────
for ext in ("png", "pdf"):
    out = os.path.join(OUT_DIR, f"eeg_waveform.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    print(f"Saved {out}")

plt.close(fig)
