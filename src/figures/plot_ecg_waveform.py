#!/usr/bin/env python3
"""Generate a bare ECG waveform with transparent background (YAAD dataset).

Outputs:
    figures/ecg_waveform.png  (300 DPI, transparent)
    figures/ecg_waveform.pdf  (vector, transparent)

Usage:
    python figures/plot_ecg_waveform.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────
FS = 128          # Hz (approx: 5000 samples / 39 s)
T_START = 5.0
T_DURATION = 9.0

DATA_FILE = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "Datasets", "YAAD", "ECG_GSR_Emotions",
    "Raw Data", "Multimodal", "ECG", "ECGdata_s1p10v1.dat",
)
OUT_DIR = os.path.dirname(__file__)

# ── Load data ──────────────────────────────────────────────────────────────
sig = np.loadtxt(DATA_FILE, delimiter=",").astype(np.float64)

start = int(T_START * FS)
end   = int((T_START + T_DURATION) * FS)
sig   = sig[start:end]

# Z-score normalise so amplitude is centred and scaled consistently
sig = (sig - sig.mean()) / (sig.std() + 1e-8)

time = np.arange(len(sig)) / FS

# ── Plot: bare waveform, no axes ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 2.5))
fig.patch.set_alpha(0.0)
ax.set_facecolor("none")

ax.plot(time, sig, color="#782121", linewidth=3.2)
ax.axis("off")
ax.set_xlim(time[0], time[-1])

# ── Save ───────────────────────────────────────────────────────────────────
for ext in ("png", "pdf"):
    out = os.path.join(OUT_DIR, f"ecg_waveform.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True)
    print(f"Saved {out}")

plt.close(fig)
