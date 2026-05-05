#!/usr/bin/env python3
"""Generate a simplified circular Valence-Arousal circumplex figure.

Blended color regions inside a circle, with axis labels only.

Outputs:
    figures/va_circumplex_simple.png  (300 DPI raster)
    figures/va_circumplex_simple.pdf  (vector)

Usage:
    python figures/plot_va_circumplex_simple.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Quadrant color anchors — placed at quadrant centers
# (valence, arousal, R, G, B)
# ---------------------------------------------------------------------------
COLOR_ANCHORS = [
    ( 0.55,  0.55, 0.95, 0.75, 0.30),  # top-right: yellow-orange (happy)
    (-0.55,  0.55, 0.90, 0.40, 0.40),  # top-left: red/warm (angry)
    (-0.55, -0.55, 0.30, 0.35, 0.75),  # bottom-left: dark blue (sad)
    ( 0.55, -0.55, 0.40, 0.75, 0.70),  # bottom-right: green-teal (calm)
]

_RES = 600
_SIGMA = 0.55  # larger sigma = bigger blended areas


def _build_circular_heatmap():
    """Build a blended color heatmap masked to a circle."""
    R = 1.0
    x = np.linspace(-R, R, _RES)
    y = np.linspace(-R, R, _RES)
    X, Y = np.meshgrid(x, y)

    # Start with white
    img = np.ones((_RES, _RES, 3))

    # Blend each color anchor as a Gaussian blob
    for vx, vy, cr, cg, cb in COLOR_ANCHORS:
        d2 = (X - vx) ** 2 + (Y - vy) ** 2
        w = np.exp(-d2 / (2.0 * _SIGMA ** 2))
        color = np.array([cr, cg, cb])
        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1 - w * 0.55) + color[c] * w * 0.55

    # Hard circular mask (colors fill to the edge)
    dist = np.sqrt(X ** 2 + Y ** 2)
    alpha = (dist <= R).astype(float)

    # Build RGBA
    rgba = np.ones((_RES, _RES, 4))
    rgba[:, :, :3] = np.clip(img, 0.0, 1.0)
    rgba[:, :, 3] = alpha

    return rgba


def create_va_circumplex_simple():
    """Create a clean circular VA circumplex figure."""
    fig, ax = plt.subplots(figsize=(5, 5))

    # Blended heatmap background (circular)
    heatmap = _build_circular_heatmap()
    ax.imshow(
        heatmap,
        extent=[-1.0, 1.0, -1.0, 1.0],
        origin="lower",
        aspect="equal",
        zorder=0,
    )

    # Circle border
    circle = mpatches.Circle((0, 0), 1.0, fill=False,
                             edgecolor="#555555", linewidth=1.2, zorder=4)
    ax.add_patch(circle)

    # Bidirectional arrow axes inside the circle
    arrow_kw = dict(arrowstyle="<->, head_width=0.2, head_length=0.1",
                    color="#333333", linewidth=1.0, zorder=5)
    ext = 0.97  # stay inside circle
    ax.annotate("", xy=(ext, 0), xytext=(-ext, 0), arrowprops=arrow_kw)
    ax.annotate("", xy=(0, ext), xytext=(0, -ext), arrowprops=arrow_kw)

    # Axis endpoint labels (outside circle)
    label_kw = dict(fontsize=14, fontweight="bold", color="#000000", zorder=10)
    ax.text(0, 1.08, "Aroused", ha="center", va="bottom", **label_kw)
    ax.text(0, -1.08, "Calm", ha="center", va="top", **label_kw)
    ax.text(1.08, 0, "Positive", ha="left", va="center", **label_kw)
    ax.text(-1.08, 0, "Negative", ha="right", va="center", **label_kw)

    # Clean up
    margin = 0.45
    ax.set_xlim(-1.0 - margin, 1.0 + margin)
    ax.set_ylim(-1.0 - margin, 1.0 + margin)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout(pad=0.5)
    return fig


if __name__ == "__main__":
    fig = create_va_circumplex_simple()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    png_path = os.path.join(out_dir, "va_circumplex_simple.png")
    pdf_path = os.path.join(out_dir, "va_circumplex_simple.pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
