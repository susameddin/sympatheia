#!/usr/bin/env python3
"""Generate a publication-ready Valence-Arousal circumplex emotion wheel figure.

Outputs:
    figures/va_circumplex.png  (300 DPI raster)
    figures/va_circumplex.pdf  (vector)

Usage:
    python figures/plot_va_circumplex.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Emotion anchors — placed around the circumplex perimeter
# ---------------------------------------------------------------------------
EMOTION_ANCHORS = {
    "Happy":      ( 0.70,  0.35),
    "Excited":    ( 0.50,  0.75),
    "Angry":      (-0.70,  0.70),
    "Anxious":    (-0.40,  0.60),
    "Disgusted":  (-0.80, -0.20),
    "Frustrated": (-0.80,  0.25),
    "Sad":        (-0.65, -0.60),
    "Tired":      (-0.15, -0.75),
    "Relaxed":    ( 0.25, -0.55),
    "Surprised":  ( 0.10,  0.85),
    "Neutral":    ( 0.00,  0.00),
    "Content":    ( 0.55, -0.20),
}

EMOTION_COLORS = {
    "Happy":      (0.90, 0.75, 0.10),
    "Excited":    (1.00, 0.65, 0.00),
    "Angry":      (0.85, 0.10, 0.10),
    "Anxious":    (0.50, 0.10, 0.70),
    "Disgusted":  (0.55, 0.30, 0.55),
    "Frustrated": (0.75, 0.30, 0.30),
    "Sad":        (0.30, 0.45, 0.85),
    "Tired":      (0.50, 0.50, 0.70),
    "Relaxed":    (0.20, 0.65, 0.55),
    "Surprised":  (0.95, 0.50, 0.80),
    "Neutral":    (0.60, 0.60, 0.60),
    "Content":    (0.40, 0.80, 0.20),
}

# ---------------------------------------------------------------------------
# Heatmap parameters
# ---------------------------------------------------------------------------
_HEATMAP_SIGMA = 0.40
_HEATMAP_RES   = 500
_HEATMAP_ALPHA = 0.55


def _rounded_rect_sdf(X, Y, half_w, half_h, radius):
    """Signed distance field for a rounded rectangle centered at origin."""
    # Distance to the inner rectangle edges, then round the corners
    dx = np.abs(X) - (half_w - radius)
    dy = np.abs(Y) - (half_h - radius)
    dx_pos = np.maximum(dx, 0.0)
    dy_pos = np.maximum(dy, 0.0)
    corner_dist = np.sqrt(dx_pos ** 2 + dy_pos ** 2) - radius
    inner_dist = np.minimum(np.maximum(dx, dy), 0.0)
    return corner_dist + inner_dist


def _build_heatmap():
    """Precompute Gaussian blob clouds with rounded-rect soft fade."""
    names = list(EMOTION_ANCHORS.keys())
    coords = np.array([EMOTION_ANCHORS[n] for n in names])
    colors = np.array([EMOTION_COLORS[n] for n in names])

    R = 1.3
    x = np.linspace(-R, R, _HEATMAP_RES)
    y = np.linspace(-R, R, _HEATMAP_RES)
    X, Y = np.meshgrid(x, y)

    img = np.ones((_HEATMAP_RES, _HEATMAP_RES, 3))

    for i in range(len(names)):
        d2 = (X - coords[i, 0]) ** 2 + (Y - coords[i, 1]) ** 2
        w = np.exp(-d2 / (2.0 * _HEATMAP_SIGMA ** 2))
        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1 - w * 0.45) + colors[i, c] * w * 0.45

    # Soft fade using rounded-rect SDF
    sdf = _rounded_rect_sdf(X, Y, half_w=1.0, half_h=1.0, radius=0.25)
    fade_width = 0.20
    alpha_channel = np.clip(1.0 - sdf / fade_width, 0.0, 1.0)
    # Smooth step for nicer falloff
    alpha_channel = alpha_channel * alpha_channel * (3 - 2 * alpha_channel)

    # Blend towards white in the fade region
    for c in range(3):
        img[:, :, c] = img[:, :, c] * alpha_channel + 1.0 * (1 - alpha_channel)

    # Combine into RGBA
    rgba = np.ones((_HEATMAP_RES, _HEATMAP_RES, 4))
    rgba[:, :, :3] = np.clip(img, 0.0, 1.0)
    rgba[:, :, 3] = alpha_channel

    return rgba


def create_va_circumplex():
    """Create a clean, compact VA circumplex figure."""
    fig, ax = plt.subplots(figsize=(5, 5))

    # Heatmap background (rounded square, soft fade)
    heatmap = _build_heatmap()
    ax.imshow(
        heatmap,
        extent=[-1.3, 1.3, -1.3, 1.3],
        origin="lower",
        aspect="equal",
        alpha=_HEATMAP_ALPHA,
        zorder=0,
    )

    # Cross-hair axes (thin lines)
    ax.plot([-1.05, 1.05], [0, 0], color="#999999", linewidth=0.7, zorder=1)
    ax.plot([0, 0], [-1.05, 1.05], color="#999999", linewidth=0.7, zorder=1)

    # Axis endpoint labels
    ax.text( 0,  1.10, "Aroused", ha="center", va="bottom", fontsize=9,
            fontweight="bold", color="#444444", zorder=10)
    ax.text( 0, -1.10, "Calm", ha="center", va="top", fontsize=9,
            fontweight="bold", color="#444444", zorder=10)
    ax.text( 1.03, -0.03, "Positive", ha="right", va="top", fontsize=9,
            fontweight="bold", color="#444444", zorder=10)
    ax.text(-1.03, -0.03, "Negative", ha="left", va="top", fontsize=9,
            fontweight="bold", color="#444444", zorder=10)

    # Emotion dots + labels below
    for name, (v, a) in EMOTION_ANCHORS.items():
        color = EMOTION_COLORS[name]
        # Darker edge: same hue but reduced brightness
        edge_color = tuple(c * 0.6 for c in color)
        ax.plot(
            v, a, "o",
            color=color,
            markersize=7,
            markeredgecolor=edge_color,
            markeredgewidth=1.2,
            zorder=6,
        )
        # Label just below the dot
        label_color = tuple(c * 0.65 for c in color)
        ax.text(
            v, a - 0.09, name,
            ha="center", va="top",
            fontsize=9,
            fontweight="bold",
            color=label_color,
            zorder=7,
        )

    # Small axis indicator in bottom-right corner (inside the heatmap)
    ax_x, ax_y = 1.0, -1.0  # corner origin
    ax_len = 0.22
    ax.plot([ax_x, ax_x - ax_len], [ax_y, ax_y], color="#333333", lw=1.5, zorder=10)
    ax.plot([ax_x, ax_x], [ax_y, ax_y + ax_len], color="#333333", lw=1.5, zorder=10)
    ax.text(ax_x - ax_len / 2 - 0.04, ax_y - 0.02, "Valence", ha="center", va="top",
            fontsize=9, fontweight="bold", color="#444444", zorder=10)
    ax.text(ax_x + 0.02, ax_y + ax_len / 2 + 0.04, "Arousal", ha="left", va="center",
            fontsize=9, fontweight="bold", color="#444444", zorder=10,
            rotation=90)

    # Remove all spines, ticks, labels
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout(pad=0.5)
    return fig


if __name__ == "__main__":
    fig = create_va_circumplex()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    png_path = os.path.join(out_dir, "va_circumplex.png")
    pdf_path = os.path.join(out_dir, "va_circumplex.pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
