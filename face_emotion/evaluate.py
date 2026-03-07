"""Evaluate trained face emotion model on test set.

Usage:
    python -m face_emotion.evaluate
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import BATCH_SIZE, CACHE_DIR, DEVICE, EMOTION_NAMES, EMOTION_TO_VA, NUM_CLASSES
from .dataset import get_dataset
from .models import FaceEmotionModel


@torch.no_grad()
def run_evaluation(device: str = DEVICE):
    os.makedirs(CACHE_DIR, exist_ok=True)

    model = FaceEmotionModel()
    ckpt_path = os.path.join(CACHE_DIR, "face_emotion.pt")
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    model.to(device).eval()

    test_ds = get_dataset("test")
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    all_preds = []
    all_labels = []
    for images, labels in loader:
        logits = model(images.to(device))
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall accuracy
    acc = (all_preds == all_labels).mean()
    print(f"Test accuracy: {acc:.3f}")

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for i, name in enumerate(EMOTION_NAMES):
        mask = all_labels == i
        cls_acc = (all_preds[mask] == i).mean() if mask.sum() > 0 else 0.0
        print(f"  {name:10s}: {cls_acc:.3f}")

    # Confusion matrix
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot confusion matrix
    im = ax1.imshow(cm, interpolation="nearest", cmap="Blues")
    ax1.set_xticks(range(NUM_CLASSES))
    ax1.set_yticks(range(NUM_CLASSES))
    short = [n[:3] for n in EMOTION_NAMES]
    ax1.set_xticklabels(short, rotation=45)
    ax1.set_yticklabels(short)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_title("Confusion Matrix")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)

    # VA scatter plot: predicted VA colored by true emotion
    va_matrix = np.array([EMOTION_TO_VA[e] for e in EMOTION_NAMES], dtype=np.float32)
    # Get softmax probabilities for soft VA mapping
    model.eval()
    all_va = []
    all_true_emo = []
    for images, labels in loader:
        logits = model(images.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        va = probs @ va_matrix
        all_va.append(va)
        all_true_emo.extend(labels.numpy())

    all_va = np.concatenate(all_va)
    all_true_emo = np.array(all_true_emo)

    colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
    for i, name in enumerate(EMOTION_NAMES):
        mask = all_true_emo == i
        ax2.scatter(all_va[mask, 0], all_va[mask, 1], c=[colors[i]], label=name, alpha=0.3, s=5)
    ax2.set_xlabel("Valence")
    ax2.set_ylabel("Arousal")
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.legend(fontsize=7, markerscale=3)
    ax2.set_title("Predicted VA by True Emotion")

    plt.tight_layout()
    out_path = os.path.join(CACHE_DIR, "evaluation.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nEvaluation plots saved to {out_path}")


if __name__ == "__main__":
    run_evaluation()
