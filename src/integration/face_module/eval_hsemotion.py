"""Evaluate the HSEmotion pretrained model on AffectNet+ val set.

Loads samples_val.pkl (built by dataset.py) and runs enet_b0_8_va_mtl,
then reports per-class accuracy and VA RMSE alongside the baseline (58.9%).

Usage:
    python -m face_emotion.eval_hsemotion
"""

import os
import pickle

import numpy as np
from PIL import Image

from .config import CACHE_DIR, EMOTION_NAMES

# HSEmotion class order (same indices, slightly different name strings)
_HSEMO_IDX_TO_NAME = {
    0: "Anger", 1: "Contempt", 2: "Disgust", 3: "Fear",
    4: "Happy", 5: "Neutral", 6: "Sad", 7: "Surprise",
}

BASELINE_ACC = {
    "overall": 0.589,
    "Anger":    0.546, "Contempt": None,   "Disgust": 0.471,
    "Fear":     0.688, "Happy":    0.806,  "Neutral": 0.498,
    "Sad":      0.654, "Surprise": 0.576,
}


def _load_val_samples():
    cache_path = os.path.join(CACHE_DIR, "samples_val.pkl")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Val cache not found at {cache_path}. "
            "Run `python -m face_emotion.dataset` or train first to build it."
        )
    with open(cache_path, "rb") as f:
        samples = pickle.load(f)
    print(f"Loaded {len(samples)} val samples from cache.")
    return samples


def run_eval(batch_size: int = 64, device: str = "cuda"):
    from emotiefflib.facial_analysis import EmotiEffLibRecognizer

    recognizer = EmotiEffLibRecognizer(
        engine="torch", model_name="enet_b0_8_va_mtl", device=device
    )
    print("HSEmotion enet_b0_8_va_mtl loaded.")

    samples = _load_val_samples()
    # samples: list of (img_path, class_idx, valence, arousal)

    all_preds, all_labels = [], []
    all_pred_v, all_pred_a = [], []
    all_true_v, all_true_a = [], []

    n = len(samples)
    for start in range(0, n, batch_size):
        batch = samples[start : start + batch_size]

        imgs = []
        for img_path, cls, v_gt, a_gt in batch:
            img = np.array(Image.open(img_path).convert("RGB"))
            imgs.append(img)
            all_labels.append(cls)
            all_true_v.append(v_gt)
            all_true_a.append(a_gt)

        _, scores = recognizer.predict_emotions(imgs, logits=False)
        # scores: (B, 10) — first 8 are emotion probs, last 2 are [valence, arousal]
        emo_probs = scores[:, :8]
        pred_cls = np.argmax(emo_probs, axis=1)
        all_preds.extend(pred_cls.tolist())
        all_pred_v.extend(scores[:, 8].tolist())
        all_pred_a.extend(scores[:, 9].tolist())

        if (start // batch_size + 1) % 10 == 0:
            done = min(start + batch_size, n)
            print(f"  {done}/{n} ({100*done/n:.0f}%)")

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_pred_v = np.array(all_pred_v)
    all_pred_a = np.array(all_pred_a)
    all_true_v = np.array(all_true_v)
    all_true_a = np.array(all_true_a)

    # --- Overall accuracy ---
    acc = (all_preds == all_labels).mean()
    base = BASELINE_ACC["overall"]
    print(f"\nOverall accuracy: {acc:.3f}  (baseline: {base:.3f}, delta: {acc-base:+.3f})")

    # --- Per-class accuracy ---
    print("\nPer-class accuracy:")
    print(f"  {'Emotion':12s}  {'HSEmotion':>10s}  {'Baseline':>10s}  {'Delta':>8s}")
    for idx, name in _HSEMO_IDX_TO_NAME.items():
        mask = all_labels == idx
        if mask.sum() == 0:
            continue
        cls_acc = (all_preds[mask] == idx).mean()
        base_cls = BASELINE_ACC.get(name)
        delta_str = f"{cls_acc - base_cls:+.3f}" if base_cls is not None else "  n/a"
        base_str  = f"{base_cls:.3f}" if base_cls is not None else "   n/a"
        print(f"  {name:12s}  {cls_acc:10.3f}  {base_str:>10s}  {delta_str:>8s}")

    # --- VA metrics ---
    v_rmse = float(np.sqrt(np.mean((all_pred_v - all_true_v) ** 2)))
    a_rmse = float(np.sqrt(np.mean((all_pred_a - all_true_a) ** 2)))
    print(f"\nValence RMSE:  {v_rmse:.4f}")
    print(f"Arousal RMSE:  {a_rmse:.4f}")
    print(f"Pred V range:  [{all_pred_v.min():.3f}, {all_pred_v.max():.3f}]")
    print(f"Pred A range:  [{all_pred_a.min():.3f}, {all_pred_a.max():.3f}]")
    print(f"True V range:  [{all_true_v.min():.3f}, {all_true_v.max():.3f}]")
    print(f"True A range:  [{all_true_a.min():.3f}, {all_true_a.max():.3f}]")

    return acc, v_rmse, a_rmse


if __name__ == "__main__":
    run_eval()
