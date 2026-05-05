#!/usr/bin/env python3
"""Pre-compute HSEmotion predictions on sampled AffectNet+ val images.

Loads samples_val.pkl, samples N images per emotion class, runs
enet_b0_8_va_mtl (PyTorch backend via emotiefflib), and saves predictions
to JSON for use by eval/generate_responses/sensing/generate_responses_face.py.

Usage:
    conda run -n s --no-capture-output python -m face_emotion.precompute_hsemotion

    # Custom sampling:
    conda run -n s --no-capture-output python -m face_emotion.precompute_hsemotion \\
        --n-per-class 10 --seed 42 \\
        --output face_emotion/cache/hsemotion_predictions.json
"""

import argparse
import json
import os
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.face_module.config import EMOTION_NAMES, CACHE_DIR
from src.constants import SPEECH_ANCHORS, _FACE_TO_SPEECH

DEFAULT_VAL_CACHE = os.path.join(CACHE_DIR, "samples_val.pkl")
DEFAULT_OUTPUT = os.path.join(CACHE_DIR, "face_predictions.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-compute HSEmotion predictions on AffectNet+ val images"
    )
    parser.add_argument(
        "--val-cache", type=str, default=DEFAULT_VAL_CACHE,
        help=f"Path to samples_val.pkl. Default: {DEFAULT_VAL_CACHE}",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help=f"Output JSON path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--n-per-class", type=int, default=6,
        help="Images to sample per emotion class (default: 6)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for emotiefflib inference (default: cuda)",
    )
    return parser.parse_args()


def load_val_cache(cache_path: str) -> list:
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Val cache not found: {cache_path}\n"
            "Run `python -m face_emotion.dataset` to build it."
        )
    with open(cache_path, "rb") as f:
        samples = pickle.load(f)
    print(f"Loaded {len(samples)} val samples from cache.")
    return samples


def sample_images(cache_samples: list, n_per_class: int, seed: int) -> list:
    """Sample n_per_class images per class. Returns list of dicts."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for img_path, class_idx, v, a in cache_samples:
        by_class[class_idx].append((img_path, v, a))

    result = []
    for class_idx in sorted(by_class.keys()):
        emotion = EMOTION_NAMES[class_idx]
        pool = by_class[class_idx]
        chosen = rng.sample(pool, min(n_per_class, len(pool)))
        for i, (img_path, v, a) in enumerate(sorted(chosen)):
            result.append({
                "id": f"{emotion.lower()}_{i:02d}",
                "emotion": emotion,
                "image_path": img_path,
            })
        print(f"  {emotion:<12}: {len(chosen)} images")
    return result


def main():
    args = parse_args()

    print(f"Loading val cache: {args.val_cache}")
    cache_samples = load_val_cache(args.val_cache)

    print(f"\nSampling {args.n_per_class} images per class (seed={args.seed})...")
    samples = sample_images(cache_samples, args.n_per_class, args.seed)
    print(f"Total: {len(samples)} images\n")

    # Build GT VA lookup from speech anchors (same values judge_qwen3omni_emotional.py will use)
    gt_va = {
        emo: SPEECH_ANCHORS[anchor]
        for emo, anchor in _FACE_TO_SPEECH.items()
    }
    # Contempt not in _FACE_TO_SPEECH; map to Disgust anchor
    gt_va.setdefault("Contempt", SPEECH_ANCHORS["disgusted"])

    print(f"Loading HSEmotion enet_b0_8_va_mtl (device={args.device})...")
    from emotiefflib.facial_analysis import EmotiEffLibRecognizer
    recognizer = EmotiEffLibRecognizer(
        engine="torch", model_name="enet_b0_8_va_mtl", device=args.device
    )
    print("HSEmotion loaded.\n")

    # HSEmotion class order (same as _HSEMO_IDX_TO_NAME in models.py)
    hsemo_names = ["Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    # Speech VA matrix for weighted-sum (same logic as HSEmotionFacePredictor.predict_va)
    speech_va_matrix = np.array(
        [gt_va[name] for name in hsemo_names], dtype=np.float32
    )

    print("Running HSEmotion predictions...")
    predictions = []
    for i, s in enumerate(samples):
        img = np.array(Image.open(s["image_path"]).convert("RGB"))
        _, scores = recognizer.predict_emotions([img], logits=False)
        # scores: (1, 10) — first 8 are emotion probs, last 2 are [valence, arousal]
        emo_probs = scores[0, :8]
        top_idx = int(np.argmax(emo_probs))
        top_emotion = hsemo_names[top_idx]

        # Weighted sum against speech anchors (same as HSEmotionFacePredictor.predict_va)
        va = emo_probs @ speech_va_matrix
        pred_v = float(np.clip(va[0], -1, 1))
        pred_a = float(np.clip(va[1], -1, 1))

        gt_v, gt_a = gt_va.get(s["emotion"], (0.0, 0.0))

        probs_dict = {name: float(p) for name, p in zip(hsemo_names, emo_probs)}

        predictions.append({
            "id": s["id"],
            "emotion": s["emotion"],
            "image_path": s["image_path"],
            "top_emotion": top_emotion,
            "predicted_valence": pred_v,
            "predicted_arousal": pred_a,
            "gt_valence": gt_v,
            "gt_arousal": gt_a,
            "probs": probs_dict,
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  [{i+1}/{len(samples)}] {s['id']}: GT={s['emotion']:<10} "
                f"pred={top_emotion:<10} VA=({pred_v:+.2f}, {pred_a:+.2f})"
            )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"\nSaved {len(predictions)} predictions → {args.output}")

    # Quick accuracy summary
    correct = sum(1 for p in predictions if p["top_emotion"] == p["emotion"])
    print(f"Top-1 accuracy: {correct}/{len(predictions)} = {correct/len(predictions):.1%}")


if __name__ == "__main__":
    main()
