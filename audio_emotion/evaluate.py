#!/usr/bin/env python3
"""Evaluate audio emotion recognition on the OpenS2S_11Emo dataset.

Runs the audeering VA predictor on all eval samples and computes:
- Nearest-anchor classification accuracy (overall + per-emotion)
- VA MAE / RMSE vs assigned anchor coordinates
- 11x11 confusion matrix
- VA scatter plot with emotion anchors

Usage:
    python -m audio_emotion.evaluate \
        --metadata /engram/naplab/users/sd3705/Datasets/OpenS2S_11Emo/metadata/sampled_eval.jsonl \
        --audio-dir /engram/naplab/users/sd3705/Datasets/OpenS2S_11Emo/audio/eval \
        --output-dir results/audio_emotion_eval/
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from .models import AudeeringVAPredictor
from .config import EMOTION_VA_MAPPING, ALL_EMOTIONS


def find_nearest_emotion(valence: float, arousal: float) -> str:
    """Find the nearest emotion anchor to a (valence, arousal) point."""
    best_emotion = None
    best_dist = float("inf")
    for emotion, (v, a) in EMOTION_VA_MAPPING.items():
        dist = np.sqrt((valence - v) ** 2 + (arousal - a) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_emotion = emotion
    return best_emotion


def build_audio_path(audio_dir: Path, query_emotion: str, index: int) -> Path:
    """Construct audio path from metadata fields.

    Audio is stored as: audio_dir/{emotion_lower}_query/{index}.wav
    """
    emotion_folder = query_emotion.lower() + "_query"
    return audio_dir / emotion_folder / f"{index}.wav"


def plot_va_scatter(predictions: list, output_path: Path):
    """Plot all predictions on the VA plane, colored by ground truth emotion."""
    # Color map for 11 emotions
    cmap = plt.cm.get_cmap("tab20", len(ALL_EMOTIONS))
    emotion_colors = {emo: cmap(i) for i, emo in enumerate(ALL_EMOTIONS)}

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot predictions grouped by ground truth emotion
    for emo in ALL_EMOTIONS:
        pts = [(p["pred_valence"], p["pred_arousal"]) for p in predictions if p["gt_emotion"] == emo]
        if pts:
            vs, ars = zip(*pts)
            ax.scatter(vs, ars, c=[emotion_colors[emo]], alpha=0.3, s=15, label=f"{emo} (n={len(pts)})")

    # Plot emotion anchors as large markers
    for emo, (v, a) in EMOTION_VA_MAPPING.items():
        ax.scatter(v, a, c="black", s=200, marker="X", zorder=10)
        ax.annotate(emo, (v, a), textcoords="offset points", xytext=(8, 8),
                    fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    ax.set_xlabel("Valence", fontsize=12)
    ax.set_ylabel("Arousal", fontsize=12)
    ax.set_title("Audeering VA Predictions vs Ground Truth Emotion Anchors", fontsize=14)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved VA scatter plot to {output_path}")


def plot_confusion_matrix(y_true: list, y_pred: list, labels: list, output_path: Path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45)
    ax.set_title("Nearest-Anchor Classification Confusion Matrix", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate audio emotion recognition")
    parser.add_argument("--metadata", type=str, required=True,
                        help="Path to sampled_eval.jsonl")
    parser.add_argument("--audio-dir", type=str, required=True,
                        help="Path to audio/eval/ directory")
    parser.add_argument("--output-dir", type=str, default="results/audio_emotion_eval/",
                        help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (for quick testing)")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    print(f"Loading metadata from {args.metadata}")
    samples = []
    with open(args.metadata) as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"Loaded {len(samples)} samples")

    if args.max_samples:
        samples = samples[:args.max_samples]
        print(f"Limited to {len(samples)} samples")

    # Load model
    print("Loading audeering model...")
    predictor = AudeeringVAPredictor(device=args.device)
    print("Model loaded.")

    # Run predictions
    predictions = []
    skipped = 0
    for i, sample in enumerate(samples):
        gt_emotion = sample["query_emotion"]
        index = sample["index"]
        audio_path = build_audio_path(audio_dir, gt_emotion, index)

        if not audio_path.exists():
            skipped += 1
            if skipped <= 5:
                print(f"  Warning: Missing audio {audio_path}")
            continue

        result = predictor.predict(str(audio_path))
        nearest = find_nearest_emotion(result["valence"], result["arousal"])

        # Ground truth VA from the emotion mapping
        gt_v, gt_a = EMOTION_VA_MAPPING[gt_emotion]

        predictions.append({
            "index": index,
            "gt_emotion": gt_emotion,
            "gt_valence": gt_v,
            "gt_arousal": gt_a,
            "pred_valence": result["valence"],
            "pred_arousal": result["arousal"],
            "pred_dominance": result["dominance"],
            "raw_valence": result["raw_valence"],
            "raw_arousal": result["raw_arousal"],
            "raw_dominance": result["raw_dominance"],
            "nearest_emotion": nearest,
            "audio_path": str(audio_path),
        })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(samples)}")

    print(f"\nProcessed {len(predictions)} samples ({skipped} skipped)")

    if not predictions:
        print("No predictions to evaluate!", file=sys.stderr)
        sys.exit(1)

    # Save raw predictions
    pred_path = output_dir / "predictions.jsonl"
    with open(pred_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    print(f"Saved predictions to {pred_path}")

    # -----------------------------------------------------------------------
    # Compute metrics
    # -----------------------------------------------------------------------
    gt_emotions = [p["gt_emotion"] for p in predictions]
    pred_emotions = [p["nearest_emotion"] for p in predictions]

    # Overall nearest-anchor accuracy
    correct = sum(1 for gt, pr in zip(gt_emotions, pred_emotions) if gt == pr)
    overall_acc = correct / len(predictions)

    # Per-emotion accuracy
    per_emotion_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for p in predictions:
        emo = p["gt_emotion"]
        per_emotion_stats[emo]["total"] += 1
        if p["nearest_emotion"] == emo:
            per_emotion_stats[emo]["correct"] += 1

    per_emotion_acc = {}
    for emo in ALL_EMOTIONS:
        stats = per_emotion_stats[emo]
        if stats["total"] > 0:
            per_emotion_acc[emo] = stats["correct"] / stats["total"]
        else:
            per_emotion_acc[emo] = None

    # VA error metrics
    v_errors = [p["pred_valence"] - p["gt_valence"] for p in predictions]
    a_errors = [p["pred_arousal"] - p["gt_arousal"] for p in predictions]

    v_mae = np.mean(np.abs(v_errors))
    a_mae = np.mean(np.abs(a_errors))
    v_rmse = np.sqrt(np.mean(np.square(v_errors)))
    a_rmse = np.sqrt(np.mean(np.square(a_errors)))

    # Per-emotion mean predicted VA (to see systematic offset)
    per_emotion_mean_va = {}
    for emo in ALL_EMOTIONS:
        emo_preds = [p for p in predictions if p["gt_emotion"] == emo]
        if emo_preds:
            mean_v = np.mean([p["pred_valence"] for p in emo_preds])
            mean_a = np.mean([p["pred_arousal"] for p in emo_preds])
            gt_v, gt_a = EMOTION_VA_MAPPING[emo]
            per_emotion_mean_va[emo] = {
                "pred_valence_mean": round(float(mean_v), 4),
                "pred_arousal_mean": round(float(mean_a), 4),
                "gt_valence": gt_v,
                "gt_arousal": gt_a,
                "valence_offset": round(float(mean_v - gt_v), 4),
                "arousal_offset": round(float(mean_a - gt_a), 4),
            }

    # Compile metrics
    metrics = {
        "total_samples": len(predictions),
        "skipped_samples": skipped,
        "overall_nearest_anchor_accuracy": round(overall_acc, 4),
        "valence_mae": round(float(v_mae), 4),
        "arousal_mae": round(float(a_mae), 4),
        "valence_rmse": round(float(v_rmse), 4),
        "arousal_rmse": round(float(a_rmse), 4),
        "per_emotion_accuracy": {k: round(v, 4) if v is not None else None for k, v in per_emotion_acc.items()},
        "per_emotion_mean_va": per_emotion_mean_va,
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Samples evaluated: {len(predictions)}")
    print(f"Overall nearest-anchor accuracy: {overall_acc:.1%}")
    print(f"Valence MAE: {v_mae:.4f}  RMSE: {v_rmse:.4f}")
    print(f"Arousal MAE: {a_mae:.4f}  RMSE: {a_rmse:.4f}")
    print(f"\nPer-emotion accuracy:")
    for emo in ALL_EMOTIONS:
        acc = per_emotion_acc[emo]
        stats = per_emotion_stats[emo]
        acc_str = f"{acc:.1%}" if acc is not None else "N/A"
        print(f"  {emo:<12} {acc_str:>6}  ({stats['correct']}/{stats['total']})")

    print(f"\nPer-emotion mean predicted VA vs assigned anchors:")
    print(f"  {'Emotion':<12} {'Pred V':>8} {'Pred A':>8} {'GT V':>8} {'GT A':>8} {'V offset':>9} {'A offset':>9}")
    for emo in ALL_EMOTIONS:
        if emo in per_emotion_mean_va:
            m = per_emotion_mean_va[emo]
            print(f"  {emo:<12} {m['pred_valence_mean']:>+8.3f} {m['pred_arousal_mean']:>+8.3f} "
                  f"{m['gt_valence']:>+8.2f} {m['gt_arousal']:>+8.2f} "
                  f"{m['valence_offset']:>+9.3f} {m['arousal_offset']:>+9.3f}")

    # Generate plots
    plot_va_scatter(predictions, output_dir / "va_scatter.png")

    # Confusion matrix using only emotions present in predictions
    present_emotions = sorted(set(gt_emotions) | set(pred_emotions),
                              key=lambda e: ALL_EMOTIONS.index(e) if e in ALL_EMOTIONS else 999)
    plot_confusion_matrix(gt_emotions, pred_emotions, present_emotions,
                          output_dir / "confusion_matrix.png")

    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
