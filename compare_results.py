#!/usr/bin/env python3
"""
Compare evaluation metrics between two model runs (e.g., base vs. fine-tuned).

Reads two metrics.json files produced by evaluate_model.py and prints a
formatted comparison table organized by metric category.

Usage:
    python compare_results.py \\
        --base results/eval_base/metrics.json \\
        --finetuned results/eval_ckpt700/metrics.json \\
        --base-label "GLM-4-Voice Base" \\
        --finetuned-label "GLM-4-Voice + LoRA (ckpt-700)"
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import ALL_EMOTIONS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare base vs. fine-tuned model evaluation metrics"
    )
    parser.add_argument("--base", type=str, required=True,
                        help="Path to base model metrics.json")
    parser.add_argument("--finetuned", type=str, required=True,
                        help="Path to fine-tuned model metrics.json")
    parser.add_argument("--base-label", type=str, default="Base",
                        help="Display label for base model column")
    parser.add_argument("--finetuned-label", type=str, default="Fine-tuned",
                        help="Display label for fine-tuned model column")
    return parser.parse_args()


def load_metrics(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def fmt(value) -> str:
    """Format a metric value for display."""
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def delta_str(base_val, ft_val, lower_is_better: bool = False) -> str:
    """Return a formatted delta string with directional arrow.

    lower_is_better=True  → negative delta is good (↓ = improvement)
    lower_is_better=False → positive delta is good (↑ = improvement)
    """
    if base_val is None or ft_val is None:
        return "N/A"
    delta = ft_val - base_val
    sign = "+" if delta >= 0 else ""
    if lower_is_better:
        arrow = "↓" if delta < 0 else ("↑" if delta > 0 else "=")
    else:
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
    return f"{sign}{delta:.4f} {arrow}"


def row(label: str, base_val, ft_val, lower_is_better: bool = False, col_w: int = 14) -> str:
    """Format a single metric row."""
    b = fmt(base_val)
    f = fmt(ft_val)
    d = delta_str(base_val, ft_val, lower_is_better)
    return f"  {label:<35} {b:>{col_w}} {f:>{col_w}} {d:>{col_w}}"


def section_header(title: str, base_label: str, ft_label: str, col_w: int = 14) -> str:
    sep = "-" * (35 + col_w * 3 + 8)
    header = f"  {'Metric':<35} {base_label:>{col_w}} {ft_label:>{col_w}} {'Delta':>{col_w}}"
    return f"\n[{title}]\n{sep}\n{header}\n{sep}"


def main():
    args = parse_args()

    base_m = load_metrics(args.base)
    ft_m = load_metrics(args.finetuned)

    col_w = max(14, len(args.base_label) + 2, len(args.finetuned_label) + 2)
    sep = "=" * (35 + col_w * 3 + 8)

    print(f"\n{sep}")
    print(f"  EMOTION MODEL EVALUATION COMPARISON")
    print(f"{sep}")
    print(f"  Base:        {args.base}")
    print(f"  Fine-tuned:  {args.finetuned}")

    # Samples info
    print(f"\n  Samples evaluated:  base={base_m.get('total_samples', 'N/A')}  "
          f"ft={ft_m.get('total_samples', 'N/A')}")
    print(f"  Failed samples:     base={base_m.get('failed_samples', 'N/A')}  "
          f"ft={ft_m.get('failed_samples', 'N/A')}")

    # -----------------------------------------------------------------------
    # Speech Quality
    # -----------------------------------------------------------------------
    print(section_header("Speech Quality", args.base_label, args.finetuned_label, col_w))
    print(row("WER ↓",
              base_m.get("wer_mean"), ft_m.get("wer_mean"),
              lower_is_better=True, col_w=col_w))
    print(row("UTMOS ↑",
              base_m.get("utmos_mean"), ft_m.get("utmos_mean"),
              lower_is_better=False, col_w=col_w))

    # -----------------------------------------------------------------------
    # Response Coherence
    # -----------------------------------------------------------------------
    print(section_header("Response Coherence", args.base_label, args.finetuned_label, col_w))
    print(row("BERTScore F1 ↑",
              base_m.get("bertscore_f1_mean"), ft_m.get("bertscore_f1_mean"),
              lower_is_better=False, col_w=col_w))
    print(row("ROUGE-L ↑",
              base_m.get("rougeL_mean"), ft_m.get("rougeL_mean"),
              lower_is_better=False, col_w=col_w))

    # -----------------------------------------------------------------------
    # Per-Emotion Accuracy
    # -----------------------------------------------------------------------
    print(section_header("Per-Emotion Nearest-Anchor Accuracy", args.base_label,
                          args.finetuned_label, col_w))

    base_per = base_m.get("per_emotion_accuracy", {})
    ft_per = ft_m.get("per_emotion_accuracy", {})

    base_vals, ft_vals = [], []
    for emo in ALL_EMOTIONS:
        b = base_per.get(emo)
        f = ft_per.get(emo)
        print(row(f"{emo}", b, f, lower_is_better=False, col_w=col_w))
        if b is not None:
            base_vals.append(b)
        if f is not None:
            ft_vals.append(f)

    if base_vals and ft_vals:
        mean_b = sum(base_vals) / len(base_vals)
        mean_f = sum(ft_vals) / len(ft_vals)
        dash = "-" * (35 + col_w * 3 + 8)
        print(f"  {dash}")
        print(row("Mean", mean_b, mean_f, lower_is_better=False, col_w=col_w))

    # -----------------------------------------------------------------------
    # Per-Emotion WER (if available)
    # -----------------------------------------------------------------------
    base_wer_per = base_m.get("wer_per_emotion")
    ft_wer_per = ft_m.get("wer_per_emotion")
    if base_wer_per or ft_wer_per:
        print(section_header("Per-Emotion WER", args.base_label, args.finetuned_label, col_w))
        b_dict = base_wer_per or {}
        f_dict = ft_wer_per or {}
        for emo in ALL_EMOTIONS:
            print(row(emo, b_dict.get(emo), f_dict.get(emo),
                      lower_is_better=True, col_w=col_w))

    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
