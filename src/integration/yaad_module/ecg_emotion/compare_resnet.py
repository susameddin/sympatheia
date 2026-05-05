"""Compare ResNet1D baseline vs ResNet1D+Deep across multiple seeds.

Runs 5-fold stratified CV with 3 different random seeds per model,
giving 15 fold measurements per model for a reliable comparison.

Usage:
    python -m ecg_emotion.compare_resnet [--epochs N] [--device cpu|cuda]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from .config import CACHE_DIR, NUM_CLASSES, RANDOM_STATE
from .features import build_all_feature_matrix, build_feature_matrix
from .loader import load_all_data
from .models import ECGResNet1D, ECGResNet1DDeep
from .experiments import run_pytorch_cv


SEEDS = [42, 123, 456]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    records = load_all_data()
    raw = np.stack([r["ecg"] for r in records]).astype("float32")
    raw_norm = (raw - raw.mean(axis=1, keepdims=True)) / (raw.std(axis=1, keepdims=True) + 1e-8)
    _, y = build_feature_matrix(records, cache_dir=CACHE_DIR)
    print(f"  {len(records)} records, signal shape: {raw_norm.shape}\n")

    models = [
        ("ResNet1D (baseline)", ECGResNet1D),
        ("ResNet1D+Deep",       ECGResNet1DDeep),
    ]

    all_results: dict[str, list[float]] = {name: [] for name, _ in models}

    for seed in SEEDS:
        print(f"=== Seed {seed} ===")
        for name, model_cls in models:
            print(f"  {name} ...")
            accs = run_pytorch_cv(raw_norm, y, model_cls, device,
                                  n_folds=5, epochs=args.epochs, seed=seed)
            all_results[name].extend(accs)
            print(f"    seed mean: {np.mean(accs):.3f} ± {np.std(accs):.3f}\n")

    print("=" * 54)
    print(f"{'Model':<24} {'Acc':>8}  {'±Std':>6}  {'Min':>6}  {'Max':>6}  {'n':>4}")
    print("-" * 54)
    for name, _ in models:
        accs = all_results[name]
        print(f"{name:<24} {np.mean(accs):>8.3f}  {np.std(accs):>6.3f}"
              f"  {min(accs):>6.3f}  {max(accs):>6.3f}  {len(accs):>4}")
    print("=" * 54)
    print(f"(15 folds per model = 3 seeds × 5-fold CV, chance = {1/NUM_CLASSES:.3f})")


if __name__ == "__main__":
    main()
