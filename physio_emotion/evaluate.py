"""Evaluate a trained PhysioVAModel on the DEAP test set.

Produces:
  - Per-dimension CCC, Pearson r, RMSE, MAE (printed + saved to metrics.json)
  - Scatter plots: predicted vs ground-truth valence and arousal
  - Error distribution histograms

Usage:
    python -m physio_emotion.evaluate
    python -m physio_emotion.evaluate --weights physio_emotion/cache/physio_va_deap.pt
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import (
    BATCH_SIZE,
    CACHE_DIR,
    DEAP_DATA_DIR,
    DEVICE,
    HOP_SAMPLES,
    TEST_SUBJECTS,
    WINDOW_SAMPLES,
)
from .dataset import DEAPPhysioDataset
from .models import PhysioVAModel


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    preds_all, labels_all = [], []
    for bvp, gsr, y in loader:
        preds_all.append(model(bvp.to(device), gsr.to(device)).cpu().numpy())
        labels_all.append(y.numpy())
    return np.concatenate(preds_all), np.concatenate(labels_all)


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    metrics = {}
    for i, dim in enumerate(["valence", "arousal"]):
        p, t = preds[:, i], labels[:, i]
        pm, tm = p.mean(), t.mean()
        pv, tv = p.var(),  t.var()
        cov    = ((p - pm) * (t - tm)).mean()
        ccc    = float(2.0 * cov / (pv + tv + (pm - tm) ** 2 + 1e-8))
        r      = float(np.corrcoef(p, t)[0, 1])
        rmse   = float(np.sqrt(((p - t) ** 2).mean()))
        mae    = float(np.abs(p - t).mean())
        metrics[dim] = {"ccc": ccc, "r": r, "rmse": rmse, "mae": mae}
    return metrics


def plot_scatter(preds, labels, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for i, (dim, ax) in enumerate(zip(["Valence", "Arousal"], axes)):
        p, t = preds[:, i], labels[:, i]
        ax.scatter(t, p, alpha=0.15, s=8, color="darkorange")
        lims = [-1.1, 1.1]
        ax.plot(lims, lims, "k--", linewidth=1, label="ideal")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(f"Ground-truth {dim}"); ax.set_ylabel(f"Predicted {dim}")
        ax.set_title(f"{dim} — Predicted vs Ground Truth (BVP+GSR)")
        ax.set_aspect("equal"); ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "va_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved scatter plot → {path}")


def plot_error_hist(preds, labels, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, (dim, ax) in enumerate(zip(["Valence", "Arousal"], axes)):
        errors = preds[:, i] - labels[:, i]
        ax.hist(errors, bins=50, color="darkorange", edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="k", linestyle="--")
        ax.set_xlabel("Prediction error"); ax.set_ylabel("Count")
        ax.set_title(f"{dim} error distribution  (mean={errors.mean():.3f})")
    fig.tight_layout()
    path = os.path.join(output_dir, "error_hist.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved error histogram → {path}")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Loading test dataset…")
    ds_test = DEAPPhysioDataset(TEST_SUBJECTS, args.data_dir, WINDOW_SAMPLES, HOP_SAMPLES, augment=False)
    loader  = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"  {len(ds_test):,} windows from subjects {TEST_SUBJECTS}")

    print("Loading model…")
    model = PhysioVAModel().to(device)
    model.load_state_dict(
        torch.load(args.weights, map_location=device, weights_only=True)
    )

    preds, labels = collect_predictions(model, loader, device)
    metrics = compute_metrics(preds, labels)

    print("\n=== EVALUATION RESULTS ===")
    for dim in ["valence", "arousal"]:
        m = metrics[dim]
        print(f"  {dim:8s}:  CCC={m['ccc']:+.4f}  r={m['r']:+.4f}  "
              f"RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}")

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"model": args.weights, "subjects": TEST_SUBJECTS, **metrics}, f, indent=2)
    print(f"Saved metrics → {metrics_path}")

    plot_scatter(preds, labels, args.output_dir)
    plot_error_hist(preds, labels, args.output_dir)


if __name__ == "__main__":
    default_weights = os.path.join(CACHE_DIR, "physio_va_deap.pt")
    default_out     = "results/physio_emotion_eval"
    parser = argparse.ArgumentParser(description="Evaluate PhysioVAModel on DEAP test set")
    parser.add_argument("--weights",    default=default_weights)
    parser.add_argument("--data_dir",   default=DEAP_DATA_DIR)
    parser.add_argument("--output_dir", default=default_out)
    parser.add_argument("--device",     default=DEVICE)
    main(parser.parse_args())
