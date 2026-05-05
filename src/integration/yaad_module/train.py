"""Train the GSR-only ResNet1D+Deep model on YAAD multimodal records.

The ECG model is already trained (ecg_emotion.train).  This script trains
the same architecture on GSR signals from the 391 multimodal records and
saves the checkpoint to cache/gsr_resnet_deep_model.pt.

Usage:
    python -m integration.yaad_module.train [--epochs 200] [--device cuda]
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .ecg_emotion.loader import load_multimodal_data
from .ecg_emotion.models import ECGResNet1DDeep
from .ecg_emotion.experiments import _class_weights, _augment, eval_model

from .config import CACHE_DIR, EMOTION_NAMES, GSR_WEIGHTS, NUM_CLASSES


def _znorm(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)


def train_gsr(
    gsr_norm: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 30,
    seed: int = 42,
) -> ECGResNet1DDeep:
    from torch.utils.data import DataLoader, TensorDataset

    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y
    )
    torch.manual_seed(seed)

    X_tr = torch.from_numpy(gsr_norm[train_idx, None, :])
    X_te = torch.from_numpy(gsr_norm[test_idx,  None, :])
    y_tr = torch.from_numpy(y[train_idx].astype(np.int64))
    y_te = torch.from_numpy(y[test_idx].astype(np.int64))

    train_loader = torch.utils.data.DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        TensorDataset(X_te, y_te), batch_size=256, shuffle=False
    )

    model = ECGResNet1DDeep(n_classes=NUM_CLASSES, in_channels=1).to(device)
    cw    = _class_weights(y[train_idx], NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc, best_state, patience_left = 0.0, None, patience

    for ep in range(1, epochs + 1):
        model.train()
        for x, lbl in train_loader:
            x, lbl = x.to(device), lbl.to(device)
            x = _augment(x)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, lbl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        val_acc = eval_model(model, test_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"  Early stop at epoch {ep}")
                break

        if ep % 20 == 0:
            print(f"  epoch {ep:>3}/{epochs}  val_acc={val_acc:.3f}  best={best_acc:.3f}")

    model.load_state_dict(best_state)
    print(f"  Final test accuracy: {eval_model(model, test_loader, device):.3f}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading multimodal records ...")
    records = load_multimodal_data()
    print(f"  {len(records)} records")

    le = LabelEncoder().fit(EMOTION_NAMES)
    y  = np.array([le.transform([r["emotion"]])[0] for r in records], dtype=np.int32)
    gsr = np.stack([r["gsr"] for r in records]).astype(np.float32)
    gsr_norm = _znorm(gsr)

    unique, counts = np.unique(y, return_counts=True)
    print("  Class distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"    {EMOTION_NAMES[cls]:10s}: {cnt}")

    print("\n--- ResNet1D+Deep on GSR ---")
    model = train_gsr(gsr_norm, y, device, epochs=args.epochs)

    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save(model.state_dict(), GSR_WEIGHTS)
    print(f"GSR model saved → {GSR_WEIGHTS}")

    # Save held-out test indices so precompute can avoid training data
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )
    test_idx_path = os.path.join(CACHE_DIR, "gsr_test_indices.npy")
    np.save(test_idx_path, test_idx)
    print(f"Test indices saved → {test_idx_path}")


if __name__ == "__main__":
    main()
