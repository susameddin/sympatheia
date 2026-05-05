"""Train YAAD ECG emotion classifier.

Usage:
    python -m ecg_emotion.train [--force] [--epochs N] [--device cpu|cuda]

Trains ResNet1D+Deep on the full YAAD dataset (80/20 split) and saves the
checkpoint to cache/ecg_resnet_deep_model.pt.

Also trains RF + DT on HRV features for comparison and saves the RF to
cache/ecg_rf_model.pkl.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from .config import CACHE_DIR, EMOTION_NAMES, NUM_CLASSES, RANDOM_STATE, TEST_SIZE
from .experiments import _class_weights, _augment, eval_model
from .features import build_feature_matrix
from .loader import load_all_data
from .models import (
    ECGResNet1DDeep, _MODEL_FILENAME, _RESNET_FILENAME, save_model, train_rf,
)


def train_resnet(
    raw_norm: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 30,
    seed: int = RANDOM_STATE,
) -> ECGResNet1DDeep:
    """Train ResNet1D+Deep on an 80/20 split, return the best model."""
    from torch.utils.data import DataLoader, TensorDataset

    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=TEST_SIZE, random_state=seed, stratify=y
    )
    torch.manual_seed(seed)

    X_tr = torch.from_numpy(raw_norm[train_idx, None, :])
    X_te = torch.from_numpy(raw_norm[test_idx,  None, :])
    y_tr = torch.from_numpy(y[train_idx].astype(np.int64))
    y_te = torch.from_numpy(y[test_idx].astype(np.int64))

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_te, y_te), batch_size=256, shuffle=False)

    model = ECGResNet1DDeep(n_classes=NUM_CLASSES).to(device)
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
    parser.add_argument("--force",  action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading YAAD data ...")
    records = load_all_data(force=args.force)
    print(f"  {len(records)} records")

    print("Extracting features ...")
    X, y = build_feature_matrix(records, cache_dir=CACHE_DIR, force=args.force)
    raw  = np.stack([r["ecg"] for r in records]).astype(np.float32)
    raw_norm = (raw - raw.mean(axis=1, keepdims=True)) / (raw.std(axis=1, keepdims=True) + 1e-8)

    unique, counts = np.unique(y, return_counts=True)
    print("  Class distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"    {EMOTION_NAMES[cls]:10s}: {cnt}")

    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain: {len(train_idx)}  Test: {len(test_idx)}")

    # ---- Decision Tree (HRV baseline) ----
    print("\n--- Decision Tree ---")
    dt = DecisionTreeClassifier(criterion="entropy", max_depth=12, random_state=RANDOM_STATE)
    dt.fit(X[train_idx], y[train_idx])
    print(classification_report(y[test_idx], dt.predict(X[test_idx]),
                                target_names=EMOTION_NAMES, zero_division=0))

    # ---- Random Forest (HRV baseline) ----
    print("--- Random Forest ---")
    rf = train_rf(X[train_idx], y[train_idx])
    print(classification_report(y[test_idx], rf.predict(X[test_idx]),
                                target_names=EMOTION_NAMES, zero_division=0))
    rf_path = os.path.join(CACHE_DIR, _MODEL_FILENAME)
    save_model(rf, rf_path)
    print(f"RF saved → {rf_path}")

    # ---- ResNet1D+Deep (production model) ----
    print("\n--- ResNet1D+Deep (PyTorch) ---")
    model = train_resnet(raw_norm, y, device, epochs=args.epochs)
    resnet_path = os.path.join(CACHE_DIR, _RESNET_FILENAME)
    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save(model.state_dict(), resnet_path)
    print(f"ResNet1D+Deep saved → {resnet_path}")

    # Save held-out test indices so precompute can avoid training data
    test_idx_path = os.path.join(CACHE_DIR, "ecg_test_indices.npy")
    np.save(test_idx_path, test_idx)
    print(f"Test indices saved → {test_idx_path}")


if __name__ == "__main__":
    main()
