"""Train YAAD ECG and GSR models with 5-fold stratified cross-validation.

Each fold saves:
  cache/ecg_fold{k}_model.pt
  cache/ecg_fold{k}_test_indices.npy
  cache/gsr_fold{k}_model.pt
  cache/gsr_fold{k}_test_indices.npy

These are consumed by precompute_yaad.py --cv to produce ~700 ECG /
~391 GSR predictions with no train/test contamination.

Usage:
    python -m integration.yaad_module.train_cv [--modality ecg|gsr|both] [--folds 5] [--epochs 200]
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from .ecg_emotion.loader import load_all_data, load_multimodal_data
from .ecg_emotion.models import ECGResNet1DDeep
from .ecg_emotion.experiments import _class_weights, _augment, eval_model

from .config import CACHE_DIR, EMOTION_NAMES, NUM_CLASSES


def _znorm(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)


def train_fold(
    signals: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    device: torch.device,
    epochs: int,
    seed: int,
) -> ECGResNet1DDeep:
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    X_tr = torch.from_numpy(signals[train_idx, None, :])
    X_te = torch.from_numpy(signals[test_idx,  None, :])
    y_tr = torch.from_numpy(y[train_idx].astype(np.int64))
    y_te = torch.from_numpy(y[test_idx].astype(np.int64))

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_te, y_te), batch_size=256, shuffle=False)

    model     = ECGResNet1DDeep(n_classes=NUM_CLASSES, in_channels=1).to(device)
    cw        = _class_weights(y[train_idx], NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc, best_state, patience, patience_left = 0.0, None, 30, 30

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
                print(f"    Early stop at epoch {ep}")
                break

        if ep % 20 == 0:
            print(f"    epoch {ep:>3}/{epochs}  val_acc={val_acc:.3f}  best={best_acc:.3f}")

    model.load_state_dict(best_state)
    print(f"    Fold test accuracy: {eval_model(model, test_loader, device):.3f}")
    return model


def run_cv(modality: str, signals: np.ndarray, y: np.ndarray,
           device: torch.device, n_folds: int, epochs: int) -> None:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"\n=== {modality.upper()} {n_folds}-fold CV ===")
    for k, (train_idx, test_idx) in enumerate(skf.split(signals, y)):
        print(f"\n  Fold {k}/{n_folds-1}  (train={len(train_idx)}, test={len(test_idx)})")
        model = train_fold(signals, y, train_idx, test_idx, device,
                           epochs=epochs, seed=42 + k)

        model_path = os.path.join(CACHE_DIR, f"{modality}_fold{k}_model.pt")
        idx_path   = os.path.join(CACHE_DIR, f"{modality}_fold{k}_test_indices.npy")
        torch.save(model.state_dict(), model_path)
        np.save(idx_path, test_idx)
        print(f"    Saved → {model_path}")
        print(f"    Saved → {idx_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["ecg", "gsr", "both"], default="both")
    parser.add_argument("--folds",   type=int, default=5)
    parser.add_argument("--epochs",  type=int, default=200)
    parser.add_argument("--device",  type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    le = LabelEncoder().fit(EMOTION_NAMES)

    if args.modality in ("ecg", "both"):
        print("\nLoading ECG records ...")
        ecg_records = load_all_data()
        print(f"  {len(ecg_records)} records")
        y_ecg = np.array([le.transform([r["emotion"]])[0] for r in ecg_records], dtype=np.int32)
        ecg = np.stack([r["ecg"] for r in ecg_records]).astype(np.float32)
        ecg_norm = _znorm(ecg)
        run_cv("ecg", ecg_norm, y_ecg, device, args.folds, args.epochs)

    if args.modality in ("gsr", "both"):
        print("\nLoading multimodal (GSR) records ...")
        mm_records = load_multimodal_data()
        print(f"  {len(mm_records)} records")
        y_gsr = np.array([le.transform([r["emotion"]])[0] for r in mm_records], dtype=np.int32)
        gsr = np.stack([r["gsr"] for r in mm_records]).astype(np.float32)
        gsr_norm = _znorm(gsr)
        run_cv("gsr", gsr_norm, y_gsr, device, args.folds, args.epochs)

    print("\nDone. Run precompute with --cv to aggregate predictions across folds:")
    print("  python -m integration.yaad_module.precompute_yaad --modality ecg --cv")
    print("  python -m integration.yaad_module.precompute_yaad --modality gsr --cv")


if __name__ == "__main__":
    main()
