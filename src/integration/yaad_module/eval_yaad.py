"""Evaluate YAAD ECG / GSR predictors on the YAAD dataset.

Uses 5-fold stratified cross-validation to avoid train/test contamination
(the production models were trained on the full dataset, so we use CV
to get an unbiased accuracy estimate and compare modalities fairly).

Reports:
  - Overall accuracy + per-class accuracy
  - VA RMSE: predicted speech-space VA vs. ground-truth speech anchor VA

Usage:
    python -m integration.yaad_module.eval_yaad --modality ecg
    python -m integration.yaad_module.eval_yaad --modality gsr
    python -m integration.yaad_module.eval_yaad --modality fusion
    python -m integration.yaad_module.eval_yaad --modality all
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from .ecg_emotion.loader import load_all_data, load_multimodal_data
from .ecg_emotion.models import ECGResNet1DDeep
from .ecg_emotion.experiments import run_pytorch_cv
from src.constants import EMOTION_VA_MAPPING as SPEECH_ANCHORS

from .config import (
    CACHE_DIR, ECG_WEIGHTS, GSR_WEIGHTS, EMOTION_NAMES,
    NUM_CLASSES, YAAD_TO_SPEECH,
)

# Ground-truth VA per emotion (speech anchor coordinates)
_GT_VA = np.array(
    [SPEECH_ANCHORS[YAAD_TO_SPEECH[e]] for e in EMOTION_NAMES], dtype=np.float32
)

# Reference results from experiments.py
BASELINES = {
    "ecg":    {"acc": 0.401, "label": "ECG full-dataset 5-fold CV"},
    "gsr":    {"acc": 0.473, "label": "GSR multimodal 5-fold CV"},
    "fusion": {"acc": None,  "label": "Fusion (new)"},
}


def _encode_labels(records: list[dict]) -> np.ndarray:
    le = LabelEncoder().fit(EMOTION_NAMES)
    return np.array([le.transform([r["emotion"]])[0] for r in records], dtype=np.int32)


def _znorm(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)


def eval_modality(modality: str, device: torch.device, n_folds: int = 5) -> dict:
    """Run 5-fold CV evaluation for the given modality. Returns metrics dict."""
    from functools import partial
    from torch.utils.data import DataLoader, TensorDataset

    print(f"\n{'='*60}")
    print(f"Evaluating: {modality.upper()}")
    print(f"{'='*60}")

    if modality in ("ecg",):
        records  = load_all_data()
        signals  = _znorm(np.stack([r["ecg"] for r in records]).astype(np.float32))
        y        = _encode_labels(records)
        weights  = ECG_WEIGHTS
        n_ch     = 1
    elif modality == "gsr":
        records  = load_multimodal_data()
        signals  = _znorm(np.stack([r["gsr"] for r in records]).astype(np.float32))
        y        = _encode_labels(records)
        weights  = GSR_WEIGHTS
        n_ch     = 1
    elif modality == "fusion":
        records  = load_multimodal_data()
        ecg_norm = _znorm(np.stack([r["ecg"] for r in records]).astype(np.float32))
        gsr_norm = _znorm(np.stack([r["gsr"] for r in records]).astype(np.float32))
        y        = _encode_labels(records)

    print(f"  Records: {len(records)}  Classes: {NUM_CLASSES}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_preds, all_labels = [], []
    all_pred_va, all_gt_va = [], []
    fold_accs = []

    # VA matrix (8, 2) in speech coordinates
    from src.constants import EMOTION_VA_MAPPING as _SA
    va_matrix = np.array(
        [_SA[YAAD_TO_SPEECH[e]] for e in EMOTION_NAMES], dtype=np.float32
    )

    for fold, (train_idx, test_idx) in enumerate(skf.split(y, y)):
        torch.manual_seed(42 + fold)

        if modality == "fusion":
            # Train ECG model on fold
            ecg_model = ECGResNet1DDeep(n_classes=NUM_CLASSES, in_channels=1).to(device)
            gsr_model = ECGResNet1DDeep(n_classes=NUM_CLASSES, in_channels=1).to(device)

            for model, sig in [(ecg_model, ecg_norm), (gsr_model, gsr_norm)]:
                _train_fold(model, sig, y, train_idx, device)

            # Evaluate with soft voting
            ecg_model.eval(); gsr_model.eval()
            with torch.no_grad():
                x_ecg = torch.from_numpy(ecg_norm[test_idx, None, :]).to(device)
                x_gsr = torch.from_numpy(gsr_norm[test_idx, None, :]).to(device)
                p_ecg = torch.softmax(ecg_model(x_ecg), dim=1).cpu().numpy()
                p_gsr = torch.softmax(gsr_model(x_gsr), dim=1).cpu().numpy()
                probs = (p_ecg + p_gsr) / 2.0
        else:
            model = ECGResNet1DDeep(n_classes=NUM_CLASSES, in_channels=n_ch).to(device)
            _train_fold(model, signals, y, train_idx, device)
            model.eval()
            with torch.no_grad():
                x = torch.from_numpy(signals[test_idx, None, :]).to(device)
                probs = torch.softmax(model(x), dim=1).cpu().numpy()

        preds = probs.argmax(axis=1)
        labels = y[test_idx]
        acc = (preds == labels).mean()
        fold_accs.append(acc)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

        pred_va = probs @ va_matrix        # (N_test, 2)
        gt_va   = _GT_VA[labels]           # (N_test, 2)
        all_pred_va.append(pred_va)
        all_gt_va.append(gt_va)

        print(f"  fold {fold+1}/{n_folds}: acc={acc:.3f}")

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_pred_va = np.vstack(all_pred_va)
    all_gt_va   = np.vstack(all_gt_va)

    overall_acc = (all_preds == all_labels).mean()
    baseline = BASELINES.get(modality, {}).get("acc")
    delta_str = f"  (baseline {baseline:.3f}, delta {overall_acc-baseline:+.3f})" if baseline else ""
    print(f"\nOverall accuracy:  {overall_acc:.3f} ± {np.std(fold_accs):.3f}{delta_str}")

    print("\nPer-class accuracy:")
    print(f"  {'Emotion':12s}  {'Acc':>8s}  {'N':>6s}")
    for i, name in enumerate(EMOTION_NAMES):
        mask = all_labels == i
        if mask.sum() == 0:
            continue
        cls_acc = (all_preds[mask] == i).mean()
        print(f"  {name:12s}  {cls_acc:8.3f}  {mask.sum():6d}")

    v_rmse = float(np.sqrt(np.mean((all_pred_va[:, 0] - all_gt_va[:, 0]) ** 2)))
    a_rmse = float(np.sqrt(np.mean((all_pred_va[:, 1] - all_gt_va[:, 1]) ** 2)))
    print(f"\nValence RMSE:  {v_rmse:.4f}")
    print(f"Arousal RMSE:  {a_rmse:.4f}")

    return {"modality": modality, "accuracy": overall_acc, "std": np.std(fold_accs),
            "v_rmse": v_rmse, "a_rmse": a_rmse}


def _train_fold(model, signals, y, train_idx, device, epochs=100, patience=20):
    """Train model on one fold (fast training for eval purposes)."""
    from torch.utils.data import DataLoader, TensorDataset
    from .ecg_emotion.experiments import _class_weights, _augment

    X_tr = torch.from_numpy(signals[train_idx, None, :])
    y_tr = torch.from_numpy(y[train_idx].astype(np.int64))
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)

    cw = _class_weights(y[train_idx], NUM_CLASSES).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss, patience_left = float("inf"), patience
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for x, lbl in loader:
            x, lbl = x.to(device), lbl.to(device)
            x = _augment(x)
            optimizer.zero_grad()
            loss = criterion(model(x), lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
        scheduler.step()
        if ep_loss < best_loss:
            best_loss = ep_loss
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["ecg", "gsr", "fusion", "all"],
                        default="ecg")
    parser.add_argument("--folds",  type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    modalities = ["ecg", "gsr", "fusion"] if args.modality == "all" else [args.modality]
    results = []
    for mod in modalities:
        results.append(eval_modality(mod, device, n_folds=args.folds))

    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"{'Modality':<10}  {'Acc':>8}  {'±Std':>6}  {'V-RMSE':>8}  {'A-RMSE':>8}")
        print(f"{'-'*60}")
        for r in results:
            print(f"{r['modality']:<10}  {r['accuracy']:>8.3f}  "
                  f"{r['std']:>6.3f}  {r['v_rmse']:>8.4f}  {r['a_rmse']:>8.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
