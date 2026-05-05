"""Benchmark multiple models on YAAD ECG emotion data.

Compares:
  - sklearn (RF, SVM-RBF, GBM, MLP) on 6 and 17 HRV features
  - PyTorch 1D-CNN and ResNet1D variants on raw ECG signal
  - ResNet ablations: SE blocks, deeper arch, focal loss, strong augmentation
  - Hybrid: ResNet1D body + HRV features concatenated

All evaluated with 5-fold stratified cross-validation.

Usage:
    python -m ecg_emotion.experiments [--folds N] [--epochs N] [--device cpu|cuda]
    python -m ecg_emotion.experiments --ablation   # ResNet ablations only
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                               VotingClassifier)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from .config import CACHE_DIR, EMOTION_NAMES, NUM_CLASSES, RANDOM_STATE
from .features import build_all_feature_matrix, build_feature_matrix
from .loader import load_all_data, load_multimodal_data
from .models import (ECG1DCNN, ECGResNet1D, ECGResNet1DSE,
                     ECGResNet1DDeep, ECGResNet1DDeepSE, ECGResNet1DHybrid)


# ---------------------------------------------------------------------------
# Sklearn cross-validation
# ---------------------------------------------------------------------------

def run_sklearn_cv(
    X: np.ndarray,
    y: np.ndarray,
    clf,
    n_folds: int = 5,
    scale: bool = False,
    seed: int = RANDOM_STATE,
) -> list[float]:
    """Return per-fold test accuracy list."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        if scale:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr, y_tr)
        accs.append(accuracy_score(y_te, clf.predict(X_te)))
    return accs


# ---------------------------------------------------------------------------
# PyTorch helpers
# ---------------------------------------------------------------------------

def _class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)
    w = 1.0 / counts
    return torch.tensor(w / w.sum() * n_classes, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def _augment(x: torch.Tensor) -> torch.Tensor:
    """Gaussian noise + random amplitude scaling."""
    noise = torch.randn_like(x) * 0.01 * x.std()
    scale = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(0.85, 1.15)
    return (x + noise) * scale


def _augment_strong(x: torch.Tensor) -> torch.Tensor:
    """Noise + amplitude scale + time shift + random masking."""
    noise = torch.randn_like(x) * 0.01 * x.std()
    scale = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(0.85, 1.15)
    x = (x + noise) * scale
    # Time shift: roll each sample by a random offset up to ±200 samples
    shift = int(torch.randint(-200, 201, (1,)).item())
    x = torch.roll(x, shift, dims=2)
    # Random masking: zero out a window of 100–500 samples
    T = x.size(2)
    mask_len = int(torch.randint(100, 501, (1,)).item())
    mask_start = int(torch.randint(0, max(1, T - mask_len), (1,)).item())
    x = x.clone()
    x[:, :, mask_start:mask_start + mask_len] = 0.0
    return x


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    augment_fn=_augment,
) -> float:
    model.train()
    correct = total = 0
    for x, lbl in loader:
        x, lbl = x.to(device), lbl.to(device)
        x = augment_fn(x)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, lbl)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        correct += (out.detach().argmax(1) == lbl).sum().item()
        total += len(lbl)
    return correct / total


@torch.no_grad()
def eval_model(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    correct = total = 0
    for x, lbl in loader:
        x, lbl = x.to(device), lbl.to(device)
        correct += (model(x).argmax(1) == lbl).sum().item()
        total += len(lbl)
    return correct / total


def run_pytorch_cv(
    raw: np.ndarray,
    y: np.ndarray,
    model_cls,
    device: torch.device,
    n_folds: int = 5,
    epochs: int = 150,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 25,
    seed: int = RANDOM_STATE,
    augment_fn=_augment,
    use_focal: bool = False,
) -> list[float]:
    """5-fold stratified CV for a PyTorch model class."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(raw, y)):
        torch.manual_seed(seed + fold)

        # Support both (N, T) single-channel and (N, C, T) multi-channel inputs
        if raw.ndim == 2:
            X_tr = torch.from_numpy(raw[train_idx, None, :])
            X_te = torch.from_numpy(raw[test_idx,  None, :])
        else:
            X_tr = torch.from_numpy(raw[train_idx])
            X_te = torch.from_numpy(raw[test_idx])
        y_tr = torch.from_numpy(y[train_idx].astype(np.int64))
        y_te = torch.from_numpy(y[test_idx].astype(np.int64))

        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(TensorDataset(X_te, y_te), batch_size=256, shuffle=False)

        model = model_cls(n_classes=NUM_CLASSES).to(device)
        cw    = _class_weights(y[train_idx], NUM_CLASSES).to(device)
        criterion = FocalLoss(weight=cw) if use_focal else nn.CrossEntropyLoss(weight=cw)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_acc, best_state, patience_left = 0.0, None, patience

        for ep in range(1, epochs + 1):
            train_one_epoch(model, train_loader, optimizer, criterion, device, augment_fn)
            scheduler.step()
            val_acc = eval_model(model, test_loader, device)
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_left = patience
            else:
                patience_left -= 1
                if patience_left == 0:
                    break

        model.load_state_dict(best_state)
        final_acc = eval_model(model, test_loader, device)
        accs.append(final_acc)
        print(f"    fold {fold+1}/{n_folds}: {final_acc:.3f}  (best {best_acc:.3f})")

    return accs


# ---------------------------------------------------------------------------
# Hybrid (signal + HRV features) CV
# ---------------------------------------------------------------------------

def _train_hybrid_epoch(
    model: ECGResNet1DHybrid,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    augment_fn=_augment,
) -> None:
    model.train()
    for x_sig, x_hrv, lbl in loader:
        x_sig, x_hrv, lbl = x_sig.to(device), x_hrv.to(device), lbl.to(device)
        x_sig = augment_fn(x_sig)
        optimizer.zero_grad()
        out = model(x_sig, x_hrv)
        loss = criterion(out, lbl)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


@torch.no_grad()
def _eval_hybrid(
    model: ECGResNet1DHybrid,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = total = 0
    for x_sig, x_hrv, lbl in loader:
        x_sig, x_hrv, lbl = x_sig.to(device), x_hrv.to(device), lbl.to(device)
        correct += (model(x_sig, x_hrv).argmax(1) == lbl).sum().item()
        total += len(lbl)
    return correct / total


def run_pytorch_hybrid_cv(
    raw: np.ndarray,
    hrv: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    n_folds: int = 5,
    epochs: int = 150,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 25,
    seed: int = RANDOM_STATE,
) -> list[float]:
    """5-fold CV for ECGResNet1DHybrid (raw signal + HRV features)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs = []
    n_hrv = hrv.shape[1]

    for fold, (train_idx, test_idx) in enumerate(skf.split(raw, y)):
        torch.manual_seed(seed + fold)

        # z-score HRV features per-fold to avoid leakage
        sc = StandardScaler()
        hrv_tr = sc.fit_transform(hrv[train_idx]).astype(np.float32)
        hrv_te = sc.transform(hrv[test_idx]).astype(np.float32)

        X_sig_tr = torch.from_numpy(raw[train_idx, None, :])
        X_sig_te = torch.from_numpy(raw[test_idx,  None, :])
        X_hrv_tr = torch.from_numpy(hrv_tr)
        X_hrv_te = torch.from_numpy(hrv_te)
        y_tr = torch.from_numpy(y[train_idx].astype(np.int64))
        y_te = torch.from_numpy(y[test_idx].astype(np.int64))

        train_loader = DataLoader(TensorDataset(X_sig_tr, X_hrv_tr, y_tr),
                                  batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(TensorDataset(X_sig_te, X_hrv_te, y_te),
                                  batch_size=256, shuffle=False)

        model = ECGResNet1DHybrid(n_hrv=n_hrv, n_classes=NUM_CLASSES).to(device)
        cw    = _class_weights(y[train_idx], NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_acc, best_state, patience_left = 0.0, None, patience

        for ep in range(1, epochs + 1):
            _train_hybrid_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            val_acc = _eval_hybrid(model, test_loader, device)
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_left = patience
            else:
                patience_left -= 1
                if patience_left == 0:
                    break

        model.load_state_dict(best_state)
        final_acc = _eval_hybrid(model, test_loader, device)
        accs.append(final_acc)
        print(f"    fold {fold+1}/{n_folds}: {final_acc:.3f}  (best {best_acc:.3f})")

    return accs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds",    type=int,  default=5)
    parser.add_argument("--epochs",   type=int,  default=150)
    parser.add_argument("--device",   type=str,  default="cuda")
    parser.add_argument("--force",    action="store_true")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ResNet ablations only (skip sklearn + baseline CNN)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading YAAD records ...")
    records = load_all_data(force=args.force)
    print(f"  {len(records)} records")

    print("Building feature matrices ...")
    X6,  y  = build_feature_matrix(records,     cache_dir=CACHE_DIR, force=args.force)
    X17, _  = build_all_feature_matrix(records, cache_dir=CACHE_DIR, force=args.force)
    raw     = np.stack([r["ecg"] for r in records]).astype(np.float32)
    raw_norm = (raw - raw.mean(axis=1, keepdims=True)) / (raw.std(axis=1, keepdims=True) + 1e-8)

    print(f"  X6={X6.shape}  X17={X17.shape}  raw={raw.shape}\n")

    results: dict[str, list[float]] = {}

    # ---- sklearn models (skip in ablation mode) ----
    if not args.ablation:
        sklearn_suite = [
            ("RF  (6 feat)",
             RandomForestClassifier(100, class_weight="balanced", random_state=RANDOM_STATE),
             X6, False),
            ("RF  (17 feat)",
             RandomForestClassifier(200, class_weight="balanced", random_state=RANDOM_STATE),
             X17, False),
            ("SVM-RBF",
             SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"),
             X17, True),
            ("GBM (17 feat)",
             GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                        learning_rate=0.05, subsample=0.8,
                                        random_state=RANDOM_STATE),
             X17, False),
            ("MLP (17 feat)",
             MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=1000,
                           early_stopping=True, random_state=RANDOM_STATE),
             X17, True),
            ("Ensemble (RF+GBM+SVM)",
             VotingClassifier(estimators=[
                 ("rf",  RandomForestClassifier(200, class_weight="balanced",
                                                random_state=RANDOM_STATE)),
                 ("gbm", GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                                    learning_rate=0.05, subsample=0.8,
                                                    random_state=RANDOM_STATE)),
                 ("svm", Pipeline([("sc", StandardScaler()),
                                    ("clf", SVC(kernel="rbf", C=10, gamma="scale",
                                               class_weight="balanced", probability=True))])),
             ], voting="soft"),
             X17, False),
        ]
        for name, clf, X, scale in sklearn_suite:
            print(f"{name} ...")
            accs = run_sklearn_cv(X, y, clf, n_folds=args.folds, scale=scale)
            results[name] = accs
            print(f"  → {np.mean(accs):.3f} ± {np.std(accs):.3f}\n")

        # baseline CNNs
        for name, signal, model_cls in [
            ("1D-CNN (raw)",     raw,      ECG1DCNN),
            ("1D-CNN (znorm)",   raw_norm, ECG1DCNN),
        ]:
            print(f"{name} ({args.epochs} epochs, {args.folds}-fold CV) ...")
            accs = run_pytorch_cv(signal, y, model_cls, device,
                                  n_folds=args.folds, epochs=args.epochs)
            results[name] = accs
            print(f"  → {np.mean(accs):.3f} ± {np.std(accs):.3f}\n")

    # ---- ResNet ablations ----
    resnet_suite = [
        ("ResNet1D (baseline)",  raw_norm, ECGResNet1D,    {}),
        ("ResNet1D+SE",          raw_norm, ECGResNet1DSE,  {}),
        ("ResNet1D+Deep",        raw_norm, ECGResNet1DDeep, {}),
        ("ResNet1D+Deep+SE",     raw_norm, ECGResNet1DDeepSE, {}),
        ("ResNet1D+FocalLoss",   raw_norm, ECGResNet1D,    {"use_focal": True}),
        ("ResNet1D+StrongAug",   raw_norm, ECGResNet1D,    {"augment_fn": _augment_strong}),
    ]
    for name, signal, model_cls, kwargs in resnet_suite:
        print(f"{name} ({args.epochs} epochs, {args.folds}-fold CV) ...")
        accs = run_pytorch_cv(signal, y, model_cls, device,
                              n_folds=args.folds, epochs=args.epochs, **kwargs)
        results[name] = accs
        print(f"  → {np.mean(accs):.3f} ± {np.std(accs):.3f}\n")

    # ---- Hybrid (ResNet1D + HRV features) ----
    print(f"ResNet1D+Hybrid ({args.epochs} epochs, {args.folds}-fold CV) ...")
    accs = run_pytorch_hybrid_cv(raw_norm, X17, y, device,
                                 n_folds=args.folds, epochs=args.epochs)
    results["ResNet1D+Hybrid"] = accs
    print(f"  → {np.mean(accs):.3f} ± {np.std(accs):.3f}\n")

    # ---- GSR experiments (multimodal subset only) ----
    print("Loading multimodal records (ECG + GSR) ...")
    mm_records = load_multimodal_data(force=args.force)
    print(f"  {len(mm_records)} multimodal records\n")

    from sklearn.preprocessing import LabelEncoder
    _le = LabelEncoder().fit(EMOTION_NAMES)
    y_mm = np.array([_le.transform([r["emotion"]])[0] for r in mm_records], dtype=np.int32)

    raw_mm  = np.stack([r["ecg"] for r in mm_records]).astype(np.float32)
    gsr_mm  = np.stack([r["gsr"] for r in mm_records]).astype(np.float32)

    # per-sample z-score
    def _znorm(x):
        return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)

    ecg_norm = _znorm(raw_mm)
    gsr_norm = _znorm(gsr_mm)
    ecg_gsr  = np.stack([ecg_norm, gsr_norm], axis=1)  # (N, 2, 5000)

    from functools import partial
    ResNet1DDeep2ch = partial(ECGResNet1DDeep, in_channels=2)

    gsr_suite = [
        ("ECG only (MM subset)",  ecg_norm, ECGResNet1DDeep,   {}),
        ("GSR only",              gsr_norm, ECGResNet1DDeep,   {}),
        ("ECG+GSR (2-channel)",   ecg_gsr,  ResNet1DDeep2ch,   {}),
    ]
    for name, signal, model_cls, kwargs in gsr_suite:
        print(f"{name} ({args.epochs} epochs, {args.folds}-fold CV) ...")
        accs = run_pytorch_cv(signal, y_mm, model_cls, device,
                              n_folds=args.folds, epochs=args.epochs, **kwargs)
        results[name] = accs
        print(f"  → {np.mean(accs):.3f} ± {np.std(accs):.3f}\n")

    # ---- summary ----
    print("=" * 54)
    print(f"{'Model':<24} {'Acc':>8}  {'±Std':>6}  {'Min':>6}  {'Max':>6}")
    print("-" * 54)
    for name, accs in sorted(results.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"{name:<24} {np.mean(accs):>8.3f}  {np.std(accs):>6.3f}"
              f"  {min(accs):>6.3f}  {max(accs):>6.3f}")
    print("=" * 54)
    print(f"(Chance level: {1/NUM_CLASSES:.3f})")


if __name__ == "__main__":
    main()
