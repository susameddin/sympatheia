"""Train ResNet18 on AffectNet for face emotion classification.

Usage:
    python -m face_emotion.train
    python -m face_emotion.train --epochs 50 --batch_size 32
"""

import argparse
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import (
    BATCH_SIZE, CACHE_DIR, DEVICE, EMOTION_NAMES, EMOTION_TO_VA,
    EPOCHS, LR, NUM_CLASSES, PATIENCE, SEED, WEIGHT_DECAY,
)
from .dataset import get_dataset
from .models import FaceEmotionModel


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(dataset) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss."""
    targets = [s[1] for s in dataset.samples]
    counts = np.bincount(targets, minlength=NUM_CLASSES).astype(np.float32)
    weights = 1.0 / (counts + 1e-6)
    weights /= weights.sum()
    weights *= NUM_CLASSES
    return torch.from_numpy(weights)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    per_class_acc = {}
    for i, name in enumerate(EMOTION_NAMES):
        mask = all_labels == i
        if mask.sum() > 0:
            per_class_acc[name] = float((all_preds[mask] == i).mean())

    return total_loss / total, correct / total, per_class_acc


def main(args):
    set_seed(args.seed)
    os.makedirs(CACHE_DIR, exist_ok=True)
    device = args.device

    print("Loading AffectNet datasets...")
    train_ds = get_dataset("train")
    val_ds = get_dataset("val")
    test_ds = get_dataset("test")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Class weights for imbalanced Sad class
    class_weights = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = FaceEmotionModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"\nTraining FaceEmotionModel (ResNet18) for {args.epochs} epochs...")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
        )
        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
            f"lr={lr_now:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(CACHE_DIR, "face_emotion.pt"),
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")

    # Load best model for test evaluation
    model.load_state_dict(
        torch.load(
            os.path.join(CACHE_DIR, "face_emotion.pt"),
            map_location=device, weights_only=True,
        )
    )

    test_loss, test_acc, per_class = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Results:")
    print(f"  Overall accuracy: {test_acc:.3f}")
    print(f"  Loss: {test_loss:.4f}")
    print(f"\n  Per-class accuracy:")
    for name, acc in per_class.items():
        va = EMOTION_TO_VA[name]
        print(f"    {name:10s}: {acc:.3f}  (V={va[0]:+.2f}, A={va[1]:+.2f})")

    # Save training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="train")
    ax1.plot(val_losses, label="val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Loss")

    ax2.plot(train_accs, label="train")
    ax2.plot(val_accs, label="val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.set_title("Accuracy")

    plt.tight_layout()
    plt.savefig(os.path.join(CACHE_DIR, "training_curves.png"), dpi=150)
    print(f"\nTraining curves saved to {CACHE_DIR}/training_curves.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train face emotion model")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()
    main(args)
