"""Train EEG emotion classifier on SEED-VII.

Usage:
    # Within-subject (primary, stratified 70/10/20 video split per subject)
    python -m eeg_emotion.train --mode within_subject

    # Cross-subject (leave-one-subject-out, then train on all)
    python -m eeg_emotion.train --mode cross_subject
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import (
    BATCH_SIZE,
    CACHE_DIR,
    DEVICE,
    EMOTION_NAMES,
    EPOCHS,
    LABEL_SMOOTHING,
    LR,
    N_SUBJECTS,
    NUM_CLASSES,
    PATIENCE,
    SEED,
    SEED_FEATURES_DIR,
    WEIGHT_DECAY,
    WS_BATCH_SIZE,
    WS_EPOCHS,
    WS_LR,
    WS_PATIENCE,
    WS_WEIGHT_DECAY,
)
from .dataset import (
    SEEDEEGDataset, get_normalization_stats, get_paper_fold_split,
    compute_session_norm_stats, get_session_split,
)
from .models import MAETEEGClassifier


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(dataset: SEEDEEGDataset, n_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss."""
    counts = np.zeros(n_classes)
    for y in dataset.y_list:
        counts[y] += 1
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, n_correct, n_total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * x.size(0)
        n_correct += (logits.argmax(1) == y).sum().item()
        n_total += x.size(0)
    return total_loss / max(n_total, 1), n_correct / max(n_total, 1)


@torch.no_grad()
def evaluate(model, loader, device, n_classes: int = NUM_CLASSES):
    """Returns (loss, accuracy, per_class_acc, confusion_matrix)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, n_correct, n_total = 0.0, 0, 0
    confusion = np.zeros((n_classes, n_classes), dtype=int)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        preds = logits.argmax(1)

        total_loss += loss.item() * x.size(0)
        n_correct += (preds == y).sum().item()
        n_total += x.size(0)

        for t, p in zip(y.cpu().numpy(), preds.cpu().numpy()):
            confusion[t, p] += 1

    acc = n_correct / max(n_total, 1)
    per_class_acc = np.zeros(n_classes)
    for c in range(n_classes):
        total_c = confusion[c].sum()
        per_class_acc[c] = confusion[c, c] / max(total_c, 1)

    return total_loss / max(n_total, 1), acc, per_class_acc, confusion


def _train_model(
    model, train_loader, n_epochs, lr, weight_decay,
    class_weights, device, label_smoothing=LABEL_SMOOTHING,
    val_loader=None, patience=None,
):
    """Train model with cosine LR schedule. If val_loader+patience are given,
    applies early stopping on validation loss; otherwise trains for exactly n_epochs."""
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device), label_smoothing=label_smoothing
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(n_epochs, 1)
    )

    best_val_loss = float("inf")
    best_state = None
    p_counter = patience if patience is not None else n_epochs

    for epoch in range(1, n_epochs + 1):
        train_one_epoch(model, train_loader, optimizer, None, criterion, device)
        scheduler.step()

        if val_loader is not None:
            val_loss, _, _, _ = evaluate(model, val_loader, device)
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                p_counter = patience
            else:
                p_counter -= 1
                if p_counter == 0:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Within-subject training: stratified 70/10/20 video split
# ---------------------------------------------------------------------------

def _make_loaders(subject_id, args, train_vids, val_vids, test_vids, norm_stats=None):
    """Build train/val/test DataLoaders for one subject. Returns None on empty split.

    norm_stats: pre-computed {session_id: (mean, std)} from compute_session_norm_stats().
    When provided, all splits share the same normalization statistics (no train/test
    distribution mismatch from computing stats per-split).
    """
    ds_train = SEEDEEGDataset([subject_id], args.features_dir, video_ids=train_vids,
                               session_norm_stats=norm_stats)
    ds_val   = SEEDEEGDataset([subject_id], args.features_dir, video_ids=val_vids,
                               session_norm_stats=norm_stats)
    ds_test  = SEEDEEGDataset([subject_id], args.features_dir, video_ids=test_vids,
                               session_norm_stats=norm_stats)
    if len(ds_train) == 0 or len(ds_val) == 0 or len(ds_test) == 0:
        return None, None, None, None
    cw = compute_class_weights(ds_train)
    kw = dict(num_workers=2, pin_memory=True)
    train_loader = DataLoader(ds_train, batch_size=args.ws_batch_size, shuffle=True, drop_last=len(ds_train) >= args.ws_batch_size, **kw)
    val_loader   = DataLoader(ds_val,   batch_size=args.ws_batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(ds_test,  batch_size=args.ws_batch_size, shuffle=False, **kw)
    return train_loader, val_loader, test_loader, cw


def train_within_subject_single(subject_id, args, device):
    """Leave-one-session-out (LOSO) CV for one subject (standard SEED protocol).

    4 folds: each fold holds out one session (20 videos) for test, trains on
    the other 3 sessions (60 videos). Per-session normalization is consistent
    because each session is always normalized with its own full 20-video stats.
    Train for exactly WS_EPOCHS with cosine LR decay (no early stopping for CV).

    Returns metrics dict with 'acc' (mean LOSO accuracy) and 'per_class',
    and saves the deployment checkpoint trained on sessions 2-4 (val=session 1).
    """
    fold_accs = []
    fold_per_class = []

    for test_session in range(1, 5):
        train_vids, test_vids = get_session_split(test_session)

        # Each session self-normalizes with its own 20-video stats → no mismatch.
        # norm_stats=None → dataset computes stats from the videos it contains.
        train_loader, _, test_loader, cw = _make_loaders(
            subject_id, args, train_vids, test_vids[:1], test_vids,  # dummy val, unused
            norm_stats=None,
        )
        if train_loader is None:
            continue

        model = MAETEEGClassifier().to(device)
        # Fixed-epoch training on all 60 train videos (no early stopping for CV eval)
        model = _train_model(
            model, train_loader, args.ws_epochs, args.ws_lr, args.ws_weight_decay,
            cw, device,
        )
        _, acc, per_class_acc, _ = evaluate(model, test_loader, device)
        fold_accs.append(acc)
        fold_per_class.append(per_class_acc)

    if not fold_accs:
        return None

    mean_acc = float(np.mean(fold_accs))
    mean_per_class = np.mean(fold_per_class, axis=0)

    # --- Deploy model: train on sessions 2-4, val=session 1 for early stopping ---
    deploy_train_vids, deploy_val_vids = get_session_split(test_session=1)

    deploy_train_ds = SEEDEEGDataset([subject_id], args.features_dir,
                                      video_ids=deploy_train_vids)
    deploy_val_ds   = SEEDEEGDataset([subject_id], args.features_dir,
                                      video_ids=deploy_val_vids)
    cw_deploy = compute_class_weights(deploy_train_ds)
    deploy_loader = DataLoader(
        deploy_train_ds, batch_size=args.ws_batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    deploy_val_loader = DataLoader(
        deploy_val_ds, batch_size=args.ws_batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    model_deploy = MAETEEGClassifier().to(device)
    model_deploy = _train_model(
        model_deploy, deploy_loader, args.ws_epochs, args.ws_lr, args.ws_weight_decay,
        cw_deploy, device, val_loader=deploy_val_loader, patience=args.ws_patience,
    )
    ckpt_path = os.path.join(args.cache_dir, f"eeg_seed_s{subject_id:02d}.pt")
    torch.save(model_deploy.state_dict(), ckpt_path)

    # Save per-session norm stats for inference
    norm_stats = compute_session_norm_stats(subject_id, args.features_dir)
    norm_path = os.path.join(args.cache_dir, f"eeg_seed_norm_s{subject_id:02d}.npz")
    means = np.stack([norm_stats[s][0] for s in sorted(norm_stats)])
    stds  = np.stack([norm_stats[s][1] for s in sorted(norm_stats)])
    sess_ids = np.array(sorted(norm_stats.keys()), dtype=np.int32)
    np.savez(norm_path, means=means, stds=stds, session_ids=sess_ids)

    return {"acc": mean_acc, "per_class": mean_per_class}


def train_within_subject(args):
    """Train per-subject models for all 20 SEED-VII subjects (4-fold session CV)."""
    set_seed()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.cache_dir, exist_ok=True)

    print(f"Within-subject training (4-fold session CV, MAET) on {N_SUBJECTS} subjects")
    print(f"  device={device}  epochs≤{args.ws_epochs}  batch={args.ws_batch_size}  lr={args.ws_lr}")

    all_metrics = {}
    t_total = time.time()

    for sid in range(1, N_SUBJECTS + 1):
        t0 = time.time()
        m = train_within_subject_single(sid, args, device)
        if m is None:
            print(f"  s{sid:02d} | SKIP")
            continue
        all_metrics[sid] = m
        per_class_str = "  ".join(
            f"{EMOTION_NAMES[i][:3]}={m['per_class'][i]:.3f}"
            for i in range(NUM_CLASSES)
        )
        print(f"  s{sid:02d} | acc={m['acc']:.4f} | {per_class_str} | {time.time()-t0:.1f}s")

    # Aggregate
    print(f"\n=== WITHIN-SUBJECT AGGREGATE (n={len(all_metrics)} subjects) ===")
    all_accs = [all_metrics[s]["acc"] for s in all_metrics]
    all_per_class = np.array([all_metrics[s]["per_class"] for s in all_metrics])
    print(f"  Overall accuracy: {np.mean(all_accs):.4f} ± {np.std(all_accs):.4f}")
    print(f"  Per-class accuracy (mean across subjects):")
    for i, name in enumerate(EMOTION_NAMES):
        print(f"    {name:10s}: {np.mean(all_per_class[:, i]):.4f} ± {np.std(all_per_class[:, i]):.4f}")
    print(f"  Total time: {(time.time()-t_total)/60:.1f} min")
    print(f"  Checkpoints: {args.cache_dir}/eeg_seed_s{{01..20}}.pt")
    print(f"  Norm stats:  {args.cache_dir}/eeg_seed_norm_s{{01..20}}.npz")


# ---------------------------------------------------------------------------
# Cross-subject training: leave-one-subject-out
# ---------------------------------------------------------------------------

def train_cross_subject(args):
    """Train cross-subject model: LOSO evaluation, then train on all."""
    set_seed()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.cache_dir, exist_ok=True)

    print(f"Cross-subject training (leave-one-subject-out) on {N_SUBJECTS} subjects")
    print(f"  device={device}  epochs≤{args.epochs}  batch={args.batch_size}  lr={args.lr}")

    # Use fold-3 as test set for cross-subject evaluation (last 5 videos/session)
    train_vids, test_vids = get_paper_fold_split(fold_idx=3)
    val_vids_set = set(get_paper_fold_split(fold_idx=2)[1])
    val_vids = list(val_vids_set)
    train_vids = [v for v in train_vids if v not in val_vids_set]

    loso_accs = []
    loso_per_class = []
    t_total = time.time()

    for test_sid in range(1, N_SUBJECTS + 1):
        t0 = time.time()
        train_sids = [s for s in range(1, N_SUBJECTS + 1) if s != test_sid]

        ds_train = SEEDEEGDataset(train_sids, args.features_dir, video_ids=train_vids)
        ds_val = SEEDEEGDataset(train_sids, args.features_dir, video_ids=val_vids)
        ds_test = SEEDEEGDataset([test_sid], args.features_dir, video_ids=test_vids)

        class_weights = compute_class_weights(ds_train)

        train_loader = DataLoader(
            ds_train, batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            ds_val, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
        )
        test_loader = DataLoader(
            ds_test, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
        )

        model = MAETEEGClassifier().to(device)
        model = _train_model(
            model, train_loader, args.epochs, args.lr, args.weight_decay,
            class_weights, device, val_loader=val_loader, patience=args.patience,
        )

        _, acc, per_class_acc, _ = evaluate(model, test_loader, device)
        loso_accs.append(acc)
        loso_per_class.append(per_class_acc)

        per_class_str = "  ".join(
            f"{EMOTION_NAMES[i][:3]}={per_class_acc[i]:.3f}"
            for i in range(NUM_CLASSES)
        )
        print(f"  LOSO s{test_sid:02d} | acc={acc:.4f} | {per_class_str} | {time.time()-t0:.1f}s")

    # Aggregate LOSO results
    loso_per_class = np.array(loso_per_class)
    print(f"\n=== CROSS-SUBJECT LOSO RESULTS (n={N_SUBJECTS}) ===")
    print(f"  Overall accuracy: {np.mean(loso_accs):.4f} ± {np.std(loso_accs):.4f}")
    for i, name in enumerate(EMOTION_NAMES):
        print(f"    {name:10s}: {np.mean(loso_per_class[:, i]):.4f} ± {np.std(loso_per_class[:, i]):.4f}")

    # Train final model on ALL subjects → save
    print("\nTraining final model on all subjects...")
    all_sids = list(range(1, N_SUBJECTS + 1))
    ds_all_train = SEEDEEGDataset(all_sids, args.features_dir, video_ids=train_vids)
    ds_all_val = SEEDEEGDataset(all_sids, args.features_dir, video_ids=val_vids)
    class_weights = compute_class_weights(ds_all_train)
    all_train_loader = DataLoader(
        ds_all_train, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    all_val_loader = DataLoader(
        ds_all_val, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    final_model = MAETEEGClassifier().to(device)
    final_model = _train_model(
        final_model, all_train_loader, args.epochs, args.lr, args.weight_decay,
        class_weights, device, val_loader=all_val_loader, patience=args.patience,
    )

    ckpt_path = os.path.join(args.cache_dir, "eeg_seed_cross.pt")
    torch.save(final_model.state_dict(), ckpt_path)
    print(f"Saved cross-subject model → {ckpt_path}")
    print(f"Total time: {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EEG emotion classifier on SEED-VII")
    parser.add_argument(
        "--mode", choices=["cross_subject", "within_subject"], default="within_subject",
        help="'within_subject' = per-subject stratified split; 'cross_subject' = LOSO",
    )
    parser.add_argument("--features_dir", default=SEED_FEATURES_DIR)
    parser.add_argument("--cache_dir", default=CACHE_DIR)
    # Cross-subject args
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--device", default=DEVICE)
    # Within-subject args
    parser.add_argument("--ws_batch_size", type=int, default=WS_BATCH_SIZE)
    parser.add_argument("--ws_epochs", type=int, default=WS_EPOCHS)
    parser.add_argument("--ws_lr", type=float, default=WS_LR)
    parser.add_argument("--ws_weight_decay", type=float, default=WS_WEIGHT_DECAY)
    parser.add_argument("--ws_patience", type=int, default=WS_PATIENCE)

    args = parser.parse_args()
    if args.mode == "within_subject":
        train_within_subject(args)
    else:
        train_cross_subject(args)
