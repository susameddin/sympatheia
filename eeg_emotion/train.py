"""Train EEGNetVA on DEAP for continuous valence-arousal prediction.

Usage:
    python -m eeg_emotion.train
    python -m eeg_emotion.train --epochs 100 --batch_size 128
"""

import argparse
import math
import os
import random
import time

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
    DROPOUT,
    EPOCHS,
    F1,
    D,
    F2,
    HOP_SAMPLES,
    LR,
    N_EEG_CHANNELS,
    N_SUBJECTS,
    PATIENCE,
    SEED,
    TEST_SUBJECTS,
    TRAIN_SUBJECTS,
    VAL_SUBJECTS,
    WEIGHT_DECAY,
    WINDOW_SAMPLES,
    WS_BATCH_SIZE,
    WS_EPOCHS,
    WS_LR,
    WS_WEIGHT_DECAY,
    WS_PATIENCE,
)
from .dataset import DEAPEEGDataset, DEAPEEGWithinSubjectDataset
from .models import EEGNetVA, EEGDEModel, ccc_loss, combined_loss


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = combined_loss(model(x), y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Returns (val_loss, metrics_dict) where metrics are per-dimension CCC/r/RMSE."""
    model.eval()
    preds_all, labels_all = [], []
    for x, y in loader:
        preds_all.append(model(x.to(device)).cpu().numpy())
        labels_all.append(y.numpy())

    preds  = np.concatenate(preds_all)   # (N, 2)
    labels = np.concatenate(labels_all)  # (N, 2)

    metrics = {}
    for i, dim in enumerate(["valence", "arousal"]):
        p, t = preds[:, i], labels[:, i]
        pm, tm    = p.mean(), t.mean()
        pv, tv    = p.var(),  t.var()
        cov       = ((p - pm) * (t - tm)).mean()
        ccc       = 2.0 * cov / (pv + tv + (pm - tm) ** 2 + 1e-8)
        r         = float(np.corrcoef(p, t)[0, 1])
        rmse      = float(np.sqrt(((p - t) ** 2).mean()))
        metrics[dim] = {"ccc": float(ccc), "r": r, "rmse": rmse}

    val_loss = ccc_loss(
        torch.tensor(preds), torch.tensor(labels)
    ).item()
    return val_loss, metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(args):
    set_seed()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.cache_dir, exist_ok=True)

    print("Loading DEAP datasets…")
    ds_train = DEAPEEGDataset(TRAIN_SUBJECTS, args.data_dir, WINDOW_SAMPLES, HOP_SAMPLES, augment=True)
    ds_val   = DEAPEEGDataset(VAL_SUBJECTS,   args.data_dir, WINDOW_SAMPLES, HOP_SAMPLES, augment=False)
    ds_test  = DEAPEEGDataset(TEST_SUBJECTS,  args.data_dir, WINDOW_SAMPLES, HOP_SAMPLES, augment=False)
    print(f"  train={len(ds_train):,}  val={len(ds_val):,}  test={len(ds_test):,} windows")

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = EEGNetVA(
        chans=N_EEG_CHANNELS, samples=WINDOW_SAMPLES,
        F1=F1, D=D, F2=F2, dropout=DROPOUT,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EEGNetVA: {n_params:,} trainable parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(ds_train) / args.batch_size) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    best_val   = float("inf")
    best_state = None
    patience   = args.patience
    history    = []

    print("\nTraining…")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_m = evaluate(model, val_loader, device)

        v_ccc = val_m["valence"]["ccc"]
        a_ccc = val_m["arousal"]["ccc"]
        print(
            f"Epoch {epoch:03d} | train={tr_loss:.4f}  val={val_loss:.4f} | "
            f"V-CCC={v_ccc:+.4f}  A-CCC={a_ccc:+.4f} | {time.time()-t0:.1f}s"
        )
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "val_v_ccc": v_ccc,
            "val_a_ccc": a_ccc,
        })

        if val_loss < best_val - 1e-4:
            best_val   = val_loss
            best_state = {"model": model.state_dict(), "epoch": epoch}
            patience   = args.patience
        else:
            patience -= 1
            if patience == 0:
                print(f"Early stopping at epoch {epoch}.")
                break

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        print(f"Restored best checkpoint from epoch {best_state['epoch']}.")

    weights_path = os.path.join(args.cache_dir, "eeg_va_deap.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"Saved weights → {weights_path}")

    # Final test evaluation
    _, test_m = evaluate(model, test_loader, device)
    print("\n=== TEST RESULTS ===")
    for dim in ["valence", "arousal"]:
        m = test_m[dim]
        print(f"  {dim:8s}:  CCC={m['ccc']:+.4f}  r={m['r']:+.4f}  RMSE={m['rmse']:.4f}")

    # Training curves plot
    epochs_h  = [h["epoch"]      for h in history]
    tr_losses = [h["train_loss"] for h in history]
    va_losses = [h["val_loss"]   for h in history]
    v_cccs    = [h["val_v_ccc"]  for h in history]
    a_cccs    = [h["val_a_ccc"]  for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs_h, tr_losses, label="train")
    axes[0].plot(epochs_h, va_losses, label="val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("CCC Loss (1 − CCC)")
    axes[0].set_title("EEG → VA: Loss"); axes[0].legend()
    axes[1].plot(epochs_h, v_cccs, label="valence")
    axes[1].plot(epochs_h, a_cccs, label="arousal")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("CCC")
    axes[1].set_title("EEG → VA: Validation CCC"); axes[1].legend()
    fig.tight_layout()
    plot_path = os.path.join(args.cache_dir, "training_curves.png")
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Saved training curves → {plot_path}")


# ---------------------------------------------------------------------------
# Within-subject training (per-subject DE models)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_de(model, loader, device):
    """Returns (val_loss, metrics_dict) for an EEGDEModel."""
    model.eval()
    preds_all, labels_all = [], []
    for x, y in loader:
        preds_all.append(model(x.to(device)).cpu().numpy())
        labels_all.append(y.numpy())

    preds  = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)

    metrics = {}
    for i, dim in enumerate(["valence", "arousal"]):
        p, t = preds[:, i], labels[:, i]
        pm, tm = p.mean(), t.mean()
        pv, tv = p.var(),  t.var()
        cov    = ((p - pm) * (t - tm)).mean()
        ccc    = 2.0 * cov / (pv + tv + (pm - tm) ** 2 + 1e-8)
        r      = float(np.corrcoef(p, t)[0, 1])
        rmse   = float(np.sqrt(((p - t) ** 2).mean()))
        # Binary accuracy: threshold predictions and labels at 0.0
        bin_acc = float(np.mean((p >= 0.0) == (t >= 0.0)))
        metrics[dim] = {"ccc": float(ccc), "r": r, "rmse": rmse, "bin_acc": bin_acc}

    val_loss = ccc_loss(torch.tensor(preds), torch.tensor(labels)).item()
    return val_loss, metrics


def _train_epochs(model, train_loader, n_epochs, lr, weight_decay, patience, device):
    """Train model for up to n_epochs with early stopping; use train loss as early-stop signal."""
    optimizer   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * n_epochs
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    best_loss  = float("inf")
    best_state = None
    p_counter  = patience

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = combined_loss(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        if avg_loss < best_loss - 1e-4:
            best_loss  = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            p_counter  = patience
        else:
            p_counter -= 1
            if p_counter == 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_within_subject_single(subject_id, args, device):
    """Train one per-subject EEGDEModel and return its test metrics.

    Step 1: Evaluate on 80/20 held-out split to measure performance.
    Step 2: Re-train on ALL 40 trials → save as deployed checkpoint.
    The saved checkpoint is thus trained on max available data.
    """
    ds_train = DEAPEEGWithinSubjectDataset(
        subject_id, "train", args.data_dir, WINDOW_SAMPLES, HOP_SAMPLES
    )
    ds_test = DEAPEEGWithinSubjectDataset(
        subject_id, "test", args.data_dir, WINDOW_SAMPLES, HOP_SAMPLES
    )

    if len(ds_train) == 0 or len(ds_test) == 0:
        print(f"  [s{subject_id:02d}] SKIP — empty split")
        return None

    train_loader = DataLoader(
        ds_train, batch_size=args.ws_batch_size, shuffle=True,  num_workers=2, pin_memory=True
    )
    test_loader  = DataLoader(
        ds_test,  batch_size=args.ws_batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # Step 1: train on 80% trials, evaluate on 20% → metrics for reporting
    model = EEGDEModel().to(device)
    _train_epochs(
        model, train_loader, args.ws_epochs, args.ws_lr, args.ws_weight_decay,
        args.ws_patience, device
    )
    _, test_m = evaluate_de(model, test_loader, device)

    # Step 2: re-train on ALL 40 trials → best deployed model
    ds_all    = DEAPEEGWithinSubjectDataset(
        subject_id, "all", args.data_dir, WINDOW_SAMPLES, HOP_SAMPLES
    )
    all_loader = DataLoader(
        ds_all, batch_size=args.ws_batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    model_all = EEGDEModel().to(device)
    _train_epochs(
        model_all, all_loader, args.ws_epochs, args.ws_lr, args.ws_weight_decay,
        args.ws_patience, device
    )

    ckpt_path = os.path.join(args.cache_dir, f"eeg_de_s{subject_id:02d}.pt")
    torch.save(model_all.state_dict(), ckpt_path)

    return test_m


def train_within_subject(args):
    """Train per-subject EEGDEModel for all 32 DEAP subjects."""
    set_seed()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.cache_dir, exist_ok=True)

    print(f"Within-subject DE training on all {N_SUBJECTS} subjects…")
    print(f"  device={device}  epochs≤{args.ws_epochs}  batch={args.ws_batch_size}  lr={args.ws_lr}")

    all_metrics = {}
    t_total = time.time()
    for sid in range(1, N_SUBJECTS + 1):
        t0 = time.time()
        m  = train_within_subject_single(sid, args, device)
        if m is None:
            continue
        all_metrics[sid] = m
        v_ccc    = m["valence"]["ccc"]
        a_ccc    = m["arousal"]["ccc"]
        v_binacc = m["valence"]["bin_acc"]
        a_binacc = m["arousal"]["bin_acc"]
        print(
            f"  s{sid:02d} | V-CCC={v_ccc:+.4f}  A-CCC={a_ccc:+.4f} | "
            f"V-acc={v_binacc:.3f}  A-acc={a_binacc:.3f} | {time.time()-t0:.1f}s"
        )

    # Aggregate across subjects
    print(f"\n=== WITHIN-SUBJECT AGGREGATE (n={len(all_metrics)} subjects) ===")
    for dim in ["valence", "arousal"]:
        cccs    = [all_metrics[s][dim]["ccc"]     for s in all_metrics]
        binacc  = [all_metrics[s][dim]["bin_acc"] for s in all_metrics]
        rmses   = [all_metrics[s][dim]["rmse"]    for s in all_metrics]
        print(
            f"  {dim:8s}: CCC={np.mean(cccs):+.4f}±{np.std(cccs):.4f}  "
            f"BinAcc={np.mean(binacc):.4f}±{np.std(binacc):.4f}  "
            f"RMSE={np.mean(rmses):.4f}±{np.std(rmses):.4f}"
        )
    print(f"Total training time: {(time.time()-t_total)/60:.1f} min")
    print(f"Checkpoints saved to: {args.cache_dir}/eeg_de_s{{01..32}}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EEG emotion models on DEAP")
    parser.add_argument("--mode",         choices=["cross", "within_subject"], default="within_subject",
                        help="'cross' = original cross-subject EEGNet; 'within_subject' = per-subject DE-MLP")
    parser.add_argument("--data_dir",     default=DEAP_DATA_DIR)
    parser.add_argument("--cache_dir",    default=CACHE_DIR)
    # Cross-subject args
    parser.add_argument("--epochs",       type=int,   default=EPOCHS)
    parser.add_argument("--batch_size",   type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",           type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--patience",     type=int,   default=PATIENCE)
    parser.add_argument("--device",       default=DEVICE)
    # Within-subject args
    parser.add_argument("--ws_batch_size",   type=int,   default=WS_BATCH_SIZE)
    parser.add_argument("--ws_epochs",       type=int,   default=WS_EPOCHS)
    parser.add_argument("--ws_lr",           type=float, default=WS_LR)
    parser.add_argument("--ws_weight_decay", type=float, default=WS_WEIGHT_DECAY)
    parser.add_argument("--ws_patience",     type=int,   default=WS_PATIENCE)

    args = parser.parse_args()
    if args.mode == "within_subject":
        train_within_subject(args)
    else:
        main(args)
