#!/usr/bin/env python3
"""Train MAET models on SEED-VII with checkpoint saving.

Trains 3 modalities (eeg, eye, both) × 20 subjects = 60 checkpoints.
Uses the exact same training loop as Datasets/MAET/train.py to reproduce
the same accuracies.

Saves:
    cache/maet_{modality}_s{id:02d}.pt       — model state_dict + best_acc
    cache/maet_norm_s{id:02d}.npz            — per-subject norm stats (from 'both' run)

Usage:
    # Train all 3 modalities for all 20 subjects:
    conda run -n s --no-capture-output python -m eeg_emotion.train_maet

    # Train only one modality:
    conda run -n s --no-capture-output python -m eeg_emotion.train_maet --modality eeg

    # Resume, skipping already-trained subjects/modalities:
    conda run -n s --no-capture-output python -m eeg_emotion.train_maet --skip-existing
"""

import argparse
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Import MAET model and dataset directly from Datasets/MAET/
MAET_DIR = Path(__file__).resolve().parents[3] / "Datasets" / "MAET"
sys.path.insert(0, str(MAET_DIR))
from dataset import ALL_VIDEOS, EMOTION_CLASSES, SEEDVIIDataset, video_emotion_label
from model import MAET

CACHE_DIR = Path(__file__).resolve().parent / "cache"

MODEL_CONFIGS = {
    "small": dict(embed_dim=32, depth=3, num_heads=4, mixffn_start_layer_index=2,
                  drop_rate=0.1, drop_path_rate=0.1),
    "large": dict(embed_dim=64, depth=4, num_heads=8, mixffn_start_layer_index=3,
                  drop_rate=0.2, drop_path_rate=0.2),
}


def make_model(device, model_size="small"):
    cfg = MODEL_CONFIGS[model_size]
    model = MAET(
        embed_dim=cfg["embed_dim"],
        num_classes=7,
        eeg_seq_len=5,
        eye_seq_len=5,
        eeg_dim=310,
        eye_dim=33,
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        qkv_bias=True,
        mixffn_start_layer_index=cfg["mixffn_start_layer_index"],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_rate=cfg["drop_rate"],
        drop_path_rate=cfg["drop_path_rate"],
    )
    return model.to(device)


def train_subject(subject_id, modality, epochs, batch_size, lr, device, seed=42, model_size="small",
                  filter_quantile=0.5):
    """Exact replica of Datasets/MAET/train.py train_subject, returns (best_acc, norm_stats)."""
    video_labels = [video_emotion_label(v) for v in ALL_VIDEOS]
    train_vids, test_vids = train_test_split(
        ALL_VIDEOS,
        test_size=0.2,
        random_state=seed,
        stratify=video_labels,
    )

    train_set = SEEDVIIDataset(subject_id, train_vids, filter_quantile=filter_quantile)
    norm_stats = train_set.get_norm_stats()
    test_set = SEEDVIIDataset(subject_id, test_vids, normalize_stats=norm_stats,
                              filter_quantile=filter_quantile)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=0)

    model = make_model(device, model_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for eeg, eye, label in train_loader:
            eeg = eeg.to(device)
            eye = eye.to(device)
            label = label.to(device)

            eeg_in = eeg if modality in ('eeg', 'both') else None
            eye_in = eye if modality in ('eye', 'both') else None
            out = model(eeg=eeg_in, eye=eye_in)
            loss = F.cross_entropy(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for eeg, eye, label in test_loader:
                eeg = eeg.to(device)
                eye = eye.to(device)
                eeg_in = eeg if modality in ('eeg', 'both') else None
                eye_in = eye if modality in ('eye', 'both') else None
                out = model(eeg=eeg_in, eye=eye_in)
                all_preds.append(torch.argmax(out, dim=-1).cpu().numpy())
                all_labels.append(label.numpy())

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs:
            avg_loss = total_loss / len(train_loader)
            print(f'  S{subject_id:02d} [{modality}] epoch {epoch:3d}/{epochs}  '
                  f'loss={avg_loss:.4f}  acc={acc:.4f}  best={best_acc:.4f}')

    return best_acc, best_state, norm_stats


def main():
    parser = argparse.ArgumentParser(
        description="Train MAET models on SEED-VII with checkpoint saving"
    )
    parser.add_argument('--subjects', type=int, nargs='+', default=list(range(1, 21)))
    parser.add_argument('--modality', type=str, default=None,
                        choices=['eeg', 'eye', 'both'],
                        help='Single modality to train. Default: train all 3 in sequence.')
    parser.add_argument('--model-size', type=str, default='small',
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Model size preset (default: small = paper config)')
    parser.add_argument('--filter-quantile', type=float, default=0.5,
                        help='Keep windows where cont_label > this quantile (default: 0.5 = top 50%%)')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip (subject, modality) pairs whose checkpoint already exists')
    parser.add_argument('--cache-dir', type=str, default=str(CACHE_DIR))
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    modalities = [args.modality] if args.modality else ['eeg', 'eye', 'both']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Device:     {device}')
    print(f'Subjects:   {args.subjects}')
    print(f'Modalities: {modalities}')
    print(f'Model size: {args.model_size} {MODEL_CONFIGS[args.model_size]}')
    print(f'Filter:     top {100*(1-args.filter_quantile):.0f}% windows (quantile={args.filter_quantile})')
    print(f'Epochs:     {args.epochs}  LR: {args.lr}  Batch: {args.batch_size}')
    print(f'Cache dir:  {cache_dir}')
    print()

    results = {mod: {} for mod in modalities}

    for mod in modalities:
        print(f'{"="*60}')
        print(f'Modality: {mod}')
        print(f'{"="*60}')

        for sid in args.subjects:
            fq_tag = f'fq{int(args.filter_quantile*100):02d}'
            ckpt_path = cache_dir / f'maet_{mod}_{args.model_size}_{fq_tag}_s{sid:02d}.pt'

            if args.skip_existing and ckpt_path.exists():
                # Load existing best_acc to include in summary
                ckpt = torch.load(ckpt_path, map_location='cpu')
                results[mod][sid] = ckpt.get('best_acc', float('nan'))
                print(f'  S{sid:02d} [{mod}] already exists (best_acc={results[mod][sid]:.4f}), skipping')
                continue

            print(f'--- Subject {sid:02d} [{mod}] ---')
            best_acc, best_state, norm_stats = train_subject(
                subject_id=sid,
                modality=mod,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                seed=args.seed,
                model_size=args.model_size,
                filter_quantile=args.filter_quantile,
            )
            results[mod][sid] = best_acc
            print(f'  Subject {sid:02d} [{mod}] best accuracy: {best_acc:.4f}\n')

            # Save checkpoint
            torch.save({
                'state_dict': best_state,
                'best_acc': best_acc,
                'subject_id': sid,
                'modality': mod,
                'model_size': args.model_size,
                'filter_quantile': args.filter_quantile,
            }, ckpt_path)

            # Save norm stats from 'both' run (all modalities share the same train split,
            # so stats are identical regardless of which run saves them)
            if mod == 'both':
                eeg_mean, eeg_std, eye_mean, eye_std = norm_stats
                norm_path = cache_dir / f'maet_norm_{args.model_size}_{fq_tag}_s{sid:02d}.npz'
                np.savez(
                    norm_path,
                    eeg_mean=eeg_mean,
                    eeg_std=eeg_std,
                    eye_mean=eye_mean,
                    eye_std=eye_std,
                )

    # Summary
    print()
    print('=' * 60)
    print('RESULTS SUMMARY')
    print('=' * 60)
    for mod in modalities:
        accs = [results[mod][sid] for sid in args.subjects if sid in results[mod]]
        if accs:
            print(f'\nModality: {mod}')
            for sid in args.subjects:
                if sid in results[mod]:
                    print(f'  S{sid:02d}: {results[mod][sid]:.4f}')
            print(f'  Mean: {np.mean(accs):.4f}  Std: {np.std(accs):.4f}')
    print(f'\nChance level (7-class): {1/7:.4f}')


if __name__ == '__main__':
    main()
