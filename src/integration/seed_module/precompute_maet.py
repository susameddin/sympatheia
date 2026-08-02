#!/usr/bin/env python3
"""Pre-compute MAET predictions on SEED-VII test windows.

Reconstructs the same stratified 80/20 test split used during training
(seed=42), pools test windows across all 20 subjects, samples N_per_class
windows per emotion class, then runs EEG-only, Eye-only, and combined MAET
predictions for each window.

Output JSON mirrors face_emotion/cache/hsemotion_predictions.json format
but with 3 modality conditions per record instead of one.

Prerequisite:
    python -m eeg_emotion.train_maet

Usage:
    python -m eeg_emotion.precompute_maet
    python -m eeg_emotion.precompute_maet --n-per-class 10
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# Import MAET model and dataset directly from Datasets/MAET/
MAET_DIR = Path(__file__).resolve().parents[3] / "Datasets" / "MAET"
sys.path.insert(0, str(MAET_DIR))
from dataset import (
    ALL_VIDEOS,
    CONT_LABELS_DIR,
    EEG_DIR,
    EMOTION_CLASSES,
    EYE_DIR,
    video_emotion_label,
)
from model import MAET

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))
from src.integration.seed_module.va_utils import EEG_SPEECH_VA, EEGVASpeechMapper

CACHE_DIR = Path(__file__).resolve().parent / "cache"
DEFAULT_OUTPUT = str(CACHE_DIR / "eeg_predictions.json")

# Modality keys used in output JSON and for checkpoint filenames
MODALITIES = ["eeg_only", "eye_only", "combined"]
# Map output key → training modality name
_MOD_TO_CKPT = {"eeg_only": "eeg", "eye_only": "eye", "combined": "both"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-compute MAET predictions on SEED-VII test windows"
    )
    parser.add_argument(
        "--subjects", type=int, nargs="+", default=list(range(1, 21)),
        help="Subject IDs to include in pool (default: 1-20)",
    )
    parser.add_argument(
        "--n-per-class", type=int, default=6,
        help="Windows to sample per emotion class (default: 6)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42). "
             "Also used for train/test split to match training.",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for inference (default: cuda)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=str(CACHE_DIR),
        help=f"Directory with trained checkpoints (default: {CACHE_DIR})",
    )
    return parser.parse_args()


def make_model(device):
    model = MAET(
        embed_dim=32,
        num_classes=7,
        eeg_seq_len=5,
        eye_seq_len=5,
        eeg_dim=310,
        eye_dim=33,
        depth=3,
        num_heads=4,
        qkv_bias=True,
        mixffn_start_layer_index=2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_rate=0.0,
        drop_path_rate=0.0,
    )
    return model.to(device)


def load_model(modality_key, subject_id, cache_dir, device):
    """Load trained MAET checkpoint for (modality, subject_id)."""
    ckpt_name = _MOD_TO_CKPT[modality_key]
    ckpt_path = cache_dir / f"maet_{ckpt_name}_s{subject_id:02d}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run first:\n"
            "  python -m eeg_emotion.train_maet"
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    model = make_model(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def build_test_pool(subject_ids, cache_dir, seed=42):
    """Build pool of normalized test windows from all subjects.

    Reconstructs the exact same 80/20 stratified video split used in training
    (same seed) so that test windows are truly held-out.

    Returns:
        pool: dict mapping emotion_class_idx -> list of (eeg_norm, eye_norm, sid, vid)
    """
    video_labels = [video_emotion_label(v) for v in ALL_VIDEOS]
    _, test_vids = train_test_split(
        ALL_VIDEOS,
        test_size=0.2,
        random_state=seed,
        stratify=video_labels,
    )
    test_vids_set = set(test_vids)

    pool = defaultdict(list)

    for sid in subject_ids:
        norm_path = cache_dir / f"maet_norm_s{sid:02d}.npz"
        if not norm_path.exists():
            raise FileNotFoundError(
                f"Norm stats not found: {norm_path}\n"
                "Run first:\n"
                "  python -m eeg_emotion.train_maet"
            )
        norm_data = np.load(str(norm_path))
        eeg_mean = norm_data["eeg_mean"]
        eeg_std = norm_data["eeg_std"]
        eye_mean = norm_data["eye_mean"]
        eye_std = norm_data["eye_std"]

        eeg_mat = sio.loadmat(f"{EEG_DIR}/{sid}.mat")
        eye_mat = sio.loadmat(f"{EYE_DIR}/{sid}.mat")
        cont_mat = sio.loadmat(f"{CONT_LABELS_DIR}/{sid}.mat")

        for vid in test_vids_set:
            eeg_feat = eeg_mat[f"de_LDS_{vid}"]           # (N, 5, 62)
            n = eeg_feat.shape[0]
            eeg_feat = eeg_feat.reshape(n, -1).astype(np.float32)  # (N, 310)

            eye_feat = eye_mat[str(vid)].astype(np.float32)  # (N, 33)

            cont = cont_mat[str(vid)].squeeze().astype(np.float32)
            mask = cont > cont.mean()
            if mask.sum() == 0:
                mask = np.ones(n, dtype=bool)

            eeg_feat = eeg_feat[mask]
            eye_feat = eye_feat[mask]

            # Apply per-subject training normalization
            eeg_norm = (eeg_feat - eeg_mean) / eeg_std
            eye_norm = (eye_feat - eye_mean) / eye_std

            label = video_emotion_label(vid)
            for i in range(len(eeg_norm)):
                pool[label].append((eeg_norm[i], eye_norm[i], sid, vid))

    return pool, test_vids


@torch.no_grad()
def predict_modality(eeg_np, eye_np, model, modality_key, device):
    """Run MAET forward pass for one window, return probs dict and top emotion."""
    eeg_t = torch.tensor(eeg_np, dtype=torch.float32).unsqueeze(0).to(device)
    eye_t = torch.tensor(eye_np, dtype=torch.float32).unsqueeze(0).to(device)

    eeg_in = eeg_t if modality_key in ("eeg_only", "combined") else None
    eye_in = eye_t if modality_key in ("eye_only", "combined") else None

    logits = model(eeg=eeg_in, eye=eye_in)  # (1, 7)
    probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # (7,)

    top_idx = int(np.argmax(probs))
    top_emotion = EMOTION_CLASSES[top_idx]
    probs_dict = {EMOTION_CLASSES[i]: float(probs[i]) for i in range(len(EMOTION_CLASSES))}
    return probs, probs_dict, top_emotion


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Building test pool from {len(args.subjects)} subjects (seed={args.seed})...")

    pool, test_vids = build_test_pool(args.subjects, cache_dir, seed=args.seed)

    print(f"Test videos ({len(test_vids)}): {sorted(test_vids)}")
    print(f"\nPool sizes per emotion class:")
    for cls_idx, emotion in enumerate(EMOTION_CLASSES):
        print(f"  {emotion:<10}: {len(pool[cls_idx])} windows")

    print(f"\nSampling {args.n_per_class} windows per class (seed={args.seed})...")
    rng = random.Random(args.seed)
    samples = []
    for cls_idx, emotion in enumerate(EMOTION_CLASSES):
        entries = pool[cls_idx]
        if not entries:
            print(f"  WARNING: no test windows for {emotion}, skipping")
            continue
        chosen = rng.sample(entries, min(args.n_per_class, len(entries)))
        for i, (eeg_np, eye_np, sid, vid) in enumerate(chosen):
            samples.append({
                "id": f"{emotion.lower()}_{i:02d}",
                "emotion": emotion,
                "cls_idx": cls_idx,
                "subject_id": sid,
                "video_id": vid,
                "eeg_np": eeg_np,
                "eye_np": eye_np,
            })
        print(f"  {emotion:<10}: {len(chosen)} windows sampled")

    print(f"\nTotal: {len(samples)} samples")

    # VA mapper (speech-model coordinates, same as face pipeline)
    va_mapper = EEGVASpeechMapper()

    # Model cache: (modality_key, subject_id) -> model
    model_cache = {}

    print("\nRunning MAET predictions...")
    predictions = []
    for i, s in enumerate(samples):
        sid = s["subject_id"]
        eeg_np = s["eeg_np"]
        eye_np = s["eye_np"]
        emotion = s["emotion"]

        gt_v, gt_a = EEG_SPEECH_VA[emotion]

        record = {
            "id": s["id"],
            "emotion": emotion,
            "subject_id": sid,
            "video_id": s["video_id"],
            "gt_valence": float(gt_v),
            "gt_arousal": float(gt_a),
        }

        for mod_key in MODALITIES:
            cache_key = (mod_key, sid)
            if cache_key not in model_cache:
                model_cache[cache_key] = load_model(mod_key, sid, cache_dir, device)
            model = model_cache[cache_key]

            probs, probs_dict, top_emotion = predict_modality(
                eeg_np, eye_np, model, mod_key, device
            )
            pred_v, pred_a = va_mapper.adapt_from_probs(probs)

            record[mod_key] = {
                "predicted_valence": pred_v,
                "predicted_arousal": pred_a,
                "top_emotion": top_emotion,
                "probs": probs_dict,
            }

        if (i + 1) % 5 == 0 or i == 0:
            print(
                f"  [{i+1}/{len(samples)}] {s['id']}: GT={emotion:<10} "
                f"eeg={record['eeg_only']['top_emotion']:<10} "
                f"eye={record['eye_only']['top_emotion']:<10} "
                f"both={record['combined']['top_emotion']:<10}"
            )

        predictions.append(record)

    # Clean up model cache
    model_cache.clear()

    # Save output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"\nSaved {len(predictions)} predictions → {args.output}")

    # Accuracy summary
    print("\nTop-1 accuracy summary:")
    for mod_key in MODALITIES:
        correct = sum(1 for p in predictions if p[mod_key]["top_emotion"] == p["emotion"])
        print(f"  {mod_key:<12}: {correct}/{len(predictions)} = {correct/len(predictions):.1%}")


if __name__ == "__main__":
    main()
