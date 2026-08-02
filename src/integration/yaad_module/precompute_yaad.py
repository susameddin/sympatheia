"""Pre-compute YAAD ECG/GSR predictions and save to JSON.

Samples N records per emotion class from YAAD, runs the predictor,
and saves predictions to cache/yaad_predictions_{modality}.json.

--cv mode: iterates over all CV folds (trained by train_cv.py), loads each
fold's model and test indices, runs inference on those held-out records, and
merges all folds into one JSON — covering the full dataset with no leakage.

Output JSON matches the face_module schema:
    {id, emotion, modality, top_emotion,
     predicted_valence, predicted_arousal,
     gt_valence, gt_arousal, probs}

Usage:
    # Single held-out split (default)
    python -m integration.yaad_module.precompute_yaad --modality ecg
    python -m integration.yaad_module.precompute_yaad --modality ecg --n-per-class 100

    # CV mode — uses all records across 5 folds (~700 ECG / ~391 GSR total)
    python -m integration.yaad_module.precompute_yaad --modality ecg --cv
    python -m integration.yaad_module.precompute_yaad --modality gsr --cv
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from .ecg_emotion.loader import load_all_data, load_multimodal_data
import sys

# Repo root on sys.path so `from src... import ...` resolves regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.constants import EMOTION_VA_MAPPING as SPEECH_ANCHORS

from .config import (
    CACHE_DIR, EMOTION_NAMES, YAAD_TO_SPEECH,
    ECG_TEST_INDICES, GSR_TEST_INDICES,
)
from .models import YAADECGPredictor, YAADGSRPredictor, YAADFusionPredictor

# Ground-truth VA per emotion (speech anchor coordinates)
_GT_VA = {e: SPEECH_ANCHORS[YAAD_TO_SPEECH[e]] for e in EMOTION_NAMES}


def sample_records(records: list[dict], n_per_class: int, seed: int) -> list[dict]:
    """Sample up to n_per_class records per emotion. Returns a flat list."""
    rng = random.Random(seed)
    by_emotion: dict[str, list] = defaultdict(list)
    for r in records:
        by_emotion[r["emotion"]].append(r)

    result = []
    for emo in EMOTION_NAMES:
        pool = by_emotion.get(emo, [])
        chosen = rng.sample(pool, min(n_per_class, len(pool)))
        for i, rec in enumerate(sorted(chosen, key=lambda x: id(x))):
            result.append({"_idx": i, "_emo": emo, "_rec": rec})
        print(f"  {emo:<12}: {len(chosen)} records")
    return result


def run_ecg(samples: list[dict], predictor: YAADECGPredictor) -> list[dict]:
    out = []
    for s in samples:
        rec = s["_rec"]
        top, probs = predictor.predict_emotion(rec["ecg"])
        v, a = predictor.predict_va(rec["ecg"])
        gv, ga = _GT_VA[s["_emo"]]
        out.append({
            "id":                 f"{s['_emo'].lower()}_{s['_idx']:02d}",
            "emotion":            s["_emo"],
            "modality":           "ecg",
            "top_emotion":        top,
            "predicted_valence":  v,
            "predicted_arousal":  a,
            "gt_valence":         gv,
            "gt_arousal":         ga,
            "probs":              probs,
        })
    return out


def run_gsr(samples: list[dict], predictor: YAADGSRPredictor) -> list[dict]:
    out = []
    for s in samples:
        rec = s["_rec"]
        top, probs = predictor.predict_emotion(rec["gsr"])
        v, a = predictor.predict_va(rec["gsr"])
        gv, ga = _GT_VA[s["_emo"]]
        out.append({
            "id":                 f"{s['_emo'].lower()}_{s['_idx']:02d}",
            "emotion":            s["_emo"],
            "modality":           "gsr",
            "top_emotion":        top,
            "predicted_valence":  v,
            "predicted_arousal":  a,
            "gt_valence":         gv,
            "gt_arousal":         ga,
            "probs":              probs,
        })
    return out


def run_fusion(samples: list[dict], predictor: YAADFusionPredictor) -> list[dict]:
    out = []
    for s in samples:
        rec = s["_rec"]
        top, probs = predictor.predict_emotion(rec["ecg"], rec["gsr"])
        v, a = predictor.predict_va(rec["ecg"], rec["gsr"])
        gv, ga = _GT_VA[s["_emo"]]
        out.append({
            "id":                 f"{s['_emo'].lower()}_{s['_idx']:02d}",
            "emotion":            s["_emo"],
            "modality":           "fusion",
            "top_emotion":        top,
            "predicted_valence":  v,
            "predicted_arousal":  a,
            "gt_valence":         gv,
            "gt_arousal":         ga,
            "probs":              probs,
        })
    return out


def _run_predictions(samples, modality, predictor):
    if modality == "ecg":
        return run_ecg(samples, predictor)
    elif modality == "gsr":
        return run_gsr(samples, predictor)
    else:
        return run_fusion(samples, predictor)


def _load_predictor_from_weights(modality, weights_path, device):
    """Load a predictor with a custom weights file (for CV fold models)."""
    from .ecg_emotion.models import ECGResNet1DDeep
    from .config import ECG_SAMPLES, NUM_CLASSES
    from .models import _znorm, _YAAD_SPEECH_VA

    import torch

    class _FoldPredictor:
        def __init__(self):
            self.model = ECGResNet1DDeep(n_classes=NUM_CLASSES, in_channels=1)
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state)
            self.model.to(device).eval()
            self._device = device

        def _probs(self, signal):
            x = np.asarray(signal, dtype=np.float32).flatten()[:ECG_SAMPLES]
            x = (x - x.mean()) / (x.std() + 1e-8)
            t = torch.from_numpy(x[None, None, :]).to(self._device)
            with torch.no_grad():
                p = torch.softmax(self.model(t), dim=1).cpu().numpy()[0]
            return p

        def predict_emotion(self, signal):
            p = self._probs(signal)
            top = EMOTION_NAMES[int(p.argmax())]
            return top, {EMOTION_NAMES[i]: float(p[i]) for i in range(len(EMOTION_NAMES))}

        def predict_va(self, signal):
            p = self._probs(signal)
            va = p @ _YAAD_SPEECH_VA
            return float(va[0]), float(va[1])

    return _FoldPredictor()


def run_cv_mode(modality: str, n_folds: int, seed: int, device: str) -> list[dict]:
    """Run CV precompute: load each fold's model + test records, merge predictions."""
    all_records = load_all_data() if modality == "ecg" else load_multimodal_data()
    all_predictions = []
    global_count = 0

    for k in range(n_folds):
        idx_path    = os.path.join(CACHE_DIR, f"{modality}_fold{k}_test_indices.npy")
        model_path  = os.path.join(CACHE_DIR, f"{modality}_fold{k}_model.pt")

        if not os.path.exists(idx_path) or not os.path.exists(model_path):
            print(f"  Fold {k}: missing files, skipping. Run train_cv.py first.")
            continue

        test_idx = np.load(idx_path)
        fold_records = [all_records[i] for i in test_idx]
        print(f"\n  Fold {k}: {len(fold_records)} test records")

        predictor = _load_predictor_from_weights(modality, model_path, device)

        # Use all records from this fold (no sampling — they're already held-out)
        samples = []
        by_emotion = defaultdict(list)
        for r in fold_records:
            by_emotion[r["emotion"]].append(r)
        for emo in EMOTION_NAMES:
            for i, rec in enumerate(by_emotion.get(emo, [])):
                samples.append({"_idx": global_count + i, "_emo": emo, "_rec": rec})
            global_count += len(by_emotion.get(emo, []))

        fold_preds = _run_predictions(samples, modality, predictor)
        all_predictions.extend(fold_preds)
        print(f"  Fold {k}: {len(fold_preds)} predictions")

    return all_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality",    choices=["ecg", "gsr", "fusion"], default="ecg")
    parser.add_argument("--n-per-class", type=int, default=6)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--cv",          action="store_true",
                        help="CV mode: pool predictions across all folds (run train_cv.py first)")
    parser.add_argument("--folds",       type=int, default=5,
                        help="Number of folds for --cv mode (default: 5)")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path. Default: cache/yaad_predictions_{modality}[_cv].json",
    )
    args = parser.parse_args()

    suffix = "_cv" if args.cv else ""
    output = args.output or os.path.join(CACHE_DIR, f"yaad_predictions_{args.modality}{suffix}.json")
    os.makedirs(CACHE_DIR, exist_ok=True)

    if args.cv:
        if args.modality == "fusion":
            print("ERROR: --cv mode does not support fusion (ECG and GSR have different record sets).")
            return
        print(f"CV mode: pooling {args.folds} folds for {args.modality.upper()} ...")
        predictions = run_cv_mode(args.modality, args.folds, args.seed, args.device)
    else:
        # Single held-out split mode
        if args.modality == "ecg":
            records = load_all_data()
            idx_path = ECG_TEST_INDICES
        else:
            records = load_multimodal_data()
            idx_path = GSR_TEST_INDICES

        if os.path.exists(idx_path):
            test_idx = np.load(idx_path)
            records = [records[i] for i in test_idx]
            print(f"Loaded {len(records)} held-out test records (from {idx_path}).")
        else:
            print(f"WARNING: test indices not found at {idx_path}. "
                  "Using full dataset — precompute may overlap with training data.")

        print(f"\nSampling {args.n_per_class} per class (seed={args.seed}) ...")
        samples = sample_records(records, args.n_per_class, args.seed)
        print(f"Total: {len(samples)} samples\n")

        print(f"Loading {args.modality.upper()} predictor (device={args.device}) ...")
        if args.modality == "ecg":
            predictor = YAADECGPredictor(device=args.device)
        elif args.modality == "gsr":
            predictor = YAADGSRPredictor(device=args.device)
        else:
            predictor = YAADFusionPredictor(device=args.device)
        predictions = _run_predictions(samples, args.modality, predictor)

    # Summary
    for i, p in enumerate(predictions):
        if i % 50 == 0 or i == len(predictions) - 1:
            print(
                f"  [{i+1}/{len(predictions)}] {p['id']}: GT={p['emotion']:<10} "
                f"pred={p['top_emotion']:<10} VA=({p['predicted_valence']:+.2f}, "
                f"{p['predicted_arousal']:+.2f})"
            )

    with open(output, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"\nSaved {len(predictions)} predictions → {output}")

    correct = sum(1 for p in predictions if p["top_emotion"] == p["emotion"])
    print(f"Top-1 accuracy: {correct}/{len(predictions)} = {correct/len(predictions):.1%}")


if __name__ == "__main__":
    main()
