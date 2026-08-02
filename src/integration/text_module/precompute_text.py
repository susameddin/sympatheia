#!/usr/bin/env python3
"""Pre-compute text-to-VA predictions on ISEAR samples.

Samples self-reported emotion descriptions from ISEAR, predicts (valence, arousal)
with TextToVAConverter, and saves to integration/text_module/cache/text_predictions.json
for use by eval/generate_responses/sensing/generate_responses_sensing.py.

Download ISEAR CSV (one-time):
    wget -O integration/cache/isear.csv \
      https://raw.githubusercontent.com/sinmaniphel/py_isear_dataset/master/isear.csv

Usage:
    python -m integration.text_module.precompute_text --n-per-class 200
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from src.integration.text_module.text_to_va import TextToVAConverter
from src.constants import SPEECH_ANCHORS

CACHE_DIR = Path(__file__).resolve().parent / "cache"
DEFAULT_DATASET_PATH = str(PROJECT_ROOT / "integration" / "cache" / "isear.csv")
DEFAULT_OUTPUT = str(CACHE_DIR / "text_predictions.json")

# ISEAR emotion → speech anchor mapping (5 clean categories only)
ISEAR_TO_ANCHOR = {
    "joy":     "happy",
    "fear":    "anxious",
    "anger":   "angry",
    "sadness": "sad",
    "disgust": "disgusted",
}

ISEAR_TO_VA = {
    isear: SPEECH_ANCHORS[anchor]
    for isear, anchor in ISEAR_TO_ANCHOR.items()
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nearest_anchor_name(v: float, a: float) -> str:
    best_dist, best_name = float("inf"), "neutral"
    for name, (av, aa) in SPEECH_ANCHORS.items():
        d = (v - av) ** 2 + (a - aa) ** 2
        if d < best_dist:
            best_dist, best_name = d, name
    return best_name


def load_isear_samples(path: str, n_per_class: int, seed: int) -> list:
    """Load ISEAR CSV; return list of {emotion, text, gt_valence, gt_arousal}."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required: pip install pandas")

    df = None
    for sep in ["|", ",", "\t"]:
        try:
            candidate = pd.read_csv(
                path, sep=sep, encoding="latin-1",
                on_bad_lines="skip", low_memory=False,
            )
            if len(candidate.columns) >= 2:
                df = candidate
                break
        except Exception:
            continue

    if df is None:
        raise ValueError(f"Could not parse ISEAR CSV at {path}")

    known = set(ISEAR_TO_ANCHOR.keys())
    emot_col = None
    for col in df.columns:
        col_vals = set(df[col].astype(str).str.lower().str.strip().unique())
        if len(col_vals & known) >= 3:
            emot_col = col
            break

    if emot_col is None:
        raise ValueError(
            f"No emotion column found. Columns: {list(df.columns)}\n"
            "Expected values like: joy, fear, anger, sadness, disgust"
        )

    text_col = None
    for candidate_name in ["SIT", "SIT1", "Field1", "situation", "text"]:
        if candidate_name in df.columns and candidate_name != emot_col:
            text_col = candidate_name
            break
    if text_col is None:
        for col in df.columns:
            if col == emot_col:
                continue
            if df[col].dtype == object:
                avg_len = df[col].dropna().astype(str).str.len().mean()
                if avg_len > 20:
                    text_col = col
                    break

    if text_col is None:
        raise ValueError(f"No text column found. Columns: {list(df.columns)}")

    print(f"ISEAR columns: emotion='{emot_col}', text='{text_col}'")

    rng = random.Random(seed)
    samples = []
    for emotion in ISEAR_TO_ANCHOR:
        mask = df[emot_col].astype(str).str.lower().str.strip() == emotion
        texts = df[mask][text_col].dropna().astype(str).str.strip().tolist()
        texts = [t for t in texts if len(t) > 10]
        if not texts:
            print(f"  WARNING: no samples for '{emotion}', skipping")
            continue
        chosen = rng.sample(texts, min(n_per_class, len(texts)))
        gt_v, gt_a = ISEAR_TO_VA[emotion]
        for i, text in enumerate(chosen):
            samples.append({
                "emotion": emotion,
                "text": text,
                "gt_valence": gt_v,
                "gt_arousal": gt_a,
            })
        print(f"  {emotion:<10}: {len(chosen)} samples")

    return samples


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-compute text-to-VA predictions on ISEAR samples"
    )
    parser.add_argument(
        "--dataset-path", default=DEFAULT_DATASET_PATH,
        help=f"Path to ISEAR CSV file. Default: {DEFAULT_DATASET_PATH}",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output JSON path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--n-per-class", type=int, default=200,
        help="ISEAR samples per emotion class (default: 200)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--method", default="hf", choices=["hf", "keyword"],
        help="TextToVAConverter method (default: hf)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"Loading ISEAR from {args.dataset_path}...")
    raw_samples = load_isear_samples(args.dataset_path, args.n_per_class, args.seed)
    if not raw_samples:
        print("ERROR: no ISEAR samples loaded.", file=sys.stderr)
        sys.exit(1)
    print(f"Total: {len(raw_samples)} samples\n")

    converter = TextToVAConverter(glm_model=None, glm_tokenizer=None)
    print(f"Predicting VA (method={args.method})...")

    predictions = []
    for i, s in enumerate(raw_samples):
        emotion = s["emotion"]
        # Assign per-class index
        class_i = sum(1 for p in predictions if p["emotion"] == emotion)
        sample_id = f"{emotion}_{class_i:03d}"

        v_pred, a_pred, _ = converter.convert(s["text"], method=args.method)
        top_emotion = nearest_anchor_name(v_pred, a_pred)

        predictions.append({
            "id": sample_id,
            "emotion": emotion,
            "gt_valence": s["gt_valence"],
            "gt_arousal": s["gt_arousal"],
            "predicted_valence": float(v_pred),
            "predicted_arousal": float(a_pred),
            "top_emotion": top_emotion,
            "input_text": s["text"],
        })

        if (i + 1) % 50 == 0 or i == 0:
            snippet = s["text"][:60] + ("..." if len(s["text"]) > 60 else "")
            print(
                f"  [{i+1}/{len(raw_samples)}] {sample_id}: {emotion:<10} "
                f"→ V={v_pred:+.2f}, A={a_pred:+.2f} → {top_emotion:<12}  \"{snippet}\""
            )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"\nSaved {len(predictions)} predictions → {args.output}")

    # Accuracy summary (top_emotion vs GT anchor)
    correct = sum(
        1 for p in predictions
        if p["top_emotion"] == ISEAR_TO_ANCHOR[p["emotion"]]
    )
    print(f"Nearest-anchor accuracy: {correct}/{len(predictions)} = {correct/len(predictions):.1%}")


if __name__ == "__main__":
    main()
