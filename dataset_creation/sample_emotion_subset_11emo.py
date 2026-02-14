#!/usr/bin/env python3
"""
Sample emotion subset from OpenS2S manifest for 11 emotions.

Filters OpenS2S dataset for 11 emotions (including Relaxed and Tired using Neutral samples)
and creates train/eval splits for GLM-4-Voice fine-tuning.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter
import sys


# Default target emotions - note Relaxed and Tired use Neutral source samples
# Disgust/DISGUSTed will be normalized to "Disgusted"
TARGET_EMOTIONS = ["Sad", "Excited", "Frustrated", "Neutral", "Happy", "Angry", "Fear", "Surprised", "Disgust"]
# Relaxed and Tired are handled specially by sampling additional Neutral samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample emotion subset from OpenS2S manifest (11 emotions)"
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        required=True,
        help="Path to OpenS2S manifest_en.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for sampled metadata",
    )
    parser.add_argument(
        "--samples-per-emotion",
        type=int,
        default=1000,
        help="Number of samples per emotion (default: 1000)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train/eval split ratio (default: 0.7)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    """Load OpenS2S manifest JSONL file."""
    print(f"Loading manifest from: {manifest_path}")
    records = []

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}", file=sys.stderr)
                continue

    print(f"Loaded {len(records)} records")
    return records


def filter_by_emotion(
    records: List[Dict[str, Any]], target_emotions: List[str]
) -> Dict[str, List[Dict[str, Any]]]:
    """Filter records by query emotion. Handles Disgust/DISGUSTed variants."""
    print(f"\nFiltering for query emotions: {target_emotions}")

    emotion_groups = {emotion: [] for emotion in target_emotions}
    # Also handle DISGUSTed variant
    emotion_groups["DISGUSTed"] = []

    for record in records:
        query_emotion = record.get("query", {}).get("emotion")
        if query_emotion in target_emotions:
            emotion_groups[query_emotion].append(record)
        elif query_emotion == "DISGUSTed":
            # Collect DISGUSTed separately for merging with Disgust
            emotion_groups["DISGUSTed"].append(record)

    # Merge Disgust and DISGUSTed
    if "Disgust" in emotion_groups and "DISGUSTed" in emotion_groups:
        disgust_samples = emotion_groups["Disgust"] + emotion_groups["DISGUSTed"]
        emotion_groups["Disgust"] = disgust_samples
        print(f"  Merged 'Disgust' and 'DISGUSTed' variants: {len(disgust_samples)} total samples")

    # Remove DISGUSTed from groups (now merged into Disgust)
    emotion_groups.pop("DISGUSTed", None)

    # Print statistics
    print("\nEmotion distribution before sampling:")
    for emotion, samples in emotion_groups.items():
        print(f"  {emotion}: {len(samples)} samples")

    return emotion_groups


def sample_and_split_11emo(
    emotion_groups: Dict[str, List[Dict[str, Any]]],
    samples_per_emotion: int,
    train_ratio: float,
    random_seed: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Sample and split into train/eval sets with special handling for Relaxed and Tired."""
    random.seed(random_seed)

    train_samples = []
    eval_samples = []

    train_count = int(samples_per_emotion * train_ratio)
    eval_count = samples_per_emotion - train_count

    print(f"\nSampling {samples_per_emotion} per emotion:")
    print(f"  Train: {train_count} samples per emotion")
    print(f"  Eval: {eval_count} samples per emotion")

    # Process standard emotions (excluding Neutral for now)
    for emotion, samples in emotion_groups.items():
        if emotion == "Neutral":
            continue  # Handle Neutral specially below

        # Normalize "Disgust" to "Disgusted" in output
        output_emotion = "Disgusted" if emotion == "Disgust" else emotion

        if len(samples) < samples_per_emotion:
            print(
                f"Warning: {emotion} has only {len(samples)} samples, "
                f"requested {samples_per_emotion}",
                file=sys.stderr,
            )
            samples_per_emotion_actual = len(samples)
            train_count_actual = int(samples_per_emotion_actual * train_ratio)
            eval_count_actual = samples_per_emotion_actual - train_count_actual
        else:
            samples_per_emotion_actual = samples_per_emotion
            train_count_actual = train_count
            eval_count_actual = eval_count

        # Shuffle and sample
        shuffled = samples.copy()
        random.shuffle(shuffled)
        sampled = shuffled[:samples_per_emotion_actual]

        # Split train/eval
        train_split = sampled[:train_count_actual]
        eval_split = sampled[train_count_actual:]

        train_samples.extend([(s, output_emotion) for s in train_split])
        eval_samples.extend([(s, output_emotion) for s in eval_split])

        print(f"  {output_emotion}: {len(train_split)} train + {len(eval_split)} eval")

    # Special handling for Neutral: need samples for Neutral, Relaxed, AND Tired
    neutral_samples = emotion_groups.get("Neutral", [])
    required_neutral = samples_per_emotion * 3  # For Neutral, Relaxed, and Tired

    if len(neutral_samples) < required_neutral:
        print(
            f"Warning: Neutral has only {len(neutral_samples)} samples, "
            f"requested {required_neutral} (for Neutral + Relaxed + Tired)",
            file=sys.stderr,
        )
        # Split available samples evenly between Neutral, Relaxed, and Tired
        per_emotion = len(neutral_samples) // 3
        train_count_actual = int(per_emotion * train_ratio)
        eval_count_actual = per_emotion - train_count_actual
    else:
        per_emotion = samples_per_emotion
        train_count_actual = train_count
        eval_count_actual = eval_count

    # Shuffle all Neutral samples
    shuffled_neutral = neutral_samples.copy()
    random.shuffle(shuffled_neutral)

    # First batch for Neutral
    neutral_sampled = shuffled_neutral[:per_emotion]
    neutral_train = neutral_sampled[:train_count_actual]
    neutral_eval = neutral_sampled[train_count_actual:per_emotion]

    train_samples.extend([(s, "Neutral") for s in neutral_train])
    eval_samples.extend([(s, "Neutral") for s in neutral_eval])
    print(f"  Neutral: {len(neutral_train)} train + {len(neutral_eval)} eval")

    # Second batch for Relaxed (non-overlapping)
    relaxed_sampled = shuffled_neutral[per_emotion:per_emotion * 2]
    relaxed_train = relaxed_sampled[:train_count_actual]
    relaxed_eval = relaxed_sampled[train_count_actual:per_emotion]

    train_samples.extend([(s, "Relaxed") for s in relaxed_train])
    eval_samples.extend([(s, "Relaxed") for s in relaxed_eval])
    print(f"  Relaxed (from Neutral): {len(relaxed_train)} train + {len(relaxed_eval)} eval")

    # Third batch for Tired (non-overlapping)
    tired_sampled = shuffled_neutral[per_emotion * 2:per_emotion * 3]
    tired_train = tired_sampled[:train_count_actual]
    tired_eval = tired_sampled[train_count_actual:per_emotion]

    train_samples.extend([(s, "Tired") for s in tired_train])
    eval_samples.extend([(s, "Tired") for s in tired_eval])
    print(f"  Tired (from Neutral): {len(tired_train)} train + {len(tired_eval)} eval")

    return train_samples, eval_samples


def extract_sample_info(record: Dict[str, Any], emotion_label: str) -> Dict[str, Any]:
    """Extract relevant fields from OpenS2S record with custom emotion label."""
    return {
        "index": record["index"],
        "query_text": record["query"]["text"],
        "query_emotion": emotion_label,  # Use provided label (important for Relaxed/Tired/Disgusted)
        "source_emotion": record["query"]["emotion"],  # Original emotion from source
        "query_audio_path": record["query"]["audio"],
        "response_text": record["response"]["text"],
        "response_emotion": record["response"]["emotion"],
        "response_audio_path": record["response"]["audio"],
    }


def save_samples(
    train_samples: List[tuple],
    eval_samples: List[tuple],
    output_dir: Path,
    random_seed: int,
):
    """Save sampled data and statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract info from samples
    train_data = [extract_sample_info(s, emotion) for s, emotion in train_samples]
    eval_data = [extract_sample_info(s, emotion) for s, emotion in eval_samples]

    # Save train samples
    train_path = output_dir / "sampled_train.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for sample in train_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(train_data)} train samples to: {train_path}")

    # Save eval samples
    eval_path = output_dir / "sampled_eval.jsonl"
    with eval_path.open("w", encoding="utf-8") as f:
        for sample in eval_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved {len(eval_data)} eval samples to: {eval_path}")

    # Calculate statistics
    train_emotions = Counter(s["query_emotion"] for s in train_data)
    eval_emotions = Counter(s["query_emotion"] for s in eval_data)

    train_text_lengths = [len(s["query_text"]) for s in train_data]
    eval_text_lengths = [len(s["query_text"]) for s in eval_data]

    stats = {
        "random_seed": random_seed,
        "emotions": ["Sad", "Excited", "Frustrated", "Neutral", "Happy", "Angry", "Fear", "Relaxed", "Surprised", "Disgusted", "Tired"],
        "train": {
            "total_samples": len(train_data),
            "emotion_distribution": dict(train_emotions),
            "query_text_length": {
                "min": min(train_text_lengths),
                "max": max(train_text_lengths),
                "mean": sum(train_text_lengths) / len(train_text_lengths),
            },
        },
        "eval": {
            "total_samples": len(eval_data),
            "emotion_distribution": dict(eval_emotions),
            "query_text_length": {
                "min": min(eval_text_lengths),
                "max": max(eval_text_lengths),
                "mean": sum(eval_text_lengths) / len(eval_text_lengths),
            },
        },
    }

    # Save statistics
    stats_path = output_dir / "sampling_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Saved statistics to: {stats_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(train_data) + len(eval_data)}")
    print(f"  Train: {len(train_data)}")
    print(f"  Eval: {len(eval_data)}")
    print("\nEmotion distribution:")
    for emotion in sorted(set(train_emotions.keys()) | set(eval_emotions.keys())):
        train_count = train_emotions.get(emotion, 0)
        eval_count = eval_emotions.get(emotion, 0)
        print(f"  {emotion}: {train_count} train + {eval_count} eval = {train_count + eval_count}")
    print("=" * 60)


def main():
    args = parse_args()

    # Load manifest
    records = load_manifest(args.input_manifest)

    # Filter by emotion (9 source emotions - Relaxed and Tired use Neutral)
    emotion_groups = filter_by_emotion(records, TARGET_EMOTIONS)

    # Check if we have enough samples
    for emotion, samples in emotion_groups.items():
        if len(samples) == 0:
            print(f"Error: No samples found for emotion '{emotion}'", file=sys.stderr)
            sys.exit(1)

    # Check Neutral has enough for Neutral, Relaxed, and Tired
    neutral_count = len(emotion_groups.get("Neutral", []))
    required = args.samples_per_emotion * 3
    if neutral_count < required:
        print(f"Warning: Neutral has {neutral_count} samples, need {required} for Neutral + Relaxed + Tired")

    # Sample and split (with special Relaxed and Tired handling)
    train_samples, eval_samples = sample_and_split_11emo(
        emotion_groups,
        args.samples_per_emotion,
        args.train_ratio,
        args.random_seed,
    )

    # Save results
    save_samples(train_samples, eval_samples, args.output_dir, args.random_seed)

    print("\nDone!")


if __name__ == "__main__":
    main()
