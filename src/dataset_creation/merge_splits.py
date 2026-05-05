#!/usr/bin/env python3
"""
Merge Emotional and Neutral splits into a single training-ready dataset.

Uses the NA variant of the Emotional split (encoded_train_na.jsonl) so
the merged dataset contains the robustness training signal where ~1/3
of emotional samples have VA masked to N/A.

Output files:
  {output_dir}/encoded_train.jsonl   (Emotional NA train + Neutral train, shuffled)
  {output_dir}/encoded_eval.jsonl    (Emotional NA eval  + Neutral eval,  shuffled)

Usage:
  python dataset_creation/merge_splits.py \\
      --emotional-dir /engram/naplab/users/sd3705/Datasets/Sympatheia-18k/Emotional \\
      --neutral-dir   /engram/naplab/users/sd3705/Datasets/Sympatheia-18k/Neutral \\
      --output-dir    /engram/naplab/users/sd3705/Datasets/Sympatheia-18k
"""

import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path):
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def merge_and_shuffle(emotional_path: Path, neutral_path: Path, seed: int):
    emotional = load_jsonl(emotional_path)
    neutral = load_jsonl(neutral_path)
    combined = emotional + neutral
    random.Random(seed).shuffle(combined)
    return combined


def main():
    parser = argparse.ArgumentParser(description="Merge Emotional and Neutral splits")
    parser.add_argument("--emotional-dir", type=Path, required=True,
                        help="Emotional split directory (must contain encoded_train_na.jsonl and encoded_eval_na.jsonl)")
    parser.add_argument("--neutral-dir", type=Path, required=True,
                        help="Neutral split directory (must contain encoded_train.jsonl and encoded_eval.jsonl)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for merged encoded_train.jsonl and encoded_eval.jsonl")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling (default: 42)")
    args = parser.parse_args()

    for split in ["train", "eval"]:
        emotional_path = args.emotional_dir / f"encoded_{split}_na.jsonl"
        neutral_path = args.neutral_dir / f"encoded_{split}.jsonl"
        output_path = args.output_dir / f"encoded_{split}.jsonl"

        if not emotional_path.exists():
            raise FileNotFoundError(f"Emotional NA file not found: {emotional_path}")
        if not neutral_path.exists():
            raise FileNotFoundError(f"Neutral file not found: {neutral_path}")

        combined = merge_and_shuffle(emotional_path, neutral_path, args.seed)
        write_jsonl(combined, output_path)
        print(f"{split}: {len(combined)} records → {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
