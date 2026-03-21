#!/usr/bin/env python3
"""
Create NA variant of a token JSONL file: randomly replace ~1/3 of samples'
VA system prompts with "User emotion N/A".

This teaches the model to handle missing VA values at inference time.

Usage:
  python dataset_creation/create_na_variant.py \
      --input-dir /engram/naplab/users/sd3705/Datasets/Sympatheia_12Emo_17k_merged/tokens \
      --na-ratio 0.333
"""

import argparse
import json
import random
import re
from pathlib import Path


VA_PATTERN = re.compile(
    r"User emotion \(valence=[\-\d.]+, arousal=[\-\d.]+\)"
)
NA_REPLACEMENT = "User emotion N/A"


def create_na_file(input_path: Path, output_path: Path, na_ratio: float, seed: int):
    """Read input JSONL, randomly mask VA → N/A, write output."""
    rng = random.Random(seed)
    total = 0
    masked = 0

    with input_path.open() as fin, output_path.open("w") as fout:
        for line in fin:
            d = json.loads(line)
            total += 1
            if rng.random() < na_ratio:
                d["text"] = VA_PATTERN.sub(NA_REPLACEMENT, d["text"])
                masked += 1
            fout.write(json.dumps(d) + "\n")

    print(f"  {input_path.name} → {output_path.name}: {masked}/{total} masked ({masked/total*100:.1f}%)")
    return total, masked


def main():
    parser = argparse.ArgumentParser(description="Create N/A variant of token JSONL files")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with train.jsonl and eval.jsonl")
    parser.add_argument("--na-ratio", type=float, default=1/3, help="Fraction of samples to mask (default: 1/3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--suffix", default="_na", help="Suffix for output files (default: _na)")
    args = parser.parse_args()

    for split in ["train", "eval"]:
        input_path = args.input_dir / f"{split}.jsonl"
        output_path = args.input_dir / f"{split}{args.suffix}.jsonl"
        if input_path.exists():
            create_na_file(input_path, output_path, args.na_ratio, args.seed)
        else:
            print(f"  WARNING: {input_path} not found, skipping")

    print("\nDone!")


if __name__ == "__main__":
    main()
