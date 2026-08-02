#!/usr/bin/env python3
"""
Create the DISCRETE-LABEL ablation variant of a token JSONL file.

Ablation for the continuous-vs-discrete study: replace the continuous
valence/arousal system clause with the discrete emotion *word*, changing
nothing else about the encoded example (audio tokens, response, splits, and the
N/A masking are all preserved verbatim).

  VA (existing):   Please respond in English. User emotion (valence=-0.75, arousal=-0.65)
  Discrete (this): Please respond in English. User emotion: Sad

Rows already masked to "User emotion N/A" are left unchanged (they don't match
the VA pattern), so the discrete model sees the exact same set of unconditioned
examples as the VA model — the only controlled difference is the conditioning
*representation*.

This mirrors `emotional/create_na_variant.py` (VA -> "N/A"); here it is
VA -> discrete word. No GPU and no audio re-encoding are needed: it is a pure
text substitution over the `text` field of the already-encoded JSONL.

The discrete label is recovered from the captured (valence, arousal) pair via a
reverse lookup of EMOTION_VA_MAPPING (the 12 anchors are a bijection, so the
mapping is unambiguous).

Usage:
  python dataset_creation/create_discrete_variant.py \
      --input-dir /path/to/Sympatheia-18k \
      --suffix _discrete

The resulting encoded_{train,eval}_discrete.jsonl are drop-in replacements for
the VA files in train_sympatheia.py's `data_files`; fine-tuning is otherwise
identical, so the two runs differ only in the conditioning representation.
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Import the single source of truth for the emotion<->VA mapping (read-only).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.constants import EMOTION_VA_MAPPING

# Capture the two VA numbers so we can map them back to the discrete label.
VA_PATTERN = re.compile(
    r"User emotion \(valence=(-?\d+\.\d+), arousal=(-?\d+\.\d+)\)"
)

# Reverse map keyed on the 2-decimal string form the prompt actually uses,
# e.g. (-0.80, 0.35) -> "Frustrated". Built from EMOTION_VA_MAPPING so it stays
# in sync with the VA pipeline.
VA_TO_EMOTION = {
    (f"{v:.2f}", f"{a:.2f}"): emotion
    for emotion, (v, a) in EMOTION_VA_MAPPING.items()
}


def _discrete_replacement(match: re.Match) -> str:
    """Map a matched 'User emotion (valence=V, arousal=A)' to 'User emotion: <Label>'."""
    key = (match.group(1), match.group(2))
    emotion = VA_TO_EMOTION.get(key)
    if emotion is None:
        raise ValueError(
            f"VA pair {key} not found in EMOTION_VA_MAPPING; cannot assign a "
            f"discrete label. Known pairs: {sorted(VA_TO_EMOTION)}"
        )
    return f"User emotion: {emotion}"


def create_discrete_file(input_path: Path, output_path: Path):
    """Read input JSONL, rewrite VA clause -> discrete label, write output."""
    total = 0
    converted = 0
    na_kept = 0

    with input_path.open() as fin, output_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            total += 1
            new_text, n = VA_PATTERN.subn(_discrete_replacement, d["text"])
            if n > 0:
                d["text"] = new_text
                converted += 1
            elif "User emotion N/A" in d["text"]:
                na_kept += 1  # left unchanged on purpose
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(
        f"  {input_path.name} -> {output_path.name}: "
        f"{converted} converted, {na_kept} kept N/A, {total} total"
    )
    return total, converted, na_kept


def main():
    parser = argparse.ArgumentParser(
        description="Create the discrete-label variant of token JSONL files"
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Directory with encoded_train.jsonl and encoded_eval.jsonl",
    )
    parser.add_argument(
        "--suffix", default="_discrete",
        help="Suffix for output files (default: _discrete)",
    )
    args = parser.parse_args()

    for split in ["train", "eval"]:
        input_path = args.input_dir / f"encoded_{split}.jsonl"
        output_path = args.input_dir / f"encoded_{split}{args.suffix}.jsonl"
        if input_path.exists():
            create_discrete_file(input_path, output_path)
        else:
            print(f"  WARNING: {input_path} not found, skipping")

    print("\nDone!")


if __name__ == "__main__":
    main()
