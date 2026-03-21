#!/usr/bin/env python3
"""
Patch old 11-emo Part 1 token data (fix swapped Frustrated/Disgusted VA,
adjust Relaxed VA) and merge with new Content data to produce 12-emo Part 1.

Usage:
  python dataset_creation/patch_and_merge_part1.py \
      --old-dir /engram/naplab/users/sd3705/Datasets/Sympatheia_11Emo_17k/tokens \
      --content-dir /engram/naplab/users/sd3705/Datasets/Sympatheia_12Emo_17k/tokens \
      --output-dir /engram/naplab/users/sd3705/Datasets/Sympatheia_12Emo_17k_merged/tokens
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


# VA substitutions in the system prompt: "valence=X, arousal=Y"
# We must swap Frustrated ↔ Disgusted and adjust Relaxed.
# Use a two-pass swap (temp placeholder) to avoid double-replacing.
VA_REPLACEMENTS = {
    # old → new  (system prompt format: "valence=V, arousal=A")
    ("0.40", "-0.45"):  ("0.25", "-0.60"),   # Relaxed
    # Frustrated ↔ Disgusted swap (need temp to avoid collision)
    ("-0.82", "-0.20"): ("-0.80", "0.35"),    # old Frustrated → new Frustrated
    ("-0.80", "0.35"):  ("-0.82", "-0.20"),   # old Disgusted  → new Disgusted
}

# Regex to find VA in system prompt
VA_PATTERN = re.compile(r"valence=([\-\d.]+), arousal=([\-\d.]+)")
# Regex to find trailing valence/arousal lines
TRAILING_V_PATTERN = re.compile(r"^valence: ([\-\d.]+)$", re.MULTILINE)
TRAILING_A_PATTERN = re.compile(r"^arousal: ([\-\d.]+)$", re.MULTILINE)


def patch_text(text: str) -> str:
    """Patch VA values in a single sample's text field."""
    m = VA_PATTERN.search(text)
    if not m:
        return text

    old_v, old_a = m.group(1), m.group(2)
    key = (old_v, old_a)
    if key not in VA_REPLACEMENTS:
        return text

    new_v, new_a = VA_REPLACEMENTS[key]

    # Replace in system prompt
    text = text.replace(
        f"valence={old_v}, arousal={old_a}",
        f"valence={new_v}, arousal={new_a}",
    )
    # Replace trailing metadata if present
    text = TRAILING_V_PATTERN.sub(f"valence: {new_v}", text)
    text = TRAILING_A_PATTERN.sub(f"arousal: {new_a}", text)
    return text


def process_file(input_path: Path, output_path: Path) -> dict:
    """Patch a JSONL file. Returns stats."""
    stats = Counter()
    with input_path.open() as fin, output_path.open("w") as fout:
        for line in fin:
            d = json.loads(line)
            text = d["text"]
            m = VA_PATTERN.search(text)
            old_va = (m.group(1), m.group(2)) if m else None

            d["text"] = patch_text(text)
            fout.write(json.dumps(d) + "\n")

            new_m = VA_PATTERN.search(d["text"])
            new_va = (new_m.group(1), new_m.group(2)) if new_m else None

            if old_va != new_va:
                stats[f"{old_va} -> {new_va}"] += 1
            else:
                stats[f"{old_va} (unchanged)"] += 1
    return dict(stats)


def merge_files(patched_path: Path, content_path: Path, merged_path: Path) -> int:
    """Concatenate patched old data + new Content data."""
    count = 0
    with merged_path.open("w") as fout:
        for src in [patched_path, content_path]:
            with src.open() as fin:
                for line in fin:
                    fout.write(line)
                    count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Patch 11-emo VA values and merge with Content")
    parser.add_argument("--old-dir", type=Path, required=True, help="Old 11-emo tokens dir")
    parser.add_argument("--content-dir", type=Path, required=True, help="New Content tokens dir")
    parser.add_argument("--output-dir", type=Path, required=True, help="Merged output dir")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    patched_dir = args.output_dir / "_patched_tmp"
    patched_dir.mkdir(exist_ok=True)

    for split in ["train", "eval"]:
        old_file = args.old_dir / f"{split}.jsonl"
        content_file = args.content_dir / f"{split}.jsonl"
        patched_file = patched_dir / f"{split}.jsonl"
        merged_file = args.output_dir / f"{split}.jsonl"

        if not old_file.exists():
            print(f"WARNING: {old_file} not found, skipping")
            continue

        # Step 1: Patch old data
        print(f"\n{'='*60}")
        print(f"Patching {split}: {old_file}")
        stats = process_file(old_file, patched_file)
        for k, v in sorted(stats.items()):
            print(f"  {k}: {v}")

        # Step 2: Merge patched + Content
        if content_file.exists():
            total = merge_files(patched_file, content_file, merged_file)
            print(f"Merged → {merged_file} ({total} samples)")
        else:
            print(f"WARNING: {content_file} not found, using patched only")
            patched_file.rename(merged_file)

    # Verify final distribution
    print(f"\n{'='*60}")
    print("Final VA distribution:")
    for split in ["train", "eval"]:
        merged_file = args.output_dir / f"{split}.jsonl"
        if not merged_file.exists():
            continue
        va_count = Counter()
        total = 0
        for line in merged_file.open():
            d = json.loads(line)
            m = VA_PATTERN.search(d["text"])
            if m:
                va_count[(m.group(1), m.group(2))] += 1
            total += 1
        print(f"\n  {split} ({total} total):")
        for (v, a), c in sorted(va_count.items(), key=lambda x: -x[1]):
            print(f"    valence={v}, arousal={a}: {c}")

    # Cleanup temp
    import shutil
    shutil.rmtree(patched_dir, ignore_errors=True)
    print(f"\nDone! Output: {args.output_dir}")


if __name__ == "__main__":
    main()
