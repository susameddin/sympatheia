#!/usr/bin/env python3
"""
Split an existing sampled_train.jsonl (that contains ALL samples) into
proper train/eval splits using embedding-clustered splitting.

This avoids re-running the full text generation pipeline just to get
the eval split.

Usage:
  conda run -n qwen3-tts4 --no-capture-output python -u \
      dataset_creation/split_train_eval.py \
      --input /engram/naplab/users/sd3705/Datasets/Sympatheia_12Emo_Emotional_v2/metadata/sampled_train.jsonl \
      --output-dir /engram/naplab/users/sd3705/Datasets/Sympatheia_12Emo_Emotional_v2/metadata/ \
      --train-ratio 0.7 \
      --dedup-threshold 0.85
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer


ALL_EMOTIONS = [
    "Sad", "Excited", "Frustrated", "Neutral", "Happy", "Angry",
    "Anxious", "Relaxed", "Surprised", "Disgusted", "Tired", "Content",
]


def main():
    parser = argparse.ArgumentParser(
        description="Split sampled_train.jsonl into train/eval using cluster-based splitting"
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to the combined JSONL file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--audio-dir", type=Path, default=None, help="Audio directory to verify alignment (optional)")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--dedup-threshold", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    # Load all samples
    all_samples = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            all_samples.append(json.loads(line))
    print(f"Loaded {len(all_samples)} samples from {args.input}")

    # Group by emotion
    emotion_samples: Dict[str, List[dict]] = defaultdict(list)
    for s in all_samples:
        emotion_samples[s["query_emotion"]].append(s)

    # Cluster-based split per emotion
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    rng = random.Random(args.seed)
    train_data: List[dict] = []
    eval_data: List[dict] = []

    print(f"\nEmotion split breakdown (cluster-based, threshold={args.dedup_threshold}):")
    for emotion in ALL_EMOTIONS:
        samples = emotion_samples.get(emotion, [])
        if not samples:
            continue

        texts = [s["query_text"] for s in samples]
        embs = emb_model.encode(texts, normalize_embeddings=True)

        # Greedy clustering
        clusters: List[List[int]] = []
        assigned: set = set()
        for i in range(len(texts)):
            if i in assigned:
                continue
            cluster = [i]
            assigned.add(i)
            for j in range(i + 1, len(texts)):
                if j in assigned:
                    continue
                if float(np.dot(embs[i], embs[j])) >= args.dedup_threshold:
                    cluster.append(j)
                    assigned.add(j)
            clusters.append(cluster)

        rng.shuffle(clusters)
        n_train_clusters = int(len(clusters) * args.train_ratio)

        train_indices: set = set()
        for c in clusters[:n_train_clusters]:
            train_indices.update(c)

        emo_train = [samples[i] for i in range(len(samples)) if i in train_indices]
        emo_eval = [samples[i] for i in range(len(samples)) if i not in train_indices]

        train_data.extend(emo_train)
        eval_data.extend(emo_eval)
        print(f"  {emotion:<12}: {len(emo_train)} train + {len(emo_eval)} eval ({len(clusters)} clusters from {len(texts)} samples)")

    rng.shuffle(train_data)
    rng.shuffle(eval_data)

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "sampled_train.jsonl"
    eval_path = args.output_dir / "sampled_eval.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for sample in train_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with eval_path.open("w", encoding="utf-8") as f:
        for sample in eval_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(train_data)} train samples → {train_path}")
    print(f"Saved {len(eval_data)} eval  samples → {eval_path}")

    # ── Verify alignment with existing audio files ────────────────────────
    if args.audio_dir and args.audio_dir.exists():
        print(f"\n{'=' * 60}")
        print("Verifying metadata ↔ audio alignment")
        print(f"{'=' * 60}")

        all_ok = True
        for split_name, split_data in [("train", train_data), ("eval", eval_data)]:
            missing_query = []
            missing_response = []
            for sample in split_data:
                emo_lower = sample["query_emotion"].lower()
                idx = sample["index"]

                query_wav = args.audio_dir / split_name / f"{emo_lower}_query" / f"{idx}.wav"
                response_wav = args.audio_dir / split_name / f"{emo_lower}_response" / f"{idx}.wav"

                if not query_wav.exists():
                    missing_query.append(idx)
                if not response_wav.exists():
                    missing_response.append(idx)

            # Also check for orphan audio files (audio exists but no metadata)
            split_audio_dir = args.audio_dir / split_name
            metadata_indices = {s["index"] for s in split_data}
            orphan_count = 0
            if split_audio_dir.exists():
                for subdir in sorted(split_audio_dir.iterdir()):
                    if not subdir.is_dir():
                        continue
                    for wav_file in subdir.glob("*.wav"):
                        stem = wav_file.stem
                        if stem not in metadata_indices:
                            orphan_count += 1

            if missing_query or missing_response or orphan_count:
                all_ok = False
                print(f"\n  [{split_name}] MISMATCHES FOUND:")
                if missing_query:
                    print(f"    Missing query  audio: {len(missing_query)} files")
                    for m in missing_query[:5]:
                        print(f"      {m}")
                    if len(missing_query) > 5:
                        print(f"      ... and {len(missing_query) - 5} more")
                if missing_response:
                    print(f"    Missing response audio: {len(missing_response)} files")
                    for m in missing_response[:5]:
                        print(f"      {m}")
                    if len(missing_response) > 5:
                        print(f"      ... and {len(missing_response) - 5} more")
                if orphan_count:
                    print(f"    Orphan audio files (no metadata): {orphan_count}")
            else:
                print(f"  [{split_name}] OK — all {len(split_data)} samples have matching audio")

        if not all_ok:
            print(f"\n  WARNING: Audio and metadata are out of sync.")
            print(f"  The audio was generated from a different train/eval split.")
            print(f"  Options:")
            print(f"    1. Re-run audio generation with --resume (only missing files)")
            print(f"    2. Split metadata to match existing audio (use --match-audio)")


if __name__ == "__main__":
    main()
