#!/usr/bin/env python3
"""
Convert Qwen3-TTS generated audio to GLM-4-Voice format with Valence-Arousal values.

Encodes audio files to tokens and creates JSONL files for GLM-4-Voice training
using valence-arousal emotion representation for 12 emotions.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vocoder import GLM4CodecEncoder
from tqdm import tqdm


# Emotion to Valence-Arousal mapping for 12 emotions
EMOTION_VA_MAPPING = {
    "Sad": (-0.75, -0.65),
    "Excited": (0.75, 0.90),
    "Frustrated": (-0.80, 0.35),
    "Neutral": (0.00, 0.00),
    "Happy": (0.85, 0.35),
    "Angry": (-0.85, 0.85),
    "Anxious": (-0.40, 0.65),
    "Relaxed": (0.25, -0.60),
    "Surprised": (0.10, 0.80),
    "Disgusted": (-0.82, -0.20),
    "Tired": (-0.15, -0.75),
    "Content": (0.60, -0.20),
}

ALL_EMOTIONS = ["Sad", "Excited", "Frustrated", "Neutral", "Happy", "Angry", "Anxious", "Relaxed", "Surprised", "Disgusted", "Tired", "Content"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Qwen3-TTS audio to GLM-4-Voice format with VA values (12 emotions)"
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        required=True,
        help="Directory containing sampled metadata JSONL files",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Directory containing generated audio files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for GLM-4-Voice JSONL files",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Log progress every N examples (default: 50)",
    )
    return parser.parse_args()


def load_metadata(metadata_dir: Path, split: str):
    """Load metadata JSONL file."""
    metadata_path = metadata_dir / f"sampled_{split}.jsonl"
    print(f"Loading {split} metadata from: {metadata_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    samples = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    print(f"Loaded {len(samples)} {split} samples")
    return samples


def encode_audio(audio_path: Path, encoder: GLM4CodecEncoder) -> str:
    """Encode audio file to token string."""
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Encode audio to tokens
    tokens = encoder([str(audio_path)])[0]  # Returns torch.Tensor

    # Convert to token string
    token_str = "".join(f"<|audio_{t}|>" for t in tokens)

    return token_str


def build_glm4voice_text_va(
    query_tokens: str,
    response_text: str,
    response_tokens: str,
    valence: float,
    arousal: float,
) -> str:
    """Build GLM-4-Voice format text with Valence-Arousal values."""
    text = (
        "<|system|>\n"
        f"Please respond in English. User emotion (valence={valence:.2f}, arousal={arousal:.2f})\n"
        "<|user|>\n"
        f"{query_tokens}\n"
        "<|assistant|>\n"
        f"{response_text}\n"
        f"{response_tokens}<|user|>"
    )

    return text


def process_split(
    samples: list,
    split: str,
    audio_dir: Path,
    output_dir: Path,
    encoder: GLM4CodecEncoder,
    log_every: int,
):
    """Process one split and create GLM-4-Voice JSONL with VA values."""
    print(f"\n{'=' * 60}")
    print(f"Processing {split} split ({len(samples)} samples)")
    print(f"{'=' * 60}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}.jsonl"

    # Statistics
    token_stats = {"query_tokens": [], "response_tokens": []}
    failed_samples = []
    processed_count = 0
    emotion_counts = {emotion: 0 for emotion in ALL_EMOTIONS}

    with output_path.open("w", encoding="utf-8") as out_f:
        for sample in tqdm(samples, desc=f"Encoding {split}"):
            try:
                # Get emotion and paths
                emotion = sample["query_emotion"]
                index = sample["index"]

                # Get VA values
                if emotion not in EMOTION_VA_MAPPING:
                    print(f"Warning: Unknown emotion '{emotion}', skipping", file=sys.stderr)
                    failed_samples.append({"index": index, "reason": f"unknown_emotion_{emotion}"})
                    continue

                valence, arousal = EMOTION_VA_MAPPING[emotion]

                # Build audio file paths
                query_audio_path = (
                    audio_dir / split / f"{emotion.lower()}_query" / f"{index}.wav"
                )
                response_audio_path = (
                    audio_dir / split / f"{emotion.lower()}_response" / f"{index}.wav"
                )

                # Check if files exist
                if not query_audio_path.exists():
                    print(
                        f"Warning: Query audio not found: {query_audio_path}",
                        file=sys.stderr,
                    )
                    failed_samples.append(
                        {"index": index, "reason": "query_audio_missing"}
                    )
                    continue

                if not response_audio_path.exists():
                    print(
                        f"Warning: Response audio not found: {response_audio_path}",
                        file=sys.stderr,
                    )
                    failed_samples.append(
                        {"index": index, "reason": "response_audio_missing"}
                    )
                    continue

                # Encode audio to tokens
                query_tokens = encode_audio(query_audio_path, encoder)
                response_tokens = encode_audio(response_audio_path, encoder)

                # Track token statistics
                query_token_count = query_tokens.count("<|audio_")
                response_token_count = response_tokens.count("<|audio_")
                token_stats["query_tokens"].append(query_token_count)
                token_stats["response_tokens"].append(response_token_count)

                # Build GLM-4-Voice format with VA values
                text = build_glm4voice_text_va(
                    query_tokens=query_tokens,
                    response_text=sample["response_text"],
                    response_tokens=response_tokens,
                    valence=valence,
                    arousal=arousal,
                )

                # Create record with VA values as separate fields
                record = {
                    "text": text,
                    "id": f"opens2s_{emotion.lower()}_{index}",
                    "valence": valence,
                    "arousal": arousal,
                }

                # Write to JSONL
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed_count += 1
                emotion_counts[emotion] += 1

                # Log progress
                if log_every and processed_count % log_every == 0:
                    print(
                        f"  Processed {processed_count} / {len(samples)} samples...",
                        flush=True,
                    )

            except Exception as e:
                print(
                    f"Error processing sample {sample.get('index', 'unknown')}: {e}",
                    file=sys.stderr,
                )
                failed_samples.append(
                    {"index": sample.get("index"), "reason": str(e)}
                )
                continue

    print(f"\nWrote {processed_count} samples to: {output_path}")

    if failed_samples:
        print(f"Failed to process {len(failed_samples)} samples")

    # Print emotion distribution
    print(f"\nEmotion distribution for {split}:")
    for emotion, count in sorted(emotion_counts.items()):
        if count > 0:
            va = EMOTION_VA_MAPPING[emotion]
            print(f"  {emotion}: {count} (V={va[0]:.2f}, A={va[1]:.2f})")

    # Calculate and print token statistics
    if token_stats["query_tokens"]:
        print(f"\nToken statistics for {split}:")
        print(
            f"  Query tokens: min={min(token_stats['query_tokens'])}, "
            f"max={max(token_stats['query_tokens'])}, "
            f"mean={sum(token_stats['query_tokens']) / len(token_stats['query_tokens']):.1f}"
        )
        print(
            f"  Response tokens: min={min(token_stats['response_tokens'])}, "
            f"max={max(token_stats['response_tokens'])}, "
            f"mean={sum(token_stats['response_tokens']) / len(token_stats['response_tokens']):.1f}"
        )

    return token_stats, failed_samples, emotion_counts


def save_encoding_stats(
    output_dir: Path,
    train_stats: Dict,
    eval_stats: Dict,
    train_failed: list,
    eval_failed: list,
    train_emotion_counts: Dict,
    eval_emotion_counts: Dict,
):
    """Save encoding statistics and conversion config."""
    stats = {
        "train": {
            "processed_samples": len(train_stats["query_tokens"]),
            "failed_samples": len(train_failed),
            "emotion_distribution": train_emotion_counts,
            "query_tokens": {
                "min": min(train_stats["query_tokens"])
                if train_stats["query_tokens"]
                else 0,
                "max": max(train_stats["query_tokens"])
                if train_stats["query_tokens"]
                else 0,
                "mean": sum(train_stats["query_tokens"])
                / len(train_stats["query_tokens"])
                if train_stats["query_tokens"]
                else 0,
            },
            "response_tokens": {
                "min": min(train_stats["response_tokens"])
                if train_stats["response_tokens"]
                else 0,
                "max": max(train_stats["response_tokens"])
                if train_stats["response_tokens"]
                else 0,
                "mean": sum(train_stats["response_tokens"])
                / len(train_stats["response_tokens"])
                if train_stats["response_tokens"]
                else 0,
            },
        },
        "eval": {
            "processed_samples": len(eval_stats["query_tokens"]),
            "failed_samples": len(eval_failed),
            "emotion_distribution": eval_emotion_counts,
            "query_tokens": {
                "min": min(eval_stats["query_tokens"])
                if eval_stats["query_tokens"]
                else 0,
                "max": max(eval_stats["query_tokens"])
                if eval_stats["query_tokens"]
                else 0,
                "mean": sum(eval_stats["query_tokens"])
                / len(eval_stats["query_tokens"])
                if eval_stats["query_tokens"]
                else 0,
            },
            "response_tokens": {
                "min": min(eval_stats["response_tokens"])
                if eval_stats["response_tokens"]
                else 0,
                "max": max(eval_stats["response_tokens"])
                if eval_stats["response_tokens"]
                else 0,
                "mean": sum(eval_stats["response_tokens"])
                / len(eval_stats["response_tokens"])
                if eval_stats["response_tokens"]
                else 0,
            },
        },
    }

    stats_path = output_dir / "encoding_stats.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved encoding statistics to: {stats_path}")

    # Save conversion config (VA mapping)
    config = {
        "emotion_to_va_mapping": {k: list(v) for k, v in EMOTION_VA_MAPPING.items()},
        "add_noise": False,
        "noise_scale": None,
    }

    config_path = output_dir / "conversion_config.json"
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved conversion config to: {config_path}")


def main():
    args = parse_args()

    # Initialize encoder
    print("Initializing GLM4CodecEncoder...")
    encoder = GLM4CodecEncoder()
    print("Encoder initialized successfully")

    # Process train split
    train_samples = load_metadata(args.metadata_dir, "train")
    train_stats, train_failed, train_emotion_counts = process_split(
        samples=train_samples,
        split="train",
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        encoder=encoder,
        log_every=args.log_every,
    )

    # Process eval split
    eval_samples = load_metadata(args.metadata_dir, "eval")
    eval_stats, eval_failed, eval_emotion_counts = process_split(
        samples=eval_samples,
        split="eval",
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        encoder=encoder,
        log_every=args.log_every,
    )

    # Save statistics
    save_encoding_stats(
        args.output_dir,
        train_stats,
        eval_stats,
        train_failed,
        eval_failed,
        train_emotion_counts,
        eval_emotion_counts,
    )

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print(f"Train JSONL: {args.output_dir / 'train.jsonl'}")
    print(f"Eval JSONL: {args.output_dir / 'eval.jsonl'}")


if __name__ == "__main__":
    main()
