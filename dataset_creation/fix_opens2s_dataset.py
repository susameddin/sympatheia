#!/usr/bin/env python3
"""
Fix OpenS2S dataset by adding trailing silence and re-encoding to GLM-4-Voice format.

This script:
1. Adds trailing silence to all audio files (fixes the "no end pattern" issue)
2. Re-encodes audio to tokens using GLM4CodecEncoder
3. Regenerates the JSONL files for training

Usage:
    python fix_opens2s_dataset.py --dataset-dir Datasets/OpenS2S_Qwen3TTS

This will create a new 'glm4voice_fixed' directory with the regenerated JSONL files.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fix OpenS2S dataset by adding trailing silence and re-encoding"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to OpenS2S_Qwen3TTS dataset directory",
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=1.0,
        help="Duration of trailing silence in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_fixed",
        help="Suffix for output JSONL directory (default: _fixed)",
    )
    parser.add_argument(
        "--skip-silence",
        action="store_true",
        help="Skip adding silence (use if already added)",
    )
    parser.add_argument(
        "--modify-audio-inplace",
        action="store_true",
        help="Modify audio files in-place instead of creating copies",
    )
    return parser.parse_args()


def add_silence_to_audio_file(audio_path: Path, silence_duration: float) -> bool:
    """Add trailing silence to a single audio file (in-place)."""
    try:
        audio, sr = sf.read(str(audio_path))

        # Create silence
        silence_samples = int(silence_duration * sr)
        if len(audio.shape) == 1:
            silence = np.zeros(silence_samples, dtype=audio.dtype)
        else:
            silence = np.zeros((silence_samples, audio.shape[1]), dtype=audio.dtype)

        # Concatenate and write
        audio_with_silence = np.concatenate([audio, silence])
        sf.write(str(audio_path), audio_with_silence, sr)

        return True
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return False


def add_silence_to_all_audio(audio_dir: Path, silence_duration: float):
    """Add trailing silence to all WAV files in directory."""
    wav_files = list(audio_dir.rglob("*.wav"))
    print(f"\nAdding {silence_duration}s trailing silence to {len(wav_files)} audio files...")

    success = 0
    failed = 0

    for wav_path in tqdm(wav_files, desc="Adding silence"):
        if add_silence_to_audio_file(wav_path, silence_duration):
            success += 1
        else:
            failed += 1

    print(f"  Success: {success}, Failed: {failed}")
    return failed == 0


def encode_audio_to_tokens(audio_path: Path, encoder) -> str:
    """Encode audio file to GLM token string."""
    tokens = encoder([str(audio_path)])[0]
    token_str = "".join(f"<|audio_{t}|>" for t in tokens)
    return token_str


def build_glm4voice_text(
    query_tokens: str,
    response_text: str,
    response_tokens: str,
    emotion: str,
) -> str:
    """Build GLM-4-Voice format text."""
    tone = emotion.lower()
    text = (
        "<|system|>\n"
        f"Please respond in English. Please respond with this tone: {tone}.\n"
        "<|user|>\n"
        f"{query_tokens}\n"
        "<|assistant|>\n"
        f"{response_text}\n"
        f"{response_tokens}"
    )
    return text


def regenerate_jsonl(
    metadata_dir: Path,
    audio_dir: Path,
    output_dir: Path,
    encoder,
):
    """Regenerate JSONL files with newly encoded audio tokens."""

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "eval"]:
        metadata_path = metadata_dir / f"sampled_{split}.jsonl"
        if not metadata_path.exists():
            print(f"Warning: {metadata_path} not found, skipping {split}")
            continue

        # Load metadata
        samples = []
        with open(metadata_path) as f:
            for line in f:
                samples.append(json.loads(line))

        print(f"\nRe-encoding {split} split ({len(samples)} samples)...")

        output_path = output_dir / f"{split}.jsonl"
        token_counts = {"query": [], "response": []}
        failed = []

        with open(output_path, "w") as out_f:
            for sample in tqdm(samples, desc=f"Encoding {split}"):
                try:
                    emotion = sample["query_emotion"]
                    index = sample["index"]

                    # Build audio paths
                    query_path = audio_dir / split / f"{emotion.lower()}_query" / f"{index}.wav"
                    response_path = audio_dir / split / f"{emotion.lower()}_response" / f"{index}.wav"

                    if not query_path.exists() or not response_path.exists():
                        failed.append({"index": index, "reason": "audio_missing"})
                        continue

                    # Encode audio
                    query_tokens = encode_audio_to_tokens(query_path, encoder)
                    response_tokens = encode_audio_to_tokens(response_path, encoder)

                    # Track token counts
                    q_count = query_tokens.count("<|audio_")
                    r_count = response_tokens.count("<|audio_")
                    token_counts["query"].append(q_count)
                    token_counts["response"].append(r_count)

                    # Build text
                    text = build_glm4voice_text(
                        query_tokens=query_tokens,
                        response_text=sample["response_text"],
                        response_tokens=response_tokens,
                        emotion=emotion,
                    )

                    # Write record
                    record = {
                        "text": text,
                        "id": f"opens2s_{emotion.lower()}_{index}",
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                except Exception as e:
                    failed.append({"index": sample.get("index"), "reason": str(e)})

        print(f"  Wrote: {output_path}")
        print(f"  Query tokens:    min={min(token_counts['query'])}, max={max(token_counts['query'])}, avg={sum(token_counts['query'])/len(token_counts['query']):.1f}")
        print(f"  Response tokens: min={min(token_counts['response'])}, max={max(token_counts['response'])}, avg={sum(token_counts['response'])/len(token_counts['response']):.1f}")

        if failed:
            print(f"  Failed: {len(failed)} samples")

    # Verify ending patterns
    print("\nVerifying ending patterns...")
    verify_ending_patterns(output_dir / "train.jsonl")


def verify_ending_patterns(jsonl_path: Path, num_samples: int = 50):
    """Check if samples now have repeated ending tokens."""
    repeated_count = 0

    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break

            d = json.loads(line)
            text = d.get("text", "")

            # Get response tokens
            parts = text.split("<|assistant|>")
            if len(parts) == 2:
                tokens = re.findall(r"<\|audio_(\d+)\|>", parts[1])
                if len(tokens) >= 6:
                    last_6 = tokens[-6:]
                    if len(set(last_6)) == 1:
                        repeated_count += 1

    pct = 100 * repeated_count / num_samples
    status = "✓" if repeated_count > 0 else "✗"
    print(f"  {status} Samples with repeated ending pattern: {repeated_count}/{num_samples} ({pct:.1f}%)")

    if repeated_count > 0:
        print("  The fix appears to be working!")
    else:
        print("  Warning: No repeated ending patterns found. The silence may not be long enough.")


def main():
    args = parse_args()

    dataset_dir = args.dataset_dir
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    audio_dir = dataset_dir / "audio"
    metadata_dir = dataset_dir / "metadata"
    output_dir = dataset_dir / f"glm4voice{args.output_suffix}"

    print("=" * 60)
    print("OpenS2S Dataset Fix")
    print("=" * 60)
    print(f"Dataset directory: {dataset_dir}")
    print(f"Silence duration: {args.silence_duration}s")
    print(f"Output JSONL dir: {output_dir}")
    print()

    # Step 1: Add trailing silence
    if not args.skip_silence:
        print("Step 1: Adding trailing silence to audio files")
        print("-" * 40)

        if not args.modify_audio_inplace:
            print("WARNING: This will modify audio files in-place!")
            print("Use --skip-silence if you've already added silence.")
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(0)

        if not add_silence_to_all_audio(audio_dir, args.silence_duration):
            print("Some files failed. Check errors above.")
    else:
        print("Step 1: Skipped (--skip-silence)")

    # Step 2: Re-encode and regenerate JSONL
    print("\nStep 2: Re-encoding audio and generating JSONL")
    print("-" * 40)

    # Import encoder (done here to avoid slow import if we abort early)
    print("Loading GLM4CodecEncoder...")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.vocoder import GLM4CodecEncoder
    encoder = GLM4CodecEncoder()
    print("Encoder ready")

    regenerate_jsonl(
        metadata_dir=metadata_dir,
        audio_dir=audio_dir,
        output_dir=output_dir,
        encoder=encoder,
    )

    print("\n" + "=" * 60)
    print("Fix complete!")
    print("=" * 60)
    print(f"\nNew JSONL files created in: {output_dir}")
    print("\nTo use the fixed dataset, update your training script:")
    print(f'  data_files = {{')
    print(f'      "train": "{output_dir}/train.jsonl",')
    print(f'      "validation": "{output_dir}/eval.jsonl",')
    print(f'  }}')


if __name__ == "__main__":
    main()
