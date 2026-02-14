#!/usr/bin/env python3
"""
Add trailing silence to OpenS2S Qwen3-TTS generated audio files.

This fixes the issue where AI-generated audio ends immediately after speech,
causing the model to not learn when to stop generating during inference.

The StyleTalk dataset has ~1 second of trailing silence which creates a
recognizable "end of speech" token pattern that the model learns.
"""

import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import shutil


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add trailing silence to audio files"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory containing audio files (will be modified in-place or backed up)",
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=1.0,
        help="Duration of silence to add in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup of original files before modifying",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Directory for backups (default: input_dir_backup)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without modifying files",
    )
    return parser.parse_args()


def add_silence_to_audio(audio_path: Path, silence_duration: float, dry_run: bool = False):
    """Add trailing silence to an audio file."""
    try:
        # Read audio
        audio, sr = sf.read(str(audio_path))

        # Create silence array
        silence_samples = int(silence_duration * sr)

        # Handle mono vs stereo
        if len(audio.shape) == 1:
            silence = np.zeros(silence_samples, dtype=audio.dtype)
        else:
            silence = np.zeros((silence_samples, audio.shape[1]), dtype=audio.dtype)

        # Concatenate audio with silence
        audio_with_silence = np.concatenate([audio, silence])

        if not dry_run:
            # Write back
            sf.write(str(audio_path), audio_with_silence, sr)

        original_duration = len(audio) / sr
        new_duration = len(audio_with_silence) / sr

        return True, original_duration, new_duration

    except Exception as e:
        return False, 0, str(e)


def process_directory(input_dir: Path, silence_duration: float,
                      backup: bool, backup_dir: Path, dry_run: bool):
    """Process all WAV files in directory tree."""

    # Find all WAV files
    wav_files = list(input_dir.rglob("*.wav"))
    print(f"Found {len(wav_files)} WAV files")

    if not wav_files:
        print("No WAV files found!")
        return

    # Create backup if requested
    if backup and not dry_run:
        if backup_dir is None:
            backup_dir = input_dir.parent / f"{input_dir.name}_backup"

        if backup_dir.exists():
            print(f"Backup directory already exists: {backup_dir}")
            response = input("Overwrite? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
            shutil.rmtree(backup_dir)

        print(f"Creating backup at: {backup_dir}")
        shutil.copytree(input_dir, backup_dir)
        print("Backup complete")

    # Process files
    success_count = 0
    fail_count = 0
    total_added_duration = 0

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Adding {silence_duration}s silence to each file...")

    for wav_path in tqdm(wav_files, desc="Processing"):
        success, orig_dur, new_dur = add_silence_to_audio(
            wav_path, silence_duration, dry_run
        )

        if success:
            success_count += 1
            total_added_duration += silence_duration
        else:
            fail_count += 1
            print(f"\nFailed: {wav_path} - {new_dur}")

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Results:")
    print(f"  Processed: {success_count} files")
    print(f"  Failed: {fail_count} files")
    print(f"  Total silence added: {total_added_duration:.1f}s ({total_added_duration/60:.1f} min)")


def main():
    args = parse_args()

    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        return

    print(f"Input directory: {args.input_dir}")
    print(f"Silence duration: {args.silence_duration}s")
    print(f"Backup: {args.backup}")
    print(f"Dry run: {args.dry_run}")
    print()

    process_directory(
        input_dir=args.input_dir,
        silence_duration=args.silence_duration,
        backup=args.backup,
        backup_dir=args.backup_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
