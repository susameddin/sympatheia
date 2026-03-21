#!/usr/bin/env python3
"""
Validate 12-Emotion Qwen3-TTS dataset.

Performs comprehensive quality checks on generated dataset including
file existence, data integrity, audio quality, and format validation
for 12 emotions with Valence-Arousal representation.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter
import sys

try:
    import soundfile as sf
    import numpy as np

    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    print("Warning: soundfile/numpy not available, skipping audio quality checks")


# All 12 emotions (lowercase for path matching)
ALL_EMOTIONS = ["sad", "excited", "frustrated", "neutral", "happy", "angry", "anxious", "relaxed", "surprised", "disgusted", "tired", "content"]

# Expected VA values for validation (updated for 12 emotions)
EXPECTED_VA_VALUES = {
    "sad": (-0.75, -0.65),
    "excited": (0.75, 0.90),
    "frustrated": (-0.80, 0.35),
    "neutral": (0.00, 0.00),
    "happy": (0.85, 0.35),
    "angry": (-0.85, 0.85),
    "anxious": (-0.40, 0.65),
    "relaxed": (0.25, -0.60),
    "surprised": (0.10, 0.80),
    "disgusted": (-0.82, -0.20),
    "tired": (-0.15, -0.75),
    "content": (0.60, -0.20),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 12-Emotion Qwen3-TTS dataset")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Root directory of the dataset (containing metadata/, audio/, glm4voice_va_format/)",
    )
    parser.add_argument(
        "--sample-audio-count",
        type=int,
        default=10,
        help="Number of audio files to sample per emotion/type (default: 10)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    return parser.parse_args()


class DatasetValidator:
    def __init__(self, dataset_dir: Path, sample_audio_count: int, random_seed: int):
        self.dataset_dir = dataset_dir
        self.sample_audio_count = sample_audio_count
        self.random_seed = random_seed
        random.seed(random_seed)

        self.metadata_dir = dataset_dir / "metadata"
        self.audio_dir = dataset_dir / "audio"
        self.glm4voice_dir = dataset_dir / "glm4voice_va_format"
        self.logs_dir = dataset_dir / "logs"

        self.validation_results = {
            "file_existence": {},
            "data_integrity": {},
            "audio_quality": {},
            "format_validation": {},
            "errors": [],
            "warnings": [],
        }

    def validate_file_existence(self):
        """Check if all required files exist."""
        print("\n" + "=" * 60)
        print("1. FILE EXISTENCE CHECK")
        print("=" * 60)

        results = {}

        # Check metadata files
        print("\nChecking metadata files...")
        for split in ["train", "eval"]:
            metadata_file = self.metadata_dir / f"sampled_{split}.jsonl"
            exists = metadata_file.exists()
            results[f"metadata_{split}"] = exists
            status = "✓" if exists else "✗"
            print(f"  {status} {metadata_file}")
            if not exists:
                self.validation_results["errors"].append(
                    f"Missing metadata file: {metadata_file}"
                )

        stats_file = self.metadata_dir / "sampling_stats.json"
        exists = stats_file.exists()
        results["sampling_stats"] = exists
        status = "✓" if exists else "✗"
        print(f"  {status} {stats_file}")

        # Check GLM4Voice JSONL files (VA format)
        print("\nChecking GLM4Voice VA format JSONL files...")
        for split in ["train", "eval"]:
            jsonl_file = self.glm4voice_dir / f"{split}.jsonl"
            exists = jsonl_file.exists()
            results[f"glm4voice_{split}"] = exists
            status = "✓" if exists else "✗"
            print(f"  {status} {jsonl_file}")
            if not exists:
                self.validation_results["errors"].append(
                    f"Missing GLM4Voice file: {jsonl_file}"
                )

        # Check audio directories and count files (12 emotions)
        print("\nChecking audio files...")
        audio_types = ["query", "response"]

        for split in ["train", "eval"]:
            for emotion in ALL_EMOTIONS:
                for audio_type in audio_types:
                    dir_path = self.audio_dir / split / f"{emotion}_{audio_type}"
                    if dir_path.exists():
                        wav_files = list(dir_path.glob("*.wav"))
                        count = len(wav_files)
                        results[f"audio_{split}_{emotion}_{audio_type}"] = count
                        print(f"  ✓ {dir_path}: {count} files")
                    else:
                        results[f"audio_{split}_{emotion}_{audio_type}"] = 0
                        print(f"  ✗ {dir_path}: directory not found")
                        self.validation_results["errors"].append(
                            f"Missing audio directory: {dir_path}"
                        )

        self.validation_results["file_existence"] = results

    def validate_data_integrity(self):
        """Validate JSONL format and data integrity."""
        print("\n" + "=" * 60)
        print("2. DATA INTEGRITY CHECK")
        print("=" * 60)

        results = {}

        for split in ["train", "eval"]:
            print(f"\nValidating {split} split...")

            # Load and validate JSONL
            jsonl_file = self.glm4voice_dir / f"{split}.jsonl"
            if not jsonl_file.exists():
                continue

            records = []
            ids = []
            line_errors = []
            va_errors = []

            with jsonl_file.open("r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        record = json.loads(line.strip())
                        records.append(record)

                        # Check required fields
                        if "text" not in record:
                            line_errors.append(
                                f"Line {line_num}: missing 'text' field"
                            )
                        if "id" not in record:
                            line_errors.append(f"Line {line_num}: missing 'id' field")
                        else:
                            ids.append(record["id"])

                        # Check VA fields
                        if "valence" not in record:
                            va_errors.append(f"Line {line_num}: missing 'valence' field")
                        if "arousal" not in record:
                            va_errors.append(f"Line {line_num}: missing 'arousal' field")

                    except json.JSONDecodeError as e:
                        line_errors.append(f"Line {line_num}: JSON decode error - {e}")

            # Check for duplicate IDs
            id_counts = Counter(ids)
            duplicates = [id_ for id_, count in id_counts.items() if count > 1]

            if duplicates:
                self.validation_results["errors"].append(
                    f"{split}: Found {len(duplicates)} duplicate IDs"
                )
                print(f"  ✗ Found {len(duplicates)} duplicate IDs")
            else:
                print(f"  ✓ No duplicate IDs")

            # Extract emotion distribution from IDs
            emotion_counts = Counter()
            for id_ in ids:
                # ID format: opens2s_{emotion}_{index}
                parts = id_.split("_")
                if len(parts) >= 2:
                    emotion = parts[1]
                    emotion_counts[emotion] += 1

            # Validate format (VA format with system/user/assistant markers)
            format_valid = 0
            format_invalid = 0
            va_format_valid = 0
            for record in records:
                text = record.get("text", "")
                if (
                    "<|system|>" in text
                    and "<|user|>" in text
                    and "<|assistant|>" in text
                ):
                    format_valid += 1
                    # Check for VA format in system prompt
                    if "valence=" in text and "arousal=" in text:
                        va_format_valid += 1
                else:
                    format_invalid += 1

            results[f"{split}_total_records"] = len(records)
            results[f"{split}_line_errors"] = len(line_errors)
            results[f"{split}_va_errors"] = len(va_errors)
            results[f"{split}_duplicate_ids"] = len(duplicates)
            results[f"{split}_emotion_distribution"] = dict(emotion_counts)
            results[f"{split}_format_valid"] = format_valid
            results[f"{split}_format_invalid"] = format_invalid
            results[f"{split}_va_format_valid"] = va_format_valid

            print(f"  Total records: {len(records)}")
            print(f"  Format valid: {format_valid}")
            print(f"  VA format valid: {va_format_valid}")
            if format_invalid > 0:
                print(f"  Format invalid: {format_invalid}")
                self.validation_results["warnings"].append(
                    f"{split}: {format_invalid} records with invalid format"
                )

            print(f"  Emotion distribution:")
            for emotion in ALL_EMOTIONS:
                count = emotion_counts.get(emotion, 0)
                if count > 0:
                    va = EXPECTED_VA_VALUES.get(emotion, (None, None))
                    print(f"    {emotion}: {count} (expected V={va[0]}, A={va[1]})")

            if line_errors:
                print(f"  Line errors: {len(line_errors)}")
                for error in line_errors[:5]:  # Show first 5 errors
                    print(f"    {error}")
                if len(line_errors) > 5:
                    print(f"    ... and {len(line_errors) - 5} more errors")

            if va_errors:
                print(f"  VA field errors: {len(va_errors)}")
                for error in va_errors[:5]:
                    print(f"    {error}")
                if len(va_errors) > 5:
                    print(f"    ... and {len(va_errors) - 5} more errors")

        self.validation_results["data_integrity"] = results

    def validate_audio_quality(self):
        """Sample and check audio quality."""
        if not AUDIO_LIBS_AVAILABLE:
            print("\n" + "=" * 60)
            print("3. AUDIO QUALITY CHECK")
            print("=" * 60)
            print("Skipped: soundfile/numpy not available")
            return

        print("\n" + "=" * 60)
        print("3. AUDIO QUALITY CHECK")
        print("=" * 60)

        results = {}
        audio_types = ["query", "response"]

        for split in ["train", "eval"]:
            print(f"\nSampling {split} audio files...")

            for emotion in ALL_EMOTIONS:
                for audio_type in audio_types:
                    dir_path = self.audio_dir / split / f"{emotion}_{audio_type}"

                    if not dir_path.exists():
                        continue

                    wav_files = list(dir_path.glob("*.wav"))
                    if not wav_files:
                        continue

                    # Sample files
                    sample_count = min(self.sample_audio_count, len(wav_files))
                    sampled_files = random.sample(wav_files, sample_count)

                    durations = []
                    sample_rates = []
                    errors = []

                    for wav_file in sampled_files:
                        try:
                            data, sr = sf.read(str(wav_file))
                            duration = len(data) / sr

                            # Check for issues
                            if np.any(np.isnan(data)):
                                errors.append(f"{wav_file.name}: contains NaN values")
                            if np.any(np.isinf(data)):
                                errors.append(f"{wav_file.name}: contains inf values")
                            if duration < 0.1:
                                errors.append(
                                    f"{wav_file.name}: too short ({duration:.2f}s)"
                                )

                            durations.append(duration)
                            sample_rates.append(sr)

                        except Exception as e:
                            errors.append(f"{wav_file.name}: {str(e)}")

                    if durations:
                        key = f"{split}_{emotion}_{audio_type}"
                        results[key] = {
                            "sampled_count": len(sampled_files),
                            "duration_min": min(durations),
                            "duration_max": max(durations),
                            "duration_mean": sum(durations) / len(durations),
                            "sample_rates": list(set(sample_rates)),
                            "errors": errors,
                        }

                        print(f"\n  {emotion} {audio_type}:")
                        print(f"    Sampled: {len(sampled_files)} files")
                        print(f"    Duration: {min(durations):.2f}s - {max(durations):.2f}s (mean: {sum(durations)/len(durations):.2f}s)")
                        print(f"    Sample rates: {set(sample_rates)}")

                        if errors:
                            print(f"    Errors: {len(errors)}")
                            for error in errors[:3]:
                                print(f"      {error}")
                            if len(errors) > 3:
                                print(f"      ... and {len(errors) - 3} more")

        self.validation_results["audio_quality"] = results

    def validate_format(self):
        """Validate GLM-4-Voice VA format specifics."""
        print("\n" + "=" * 60)
        print("4. VA FORMAT VALIDATION")
        print("=" * 60)

        results = {}

        for split in ["train", "eval"]:
            jsonl_file = self.glm4voice_dir / f"{split}.jsonl"
            if not jsonl_file.exists():
                continue

            print(f"\nValidating {split} VA format...")

            va_values = {"valence": [], "arousal": []}
            audio_token_counts = {"query": [], "response": []}
            emotion_va_mapping = {}

            with jsonl_file.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        text = record.get("text", "")

                        # Get VA values from record
                        valence = record.get("valence")
                        arousal = record.get("arousal")

                        if valence is not None:
                            va_values["valence"].append(valence)
                        if arousal is not None:
                            va_values["arousal"].append(arousal)

                        # Extract emotion from ID and map to VA
                        record_id = record.get("id", "")
                        parts = record_id.split("_")
                        if len(parts) >= 2:
                            emotion = parts[1]
                            if emotion not in emotion_va_mapping:
                                emotion_va_mapping[emotion] = {"valence": [], "arousal": []}
                            if valence is not None:
                                emotion_va_mapping[emotion]["valence"].append(valence)
                            if arousal is not None:
                                emotion_va_mapping[emotion]["arousal"].append(arousal)

                        # Count audio tokens
                        if "<|assistant|>" in text:
                            query_tokens = text.split("<|assistant|>")[0].count("<|audio_")
                            response_tokens = text.split("<|assistant|>")[1].count("<|audio_")
                            audio_token_counts["query"].append(query_tokens)
                            audio_token_counts["response"].append(response_tokens)

                    except Exception as e:
                        continue

            # Validate VA ranges
            if va_values["valence"]:
                valence_min = min(va_values["valence"])
                valence_max = max(va_values["valence"])
                results[f"{split}_valence_range"] = [valence_min, valence_max]
                print(f"  Valence range: [{valence_min:.2f}, {valence_max:.2f}]")

                if valence_min < -1 or valence_max > 1:
                    self.validation_results["warnings"].append(
                        f"{split}: Valence values outside [-1, 1] range"
                    )

            if va_values["arousal"]:
                arousal_min = min(va_values["arousal"])
                arousal_max = max(va_values["arousal"])
                results[f"{split}_arousal_range"] = [arousal_min, arousal_max]
                print(f"  Arousal range: [{arousal_min:.2f}, {arousal_max:.2f}]")

                if arousal_min < -1 or arousal_max > 1:
                    self.validation_results["warnings"].append(
                        f"{split}: Arousal values outside [-1, 1] range"
                    )

            # Validate per-emotion VA values
            print(f"\n  Emotion-wise VA validation:")
            for emotion, va_data in sorted(emotion_va_mapping.items()):
                if va_data["valence"] and va_data["arousal"]:
                    avg_v = sum(va_data["valence"]) / len(va_data["valence"])
                    avg_a = sum(va_data["arousal"]) / len(va_data["arousal"])
                    expected = EXPECTED_VA_VALUES.get(emotion, (None, None))

                    if expected[0] is not None:
                        v_match = "✓" if abs(avg_v - expected[0]) < 0.01 else "✗"
                        a_match = "✓" if abs(avg_a - expected[1]) < 0.01 else "✗"
                        print(f"    {emotion}: V={avg_v:.2f} (exp={expected[0]:.2f}) {v_match}, A={avg_a:.2f} (exp={expected[1]:.2f}) {a_match}")
                    else:
                        print(f"    {emotion}: V={avg_v:.2f}, A={avg_a:.2f} (no expected values)")

            results[f"{split}_emotion_va_mapping"] = {
                e: {"avg_valence": sum(d["valence"])/len(d["valence"]) if d["valence"] else None,
                    "avg_arousal": sum(d["arousal"])/len(d["arousal"]) if d["arousal"] else None}
                for e, d in emotion_va_mapping.items()
            }

            if audio_token_counts["query"]:
                results[f"{split}_query_tokens"] = {
                    "min": min(audio_token_counts["query"]),
                    "max": max(audio_token_counts["query"]),
                    "mean": sum(audio_token_counts["query"])
                    / len(audio_token_counts["query"]),
                }

            if audio_token_counts["response"]:
                results[f"{split}_response_tokens"] = {
                    "min": min(audio_token_counts["response"]),
                    "max": max(audio_token_counts["response"]),
                    "mean": sum(audio_token_counts["response"])
                    / len(audio_token_counts["response"]),
                }

            if audio_token_counts["query"]:
                print(f"\n  Query audio tokens:")
                print(
                    f"    min={min(audio_token_counts['query'])}, "
                    f"max={max(audio_token_counts['query'])}, "
                    f"mean={sum(audio_token_counts['query'])/len(audio_token_counts['query']):.1f}"
                )

            if audio_token_counts["response"]:
                print(f"  Response audio tokens:")
                print(
                    f"    min={min(audio_token_counts['response'])}, "
                    f"max={max(audio_token_counts['response'])}, "
                    f"mean={sum(audio_token_counts['response'])/len(audio_token_counts['response']):.1f}"
                )

        self.validation_results["format_validation"] = results

    def save_report(self):
        """Save validation report."""
        report_path = self.logs_dir / "validation_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with report_path.open("w") as f:
            json.dump(self.validation_results, f, indent=2)

        print(f"\n\nValidation report saved to: {report_path}")

    def print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        errors = self.validation_results["errors"]
        warnings = self.validation_results["warnings"]

        if not errors and not warnings:
            print("✓ All validations passed!")
        else:
            if errors:
                print(f"\n✗ Errors: {len(errors)}")
                for error in errors:
                    print(f"  - {error}")

            if warnings:
                print(f"\n⚠ Warnings: {len(warnings)}")
                for warning in warnings:
                    print(f"  - {warning}")

        print("=" * 60)

    def run_all_validations(self):
        """Run all validation checks."""
        self.validate_file_existence()
        self.validate_data_integrity()
        self.validate_audio_quality()
        self.validate_format()
        self.save_report()
        self.print_summary()


def main():
    args = parse_args()

    if not args.dataset_dir.exists():
        print(f"Error: Dataset directory not found: {args.dataset_dir}", file=sys.stderr)
        sys.exit(1)

    validator = DatasetValidator(
        dataset_dir=args.dataset_dir,
        sample_audio_count=args.sample_audio_count,
        random_seed=args.random_seed,
    )

    validator.run_all_validations()


if __name__ == "__main__":
    main()
