#!/usr/bin/env python3
"""CLI tool for predicting valence-arousal from audio files.

Usage:
    # Single file
    python -m audio_emotion.predict --audio path/to/file.wav

    # Batch (directory)
    python -m audio_emotion.predict --audio-dir path/to/wavs/ --output results.jsonl

    # Use CPU
    python -m audio_emotion.predict --audio path/to/file.wav --device cpu
"""

import argparse
import json
import sys
from pathlib import Path

from .models import AudeeringVAPredictor
from .config import EMOTION_VA_MAPPING

import numpy as np


def find_nearest_emotion(valence: float, arousal: float) -> tuple:
    """Find the nearest emotion anchor to a (valence, arousal) point."""
    best_emotion = None
    best_dist = float("inf")
    for emotion, (v, a) in EMOTION_VA_MAPPING.items():
        dist = np.sqrt((valence - v) ** 2 + (arousal - a) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_emotion = emotion
    return best_emotion, best_dist


def main():
    parser = argparse.ArgumentParser(description="Predict valence-arousal from audio")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio", type=str, help="Path to a single audio file")
    group.add_argument("--audio-dir", type=str, help="Directory of audio files to process")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path (for batch mode)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()

    print("Loading model...")
    predictor = AudeeringVAPredictor(device=args.device)
    print("Model loaded.")

    if args.audio:
        # Single file mode
        result = predictor.predict(args.audio)
        nearest, dist = find_nearest_emotion(result["valence"], result["arousal"])
        print(f"\nFile: {args.audio}")
        print(f"  Valence:   {result['valence']:+.3f}  (raw: {result['raw_valence']:.3f})")
        print(f"  Arousal:   {result['arousal']:+.3f}  (raw: {result['raw_arousal']:.3f})")
        print(f"  Dominance: {result['dominance']:+.3f}  (raw: {result['raw_dominance']:.3f})")
        print(f"  Nearest emotion: {nearest} (distance: {dist:.3f})")
    else:
        # Batch mode
        audio_dir = Path(args.audio_dir)
        audio_files = sorted(
            p for p in audio_dir.rglob("*") if p.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg")
        )
        if not audio_files:
            print(f"No audio files found in {audio_dir}", file=sys.stderr)
            sys.exit(1)

        print(f"Found {len(audio_files)} audio files")
        results = []
        for i, audio_path in enumerate(audio_files):
            result = predictor.predict(str(audio_path))
            nearest, dist = find_nearest_emotion(result["valence"], result["arousal"])
            result["file"] = str(audio_path)
            result["nearest_emotion"] = nearest
            result["nearest_distance"] = dist
            results.append(result)
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(audio_files)}")

        print(f"Processed {len(results)} files")

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            print(f"Results saved to {output_path}")
        else:
            # Print summary to stdout
            for r in results:
                nearest = r["nearest_emotion"]
                print(f"{r['file']}: V={r['valence']:+.3f} A={r['arousal']:+.3f} -> {nearest}")


if __name__ == "__main__":
    main()
