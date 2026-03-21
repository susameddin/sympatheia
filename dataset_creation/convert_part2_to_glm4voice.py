#!/usr/bin/env python3
"""
Convert Part 2 v2 audio to GLM-4-Voice format with Valence-Arousal values.

All queries are neutral. VA values come from the RESPONSE emotion.
This forces the model to rely on the VA label rather than query audio emotion.

Audio path layout expected:
  audio/{split}/query/neutral/{query_index}.wav
  audio/{split}/response/{response_emotion.lower()}/{pair_index}.wav

Output JSONL record:
  {
    "text": "<|system|>\\nPlease respond in English. User emotion (valence=V, arousal=A)\\n
             <|user|>\\n{query_tokens}\\n
             <|assistant|>\\n{response_text}\\n{response_tokens}<|user|>",
    "id": "p2_Sad_00001_Happy",
    "valence": 0.85,      ← from Happy (response emotion)
    "arousal": 0.35
  }

Run:
  conda run -n qwen3-tts4 python dataset_creation/convert_part2_to_glm4voice.py \\
      --metadata-dir /engram/naplab/users/sd3705/Datasets/Sympatheia_12Emo_Neutral/metadata/ \\
      --audio-dir    /engram/naplab/users/sd3705/Datasets/Sympatheia_12Emo_Neutral/audio/ \\
      --output-dir   /engram/naplab/users/sd3705/Datasets/Sympatheia_12Emo_Neutral/
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.vocoder import GLM4CodecEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Emotion → Valence/Arousal mapping (same as Part 1)
# ─────────────────────────────────────────────────────────────────────────────
EMOTION_VA_MAPPING: Dict[str, Tuple[float, float]] = {
    "Sad":        (-0.75, -0.65),
    "Excited":    ( 0.75,  0.90),
    "Frustrated": (-0.80,  0.35),
    "Neutral":    ( 0.00,  0.00),
    "Happy":      ( 0.85,  0.35),
    "Angry":      (-0.85,  0.85),
    "Anxious":    (-0.40,  0.65),
    "Relaxed":    ( 0.25, -0.60),
    "Surprised":  ( 0.10,  0.80),
    "Disgusted":  (-0.82, -0.20),
    "Tired":      (-0.15, -0.75),
    "Content":    ( 0.60, -0.20),
}

ALL_EMOTIONS = list(EMOTION_VA_MAPPING.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Part 2 audio to GLM-4-Voice format (VA from response emotion)"
    )
    parser.add_argument("--metadata-dir", type=Path, required=True,
                        help="Directory with sampled_{split}.jsonl files")
    parser.add_argument("--audio-dir", type=Path, required=True,
                        help="Root audio directory")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for train.jsonl / eval.jsonl")
    parser.add_argument("--log-every", type=int, default=100,
                        help="Log progress every N samples (default: 100)")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> List[Dict]:
    samples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


def encode_audio(audio_path: Path, encoder: GLM4CodecEncoder) -> str:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    tokens = encoder([str(audio_path)])[0]
    return "".join(f"<|audio_{t}|>" for t in tokens)


def build_glm4voice_text(
    query_tokens: str,
    response_text: str,
    response_tokens: str,
    valence: float,
    arousal: float,
) -> str:
    return (
        "<|system|>\n"
        f"Please respond in English. User emotion (valence={valence:.2f}, arousal={arousal:.2f})\n"
        "<|user|>\n"
        f"{query_tokens}\n"
        "<|assistant|>\n"
        f"{response_text}\n"
        f"{response_tokens}<|user|>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Process one split
# ─────────────────────────────────────────────────────────────────────────────
def process_split(
    samples: List[Dict],
    split: str,
    audio_dir: Path,
    output_dir: Path,
    encoder: GLM4CodecEncoder,
    log_every: int,
):
    print(f"\n{'=' * 60}")
    print(f"Processing {split} split ({len(samples)} pairs)")
    print(f"{'=' * 60}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}.jsonl"

    token_stats = {"query_tokens": [], "response_tokens": []}
    failed = []
    processed = 0
    emotion_counts = defaultdict(int)

    with output_path.open("w", encoding="utf-8") as out_f:
        for sample in tqdm(samples, desc=f"Encoding {split}"):
            try:
                query_emotion = sample["query_emotion"]
                resp_emotion = sample["response_emotion"]
                query_index = sample["query_index"]   # e.g. p2_Sad_00001
                pair_index = sample["index"]           # e.g. p2_Sad_00001_Happy

                if resp_emotion not in EMOTION_VA_MAPPING:
                    print(f"Warning: Unknown response emotion '{resp_emotion}', skipping", file=sys.stderr)
                    failed.append({"index": pair_index, "reason": f"unknown_emotion_{resp_emotion}"})
                    continue

                # VA values from RESPONSE emotion — the key override signal
                valence, arousal = EMOTION_VA_MAPPING[resp_emotion]

                # Audio paths — all queries are neutral
                query_audio_path = (
                    audio_dir / split / "query" / "neutral" / f"{query_index}.wav"
                )
                response_audio_path = (
                    audio_dir / split / "response" / resp_emotion.lower() / f"{pair_index}.wav"
                )

                if not query_audio_path.exists():
                    print(f"Warning: Query audio missing: {query_audio_path}", file=sys.stderr)
                    failed.append({"index": pair_index, "reason": "query_audio_missing"})
                    continue

                if not response_audio_path.exists():
                    print(f"Warning: Response audio missing: {response_audio_path}", file=sys.stderr)
                    failed.append({"index": pair_index, "reason": "response_audio_missing"})
                    continue

                query_tokens = encode_audio(query_audio_path, encoder)
                response_tokens = encode_audio(response_audio_path, encoder)

                token_stats["query_tokens"].append(query_tokens.count("<|audio_"))
                token_stats["response_tokens"].append(response_tokens.count("<|audio_"))

                text = build_glm4voice_text(
                    query_tokens=query_tokens,
                    response_text=sample["response_text"],
                    response_tokens=response_tokens,
                    valence=valence,
                    arousal=arousal,
                )

                record = {
                    "text": text,
                    "id": pair_index,
                    "valence": valence,
                    "arousal": arousal,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed += 1
                emotion_counts[resp_emotion] += 1

                if log_every and processed % log_every == 0:
                    print(f"  Processed {processed}/{len(samples)}...", flush=True)

            except Exception as e:
                idx = sample.get("index", "unknown")
                print(f"Error processing {idx}: {e}", file=sys.stderr)
                failed.append({"index": idx, "reason": str(e)})

    print(f"\nWrote {processed} samples → {output_path}")
    if failed:
        print(f"Failed: {len(failed)}")

    # Print distribution by response emotion
    print(f"\nResponse-emotion distribution ({split}):")
    for emo in ALL_EMOTIONS:
        count = emotion_counts.get(emo, 0)
        if count:
            v, a = EMOTION_VA_MAPPING[emo]
            print(f"  {emo:<12}: {count:>5}  (V={v:.2f}, A={a:.2f})")

    if token_stats["query_tokens"]:
        q = token_stats["query_tokens"]
        r = token_stats["response_tokens"]
        print(f"\nToken stats ({split}):")
        print(f"  Query:    min={min(q)}, max={max(q)}, mean={sum(q)/len(q):.1f}")
        print(f"  Response: min={min(r)}, max={max(r)}, mean={sum(r)/len(r):.1f}")

    return token_stats, failed, dict(emotion_counts)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print("Initializing GLM4CodecEncoder...")
    encoder = GLM4CodecEncoder()
    print("Encoder ready")

    all_stats = {}
    for split in ["train", "eval"]:
        meta_path = args.metadata_dir / f"sampled_{split}.jsonl"
        if not meta_path.exists():
            print(f"Metadata not found: {meta_path} — skipping {split}")
            continue
        samples = load_jsonl(meta_path)
        stats, failed, emo_dist = process_split(
            samples=samples,
            split=split,
            audio_dir=args.audio_dir,
            output_dir=args.output_dir,
            encoder=encoder,
            log_every=args.log_every,
        )
        all_stats[split] = {
            "processed": len(stats["query_tokens"]),
            "failed": len(failed),
            "response_emotion_distribution": emo_dist,
            "query_tokens": {
                "min": min(stats["query_tokens"]) if stats["query_tokens"] else 0,
                "max": max(stats["query_tokens"]) if stats["query_tokens"] else 0,
                "mean": sum(stats["query_tokens"]) / len(stats["query_tokens"]) if stats["query_tokens"] else 0,
            },
            "response_tokens": {
                "min": min(stats["response_tokens"]) if stats["response_tokens"] else 0,
                "max": max(stats["response_tokens"]) if stats["response_tokens"] else 0,
                "mean": sum(stats["response_tokens"]) / len(stats["response_tokens"]) if stats["response_tokens"] else 0,
            },
        }

    # Save stats
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.output_dir / "encoding_stats_part2v2.json"
    with stats_path.open("w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nStats saved → {stats_path}")

    config_path = args.output_dir / "conversion_config_part2v2.json"
    with config_path.open("w") as f:
        json.dump(
            {
                "va_source": "response_emotion",
                "emotion_to_va_mapping": {k: list(v) for k, v in EMOTION_VA_MAPPING.items()},
            },
            f,
            indent=2,
        )
    print(f"Config saved → {config_path}")

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"  Train: {args.output_dir / 'train.jsonl'}")
    print(f"  Eval:  {args.output_dir / 'eval.jsonl'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
