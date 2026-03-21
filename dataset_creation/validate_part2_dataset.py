#!/usr/bin/env python3
"""
Validate the Part 2 v2 Sympatheia dataset (neutral-only queries).

Checks:
  1. Metadata files exist and are well-formed
  2. All queries are neutral
  3. Unique query WAV files exist (500 expected)
  4. Response pair WAV files exist (5,500 expected)
  5. Final JSONL records have VA values matching response_emotion
  6. Audio quality spot-check (duration, sample rate, no NaN/inf)
  7. Per-emotion distribution table
  8. Content differentiation spot-check

Run:
  python dataset_creation/validate_part2_dataset.py \
      --dataset-dir /engram/naplab/users/sd3705/Datasets/Sympatheia_12Emo_Neutral/
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf


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
    parser = argparse.ArgumentParser(description="Validate Part 2 v2 dataset")
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="Root of the Part 2 v2 dataset directory")
    parser.add_argument("--audio-spot-check", type=int, default=5,
                        help="Number of audio files to spot-check per emotion (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    return parser.parse_args()


def load_jsonl(path: Path) -> Tuple[List[Dict], List[str]]:
    records = []
    errors = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors.append(f"  Line {lineno}: {e}")
    return records, errors


def check_audio_file(path: Path) -> Dict:
    result = {"path": str(path), "ok": False, "issues": []}
    try:
        data, sr = sf.read(str(path))
        result["duration_s"] = len(data) / sr
        result["sample_rate"] = sr
        if np.any(np.isnan(data)):
            result["issues"].append("NaN values")
        if np.any(np.isinf(data)):
            result["issues"].append("Inf values")
        if result["duration_s"] < 0.1:
            result["issues"].append(f"Too short ({result['duration_s']:.3f}s)")
        if result["duration_s"] > 30.0:
            result["issues"].append(f"Too long ({result['duration_s']:.1f}s)")
        result["ok"] = len(result["issues"]) == 0
    except Exception as e:
        result["issues"].append(str(e))
    return result


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    dataset_dir = args.dataset_dir
    audio_dir = dataset_dir / "audio"
    metadata_dir = dataset_dir / "metadata"

    report = {"checks": [], "warnings": [], "errors": []}
    all_ok = True

    # ── 1. Metadata files ────────────────────────────────────────────────────
    section("1. Metadata files")
    meta_files = {
        "sampled_train.jsonl":        metadata_dir / "sampled_train.jsonl",
        "sampled_eval.jsonl":         metadata_dir / "sampled_eval.jsonl",
        "unique_queries_train.jsonl": metadata_dir / "unique_queries_train.jsonl",
        "unique_queries_eval.jsonl":  metadata_dir / "unique_queries_eval.jsonl",
    }
    meta_data = {}
    for name, path in meta_files.items():
        if not path.exists():
            print(f"  [MISSING] {name}")
            report["errors"].append(f"Missing metadata file: {name}")
            all_ok = False
        else:
            records, errors = load_jsonl(path)
            if errors:
                for e in errors:
                    print(f"  [JSON ERROR] {name}: {e}")
                report["errors"].extend(errors)
                all_ok = False
            meta_data[name] = records
            print(f"  [OK] {name}: {len(records)} records")

    # ── 2. All queries must be neutral ────────────────────────────────────────
    section("2. Query emotion check (all must be Neutral)")

    for split in ["train", "eval"]:
        key = f"sampled_{split}.jsonl"
        if key not in meta_data:
            continue
        pairs = meta_data[key]
        non_neutral = [p for p in pairs if p.get("query_emotion") != "Neutral"]
        if non_neutral:
            print(f"  [ERROR] {split}: {len(non_neutral)} pairs have non-neutral query_emotion!")
            for p in non_neutral[:5]:
                print(f"    {p['index']}: query_emotion={p['query_emotion']}")
            all_ok = False
        else:
            print(f"  [OK] {split}: all {len(pairs)} pairs have query_emotion=Neutral")

    # ── 3. Metadata content checks ───────────────────────────────────────────
    section("3. Metadata content")

    for split in ["train", "eval"]:
        key = f"sampled_{split}.jsonl"
        if key not in meta_data:
            continue
        pairs = meta_data[key]
        required = {"index", "query_index", "query_text", "query_emotion",
                    "response_emotion", "response_text"}
        missing_field_count = 0
        invalid_emo_count = 0
        dup_ids = set()
        seen_ids = set()
        for p in pairs:
            missing = required - set(p.keys())
            if missing:
                missing_field_count += 1
            re_ = p.get("response_emotion", "")
            if re_ not in ALL_EMOTIONS:
                invalid_emo_count += 1
            pid = p.get("index", "")
            if pid in seen_ids:
                dup_ids.add(pid)
            seen_ids.add(pid)

        status = "OK" if not (missing_field_count or invalid_emo_count or dup_ids) else "ISSUES"
        print(f"  [{status}] {split} pairs: {len(pairs)} records, "
              f"{missing_field_count} missing fields, "
              f"{invalid_emo_count} invalid emotions, "
              f"{len(dup_ids)} duplicate IDs")
        if missing_field_count or invalid_emo_count or dup_ids:
            all_ok = False

    # Distribution table
    print("\n  Per-emotion pair counts (train):")
    train_pairs = meta_data.get("sampled_train.jsonl", [])
    r_emo_count: Dict[str, int] = defaultdict(int)
    for p in train_pairs:
        r_emo_count[p.get("response_emotion", "?")] += 1
    print(f"  {'Emotion':<14} {'Response pairs':<14}")
    print(f"  {'-'*14} {'-'*14}")
    for emo in ALL_EMOTIONS:
        print(f"  {emo:<14} {r_emo_count.get(emo, 0):<14}")

    # ── 4. Audio file existence ──────────────────────────────────────────────
    section("4. Audio file existence")

    for split in ["train", "eval"]:
        # Query audio — all under neutral/
        uq_key = f"unique_queries_{split}.jsonl"
        if uq_key in meta_data:
            uq_list = meta_data[uq_key]
            missing_query = 0
            for q in uq_list:
                qi = q.get("query_index", "")
                path = audio_dir / split / "query" / "neutral" / f"{qi}.wav"
                if not path.exists():
                    missing_query += 1
            status = "OK" if missing_query == 0 else "MISSING"
            print(f"  [{status}] {split} query audio: {len(uq_list)} expected, "
                  f"{missing_query} missing")
            if missing_query:
                all_ok = False

        # Response audio
        pairs_key = f"sampled_{split}.jsonl"
        if pairs_key in meta_data:
            pairs = meta_data[pairs_key]
            missing_resp = 0
            for p in pairs:
                re_ = p.get("response_emotion", "")
                pi = p.get("index", "")
                path = audio_dir / split / "response" / re_.lower() / f"{pi}.wav"
                if not path.exists():
                    missing_resp += 1
            status = "OK" if missing_resp == 0 else "MISSING"
            print(f"  [{status}] {split} response audio: {len(pairs)} expected, "
                  f"{missing_resp} missing")
            if missing_resp:
                all_ok = False

    # ── 5. Audio quality spot-check ──────────────────────────────────────────
    section("5. Audio quality spot-check")

    n = args.audio_spot_check
    audio_issues = []

    for split in ["train", "eval"]:
        # Query audio — only neutral directory
        query_dir = audio_dir / split / "query" / "neutral"
        if query_dir.exists():
            wavs = list(query_dir.glob("*.wav"))
            sample = rng.sample(wavs, min(n, len(wavs)))
            for wav in sample:
                r = check_audio_file(wav)
                if not r["ok"]:
                    audio_issues.append(r)

        # Response audio per response emotion
        for emo in ALL_EMOTIONS:
            resp_dir = audio_dir / split / "response" / emo.lower()
            if not resp_dir.exists():
                continue
            wavs = list(resp_dir.glob("*.wav"))
            sample = rng.sample(wavs, min(n, len(wavs)))
            for wav in sample:
                r = check_audio_file(wav)
                if not r["ok"]:
                    audio_issues.append(r)

    if audio_issues:
        print(f"  [ISSUES] {len(audio_issues)} audio files with problems:")
        for r in audio_issues[:20]:
            print(f"    {r['path']}: {', '.join(r['issues'])}")
        all_ok = False
    else:
        print(f"  [OK] All spot-checked audio files passed")

    # ── 6. Final JSONL VA validation ─────────────────────────────────────────
    section("6. Final JSONL VA value validation")

    va_tol = 1e-3
    for split in ["train", "eval"]:
        jsonl_path = dataset_dir / f"{split}.jsonl"
        if not jsonl_path.exists():
            print(f"  [SKIP] {split}.jsonl not found (run conversion first)")
            continue
        records, errors = load_jsonl(jsonl_path)
        if errors:
            print(f"  [ERROR] {split}.jsonl has JSON parse errors")
            all_ok = False
            continue

        missing_va = 0
        unrecognized_va = 0
        for r in records:
            v = r.get("valence")
            a = r.get("arousal")
            if v is None or a is None:
                missing_va += 1
                continue
            matched = any(
                abs(v - ev) < va_tol and abs(a - ea) < va_tol
                for ev, ea in EMOTION_VA_MAPPING.values()
            )
            if not matched:
                unrecognized_va += 1

        status = "OK" if not (missing_va or unrecognized_va) else "ISSUES"
        print(f"  [{status}] {split}.jsonl: {len(records)} records, "
              f"{missing_va} missing VA, {unrecognized_va} unrecognized VA values")
        if missing_va or unrecognized_va:
            all_ok = False

        # Cross-check VA against response_emotion from metadata
        pairs_key = f"sampled_{split}.jsonl"
        if pairs_key in meta_data and records:
            pairs_by_id = {p["index"]: p for p in meta_data[pairs_key]}
            sample = rng.sample(records, min(200, len(records)))
            cross_mismatch = 0
            for r in sample:
                pid = r.get("id", "")
                pair = pairs_by_id.get(pid)
                if pair is None:
                    continue
                resp_emo = pair["response_emotion"]
                ev, ea = EMOTION_VA_MAPPING.get(resp_emo, (None, None))
                if ev is None:
                    continue
                if abs(r.get("valence", 999) - ev) > va_tol or abs(r.get("arousal", 999) - ea) > va_tol:
                    cross_mismatch += 1
            if cross_mismatch:
                print(f"  [ERROR] {split}: {cross_mismatch}/200 sampled records "
                      f"have VA not matching their response_emotion")
                all_ok = False
            else:
                print(f"  [OK] {split}: VA cross-check passed (200 samples)")

    # ── 7. Content differentiation spot-check ────────────────────────────────
    section("7. Content differentiation spot-check")

    for split in ["train"]:
        pairs_key = f"sampled_{split}.jsonl"
        if pairs_key not in meta_data:
            print(f"  [SKIP] No {pairs_key} available")
            continue

        pairs = meta_data[pairs_key]
        by_query: Dict[str, Dict[str, str]] = defaultdict(dict)
        query_texts: Dict[str, str] = {}
        for p in pairs:
            qi = p["query_index"]
            by_query[qi][p["response_emotion"]] = p["response_text"]
            query_texts[qi] = p["query_text"]

        # Find queries that have all 11 responses
        full_queries = [qi for qi, resps in by_query.items() if len(resps) == len(ALL_EMOTIONS)]
        if not full_queries:
            print(f"  [WARN] No queries with all 11 response emotions found")
            continue

        # Show 2 sample queries with all their responses
        n_samples = min(2, len(full_queries))
        sample_queries = rng.sample(full_queries, n_samples)

        for qi in sample_queries:
            print(f"\n  Query [{qi}]: \"{query_texts[qi]}\"")
            for emo in ALL_EMOTIONS:
                text = by_query[qi].get(emo, "[missing]")
                # Truncate long responses for display
                display = text[:120] + "..." if len(text) > 120 else text
                print(f"    [{emo:<12}] {display}")

        print(f"\n  Shown {n_samples} sample queries with all 11 response emotions.")
        print(f"  Verify that responses clearly differ by emotion for the same query.")

    # ── Summary ──────────────────────────────────────────────────────────────
    section("Summary")
    if all_ok:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED — review issues above before training")

    # Save report
    report_path = dataset_dir / "validation_report_part2v2.json"
    with report_path.open("w") as f:
        json.dump(
            {
                "all_ok": all_ok,
                "audio_issues_count": len(audio_issues),
                "audio_issues_sample": audio_issues[:10],
            },
            f,
            indent=2,
        )
    print(f"\n  Report saved → {report_path}")


if __name__ == "__main__":
    main()
