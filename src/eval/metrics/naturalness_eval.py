#!/usr/bin/env python3
"""
Naturalness evaluation using UTMOS22 MOS prediction.

Scores every generated audio response in manifest.jsonl with UTMOS22
(Unified Text-to-speech MOS prediction) and reports per-condition and
per-emotion mean ± std.

UTMOS loads via torch.hub (tarepan/SpeechMOS) — no extra install needed.
First run downloads ~392MB model weights to the torch.hub cache.

Usage:
    python -m eval.metrics.naturalness_eval \\
        --manifest eval_emotional_.../manifest.jsonl

Outputs:
    <output-dir>/metrics/naturalness_metrics.json
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import ALL_EMOTIONS


# ---------------------------------------------------------------------------
# UTMOS scoring (reused from evaluate_model.py)
# ---------------------------------------------------------------------------

def load_utmos() -> object | None:
    """Load UTMOS22 predictor via torch.hub. Returns None on failure."""
    print("Loading UTMOS22 via torch.hub (tarepan/SpeechMOS:v1.2.0)...")
    try:
        predictor = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
        )
        predictor.eval()
        if torch.cuda.is_available():
            predictor = predictor.cuda()
        print("  UTMOS loaded.")
        return predictor
    except Exception as e:
        print(f"  UTMOS load failed: {e}")
        return None


def score_utmos(audio_path: str, predictor) -> float | None:
    """Score a WAV file with UTMOS22. Returns MOS ∈ [1,5] or None."""
    if predictor is None:
        return None
    try:
        wav, sr = torchaudio.load(audio_path)
        wav_mono = wav.mean(dim=0, keepdim=True)  # (1, T)
        if sr != 16000:
            wav_mono = torchaudio.functional.resample(wav_mono, sr, 16000)
        if torch.cuda.is_available():
            wav_mono = wav_mono.cuda()
        with torch.no_grad():
            score = predictor(wav_mono, sr=16000)
        return float(score.item())
    except Exception as e:
        print(f"    UTMOS scoring failed for {audio_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(results: list[dict], conditions: list[str]) -> dict:
    """Aggregate UTMOS scores per condition (overall + per emotion)."""
    by_cond: dict[str, list] = defaultdict(list)
    by_cond_emo: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        cond = r["condition"]
        emo  = r["emotion"]
        s    = r["score"]
        if s is not None:
            by_cond[cond].append(s)
            by_cond_emo[cond][emo].append(s)

    summary = {}
    for cond in conditions:
        vals = by_cond[cond]
        per_emo = {
            emo: round(float(np.mean(by_cond_emo[cond][emo])), 4)
            if by_cond_emo[cond][emo] else None
            for emo in ALL_EMOTIONS
        }
        summary[cond] = {
            "mean": round(float(np.mean(vals)), 4) if vals else None,
            "std":  round(float(np.std(vals)), 4)  if vals else None,
            "n":    len(vals),
            "per_emotion": per_emo,
        }
    return summary


def print_summary(summary: dict, conditions: list[str]):
    print(f"\n{'='*60}")
    print("UTMOS22 NATURALNESS SCORES (1–5)")
    print(f"{'='*60}")
    header = f"{'Condition':<20} {'Mean':>6} {'Std':>6} {'N':>5}"
    print(header)
    print("-" * 40)
    for cond in conditions:
        s = summary.get(cond, {})
        mean = f"{s['mean']:.3f}" if s.get("mean") is not None else "N/A"
        std  = f"{s['std']:.3f}"  if s.get("std")  is not None else "N/A"
        n    = str(s.get("n", 0))
        print(f"{cond:<20} {mean:>6} {std:>6} {n:>5}")

    all_emotions = sorted({emo for s in summary.values()
                           for emo in s.get("per_emotion", {})})
    if all_emotions:
        print(f"\n--- Per-emotion means ---")
        hdr = f"{'Emotion':<14}" + "".join(f" {c[:12]:>13}" for c in conditions)
        print(hdr)
        print("-" * (14 + 13 * len(conditions)))
        for emo in all_emotions:
            row = f"{emo:<14}"
            for cond in conditions:
                val = summary.get(cond, {}).get("per_emotion", {}).get(emo)
                row += f" {f'{val:.3f}' if val is not None else 'N/A':>13}"
            print(row)
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="UTMOS22 naturalness evaluation for speech responses"
    )
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to manifest.jsonl")
    parser.add_argument("--conditions", type=str, nargs="+", default=None,
                        help="Conditions to score (default: auto-detect)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: manifest parent dir)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip (sample, condition) pairs already in output JSON")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else manifest_path.parent
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    stem = manifest_path.stem
    suffix = "" if stem == "manifest" else f"_{stem.removeprefix('manifest_')}"
    out_path = metrics_dir / f"naturalness_metrics{suffix}.json"

    # Load manifest
    records = []
    with open(manifest_path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} manifest records")

    # Auto-detect conditions
    if args.conditions is None:
        all_keys: set = set()
        for rec in records:
            all_keys.update(rec.keys())
        conditions = [c for c in
                      ["base", "finetuned_va", "finetuned_na", "opens2s", "osum_no_think",
                       "osum_think", "qwen3omni", "face_va", "no_va", "oracle",
                       "eeg_only", "eye_only", "combined"]
                      if f"{c}_response" in all_keys]
        print(f"Auto-detected conditions: {conditions}")
    else:
        conditions = args.conditions

    # Load existing results for resume
    existing: set = set()  # (id, condition)
    all_results: list = []
    if args.skip_existing and out_path.exists():
        with open(out_path) as f:
            saved = json.load(f)
        for r in saved.get("raw", []):
            existing.add((r["id"], r["condition"]))
            all_results.append(r)
        print(f"Resuming: {len(existing)} already scored")

    # Load UTMOS
    print(f"\nCUDA: {torch.cuda.is_available()}")
    predictor = load_utmos()
    if predictor is None:
        print("ERROR: UTMOS failed to load — cannot continue.")
        sys.exit(1)

    # Score each audio
    total = len(records) * len(conditions)
    done = 0
    t_start = time.time()

    for rec in records:
        sample_id = rec.get("id", "")
        emotion   = rec.get("emotion") or rec.get("emotion_gt", "Unknown")

        for cond in conditions:
            if (sample_id, cond) in existing:
                done += 1
                continue

            audio_path = rec.get(f"{cond}_response")
            if audio_path is None or not Path(audio_path).exists():
                done += 1
                continue

            score = score_utmos(audio_path, predictor)
            all_results.append({
                "id":        sample_id,
                "emotion":   emotion,
                "condition": cond,
                "score":     score,
            })
            done += 1
            if done % 30 == 0 or done == total:
                elapsed = time.time() - t_start
                score_str = f"{score:.3f}" if score is not None else "N/A"
                print(f"  [{done}/{total}] {sample_id}/{cond} → {score_str}  ({elapsed:.0f}s)")

    # Aggregate
    summary = aggregate(all_results, conditions)
    print_summary(summary, conditions)

    # Save
    output = {
        "conditions": conditions,
        "summary": summary,
        "raw": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_path}")
    return summary


if __name__ == "__main__":
    main()
