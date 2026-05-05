#!/usr/bin/env python3
"""
Intelligibility evaluation via ASR-based WER.

Transcribes generated audio responses ({cond}_response in manifest) with
openai-whisper, then computes WER against the model's own text output
({cond}_text from manifest). This measures how faithfully the TTS reproduced
the intended text — a genuine speech intelligibility metric.

Usage:
    conda run -n s --no-capture-output python -m eval.metrics.intelligibility_eval \\
        --manifest /path/to/manifest.jsonl

    # Use a smaller model (faster, less accurate):
    conda run -n s --no-capture-output python -m eval.metrics.intelligibility_eval \\
        --manifest /path/to/manifest.jsonl \\
        --whisper-model medium

Outputs:
    <output-dir>/metrics/intelligibility_metrics.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import jiwer
    _JIWER_OK = True
except ImportError:
    _JIWER_OK = False
    print("WARNING: jiwer not installed — install with: pip install jiwer")

try:
    import whisper as openai_whisper
    _WHISPER_OK = True
except ImportError:
    _WHISPER_OK = False
    print("WARNING: openai-whisper not installed — install with: pip install openai-whisper")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import ALL_EMOTIONS

DEFAULT_WHISPER_MODEL = "large-v3"


# ---------------------------------------------------------------------------
# WER computation
# ---------------------------------------------------------------------------

_WER_TRANSFORM = None


def _get_wer_transform():
    global _WER_TRANSFORM
    if _WER_TRANSFORM is None and _JIWER_OK:
        _WER_TRANSFORM = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.Strip(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.ReduceToListOfListOfWords(),
        ])
    return _WER_TRANSFORM


def compute_wer(reference: str, hypothesis: str) -> float | None:
    if not _JIWER_OK or not reference.strip() or not hypothesis.strip():
        return None
    transform = _get_wer_transform()
    try:
        return float(jiwer.wer(
            reference, hypothesis,
            reference_transform=transform,
            hypothesis_transform=transform,
        ))
    except Exception as e:
        print(f"  WER computation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# ASR transcription with openai-whisper
# ---------------------------------------------------------------------------

def load_whisper(model_name: str):
    """Load openai-whisper model. Returns None if unavailable."""
    if not _WHISPER_OK:
        return None
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading openai-whisper ({model_name}) on {device}...")
    try:
        model = openai_whisper.load_model(model_name, device=device)
        print("  openai-whisper loaded.")
        return model
    except Exception as e:
        print(f"  openai-whisper load failed: {e}")
        return None


def transcribe(audio_path: str, model) -> str | None:
    """Transcribe audio file with openai-whisper. Returns None on failure."""
    try:
        result = model.transcribe(audio_path, beam_size=5)
        return result["text"].strip()
    except Exception as e:
        print(f"  Transcription failed for {audio_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(results: list[dict], conditions: list[str]) -> dict:
    by_cond: dict[str, list] = defaultdict(list)
    by_cond_emo: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        cond = r["condition"]
        emo  = r["emotion"]
        wer  = r.get("wer")
        if wer is not None:
            by_cond[cond].append(wer)
            by_cond_emo[cond][emo].append(wer)

    summary = {}
    for cond in conditions:
        vals = by_cond[cond]
        per_emo = {
            emo: round(float(np.mean(by_cond_emo[cond][emo])), 4)
            if by_cond_emo[cond][emo] else None
            for emo in ALL_EMOTIONS
        }
        summary[cond] = {
            "wer_mean": round(float(np.mean(vals)), 4) if vals else None,
            "wer_std":  round(float(np.std(vals)), 4)  if vals else None,
            "n":        len(vals),
            "per_emotion": per_emo,
        }
    return summary


def print_summary(summary: dict, conditions: list[str], mode: str):
    print(f"\n{'='*60}")
    print(f"INTELLIGIBILITY — WER (lower = better)  [{mode}]")
    print(f"{'='*60}")
    header = f"{'Condition':<20} {'WER Mean':>9} {'Std':>6} {'N':>5}"
    print(header)
    print("-" * 43)
    for cond in conditions:
        s = summary.get(cond, {})
        mean = f"{s['wer_mean']:.4f}" if s.get("wer_mean") is not None else "N/A"
        std  = f"{s['wer_std']:.4f}"  if s.get("wer_std")  is not None else "N/A"
        n    = str(s.get("n", 0))
        print(f"{cond:<20} {mean:>9} {std:>6} {n:>5}")

    all_emotions = sorted({emo for s in summary.values()
                           for emo in s.get("per_emotion", {})})
    if all_emotions:
        print(f"\n--- Per-emotion WER means ---")
        hdr = f"{'Emotion':<14}" + "".join(f" {c[:12]:>13}" for c in conditions)
        print(hdr)
        print("-" * (14 + 13 * len(conditions)))
        for emo in all_emotions:
            row = f"{emo:<14}"
            for cond in conditions:
                val = summary.get(cond, {}).get("per_emotion", {}).get(emo)
                row += f" {f'{val:.4f}' if val is not None else 'N/A':>13}"
            print(row)
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="ASR-based WER intelligibility evaluation (openai-whisper)"
    )
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to manifest.jsonl")
    parser.add_argument("--conditions", type=str, nargs="+", default=None,
                        help="Conditions to evaluate (default: auto-detect)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: manifest parent dir)")
    parser.add_argument("--whisper-model", type=str, default=DEFAULT_WHISPER_MODEL,
                        help=f"openai-whisper model name (default: {DEFAULT_WHISPER_MODEL})")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not _JIWER_OK:
        print("ERROR: jiwer is required. Install with: pip install jiwer")
        sys.exit(1)

    if not _WHISPER_OK:
        print("ERROR: openai-whisper is required. Install with: pip install openai-whisper")
        sys.exit(1)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else manifest_path.parent
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    stem = manifest_path.stem
    suffix = "" if stem == "manifest" else f"_{stem.removeprefix('manifest_')}"
    out_path = metrics_dir / f"intelligibility_metrics{suffix}.json"

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

    # Load ASR model
    whisper_model = load_whisper(args.whisper_model)
    if whisper_model is None:
        print("ERROR: openai-whisper failed to load — cannot continue.")
        sys.exit(1)

    mode = f"ASR (openai-whisper/{args.whisper_model})"
    print(f"\nMode: {mode}")
    print("Reference: {cond}_text from manifest (model's intended output)")
    print("Hypothesis: Whisper transcription of {cond}_response audio")

    # Load existing results for resume
    existing: set = set()
    all_results: list = []
    if args.skip_existing and out_path.exists():
        with open(out_path) as f:
            saved = json.load(f)
        for r in saved.get("raw", []):
            existing.add((r["id"], r["condition"]))
            all_results.append(r)
        print(f"Resuming: {len(existing)} already scored")

    print(f"Computing WER for {len(records)} samples × {len(conditions)} conditions...\n")

    for i, rec in enumerate(records):
        sample_id = rec.get("id", f"sample_{i}")
        emotion   = rec.get("emotion") or rec.get("emotion_gt", "Unknown")

        for cond in conditions:
            if (sample_id, cond) in existing:
                continue

            # Reference: model's own intended text output
            reference = rec.get(f"{cond}_text", "") or ""

            # Hypothesis: Whisper transcription of the audio
            audio_path = rec.get(f"{cond}_response")
            if audio_path and Path(audio_path).exists():
                hypothesis = transcribe(audio_path, whisper_model) or ""
            else:
                hypothesis = ""

            wer_val = compute_wer(reference, hypothesis)

            all_results.append({
                "id":         sample_id,
                "emotion":    emotion,
                "condition":  cond,
                "wer":        round(float(wer_val), 4) if wer_val is not None else None,
                "mode":       mode,
            })

        if (i + 1) % 20 == 0 or i == len(records) - 1:
            print(f"  [{i+1}/{len(records)}] {sample_id}")

    # Aggregate and report
    summary = aggregate(all_results, conditions)
    print_summary(summary, conditions, mode)

    output = {
        "conditions": conditions,
        "mode": mode,
        "summary": summary,
        "raw": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_path}")
    return summary


if __name__ == "__main__":
    main()
