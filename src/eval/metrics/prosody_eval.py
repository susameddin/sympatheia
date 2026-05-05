#!/usr/bin/env python3
"""
Prosody analysis for emotion-conditioned speech responses.

Extracts acoustic features from generated audio in manifest.jsonl and
computes two analyses:

1. Reference-free: Spearman correlation between features and target arousal/valence,
   plus Mann-Whitney U emotion discriminability (arousal and valence groupings).

2. Reference-based (--eval-jsonl): Decodes ground-truth responses from eval.jsonl
   using GLM4CodecDecoder and computes per-feature MAE vs. reference profiles.

Features extracted per audio:
  f0_mean, f0_std, f0_range       — pitch (Hz), voiced frames only via librosa.pyin
  energy_mean, energy_std         — RMS energy
  speaking_rate                   — words/second from {cond}_text + audio duration
  spectral_centroid_mean          — voice brightness (Hz)

Usage:
    # Reference-free only:
    conda run -n s --no-capture-output python -m eval.metrics.prosody_eval \\
        --manifest /engram/naplab/users/sd3705/emo_recog_2025s/eval_emotional_.../manifest.jsonl

    # With reference audio comparison:
    conda run -n s --no-capture-output python -m eval.metrics.prosody_eval \\
        --manifest /path/to/manifest.jsonl \\
        --eval-jsonl new_samples/eval.jsonl \\
        --num-reference 5

Outputs:
    <output-dir>/metrics/prosody_metrics.json
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

try:
    import librosa
    _LIBROSA_OK = True
except ImportError:
    _LIBROSA_OK = False
    print("WARNING: librosa not installed — install with: pip install librosa")

try:
    from scipy.stats import spearmanr, mannwhitneyu
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False
    print("WARNING: scipy not installed — correlation/discriminability will be skipped. pip install scipy")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import EMOTION_VA_MAPPING, ALL_EMOTIONS

# Arousal groupings for Mann-Whitney discriminability
HIGH_AROUSAL = {"Angry", "Excited", "Anxious", "Surprised"}
LOW_AROUSAL  = {"Sad", "Tired", "Relaxed", "Content"}

# Valence groupings
POSITIVE_VALENCE = {"Happy", "Excited", "Content", "Relaxed"}
NEGATIVE_VALENCE = {"Angry", "Sad", "Frustrated", "Disgusted", "Anxious", "Tired"}

FEATURE_NAMES = [
    "f0_mean", "f0_std", "f0_range",
    "energy_mean", "energy_std",
    "speaking_rate",
    "spectral_centroid_mean",
]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_prosody(audio_path: str, word_count: int | None = None) -> dict | None:
    """Extract prosody features from a WAV file.

    Args:
        audio_path: path to WAV file
        word_count: number of words in the text response (for speaking rate)

    Returns dict with FEATURE_NAMES keys, or None on failure.
    """
    if not _LIBROSA_OK:
        return None
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        duration = len(y) / sr

        # --- Pitch (F0) via pyin ---
        f0, voiced_flag, _ = librosa.pyin(
            y, sr=sr,
            fmin=librosa.note_to_hz("C2"),  # ~65 Hz
            fmax=librosa.note_to_hz("C7"),  # ~2093 Hz
            frame_length=2048,
        )
        voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
        if len(voiced_f0) > 0:
            f0_mean  = float(np.mean(voiced_f0))
            f0_std   = float(np.std(voiced_f0))
            f0_range = float(np.percentile(voiced_f0, 90) - np.percentile(voiced_f0, 10))
        else:
            f0_mean = f0_std = f0_range = float("nan")

        # --- Energy (RMS) ---
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        energy_mean = float(np.mean(rms))
        energy_std  = float(np.std(rms))

        # --- Speaking rate (words/sec) ---
        if word_count is not None and duration > 0:
            speaking_rate = float(word_count / duration)
        else:
            speaking_rate = float("nan")

        # --- Spectral centroid ---
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
        spectral_centroid_mean = float(np.mean(centroid))

        return {
            "f0_mean":               f0_mean,
            "f0_std":                f0_std,
            "f0_range":              f0_range,
            "energy_mean":           energy_mean,
            "energy_std":            energy_std,
            "speaking_rate":         speaking_rate,
            "spectral_centroid_mean": spectral_centroid_mean,
            "duration":              duration,
            "voiced_fraction":       float(len(voiced_f0) / max(len(f0), 1)),
        }
    except Exception as e:
        print(f"  prosody extraction failed for {audio_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Per-sample feature cache
# ---------------------------------------------------------------------------

def load_cache(cache_path: Path) -> dict:
    """Load per-sample feature cache from JSONL. Returns {audio_path: features}."""
    cache = {}
    if not cache_path.exists():
        return cache
    with open(cache_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
                cache[entry["audio_path"]] = entry["features"]
            except Exception:
                pass
    return cache


def append_cache(cache_path: Path, audio_path: str, features: dict) -> None:
    """Append one sample's features to the cache JSONL file."""
    with open(cache_path, "a") as f:
        f.write(json.dumps({"audio_path": audio_path, "features": features}) + "\n")


def _extract_worker(args: tuple) -> tuple:
    """Module-level worker for parallel prosody extraction (must be picklable).

    Returns: (audio_path, features_or_None, skip_reason_or_None)
    """
    audio_path, word_count = args
    try:
        dur = librosa.get_duration(path=audio_path)
        if dur > 120:
            return audio_path, None, f"{dur:.0f}s > 120s"
    except Exception:
        pass
    feats = extract_prosody(audio_path, word_count)
    return audio_path, feats, None


# ---------------------------------------------------------------------------
# Reference audio decoding (optional)
# ---------------------------------------------------------------------------

def decode_reference_audio(text: str, decoder, audio_0_id: int) -> np.ndarray | None:
    """Decode the assistant's audio tokens from an eval.jsonl text field."""
    import torch
    # Extract assistant section
    m = re.search(r"<\|assistant\|>(.*)", text, re.DOTALL)
    if not m:
        return None
    assistant_text = m.group(1)

    # Parse all <|audio_XXXX|> token IDs
    token_ids = [int(x) for x in re.findall(r"<\|audio_(\d+)\|>", assistant_text)]
    if not token_ids:
        return None

    try:
        shifted = torch.tensor(
            [[t - audio_0_id for t in token_ids]], dtype=torch.long
        )
        audio = decoder(shifted)
        return audio.squeeze().cpu().numpy()
    except Exception as e:
        print(f"  reference decode failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_features(results: list) -> dict:
    """Aggregate features per emotion: mean ± std for each feature."""
    per_emotion: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        emo = r["emotion"]
        feats = r["features"]
        for fname in FEATURE_NAMES:
            v = feats.get(fname, float("nan"))
            if not np.isnan(v):
                per_emotion[emo][fname].append(v)

    summary = {}
    for emo in ALL_EMOTIONS:
        summary[emo] = {}
        for fname in FEATURE_NAMES:
            vals = per_emotion[emo].get(fname, [])
            if vals:
                summary[emo][fname] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std":  round(float(np.std(vals)), 4),
                    "n":    len(vals),
                }
            else:
                summary[emo][fname] = {"mean": None, "std": None, "n": 0}
    return summary


def compute_correlations(results: list) -> dict:
    """Spearman ρ(feature, arousal) and ρ(feature, valence) across all samples."""
    if not _SCIPY_OK:
        return {}

    # Collect flat arrays
    feature_vals: dict[str, list] = defaultdict(list)
    arousals = []
    valences = []

    for r in results:
        emo = r["emotion"]
        va = EMOTION_VA_MAPPING.get(emo)
        if va is None:
            continue
        target_v, target_a = va
        feats = r["features"]
        # Only include if all features are valid
        vals = {fname: feats.get(fname, float("nan")) for fname in FEATURE_NAMES}
        if any(np.isnan(v) for v in vals.values()):
            continue
        for fname, v in vals.items():
            feature_vals[fname].append(v)
        arousals.append(target_a)
        valences.append(target_v)

    if len(arousals) < 3:
        return {}

    correlations = {}
    for fname in FEATURE_NAMES:
        fv = feature_vals[fname]
        rho_a, p_a = spearmanr(fv, arousals)
        rho_v, p_v = spearmanr(fv, valences)
        correlations[fname] = {
            "rho_arousal":  round(float(rho_a), 4),
            "p_arousal":    round(float(p_a), 4),
            "rho_valence":  round(float(rho_v), 4),
            "p_valence":    round(float(p_v), 4),
        }
    return correlations


def compute_discriminability(results: list) -> dict:
    """Mann-Whitney U for each feature: high vs. low arousal, positive vs. negative valence."""
    if not _SCIPY_OK:
        return {}

    # Build per-group feature lists
    hi_arousal:   dict[str, list] = defaultdict(list)
    lo_arousal:   dict[str, list] = defaultdict(list)
    pos_valence:  dict[str, list] = defaultdict(list)
    neg_valence:  dict[str, list] = defaultdict(list)

    for r in results:
        emo = r["emotion"]
        feats = r["features"]
        for fname in FEATURE_NAMES:
            v = feats.get(fname, float("nan"))
            if np.isnan(v):
                continue
            if emo in HIGH_AROUSAL:
                hi_arousal[fname].append(v)
            elif emo in LOW_AROUSAL:
                lo_arousal[fname].append(v)
            if emo in POSITIVE_VALENCE:
                pos_valence[fname].append(v)
            elif emo in NEGATIVE_VALENCE:
                neg_valence[fname].append(v)

    disc = {}
    for fname in FEATURE_NAMES:
        disc[fname] = {}
        # Arousal discriminability
        hi = hi_arousal[fname]
        lo = lo_arousal[fname]
        if len(hi) >= 3 and len(lo) >= 3:
            stat, p = mannwhitneyu(hi, lo, alternative="two-sided")
            disc[fname]["arousal"] = {
                "U":          round(float(stat), 2),
                "p":          round(float(p), 4),
                "mean_high":  round(float(np.mean(hi)), 4),
                "mean_low":   round(float(np.mean(lo)), 4),
                "n_high":     len(hi),
                "n_low":      len(lo),
            }
        else:
            disc[fname]["arousal"] = None

        # Valence discriminability
        pos = pos_valence[fname]
        neg = neg_valence[fname]
        if len(pos) >= 3 and len(neg) >= 3:
            stat, p = mannwhitneyu(pos, neg, alternative="two-sided")
            disc[fname]["valence"] = {
                "U":            round(float(stat), 2),
                "p":            round(float(p), 4),
                "mean_positive": round(float(np.mean(pos)), 4),
                "mean_negative": round(float(np.mean(neg)), 4),
                "n_positive":   len(pos),
                "n_negative":   len(neg),
            }
        else:
            disc[fname]["valence"] = None

    return disc


# ---------------------------------------------------------------------------
# Reference-based comparison
# ---------------------------------------------------------------------------

def build_reference_profiles(eval_jsonl: str, num_reference: int,
                              decoder, audio_0_id: int) -> dict:
    """Decode reference responses from eval.jsonl and extract prosody profiles.

    Returns: {emotion: {feature: {"mean": float, "std": float, "n": int}}}
    """
    import soundfile as sf
    import tempfile

    by_emotion: dict[str, list] = defaultdict(list)
    with open(eval_jsonl) as f:
        for line in f:
            entry = json.loads(line)
            emo = entry["id"].split("_")[1].capitalize()
            by_emotion[emo].append(entry)

    profiles: dict[str, dict] = {}
    for emo in ALL_EMOTIONS:
        entries = by_emotion.get(emo, [])[:num_reference]
        if not entries:
            print(f"  WARNING: no reference entries for {emo}")
            continue

        feat_lists: dict[str, list] = defaultdict(list)
        for entry in entries:
            audio_np = decode_reference_audio(entry["text"], decoder, audio_0_id)
            if audio_np is None:
                continue

            # Write to temp file for librosa
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                sf.write(tmp_path, audio_np, 22050)
                # Extract reference text for word count
                m = re.search(r"<\|assistant\|>\n(.*?)\n<\|audio_", entry["text"], re.DOTALL)
                ref_text = m.group(1).strip() if m else ""
                wc = len(ref_text.split()) if ref_text else None
                feats = extract_prosody(tmp_path, word_count=wc)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            if feats is None:
                continue
            for fname in FEATURE_NAMES:
                v = feats.get(fname, float("nan"))
                if not np.isnan(v):
                    feat_lists[fname].append(v)

        profiles[emo] = {}
        for fname in FEATURE_NAMES:
            vals = feat_lists[fname]
            if vals:
                profiles[emo][fname] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std":  round(float(np.std(vals)), 4),
                    "n":    len(vals),
                }
            else:
                profiles[emo][fname] = {"mean": None, "std": None, "n": 0}

    return profiles


def compute_reference_mae(cond_profile: dict, ref_profile: dict) -> dict:
    """Per-feature MAE between condition's per-emotion means and reference means."""
    maes: dict[str, list] = defaultdict(list)
    for emo in ALL_EMOTIONS:
        if emo not in ref_profile or emo not in cond_profile:
            continue
        for fname in FEATURE_NAMES:
            cond_val = cond_profile.get(emo, {}).get(fname, {}).get("mean")
            ref_val  = ref_profile.get(emo, {}).get(fname, {}).get("mean")
            if cond_val is not None and ref_val is not None:
                maes[fname].append(abs(cond_val - ref_val))

    return {
        fname: round(float(np.mean(vals)), 4) if vals else None
        for fname, vals in maes.items()
    }


# ---------------------------------------------------------------------------
# Output printing
# ---------------------------------------------------------------------------

def print_correlation_table(correlations_by_cond: dict, conditions: list):
    """Print Spearman ρ table (feature × condition, arousal and valence)."""
    print(f"\n{'='*70}")
    print("SPEARMAN ρ: FEATURE vs. TARGET AROUSAL (reference-free)")
    print(f"{'='*70}")
    header = f"{'Feature':<26}" + "".join(f" {c[:12]:>14}" for c in conditions)
    print(header)
    print("-" * (26 + 14 * len(conditions)))
    for fname in FEATURE_NAMES:
        row = f"{fname:<26}"
        for cond in conditions:
            c = correlations_by_cond.get(cond, {}).get(fname, {})
            rho = c.get("rho_arousal")
            p   = c.get("p_arousal")
            if rho is not None:
                sig = "*" if p is not None and p < 0.05 else " "
                row += f" {f'{rho:.3f}{sig}':>14}"
            else:
                row += f" {'N/A':>14}"
        print(row)
    print("  (* p < 0.05)")

    print(f"\n{'='*70}")
    print("SPEARMAN ρ: FEATURE vs. TARGET VALENCE (reference-free)")
    print(f"{'='*70}")
    print(header)
    print("-" * (26 + 14 * len(conditions)))
    for fname in FEATURE_NAMES:
        row = f"{fname:<26}"
        for cond in conditions:
            c = correlations_by_cond.get(cond, {}).get(fname, {})
            rho = c.get("rho_valence")
            p   = c.get("p_valence")
            if rho is not None:
                sig = "*" if p is not None and p < 0.05 else " "
                row += f" {f'{rho:.3f}{sig}':>14}"
            else:
                row += f" {'N/A':>14}"
        print(row)
    print("  (* p < 0.05)\n")


def print_discriminability_table(disc_by_cond: dict, conditions: list):
    """Print emotion discriminability (Mann-Whitney p-values)."""
    print(f"\n{'='*70}")
    print("EMOTION DISCRIMINABILITY (Mann-Whitney U p-values, two-sided)")
    print("  Arousal: high={Angry,Excited,Anxious,Surprised} vs low={Sad,Tired,Relaxed,Content}")
    print("  Valence: positive={Happy,Excited,Content,Relaxed} vs negative={Angry,Sad,Frustrated,Disgusted,Anxious,Tired}")
    print(f"{'='*70}")
    for dim in ("arousal", "valence"):
        print(f"\n  --- {dim.capitalize()} discriminability ---")
        header = f"  {'Feature':<26}" + "".join(f" {c[:10]:>12}" for c in conditions)
        print(header)
        print("  " + "-" * (24 + 12 * len(conditions)))
        for fname in FEATURE_NAMES:
            row = f"  {fname:<26}"
            for cond in conditions:
                d = disc_by_cond.get(cond, {}).get(fname, {}).get(dim)
                if d:
                    p = d["p"]
                    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "  ")
                    row += f" {f'p={p:.3f}{sig}':>12}"
                else:
                    row += f" {'N/A':>12}"
            print(row)
    print("  (* p<0.05, ** p<0.01)\n")


def print_per_emotion_table(profiles_by_cond: dict, conditions: list, feature: str):
    """Print per-emotion means for one feature across conditions."""
    print(f"\n  {feature}:")
    header = f"    {'Emotion':<14}" + "".join(f" {c[:12]:>13}" for c in conditions)
    print(header)
    print("    " + "-" * (12 + 13 * len(conditions)))
    for emo in ALL_EMOTIONS:
        row = f"    {emo:<14}"
        for cond in conditions:
            v = profiles_by_cond.get(cond, {}).get(emo, {}).get(feature, {}).get("mean")
            row += f" {f'{v:.3f}' if v is not None else 'N/A':>13}"
        print(row)


def print_reference_mae_table(mae_by_cond: dict, conditions: list):
    """Print per-feature MAE vs. reference for each condition."""
    print(f"\n{'='*70}")
    print("REFERENCE-BASED PROSODY MAE (lower = closer to reference)")
    print(f"{'='*70}")
    header = f"{'Feature':<26}" + "".join(f" {c[:12]:>14}" for c in conditions)
    print(header)
    print("-" * (26 + 14 * len(conditions)))
    for fname in FEATURE_NAMES:
        row = f"{fname:<26}"
        for cond in conditions:
            v = mae_by_cond.get(cond, {}).get(fname)
            row += f" {f'{v:.4f}' if v is not None else 'N/A':>14}"
        print(row)
    print()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prosody analysis for emotion-conditioned speech responses"
    )
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to manifest.jsonl from eval/generate_responses.py")
    parser.add_argument("--conditions", type=str, nargs="+", default=None,
                        help="Conditions to evaluate (default: auto-detect from manifest)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: manifest parent dir)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip samples whose features are already computed")
    parser.add_argument("--eval-jsonl", type=str, default=None,
                        help="eval.jsonl for reference-based comparison (requires GPU)")
    parser.add_argument("--decoder-path", type=str,
                        default=str(PROJECT_ROOT / "glm-4-voice-decoder"),
                        help="Path to glm-4-voice-decoder directory")
    parser.add_argument("--num-reference", type=int, default=5,
                        help="Reference responses per emotion (default: 5)")
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1),
                        help="Parallel workers for feature extraction (default: min(8, cpu_count))")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not _LIBROSA_OK:
        print("ERROR: librosa is required. Install with: pip install librosa")
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
    out_path = metrics_dir / f"prosody_metrics{suffix}.json"

    cache_path = metrics_dir / f"prosody_cache{suffix}.jsonl"
    cache = load_cache(cache_path)
    if args.skip_existing and cache:
        print(f"Loaded {len(cache)} cached samples from {cache_path.name}")

    # Load manifest
    records = []
    with open(manifest_path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} manifest records")

    # Auto-detect conditions
    if args.conditions is None:
        all_keys = set()
        for rec in records:
            all_keys.update(rec.keys())
        conditions = [c for c in
                      ["base", "finetuned_va", "finetuned_na", "opens2s", "osum_no_think",
                       "osum_think", "qwen3omni", "qwen2_5omni", "kimiaudio", "face_va", "no_va", "oracle",
                       "eeg_only", "eye_only", "combined"]
                      if f"{c}_response" in all_keys]
        print(f"Auto-detected conditions: {conditions}")
    else:
        conditions = args.conditions

    # Extract prosody per condition
    all_results: dict[str, list] = {}  # condition -> list of {id, emotion, valence, arousal, features}
    for cond in conditions:
        print(f"\n--- Extracting prosody: {cond} ---")

        # Separate cached samples from those needing extraction
        pending = []        # (audio_path, word_count, rec) — need extraction
        feat_map: dict[str, dict] = {}  # audio_path -> features (filled from cache or extraction)

        for rec in records:
            audio_path = rec.get(f"{cond}_response")
            if audio_path is None or not Path(audio_path).exists():
                continue
            text = rec.get(f"{cond}_text", "") or ""
            word_count = len(text.split()) if text.strip() else None
            if args.skip_existing and audio_path in cache:
                feat_map[audio_path] = cache[audio_path]
            else:
                pending.append((audio_path, word_count, rec))

        n_cached = len(feat_map)
        print(f"  {n_cached} cached, {len(pending)} to extract (workers={args.num_workers})")

        # Parallel extraction of pending samples
        if pending:
            worker_args = [(ap, wc) for ap, wc, _ in pending]
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                for (audio_path, _, rec), (_, feats, skip_msg) in zip(
                    pending, executor.map(_extract_worker, worker_args)
                ):
                    if skip_msg:
                        print(f"  Skipping {rec.get('id', '')} ({cond}): {skip_msg}")
                        continue
                    if feats is None:
                        continue
                    append_cache(cache_path, audio_path, feats)
                    cache[audio_path] = feats
                    feat_map[audio_path] = feats

        # Build results in original manifest order
        cond_results = []
        for rec in records:
            audio_path = rec.get(f"{cond}_response")
            if audio_path not in feat_map:
                continue
            cond_results.append({
                "id":      rec.get("id", ""),
                "emotion": rec.get("emotion") or rec.get("emotion_gt", "Unknown"),
                "valence": rec.get("valence", 0.0),
                "arousal": rec.get("arousal", 0.0),
                "features": feat_map[audio_path],
            })

        print(f"  Done: {len(cond_results)} samples ({len(cond_results) - n_cached} newly extracted)")
        all_results[cond] = cond_results

    # Aggregate per-emotion profiles
    profiles_by_cond: dict[str, dict] = {}
    for cond, results in all_results.items():
        profiles_by_cond[cond] = aggregate_features(results)

    # Correlation analysis
    correlations_by_cond: dict[str, dict] = {}
    for cond, results in all_results.items():
        correlations_by_cond[cond] = compute_correlations(results)

    # Discriminability analysis
    disc_by_cond: dict[str, dict] = {}
    for cond, results in all_results.items():
        disc_by_cond[cond] = compute_discriminability(results)

    # Reference-based comparison (optional)
    reference_mae_by_cond: dict[str, dict] = {}
    reference_profiles: dict = {}
    if args.eval_jsonl:
        print(f"\n--- Decoding reference audio from {args.eval_jsonl} ---")
        try:
            import torch
            from src.vocoder_src import GLM4CodecDecoder
            from transformers import AutoTokenizer

            print("Loading tokenizer for audio_0_id...")
            tokenizer = AutoTokenizer.from_pretrained(
                "THUDM/glm-4-voice-9b", trust_remote_code=True
            )
            audio_0_id = tokenizer.convert_tokens_to_ids("<|audio_0|>")
            print(f"  audio_0_id = {audio_0_id}")

            print("Loading GLM4CodecDecoder...")
            decoder = GLM4CodecDecoder(args.decoder_path)
            print("  Decoder loaded.")

            reference_profiles = build_reference_profiles(
                args.eval_jsonl, args.num_reference, decoder, audio_0_id
            )
            for cond in conditions:
                reference_mae_by_cond[cond] = compute_reference_mae(
                    profiles_by_cond[cond], reference_profiles
                )
        except Exception as e:
            print(f"  Reference comparison failed: {e}")
            print("  (Skipping reference-based analysis)")

    # Print summary tables
    print_correlation_table(correlations_by_cond, conditions)
    print_discriminability_table(disc_by_cond, conditions)

    print(f"\n{'='*70}")
    print("PER-EMOTION PROSODY PROFILES (selected features)")
    print(f"{'='*70}")
    for fname in ["f0_mean", "energy_mean", "speaking_rate"]:
        print_per_emotion_table(profiles_by_cond, conditions, fname)

    if reference_mae_by_cond:
        print_reference_mae_table(reference_mae_by_cond, conditions)

    # Save to JSON
    output = {
        "conditions": conditions,
        "per_emotion_profiles": profiles_by_cond,
        "correlations": correlations_by_cond,
        "discriminability": disc_by_cond,
    }
    if reference_mae_by_cond:
        output["reference_mae"] = reference_mae_by_cond
        output["reference_profiles"] = reference_profiles

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
