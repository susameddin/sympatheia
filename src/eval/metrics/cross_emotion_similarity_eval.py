#!/usr/bin/env python3
"""
Cross-emotion response similarity evaluation.

For the same neutral query delivered under different emotion conditions,
measures how similar the model's responses are to each other.

  High similarity  →  model ignores user emotion        →  bad
  Low similarity   →  model adapts content to emotion   →  good

Metrics per condition:
  mean_rouge_l        — mean ROUGE-L F1 across all C(N,2) emotion pairs
  mean_bertscore_f1   — mean BERTScore F1 across all C(N,2) emotion pairs

Text source (default): Whisper ASR transcriptions of the response audio.
  Reuses metrics/asr_cache{suffix}.jsonl shared with differentiation_eval.
  Pass --use-text-field to skip ASR and read *_text manifest fields instead.

Accepts multiple --manifests: all are merged so conditions from different
manifest files (e.g. manifest.jsonl + manifest_base.jsonl) are evaluated
together in one output file.

Usage:
    python -m eval.metrics.cross_emotion_similarity_eval \\
        --manifests manifest.jsonl manifest_base.jsonl

    # Fast path (no Whisper):
    python -m eval.metrics.cross_emotion_similarity_eval \\
        --manifests manifest.jsonl manifest_base.jsonl \\
        --use-text-field

Outputs:
    <output-dir>/metrics/cross_emotion_similarity.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_WHISPER_MODEL = "medium"

try:
    from bert_score import score as bert_score_fn
    _BERTSCORE_OK = True
except ImportError:
    _BERTSCORE_OK = False
    print("WARNING: bert_score not installed — install with: pip install bert-score")

try:
    from rouge_score import rouge_scorer as rouge_scorer_lib
    _ROUGE_OK = True
except ImportError:
    _ROUGE_OK = False
    print("WARNING: rouge_score not installed — install with: pip install rouge-score")

KNOWN_CONDITIONS = [
    "base", "finetuned_va", "finetuned_na", "opens2s", "osum_no_think",
    "osum_think", "qwen3omni", "qwen3tts_cascaded", "qwen2_5omni",
    "kimiaudio", "osum_neutral", "face_va", "no_va", "oracle",
    "eeg_only", "eye_only", "combined",
]

# CJK Unicode blocks: Unified Ideographs, Extension A, Hiragana, Katakana, full-width
import re
_CJK_RE = re.compile(r'[一-鿿㐀-䶿぀-ヿ＀-￯]')

MIN_DURATION_SECS = 10.0  # discard truncated/garbled responses shorter than this

def is_english(text: str, max_cjk_ratio: float = 0.05) -> bool:
    """Return False if more than 5% of characters are CJK (Chinese/Japanese/Korean)."""
    stripped = text.strip()
    if not stripped:
        return False
    return len(_CJK_RE.findall(stripped)) / len(stripped) < max_cjk_ratio


def get_audio_duration(audio_path: str) -> float | None:
    """Return duration in seconds. Handles float32 WAV and other formats."""
    try:
        import soundfile as sf
        return sf.info(audio_path).duration
    except Exception:
        pass
    try:
        import wave
        with wave.open(audio_path, 'r') as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_query_index(sample_id: str) -> str:
    """'angry_00' → '00'"""
    parts = sample_id.split("_", 1)
    return parts[1] if len(parts) == 2 else sample_id


def detect_conditions(records_list: list[list[dict]], use_text_field: bool) -> list[str]:
    all_keys: set = set()
    for records in records_list:
        for rec in records:
            all_keys.update(rec.keys())
    suffix = "_text" if use_text_field else "_response"
    return [c for c in KNOWN_CONDITIONS if f"{c}{suffix}" in all_keys]


def get_text_by_condition(
    records_list: list[list[dict]],
    conditions: list[str],
    use_text_field: bool,
    cache_paths: list[Path],
    whisper_model,
    skip_existing: bool,
) -> dict[str, dict[str, str]]:
    """Return {condition: {sample_id: text}}."""
    result: dict[str, dict[str, str]] = {c: {} for c in conditions}

    if use_text_field:
        for records in records_list:
            for rec in records:
                sid = rec.get("id", "")
                for cond in conditions:
                    text = rec.get(f"{cond}_text", "") or ""
                    if text.strip():
                        result[cond][sid] = text.strip()
        return result

    # ASR mode: per-manifest transcription with shared cache
    from eval.metrics._asr_utils import transcribe_manifest
    for records, cache_path in zip(records_list, cache_paths):
        trans_by_cond = transcribe_manifest(
            records, conditions, cache_path, whisper_model,
            skip_existing=skip_existing,
        )
        for rec in records:
            sid = rec.get("id", "")
            for cond in conditions:
                audio_path = rec.get(f"{cond}_response")
                if audio_path:
                    transcript = trans_by_cond.get(cond, {}).get(audio_path, "")
                    if transcript.strip():
                        result[cond][sid] = transcript.strip()

    return result


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_pairwise_scores(
    query_data: dict[str, dict[str, str]],
) -> tuple[list, list, list[tuple[str, str, str]]]:
    """Compute ROUGE-L and BERTScore for all cross-emotion pairs.

    Returns parallel lists: rouge_scores, bertscore_list, pair_labels.
    pair_labels[i] = (query_idx, emotion_a, emotion_b).
    ROUGE-L is symmetric (average of both directions).
    """
    hyps: list[str] = []
    refs: list[str] = []
    pair_labels: list[tuple] = []

    for query_idx, emo_texts in query_data.items():
        emotions = sorted(emo_texts.keys())
        for i in range(len(emotions)):
            for j in range(i + 1, len(emotions)):
                ea, eb = emotions[i], emotions[j]
                hyps.append(emo_texts[ea])
                refs.append(emo_texts[eb])
                pair_labels.append((query_idx, ea, eb))

    if not hyps:
        return [], [], []

    # ROUGE-L (symmetric)
    rouge_scores: list = []
    if _ROUGE_OK:
        scorer = rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=True)
        for h, r in zip(hyps, refs):
            if h.strip() and r.strip():
                fwd = scorer.score(r, h)["rougeL"].fmeasure
                bwd = scorer.score(h, r)["rougeL"].fmeasure
                rouge_scores.append((fwd + bwd) / 2)
            else:
                rouge_scores.append(0.0)
    else:
        rouge_scores = [None] * len(hyps)

    # BERTScore F1 (batched)
    bertscore_list: list = []
    if _BERTSCORE_OK:
        print(f"    Computing BERTScore on {len(hyps)} pairs...")
        try:
            _, _, F = bert_score_fn(
                hyps, refs, lang="en", batch_size=64,
                model_type="distilbert-base-uncased", verbose=False,
            )
            bertscore_list = F.tolist()
        except Exception as e:
            print(f"    BERTScore failed: {e}")
            bertscore_list = [None] * len(hyps)
    else:
        bertscore_list = [None] * len(hyps)

    return rouge_scores, bertscore_list, pair_labels


def aggregate(scores: list, labels: list[tuple]) -> tuple[dict, dict]:
    """Overall mean/std and per-emotion-pair breakdown."""
    valid = [(s, lbl) for s, lbl in zip(scores, labels) if s is not None]
    if not valid:
        return {"mean": None, "std": None, "n": 0}, {}

    vals = [v for v, _ in valid]
    overall = {
        "mean": round(float(np.mean(vals)), 4),
        "std":  round(float(np.std(vals)), 4),
        "n":    len(vals),
    }
    pair_accum: dict[str, list] = defaultdict(list)
    for val, (_, ea, eb) in valid:
        pair_accum[f"{ea}|{eb}"].append(val)
    per_pair = {k: round(float(np.mean(v)), 4) for k, v in sorted(pair_accum.items())}
    return overall, per_pair


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_summary(summary: dict, conditions: list[str]):
    print(f"\n{'='*80}")
    print("CROSS-EMOTION RESPONSE SIMILARITY  (lower = more diverse = better)")
    print(f"{'='*80}")
    print(f"{'Condition':<22} {'ROUGE-L':>10} {'±':>6} {'BERTScore-F1':>14} {'±':>6}")
    print("-" * 62)
    for cond in conditions:
        s  = summary.get(cond, {})
        rl = s.get("rouge_l", {})
        bs = s.get("bertscore_f1", {})
        rl_m = f"{rl['mean']:.4f}" if rl.get("mean") is not None else "N/A"
        rl_s = f"{rl['std']:.4f}"  if rl.get("std")  is not None else ""
        bs_m = f"{bs['mean']:.4f}" if bs.get("mean") is not None else "N/A"
        bs_s = f"{bs['std']:.4f}"  if bs.get("std")  is not None else ""
        print(f"{cond:<22} {rl_m:>10} {rl_s:>6} {bs_m:>14} {bs_s:>6}")
    print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-emotion similarity: ROUGE-L + BERTScore F1 across emotion conditions."
    )
    parser.add_argument("--manifests",      type=str, nargs="+", required=True,
                        help="One or more manifest.jsonl files (conditions auto-detected)")
    parser.add_argument("--conditions",     type=str, nargs="+", default=None)
    parser.add_argument("--output-dir",     type=str, default=None)
    parser.add_argument("--whisper-model",  type=str, default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--use-text-field", action="store_true",
                        help="Read *_text fields instead of running Whisper ASR")
    parser.add_argument("--skip-existing",  action="store_true", default=True)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not _ROUGE_OK and not _BERTSCORE_OK:
        print("ERROR: neither rouge_score nor bert_score is installed.")
        sys.exit(1)

    manifest_paths = [Path(p) for p in args.manifests]
    for mp in manifest_paths:
        if not mp.exists():
            print(f"ERROR: manifest not found: {mp}")
            sys.exit(1)

    output_dir  = Path(args.output_dir) if args.output_dir else manifest_paths[0].parent
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path    = metrics_dir / "cross_emotion_similarity.json"

    # Load manifests
    records_list: list[list[dict]] = []
    for mp in manifest_paths:
        records = []
        with open(mp) as f:
            for line in f:
                records.append(json.loads(line))
        print(f"Loaded {len(records)} records from {mp.name}")
        records_list.append(records)

    conditions = args.conditions or detect_conditions(records_list, args.use_text_field)
    print(f"Conditions: {conditions}")

    # Cache paths (same suffix convention as other scripts)
    cache_paths = [
        metrics_dir / (
            "asr_cache.jsonl" if mp.stem == "manifest"
            else f"asr_cache_{mp.stem.removeprefix('manifest_')}.jsonl"
        )
        for mp in manifest_paths
    ]

    # Load / transcribe text
    if args.use_text_field:
        print("\n--- Using *_text fields (no ASR) ---")
        whisper_model = None
    else:
        print(f"\n--- ASR transcription (whisper: {args.whisper_model}) ---")
        from eval.metrics._asr_utils import load_whisper
        whisper_model = load_whisper(args.whisper_model)

    text_by_cond = get_text_by_condition(
        records_list, conditions,
        args.use_text_field, cache_paths,
        whisper_model, skip_existing=args.skip_existing,
    )

    # Build id→emotion and id→audio_path maps
    id_to_emotion: dict[str, str] = {}
    audio_by_cond_sid: dict[str, dict[str, str]] = defaultdict(dict)
    for records in records_list:
        for rec in records:
            sid = rec.get("id", "")
            emo = rec.get("emotion") or rec.get("emotion_gt", "Unknown")
            id_to_emotion[sid] = emo
            if not args.use_text_field:
                for cond in conditions:
                    path = rec.get(f"{cond}_response", "")
                    if path:
                        audio_by_cond_sid[cond][sid] = path

    # Compute metrics per condition
    summary: dict[str, dict] = {}

    for cond in conditions:
        print(f"\n--- Cross-emotion similarity: {cond} ---")
        cond_texts = text_by_cond.get(cond, {})

        # Drop non-English or too-short/garbled responses (< MIN_DURATION_SECS audio)
        n_total  = len(cond_texts)
        filtered = {}
        for sid, t in cond_texts.items():
            if not is_english(t):
                continue
            if not args.use_text_field:
                path = audio_by_cond_sid.get(cond, {}).get(sid, "")
                if path:
                    dur = get_audio_duration(path)
                    if dur is not None and dur < MIN_DURATION_SECS:
                        continue
            filtered[sid] = t
        cond_texts = filtered
        n_dropped  = n_total - len(cond_texts)
        if n_dropped:
            print(f"  Dropped {n_dropped}/{n_total} non-English or <{MIN_DURATION_SECS}s responses")

        query_data: dict[str, dict[str, str]] = defaultdict(dict)
        for sid, text in cond_texts.items():
            emo       = id_to_emotion.get(sid, "Unknown")
            query_idx = parse_query_index(sid)
            query_data[query_idx][emo] = text

        query_data = {q: d for q, d in query_data.items() if len(d) >= 2}
        print(f"  {len(query_data)} queries with ≥2 emotion responses")

        if not query_data:
            summary[cond] = {}
            continue

        rouge_scores, bertscore_list, pair_labels = compute_pairwise_scores(query_data)

        rl_overall, rl_per_pair = aggregate(rouge_scores,   pair_labels)
        bs_overall, bs_per_pair = aggregate(bertscore_list, pair_labels)

        print(f"  ROUGE-L    mean={rl_overall.get('mean')}  std={rl_overall.get('std')}  n={rl_overall.get('n')}")
        print(f"  BERTScore  mean={bs_overall.get('mean')}  std={bs_overall.get('std')}")

        summary[cond] = {
            "rouge_l":              rl_overall,
            "bertscore_f1":         bs_overall,
            "invalid_dropped":  n_dropped,
            "per_emotion_pair": {
                "rouge_l":      rl_per_pair,
                "bertscore_f1": bs_per_pair,
            },
        }

    print_summary(summary, conditions)

    output = {
        "conditions":  conditions,
        "text_source": "text_field" if args.use_text_field else f"whisper_{args.whisper_model}",
        "manifests":   [str(p) for p in manifest_paths],
        "summary":     summary,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_path}")
    return summary


if __name__ == "__main__":
    main()
