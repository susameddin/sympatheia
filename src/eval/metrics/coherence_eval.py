#!/usr/bin/env python3
"""
Coherence evaluation: BERTScore F1 + ROUGE-L.

Compares generated text responses ({cond}_text from manifest) against
reference responses from eval.jsonl using semantic and surface-level
similarity metrics.

Each manifest sample is matched to a reference response of the same
emotion from eval.jsonl. Measures whether the generated content is
semantically aligned with what an appropriate empathetic response looks like.

Requires:
    pip install bert-score rouge-score

Usage:
    conda run -n s --no-capture-output python -m eval.metrics.coherence_eval \\
        --manifest /path/to/manifest.jsonl \\
        --eval-jsonl new_samples/eval.jsonl

Outputs:
    <output-dir>/metrics/coherence_metrics.json
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import ALL_EMOTIONS

DEFAULT_EVAL_JSONL = str(
    Path("/engram/naplab/users/sd3705/Datasets/Sympatheia_12Emo_Emotional_v2/tokens/eval.jsonl")
)


# ---------------------------------------------------------------------------
# Reference text loading (same as intelligibility_eval)
# ---------------------------------------------------------------------------

def extract_reference_text(text: str) -> str:
    """Extract assistant text from eval.jsonl entry."""
    m = re.search(r"<\|assistant\|>\n(.*)", text, re.DOTALL)
    if not m:
        return ""
    return m.group(1).strip()


def load_references(eval_jsonl: str) -> dict[str, list[str]]:
    """Load reference texts per emotion from eval.jsonl."""
    by_emotion: dict[str, list] = defaultdict(list)
    with open(eval_jsonl) as f:
        for line in f:
            entry = json.loads(line)
            emo = entry["id"].split("_")[0].capitalize()
            ref_text = extract_reference_text(entry.get("text", ""))
            if ref_text:
                by_emotion[emo].append(ref_text)
    print(f"Loaded references: " +
          ", ".join(f"{e}:{len(v)}" for e, v in sorted(by_emotion.items())))
    return dict(by_emotion)


# ---------------------------------------------------------------------------
# Metric computation (adapted from evaluate_model.py)
# ---------------------------------------------------------------------------

def compute_bertscore(hypotheses: list[str], references: list[str],
                      emotions: list[str]) -> dict:
    """Compute BERTScore F1. Returns metrics dict."""
    if not _BERTSCORE_OK:
        return {"bertscore_f1_mean": None, "bertscore_f1_per_emotion": None}

    print("Computing BERTScore (batch)...")
    try:
        # transformers 5.x dropped build_inputs_with_special_tokens from
        # PreTrainedTokenizerFast; bert_score 0.3.x still calls it.
        try:
            from transformers import PreTrainedTokenizerFast
            if not hasattr(PreTrainedTokenizerFast, "build_inputs_with_special_tokens"):
                PreTrainedTokenizerFast.build_inputs_with_special_tokens = (
                    lambda self, token_ids_0, token_ids_1=None:
                    token_ids_0 if token_ids_1 is None else token_ids_0 + token_ids_1
                )
        except Exception:
            pass
        _, _, F = bert_score_fn(
            hypotheses, references,
            lang="en", batch_size=32, verbose=False,
            model_type="bert-base-uncased",
        )
        f1_list = F.tolist()
        per_emo: dict[str, list] = defaultdict(list)
        for emo, f1 in zip(emotions, f1_list):
            per_emo[emo].append(f1)

        return {
            "bertscore_f1_mean": round(float(np.mean(f1_list)), 4),
            "bertscore_f1_std":  round(float(np.std(f1_list)), 4),
            "bertscore_f1_per_emotion": {
                emo: round(float(np.mean(per_emo[emo])), 4)
                if per_emo[emo] else None
                for emo in ALL_EMOTIONS
            },
            "per_sample": [round(float(f), 4) for f in f1_list],
        }
    except Exception as e:
        print(f"  BERTScore failed: {e}")
        return {"bertscore_f1_mean": None, "bertscore_f1_per_emotion": None}


def compute_rougel(hypotheses: list[str], references: list[str],
                   emotions: list[str]) -> dict:
    """Compute ROUGE-L F-measure. Returns metrics dict."""
    if not _ROUGE_OK:
        return {"rougeL_mean": None, "rougeL_per_emotion": None}

    print("Computing ROUGE-L...")
    try:
        scorer = rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=True)
        scores = []
        per_emo: dict[str, list] = defaultdict(list)
        for emo, hyp, ref in zip(emotions, hypotheses, references):
            if hyp.strip() and ref.strip():
                f = scorer.score(ref, hyp)["rougeL"].fmeasure
            else:
                f = 0.0
            scores.append(f)
            per_emo[emo].append(f)

        return {
            "rougeL_mean": round(float(np.mean(scores)), 4),
            "rougeL_std":  round(float(np.std(scores)), 4),
            "rougeL_per_emotion": {
                emo: round(float(np.mean(per_emo[emo])), 4)
                if per_emo[emo] else None
                for emo in ALL_EMOTIONS
            },
            "per_sample": [round(float(s), 4) for s in scores],
        }
    except Exception as e:
        print(f"  ROUGE-L failed: {e}")
        return {"rougeL_mean": None, "rougeL_per_emotion": None}


# ---------------------------------------------------------------------------
# Output printing
# ---------------------------------------------------------------------------

def print_summary(summary_by_cond: dict, conditions: list[str]):
    print(f"\n{'='*70}")
    print("COHERENCE METRICS (vs. reference responses from eval.jsonl)")
    print(f"{'='*70}")
    header = f"{'Condition':<20} {'BERTScore-F1':>13} {'ROUGE-L':>9}"
    print(header)
    print("-" * 45)
    for cond in conditions:
        s = summary_by_cond.get(cond, {})
        bs = s.get("bertscore_f1_mean")
        rl = s.get("rougeL_mean")
        bs_str = f"{bs:.4f}" if bs is not None else "N/A"
        rl_str = f"{rl:.4f}" if rl is not None else "N/A"
        print(f"{cond:<20} {bs_str:>13} {rl_str:>9}")

    # Per-emotion tables for BERTScore
    all_emotions = sorted({emo for s in summary_by_cond.values()
                           for emo in (s.get("bertscore_f1_per_emotion") or {})})
    for metric_key, label in [("bertscore_f1_per_emotion", "BERTScore-F1"),
                               ("rougeL_per_emotion", "ROUGE-L")]:
        if not any(s.get(metric_key) for s in summary_by_cond.values()):
            continue
        print(f"\n--- {label} per emotion ---")
        hdr = f"{'Emotion':<14}" + "".join(f" {c[:12]:>13}" for c in conditions)
        print(hdr)
        print("-" * (14 + 13 * len(conditions)))
        for emo in all_emotions:
            row = f"{emo:<14}"
            for cond in conditions:
                val = (summary_by_cond.get(cond, {})
                       .get(metric_key, {}) or {}).get(emo)
                row += f" {f'{val:.4f}' if val is not None else 'N/A':>13}"
            print(row)
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="BERTScore + ROUGE-L coherence evaluation"
    )
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--eval-jsonl", type=str, default=DEFAULT_EVAL_JSONL,
                        help=f"eval.jsonl for reference texts (default: {DEFAULT_EVAL_JSONL})")
    parser.add_argument("--conditions", type=str, nargs="+", default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not _BERTSCORE_OK and not _ROUGE_OK:
        print("ERROR: neither bert_score nor rouge_score is installed.")
        print("  Install with: pip install bert-score rouge-score")
        sys.exit(1)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}")
        sys.exit(1)

    eval_jsonl_path = Path(args.eval_jsonl)
    if not eval_jsonl_path.exists():
        print(f"ERROR: eval.jsonl not found: {eval_jsonl_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else manifest_path.parent
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    stem = manifest_path.stem
    suffix = "" if stem == "manifest" else f"_{stem.removeprefix('manifest_')}"
    out_path = metrics_dir / f"coherence_metrics{suffix}.json"

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

    # Load references
    references_by_emotion = load_references(args.eval_jsonl)
    ref_counters: dict[str, int] = defaultdict(int)

    def get_reference(emotion: str) -> str:
        refs = references_by_emotion.get(emotion, [])
        if not refs:
            return ""
        idx = ref_counters[emotion] % len(refs)
        ref_counters[emotion] += 1
        return refs[idx]

    # Build hypothesis/reference pairs per condition
    summary_by_cond: dict[str, dict] = {}
    for cond in conditions:
        print(f"\n--- {cond} ---")
        hypotheses = []
        references = []
        emotions   = []

        for rec in records:
            hypothesis = rec.get(f"{cond}_text", "") or ""
            emotion    = rec.get("emotion") or rec.get("emotion_gt", "Unknown")
            reference  = get_reference(emotion)
            if not hypothesis.strip():
                continue
            hypotheses.append(hypothesis)
            references.append(reference)
            emotions.append(emotion)
            # Reset counter so next condition uses same reference ordering
        # Reset counters for consistent pairing across conditions
        ref_counters.clear()

        if not hypotheses:
            print(f"  No hypothesis texts found for condition {cond}")
            summary_by_cond[cond] = {}
            continue

        print(f"  {len(hypotheses)} samples")
        bs_metrics = compute_bertscore(hypotheses, references, emotions)
        rl_metrics = compute_rougel(hypotheses, references, emotions)
        summary_by_cond[cond] = {**bs_metrics, **rl_metrics}

    print_summary(summary_by_cond, conditions)

    output = {
        "conditions":    conditions,
        "eval_jsonl":    str(args.eval_jsonl),
        "metrics_by_condition": summary_by_cond,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_path}")
    return summary_by_cond


if __name__ == "__main__":
    main()
