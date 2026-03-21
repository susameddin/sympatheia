#!/usr/bin/env python3
"""
Judge model responses for emotion-adaptation quality using Qwen3-Omni.

This is the neutral-input variant of judge_qwen3omni.py.  The key difference
is the rubric: instead of rating empathy toward an emotionally expressive user,
the judge rates how well the model *adapts* to a *stated* emotion label when
the user's actual audio was neutral.

For each entry in a manifest.jsonl, Qwen3-Omni listens to each model's audio
response and rates how well it adapts to the stated emotion (1–5 scale).

Conditions judged (auto-detected from manifest):
  base, finetuned_va, finetuned_na, opens2s, qwen3omni

Outputs:
  <output-dir>/judgments.jsonl   — one record per (sample, condition)
  <output-dir>/summary.json      — aggregated mean/std per condition and emotion
  Prints a comparison table to stdout.

Usage:
    conda run -n qwen3omni python -m eval.judge_qwen3omni_neutral \\
        --manifest results/eval_neutral/manifest.jsonl

    # Judge only specific conditions:
    conda run -n qwen3omni python -m eval.judge_qwen3omni_neutral \\
        --manifest results/eval_neutral/manifest.jsonl \\
        --conditions base finetuned_va

    # Resume interrupted run:
    conda run -n qwen3omni python -m eval.judge_qwen3omni_neutral \\
        --manifest results/eval_neutral/manifest.jsonl \\
        --skip-existing
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "/engram/naplab/users/sd3705/emo_recog_2025s/Models/Qwen3-Omni"
ALL_CONDITIONS = ["base", "finetuned_va", "finetuned_na", "opens2s", "qwen3omni"]

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator of emotion-adaptive conversational AI. Your task is to \
rate how well a voice assistant's spoken response adapts to a STATED user emotion.

IMPORTANT CONTEXT: The user's audio input is emotionally NEUTRAL. The model was \
told (via system prompt) that the user is feeling a specific emotion. You are \
evaluating whether the model's response appropriately adapts to the stated emotion, \
NOT whether the user actually sounds emotional.

Rate the response on a scale from 1 to 5 using this rubric:
  1 — Response completely ignores the stated emotion, treats user as neutral
  2 — Minimal adaptation; response is mostly generic regardless of stated emotion
  3 — Some adaptation shown but inconsistent or superficial
  4 — Clear adaptation to the stated emotion; tone and content are appropriate
  5 — Excellent adaptation: response clearly tailored to the stated emotion with \
appropriate tone, word choice, and emotional support

Respond with ONLY the rating number (1-5) followed by a vertical bar and a \
brief one-sentence justification.
Format exactly: <rating>|<justification>
Example: 4|The response appropriately acknowledges the user's stated frustration and offers calm, supportive guidance.\
"""


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Judge audio responses for emotion-adaptation quality with Qwen3-Omni (neutral-input eval)"
    )
    parser.add_argument(
        "--manifest", type=str, required=True,
        help="Path to manifest.jsonl produced by eval/generate_responses_neutral.py",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for judgments.jsonl and summary.json. "
             "Defaults to the same directory as --manifest.",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Qwen3-Omni model path or HF repo ID. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--conditions", type=str, nargs="+", default=None,
        choices=ALL_CONDITIONS,
        help="Which model conditions to judge. Defaults to whichever conditions "
             "are present in the manifest. E.g.: --conditions base finetuned_va opens2s",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip (sample, condition) pairs already present in judgments.jsonl",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Qwen3-Omni helpers
# ---------------------------------------------------------------------------

def load_judge_model(model_path: str):
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    print(f"Loading Qwen3-Omni from: {model_path}")
    t0 = time.time()
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path, dtype="auto", device_map="auto"
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")
    return model, processor


def build_conversation(emotion: str, valence: float, arousal: float, audio_path: str) -> list:
    """Build the conversation dict for Qwen3-Omni judge (neutral-input variant)."""
    user_text = (
        f"The model was told the user is feeling {emotion} "
        f"(valence={valence:.2f}, arousal={arousal:.2f}), "
        "but the user's actual audio was emotionally neutral. "
        "Listen to the voice assistant's response below and rate how well "
        "it adapts to the stated emotion."
    )
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": JUDGE_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text",  "text":  user_text},
                {"type": "audio", "audio": audio_path},
            ],
        },
    ]


def run_judge(model, processor, conversation: list) -> str:
    """Run Qwen3-Omni on the conversation and return the raw text response."""
    from qwen_omni_utils import process_mm_info

    USE_AUDIO_IN_VIDEO = True
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.no_grad():
        text_ids, _ = model.generate(
            **inputs,
            speaker="Chelsie",
            thinker_return_dict_in_generate=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
        )

    decoded = processor.batch_decode(
        text_ids.sequences[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0] if decoded else ""


def parse_rating(raw_response: str) -> tuple[int | None, str]:
    """Extract (rating, justification) from the model's raw response.

    Expected format: "<digit>|<justification>"
    Falls back to searching for any 1–5 digit in the response.
    """
    # Primary: digit immediately followed by '|'
    m = re.search(r"([1-5])\s*\|(.+)", raw_response, re.DOTALL)
    if m:
        rating = int(m.group(1))
        justification = m.group(2).strip()
        return rating, justification

    # Fallback: any standalone 1–5 digit
    m = re.search(r"\b([1-5])\b", raw_response)
    if m:
        return int(m.group(1)), raw_response.strip()

    return None, raw_response.strip()


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def aggregate_judgments(judgments: list, conditions: list) -> dict:
    """Compute mean/std per condition overall and per emotion."""
    from collections import defaultdict
    import statistics

    by_cond = defaultdict(list)
    by_cond_emo = defaultdict(lambda: defaultdict(list))

    for j in judgments:
        if j["rating"] is None:
            continue
        cond = j["condition"]
        emo = j["emotion"]
        r = j["rating"]
        by_cond[cond].append(r)
        by_cond_emo[cond][emo].append(r)

    summary = {}
    for cond in conditions:
        ratings = by_cond[cond]
        if ratings:
            mean = round(sum(ratings) / len(ratings), 3)
            std = round(statistics.stdev(ratings) if len(ratings) > 1 else 0.0, 3)
        else:
            mean = std = None
        per_emotion = {}
        for emo, vals in by_cond_emo[cond].items():
            per_emotion[emo] = round(sum(vals) / len(vals), 3) if vals else None
        summary[cond] = {"mean": mean, "std": std, "n": len(ratings), "per_emotion": per_emotion}
    return summary


def print_summary_table(summary: dict, conditions: list):
    """Print a human-readable comparison table."""
    print(f"\n{'='*60}")
    print("EMOTION-ADAPTATION RATINGS (1–5)  [Neutral-Input Eval]")
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

    # Per-emotion breakdown
    all_emotions = sorted({emo for s in summary.values() for emo in s.get("per_emotion", {})})
    if all_emotions:
        print(f"\n--- Per-emotion means ---")
        header_emo = f"{'Emotion':<14}" + "".join(f" {c[:12]:>13}" for c in conditions)
        print(header_emo)
        print("-" * (14 + 14 * len(conditions)))
        for emo in all_emotions:
            row = f"{emo:<14}"
            for cond in conditions:
                val = summary.get(cond, {}).get("per_emotion", {}).get(emo)
                row += f" {f'{val:.3f}' if val is not None else 'N/A':>13}"
            print(row)
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else manifest_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # For separate manifests, use separate output files to avoid overwriting
    manifest_stem = manifest_path.stem  # e.g. "manifest" or "manifest_opens2s"
    if manifest_stem == "manifest":
        judgments_path = output_dir / "judgments.jsonl"
        summary_path   = output_dir / "summary.json"
    else:
        suffix = manifest_stem[len("manifest"):]  # e.g. "_opens2s"
        judgments_path = output_dir / f"judgments{suffix}.jsonl"
        summary_path   = output_dir / f"summary{suffix}.json"

    # Auto-detect conditions from manifest if not specified
    if args.conditions is None:
        sample_rec = {}
        with open(manifest_path) as f:
            sample_rec = json.loads(f.readline())
        detected = [
            c for c in ALL_CONDITIONS
            if f"{c}_response" in sample_rec and sample_rec[f"{c}_response"] is not None
        ]
        conditions = detected if detected else ALL_CONDITIONS[:3]
        print(f"Auto-detected conditions from manifest: {conditions}")
    else:
        conditions = args.conditions

    print(f"Manifest    : {manifest_path}")
    print(f"Output dir  : {output_dir}")
    print(f"Conditions  : {conditions}")
    print(f"Judge model : {args.model}\n")

    # Load manifest
    records = []
    with open(manifest_path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} manifest records")

    # Load existing judgments for resume
    existing_keys = set()  # (id, condition)
    all_judgments = []
    if args.skip_existing and judgments_path.exists():
        with open(judgments_path) as f:
            for line in f:
                j = json.loads(line)
                all_judgments.append(j)
                existing_keys.add((j["id"], j["condition"]))
        print(f"Resuming: {len(existing_keys)} judgments already done\n")

    # Count work to do
    todo = []
    for rec in records:
        for cond in conditions:
            resp_key = f"{cond}_response"
            audio_path = rec.get(resp_key)
            if audio_path is None:
                print(f"  SKIP {rec['id']} / {cond}: no audio path in manifest")
                continue
            if not Path(audio_path).exists():
                print(f"  SKIP {rec['id']} / {cond}: audio file not found: {audio_path}")
                continue
            if args.skip_existing and (rec["id"], cond) in existing_keys:
                continue
            todo.append((rec, cond, audio_path))

    print(f"Judgments to compute: {len(todo)}")
    if not todo:
        print("Nothing to do. Use --skip-existing to resume, or check your manifest.")
        if all_judgments:
            summary = aggregate_judgments(all_judgments, conditions)
            print_summary_table(summary, conditions)
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
        return

    # Load judge model
    print(f"\nCUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  {torch.cuda.device_count()} GPU(s): {torch.cuda.get_device_name(0)}")
    model, processor = load_judge_model(args.model)

    # Open judgments file for append (enables resuming)
    judgments_file = open(judgments_path, "a")

    try:
        for idx, (rec, cond, audio_path) in enumerate(todo):
            sample_id = rec["id"]
            emotion   = rec["emotion"]
            valence   = rec["valence"]
            arousal   = rec["arousal"]

            print(f"\n[{idx+1}/{len(todo)}] {sample_id} / {cond}  ({emotion})")

            conversation = build_conversation(emotion, valence, arousal, audio_path)

            t0 = time.time()
            try:
                raw_response = run_judge(model, processor, conversation)
            except Exception as e:
                print(f"  ERROR during generation: {e}")
                raw_response = ""

            elapsed = time.time() - t0
            rating, justification = parse_rating(raw_response)

            status = f"rating={rating}" if rating is not None else "PARSE_FAILED"
            print(f"  {status}  ({elapsed:.1f}s)")
            print(f"  Raw: {raw_response[:120]!r}")
            if rating is not None:
                print(f"  Justification: {justification[:120]}")

            judgment = {
                "id":            sample_id,
                "emotion":       emotion,
                "condition":     cond,
                "rating":        rating,
                "justification": justification,
                "raw_response":  raw_response,
            }
            all_judgments.append(judgment)
            judgments_file.write(json.dumps(judgment) + "\n")
            judgments_file.flush()

    finally:
        judgments_file.close()

    # Aggregate and save summary
    summary = aggregate_judgments(all_judgments, conditions)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_path}")

    print_summary_table(summary, conditions)
    print(f"Saved judgments: {judgments_path}")


if __name__ == "__main__":
    main()
