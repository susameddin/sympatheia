#!/usr/bin/env python3
"""
Judge sensitivity analysis responses for emotion-adaptation quality using Qwen3-Omni.

Reads a manifest produced by integration/sensitivity_analysis.py (neutral queries,
VA noise sweep). Each record has one response audio. Rates how well the model adapts
to the stated (noisy) VA on a 1–5 scale, then aggregates scores by sigma level.

Outputs:
  <output-dir>/sensitivity_judgments.jsonl  — one record per sample
  <output-dir>/sensitivity_summary.json     — mean/std grouped by sigma (and emotion×sigma)
  Prints a sigma vs. mean-score table to stdout.

Usage:
    # Note: run this in the Qwen3-Omni environment (see https://github.com/QwenLM/Qwen3)
    python -m eval.judge_sensitivity \\
        --manifest eval/eval_sensitivity/manifest.jsonl

    # Resume interrupted run:
    # Note: run this in the Qwen3-Omni environment (see https://github.com/QwenLM/Qwen3)
    python -m eval.judge_sensitivity \\
        --manifest eval/eval_sensitivity/manifest.jsonl \\
        --skip-existing
"""

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "Qwen/Qwen3-Omni"

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

JUDGE_SYSTEM_PROMPT_NEUTRAL = """\
You are an expert evaluator of conversational AI quality. Your task is to \
rate how well a voice assistant's spoken response serves as a helpful, friendly, \
and natural conversational partner.

IMPORTANT CONTEXT: The user's emotional state is NEUTRAL, and the model was \
correctly told the user is neutral. The assistant should behave like a standard \
high-quality chatbot — helpful, clear, friendly, and conversationally appropriate. \
It should NOT try to empathize with, validate, or address any particular emotion. \
A good response simply answers the user's query in a natural, pleasant manner.

Rate the response on a scale from 1 to 5 using this rubric:
  1 — Unhelpful, incoherent, or inappropriate response
  2 — Somewhat relevant but unclear, awkward, or overly terse
  3 — Adequate response; addresses the query but lacks polish or warmth
  4 — Good response: clear, helpful, friendly, and conversationally natural
  5 — Excellent response: highly helpful, naturally friendly, well-structured, \
and pleasant to listen to

Respond with ONLY the rating number (1-5) followed by a vertical bar and a \
brief one-sentence justification.
Format exactly: <rating>|<justification>
Example: 4|The response clearly and warmly answers the user's question with relevant information.\
"""


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Judge sensitivity analysis responses for emotion-adaptation quality"
    )
    parser.add_argument(
        "--manifest", type=str, required=True,
        help="Path to manifest.jsonl produced by integration/sensitivity_analysis.py",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory. Defaults to the same directory as --manifest.",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Qwen3-Omni model path. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip samples already present in sensitivity_judgments.jsonl",
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


def build_conversation(emotion: str, noisy_v: float, noisy_a: float, audio_path: str) -> list:
    """Build the conversation dict for Qwen3-Omni to judge one sensitivity response."""
    if emotion.lower() == "neutral":
        system_prompt = JUDGE_SYSTEM_PROMPT_NEUTRAL
        user_text = (
            "The model was told the user is feeling neutral "
            "(valence=0.00, arousal=0.00), and the user's actual audio was "
            "also emotionally neutral. "
            "Listen to the voice assistant's response below and rate its quality "
            "as a helpful, friendly chatbot response."
        )
    else:
        system_prompt = JUDGE_SYSTEM_PROMPT
        user_text = (
            f"The model was told the user is feeling {emotion} "
            f"(valence={noisy_v:.2f}, arousal={noisy_a:.2f}), "
            "but the user's actual audio was emotionally neutral. "
            "Listen to the voice assistant's response below and rate how well "
            "it adapts to the stated emotion."
        )
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
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


def parse_rating(raw_response: str) -> tuple:
    m = re.search(r"([1-5])\s*\|(.+)", raw_response, re.DOTALL)
    if m:
        return int(m.group(1)), m.group(2).strip()
    m = re.search(r"\b([1-5])\b", raw_response)
    if m:
        return int(m.group(1)), raw_response.strip()
    return None, raw_response.strip()


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def aggregate_judgments(judgments: list) -> dict:
    by_sigma = defaultdict(list)
    by_sigma_emo = defaultdict(lambda: defaultdict(list))

    for j in judgments:
        if j["rating"] is None:
            continue
        sigma = j["sigma"]
        emo = j["emotion"]
        r = j["rating"]
        by_sigma[sigma].append(r)
        by_sigma_emo[sigma][emo].append(r)

    summary = {}
    for sigma in sorted(by_sigma.keys()):
        ratings = by_sigma[sigma]
        mean = round(sum(ratings) / len(ratings), 3) if ratings else None
        std  = round(statistics.stdev(ratings) if len(ratings) > 1 else 0.0, 3)
        per_emotion = {
            emo: round(sum(vals) / len(vals), 3)
            for emo, vals in by_sigma_emo[sigma].items() if vals
        }
        summary[sigma] = {"mean": mean, "std": std, "n": len(ratings), "per_emotion": per_emotion}

    return summary


def print_summary_table(summary: dict):
    print(f"\n{'='*55}")
    print("SENSITIVITY ANALYSIS — Emotion-Adaptation Score (1–5)")
    print(f"{'='*55}")
    print(f"{'Sigma':>8} {'Mean':>7} {'Std':>7} {'N':>5}")
    print("-" * 30)
    for sigma in sorted(summary.keys()):
        s = summary[sigma]
        mean = f"{s['mean']:.3f}" if s["mean"] is not None else "N/A"
        std  = f"{s['std']:.3f}"
        print(f"{sigma:>8.1f} {mean:>7} {std:>7} {s['n']:>5}")

    all_emotions = sorted({emo for s in summary.values() for emo in s.get("per_emotion", {})})
    if all_emotions:
        sigmas = sorted(summary.keys())
        print(f"\n--- Per-emotion mean by sigma ---")
        header = f"{'Emotion':<14}" + "".join(f" {s:>7.1f}" for s in sigmas)
        print(header)
        print("-" * (14 + 8 * len(sigmas)))
        for emo in all_emotions:
            row = f"{emo:<14}"
            for sigma in sigmas:
                val = summary[sigma].get("per_emotion", {}).get(emo)
                row += f" {f'{val:.3f}' if val is not None else 'N/A':>7}"
            print(row)
    print(f"{'='*55}\n")


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

    judgments_path = output_dir / "sensitivity_judgments.jsonl"
    summary_path   = output_dir / "sensitivity_summary.json"

    print(f"Manifest    : {manifest_path}")
    print(f"Output dir  : {output_dir}")
    print(f"Judge model : {args.model}\n")

    records = []
    with open(manifest_path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} manifest records")

    # Load existing judgments for resume
    existing_ids = set()
    all_judgments = []
    if args.skip_existing and judgments_path.exists():
        with open(judgments_path) as f:
            for line in f:
                j = json.loads(line)
                all_judgments.append(j)
                existing_ids.add(j["id"])
        print(f"Resuming: {len(existing_ids)} judgments already done\n")

    todo = []
    for rec in records:
        audio_path = rec.get("response_audio")
        if audio_path is None:
            print(f"  SKIP {rec['id']}: no response_audio in manifest")
            continue
        if not Path(audio_path).exists():
            print(f"  SKIP {rec['id']}: audio file not found: {audio_path}")
            continue
        if args.skip_existing and rec["id"] in existing_ids:
            continue
        todo.append(rec)

    print(f"Judgments to compute: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        if all_judgments:
            summary = aggregate_judgments(all_judgments)
            print_summary_table(summary)
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
        return

    print(f"\nCUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  {torch.cuda.device_count()} GPU(s): {torch.cuda.get_device_name(0)}")
    model, processor = load_judge_model(args.model)

    judgments_file = open(judgments_path, "a")
    try:
        for idx, rec in enumerate(todo):
            sample_id = rec["id"]
            emotion   = rec["emotion"]
            sigma     = rec["sigma"]
            noisy_v   = rec["noisy_v"]
            noisy_a   = rec["noisy_a"]
            audio_path = rec["response_audio"]

            print(f"\n[{idx+1}/{len(todo)}] {sample_id}  ({emotion}, σ={sigma:.1f}, V={noisy_v:+.2f}, A={noisy_a:+.2f})")

            conversation = build_conversation(emotion, noisy_v, noisy_a, audio_path)

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
                "sigma":         sigma,
                "repeat":        rec["repeat"],
                "anchor_v":      rec["anchor_v"],
                "anchor_a":      rec["anchor_a"],
                "noisy_v":       noisy_v,
                "noisy_a":       noisy_a,
                "rating":        rating,
                "justification": justification,
                "raw_response":  raw_response,
            }
            all_judgments.append(judgment)
            judgments_file.write(json.dumps(judgment) + "\n")
            judgments_file.flush()
    finally:
        judgments_file.close()

    summary = aggregate_judgments(all_judgments)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary : {summary_path}")
    print(f"Saved judgments: {judgments_path}")
    print_summary_table(summary)


if __name__ == "__main__":
    main()
