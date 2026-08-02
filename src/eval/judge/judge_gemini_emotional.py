#!/usr/bin/env python3
"""
Independent second judge for the EMOTIONAL-query setting, using Google Gemini.

This is the Gemini counterpart to judge_qwen3omni_emotional.py. A judge from a
different model family guards against shared model-family bias with the
Qwen-heavy Sympatheia pipeline: Gemini listens to the same generated .wav audio
and rates it on the SAME 1-5 rubric with BYTE-IDENTICAL prompts, so the only
variable changed is the judge model.

Outputs are judge-tagged (judgments_gemini*.jsonl / summary_gemini*.json) and
written next to the manifest, so they can never overwrite the Qwen results.

Requires google-genai and GEMINI_API_KEY (see gemini_backend.py). Run from src/:

    GEMINI_API_KEY=... python -m eval.judge.judge_gemini_emotional \\
        --manifest /path/to/eval_emotional_.../manifest.jsonl \\
        --conditions finetuned_va

    # qwen3omni lives in its own manifest:
    GEMINI_API_KEY=... python -m eval.judge.judge_gemini_emotional \\
        --manifest /path/to/manifest_qwen3omni.jsonl --conditions qwen3omni

    # Smoke test (5 calls):
    GEMINI_API_KEY=... python -m eval.judge.judge_gemini_emotional \\
        --manifest /path/to/manifest.jsonl --conditions finetuned_va --limit 5
"""

import argparse
import sys
from pathlib import Path

import eval.judge.judge_common as jc
from eval.judge.gemini_backend import GeminiJudge, get_pricing

DEFAULT_MODEL = "gemini-3.5-flash"

# --- Prompts: copied BYTE-FOR-BYTE from judge_qwen3omni_emotional.py ---------

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator of empathetic conversational AI. Your task is to \
rate how well a voice assistant's spoken response supports and empathizes with \
the user, given the user's emotional state.

Rate the response on a scale from 1 to 5 using this rubric:
  1 — Completely ignores or contradicts the user's emotion
  2 — Minimal acknowledgment, mostly generic or off-topic
  3 — Some empathy shown but inconsistent or superficial
  4 — Clear empathy and support, appropriate to the emotion
  5 — Excellent empathy: acknowledges emotion, validates, offers meaningful support

Respond with ONLY the rating number (1–5) followed by a vertical bar and a \
brief one-sentence justification.
Format exactly: <rating>|<justification>
Example: 4|The response acknowledges the user's frustration and offers calm, supportive guidance.\
"""

JUDGE_SYSTEM_PROMPT_NEUTRAL = """\
You are an expert evaluator of conversational AI quality. Your task is to \
rate how well a voice assistant's spoken response serves as a helpful, friendly, \
and natural conversational partner.

IMPORTANT CONTEXT: The user's emotional state is NEUTRAL. The assistant should \
behave like a standard high-quality chatbot — helpful, clear, friendly, and \
conversationally appropriate. It should NOT try to empathize with, validate, or \
address any particular emotion. A good response simply answers the user's query \
in a natural, pleasant manner.

Rate the response on a scale from 1 to 5 using this rubric:
  1 — Unhelpful, incoherent, or inappropriate response
  2 — Somewhat relevant but unclear, awkward, or overly terse
  3 — Adequate response; addresses the query but lacks polish or warmth
  4 — Good response: clear, helpful, friendly, and conversationally natural
  5 — Excellent response: highly helpful, naturally friendly, well-structured, \
and pleasant to listen to

Respond with ONLY the rating number (1–5) followed by a vertical bar and a \
brief one-sentence justification.
Format exactly: <rating>|<justification>
Example: 4|The response clearly and warmly answers the user's question with relevant information.\
"""


def build_prompt(emotion: str, valence: float, arousal: float):
    """(system_prompt, user_text) — user-turn wording copied from
    judge_qwen3omni_emotional.build_conversation."""
    if emotion.lower() == "neutral":
        system_prompt = JUDGE_SYSTEM_PROMPT_NEUTRAL
        user_text = (
            "The user's emotional state is neutral. "
            "Listen to the voice assistant's response below and rate its quality "
            "as a helpful, friendly chatbot response."
        )
    else:
        system_prompt = JUDGE_SYSTEM_PROMPT
        user_text = (
            f"The user is feeling {emotion} "
            f"(valence={valence:.2f}, arousal={arousal:.2f}). "
            "Listen to the voice assistant's response below and rate its empathetic quality."
        )
    return system_prompt, user_text


def parse_args():
    p = argparse.ArgumentParser(
        description="Judge audio responses for empathetic quality with Gemini (emotional-query eval)"
    )
    p.add_argument("--manifest", required=True, help="Path to manifest.jsonl")
    p.add_argument("--output-dir", default=None,
                   help="Defaults to the manifest's own directory.")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model ID. Default: {DEFAULT_MODEL}")
    p.add_argument("--conditions", nargs="+", default=None,
                   choices=jc.ALL_CONDITIONS_EMOTIONAL,
                   help="Which conditions to judge. Default: auto-detect from manifest.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Resume: append only (id, condition) pairs not already judged.")
    p.add_argument("--limit", type=int, default=None, help="Judge at most N calls (smoke test).")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-output-tokens", type=int, default=256)
    p.add_argument("--enable-thinking", action="store_true",
                   help="Allow Gemini thinking (default: disabled to cut cost/latency).")
    return p.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)
    output_dir = Path(args.output_dir) if args.output_dir else manifest_path.parent
    conditions = args.conditions or jc.detect_conditions(manifest_path, jc.ALL_CONDITIONS_EMOTIONAL)

    print(f"Judge model : {args.model}  (temperature={args.temperature}, "
          f"thinking={'on' if args.enable_thinking else 'off'})")
    pricing = get_pricing(args.model)
    print(f"Pricing     : {pricing if pricing else 'unknown (cost printout disabled)'}")

    judge = GeminiJudge(
        model_name=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        disable_thinking=not args.enable_thinking,
    )

    jc.run_judge_over_manifest(
        manifest_path=manifest_path,
        output_dir=output_dir,
        conditions=conditions,
        all_conditions=jc.ALL_CONDITIONS_EMOTIONAL,
        build_prompt=build_prompt,
        judge_fn=judge.judge,
        judge_tag="gemini",
        title=f"GEMINI ({args.model}) EMPATHY RATINGS (1–5)",
        skip_existing=args.skip_existing,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
