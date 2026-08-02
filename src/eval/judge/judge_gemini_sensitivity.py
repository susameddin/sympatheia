#!/usr/bin/env python3
"""
Independent second judge for the VA-noise SENSITIVITY analysis, using Google Gemini.

Gemini counterpart to eval/judge/judge_sensitivity.py (Qwen3-Omni). It reads the
manifest produced by integration/sensitivity_analysis.py (neutral queries + VA
noise sweep; one response_audio per record, with `sigma`, `noisy_v`, `noisy_a`),
rates emotion-adaptation on 1-5, and aggregates by sigma.

Prompts and user-turn wording are IDENTICAL to the Qwen sensitivity judge — they
are imported from judge_gemini_neutral.build_prompt, whose text was copied
byte-for-byte from judge_qwen3omni_neutral.py, which in turn is identical to
judge_sensitivity.py. Only the judge MODEL differs, keeping the comparison
apples-to-apples.

Outputs (judge-tagged; never overwrites the Qwen results):
  <output-dir>/sensitivity_judgments_gemini.jsonl
  <output-dir>/sensitivity_summary_gemini.json

Requires google-genai and GEMINI_API_KEY (see gemini_backend.py). Run from src/:

    GEMINI_API_KEY=... python -m eval.judge.judge_gemini_sensitivity \\
        --manifest /path/to/eval_sensitivity/manifest.jsonl
"""

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from eval.judge.gemini_backend import GeminiJudge, get_pricing
from eval.judge.judge_common import FatalJudgeError, parse_rating
# Prompts/user-text imported to guarantee byte-identity with the neutral judge
# (== the Qwen sensitivity judge).
from eval.judge.judge_gemini_neutral import build_prompt

DEFAULT_MODEL = "gemini-3.5-flash"


# --- Aggregation by sigma (copied from judge_sensitivity.py) ------------------

def aggregate_judgments(judgments: list) -> dict:
    by_sigma = defaultdict(list)
    by_sigma_emo = defaultdict(lambda: defaultdict(list))
    for j in judgments:
        if j["rating"] is None:
            continue
        by_sigma[j["sigma"]].append(j["rating"])
        by_sigma_emo[j["sigma"]][j["emotion"]].append(j["rating"])
    summary = {}
    for sigma in sorted(by_sigma.keys()):
        ratings = by_sigma[sigma]
        mean = round(sum(ratings) / len(ratings), 3) if ratings else None
        std = round(statistics.stdev(ratings) if len(ratings) > 1 else 0.0, 3)
        per_emotion = {
            emo: round(sum(vals) / len(vals), 3)
            for emo, vals in by_sigma_emo[sigma].items() if vals
        }
        summary[sigma] = {"mean": mean, "std": std, "n": len(ratings), "per_emotion": per_emotion}
    return summary


def print_summary_table(summary: dict):
    print(f"\n{'='*55}")
    print("GEMINI SENSITIVITY — Emotion-Adaptation Score (1–5)")
    print(f"{'='*55}")
    print(f"{'Sigma':>8} {'Mean':>7} {'Std':>7} {'N':>5}")
    print("-" * 30)
    for sigma in sorted(summary.keys()):
        s = summary[sigma]
        mean = f"{s['mean']:.3f}" if s["mean"] is not None else "N/A"
        print(f"{sigma:>8.1f} {mean:>7} {s['std']:>7.3f} {s['n']:>5}")
    all_emotions = sorted({emo for s in summary.values() for emo in s.get("per_emotion", {})})
    if all_emotions:
        sigmas = sorted(summary.keys())
        print(f"\n--- Per-emotion mean by sigma ---")
        print(f"{'Emotion':<14}" + "".join(f" {s:>7.1f}" for s in sigmas))
        print("-" * (14 + 8 * len(sigmas)))
        for emo in all_emotions:
            row = f"{emo:<14}"
            for sigma in sigmas:
                val = summary[sigma].get("per_emotion", {}).get(emo)
                row += f" {f'{val:.3f}' if val is not None else 'N/A':>7}"
            print(row)
    print(f"{'='*55}\n")


def parse_args():
    p = argparse.ArgumentParser(description="Gemini judge for the VA-noise sensitivity analysis")
    p.add_argument("--manifest", required=True, help="manifest.jsonl from integration/sensitivity_analysis.py")
    p.add_argument("--output-dir", default=None, help="Defaults to the manifest's own directory.")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model ID. Default: {DEFAULT_MODEL}")
    p.add_argument("--skip-existing", action="store_true",
                   help="Resume: append only ids not already judged.")
    p.add_argument("--limit", type=int, default=None, help="Judge at most N calls (smoke test).")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-output-tokens", type=int, default=256)
    p.add_argument("--enable-thinking", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)
    output_dir = Path(args.output_dir) if args.output_dir else manifest_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    judgments_path = output_dir / "sensitivity_judgments_gemini.jsonl"
    summary_path = output_dir / "sensitivity_summary_gemini.json"

    # --- Hard no-clobber guard (same policy as the other Gemini judges) -------
    if judgments_path.exists() and not args.skip_existing:
        print(f"ERROR: output already exists: {judgments_path}\n"
              f"       Refusing to overwrite. Pass --skip-existing to resume "
              f"(append only), or move/remove the file first.", file=sys.stderr)
        sys.exit(1)

    print(f"Judge model : {args.model}  (temperature={args.temperature}, "
          f"thinking={'on' if args.enable_thinking else 'off'})")
    print(f"Pricing     : {get_pricing(args.model) or 'unknown'}")
    print(f"Manifest    : {manifest_path}")
    print(f"Judgments   : {judgments_path}")

    records = [json.loads(l) for l in open(manifest_path)]
    print(f"Loaded {len(records)} manifest records")

    existing_ids = set()
    all_judgments = []
    if args.skip_existing and judgments_path.exists():
        with open(judgments_path) as f:
            for line in f:
                j = json.loads(line)
                all_judgments.append(j)
                existing_ids.add(j["id"])
        print(f"Resuming: {len(existing_ids)} judgments already done")

    todo = []
    for rec in records:
        audio_path = rec.get("response_audio")
        if audio_path is None:
            print(f"  SKIP {rec['id']}: no response_audio")
            continue
        if not Path(audio_path).exists():
            print(f"  SKIP {rec['id']}: audio not found: {audio_path}")
            continue
        if args.skip_existing and rec["id"] in existing_ids:
            continue
        todo.append(rec)
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"Judgments to compute: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        if all_judgments:
            summary = aggregate_judgments(all_judgments)
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print_summary_table(summary)
        return

    judge = GeminiJudge(
        model_name=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        disable_thinking=not args.enable_thinking,
    )

    parse_failures = 0
    n_written = 0
    usage_totals = defaultdict(int)
    preexisting = judgments_path.exists()
    judgments_file = open(judgments_path, "a")
    try:
        for idx, rec in enumerate(todo):
            emotion = rec["emotion"]
            # Sensitivity uses the (noisy) stated VA; build_prompt takes (emotion, v, a)
            # and is byte-identical to the Qwen sensitivity judge's wording.
            system_prompt, user_text = build_prompt(emotion, rec["noisy_v"], rec["noisy_a"])

            t0 = time.time()
            try:
                raw_response, usage = judge.judge(system_prompt, user_text, rec["response_audio"])
            except FatalJudgeError:
                raise
            except Exception as e:
                print(f"  ERROR judging {rec['id']}: {e}")
                raw_response, usage = "", {}
            elapsed = time.time() - t0

            rating, justification = parse_rating(raw_response)
            if rating is None:
                parse_failures += 1
            for k, v in (usage or {}).items():
                if isinstance(v, (int, float)):
                    usage_totals[k] += v

            status = f"rating={rating}" if rating is not None else "PARSE_FAILED"
            print(f"[{idx+1}/{len(todo)}] {rec['id']}  ({emotion}, σ={rec['sigma']:.1f}, "
                  f"V={rec['noisy_v']:+.2f}, A={rec['noisy_a']:+.2f})  {status}  ({elapsed:.1f}s)")
            if rating is None:
                print(f"    Raw: {raw_response[:160]!r}")

            judgment = {
                "id": rec["id"], "emotion": emotion, "sigma": rec["sigma"],
                "repeat": rec["repeat"], "anchor_v": rec["anchor_v"], "anchor_a": rec["anchor_a"],
                "noisy_v": rec["noisy_v"], "noisy_a": rec["noisy_a"],
                "rating": rating, "justification": justification,
                "raw_response": raw_response, "judge": "gemini", "usage": usage,
            }
            # Carry through optional provenance fields written by manifest
            # producers that record how the coordinate was derived (absent from
            # the standard sensitivity manifest, in which case this is a no-op).
            for prov in ("arm", "snapped_to", "conditioned_on_v", "conditioned_on_a"):
                if prov in rec:
                    judgment[prov] = rec[prov]
            all_judgments.append(judgment)
            judgments_file.write(json.dumps(judgment) + "\n")
            judgments_file.flush()
            n_written += 1
    finally:
        judgments_file.close()
        if not preexisting and n_written == 0 and judgments_path.exists():
            try:
                judgments_path.unlink()
            except OSError:
                pass

    summary = aggregate_judgments(all_judgments)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary  : {summary_path}")
    print(f"Saved judgments: {judgments_path}")
    print(f"Parse failures : {parse_failures}/{len(todo)}")
    if usage_totals:
        print(f"Token usage    : {dict(usage_totals)}")
    print_summary_table(summary)


if __name__ == "__main__":
    main()
