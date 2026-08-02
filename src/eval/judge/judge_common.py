#!/usr/bin/env python3
"""
Shared, backend-agnostic mechanics for the LLM-as-a-judge scripts.

This module factors out the parts that are identical across judge backends
(rating parsing, aggregation, summary tables, manifest loading, the main
judging loop with resume + no-clobber safety) so a second judge (e.g. Gemini)
can reuse them WITHOUT touching the working Qwen3-Omni scripts.

Design notes:
  * The setting-specific *prompt strings* and the exact user-turn wording live in
    each concrete judge script rather than here. The emotional-query and
    neutral-query settings use genuinely different rubrics, and within a setting
    the Gemini script's copy is byte-identical to the Qwen3-Omni script's, so the
    only variable being tested is the judge model. Any edit to a rubric must be
    applied to both scripts of that setting.
  * A backend supplies a single callable ``judge_fn(system_prompt, user_text,
    audio_path) -> (raw_text, usage_dict)``.  The driver here handles everything
    else.
  * Outputs are written with a ``judge_tag`` (e.g. "gemini") baked into the
    filename so they can never overwrite the Qwen ``judgments*.jsonl`` /
    ``summary*.json`` files.
"""

import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path


class FatalJudgeError(Exception):
    """Non-retryable backend error (bad key, billing, permission). The driver
    re-raises this to abort the run instead of writing empty judgments."""


# Single source of truth for the condition lists. Every judge script imports
# from here — they were previously duplicated per script and drifted, so a new
# condition had to be registered in four places to be picked up.
ALL_CONDITIONS_EMOTIONAL = ["base", "finetuned_va", "finetuned_discrete", "finetuned_discrete_na", "finetuned_na", "opens2s", "osum_no_think", "osum_think", "qwen3omni", "qwen2_5omni", "kimiaudio", "face_va", "text_va", "no_va", "oracle", "eeg_only", "eye_only", "combined", "ecg_va", "gsr_va", "fusion_va", "yaad_va", "qwen3tts_cascaded", "face_image_qwen3omni", "text_qwen3omni"]
ALL_CONDITIONS_NEUTRAL = ["base", "finetuned_va", "finetuned_discrete", "finetuned_discrete_na", "finetuned_na", "opens2s", "osum_neutral_no_think", "osum_neutral_think", "qwen3omni", "qwen2_5omni", "kimiaudio", "face_va", "text_va", "no_va", "oracle", "eeg_only", "eye_only", "combined", "ecg_va", "gsr_va", "fusion_va", "yaad_va", "qwen3tts_cascaded", "face_image_qwen3omni", "text_qwen3omni"]


# ---------------------------------------------------------------------------
# Rating parsing  (copied verbatim from judge_qwen3omni_emotional.py)
# ---------------------------------------------------------------------------

def parse_rating(raw_response: str):
    """Extract (rating, justification) from the model's raw response.

    Expected format: "<digit>|<justification>"
    Falls back to searching for any 1-5 digit in the response.
    """
    # Primary: digit immediately followed by '|'
    m = re.search(r"([1-5])\s*\|(.+)", raw_response, re.DOTALL)
    if m:
        rating = int(m.group(1))
        justification = m.group(2).strip()
        return rating, justification

    # Fallback: any standalone 1-5 digit
    m = re.search(r"\b([1-5])\b", raw_response)
    if m:
        return int(m.group(1)), raw_response.strip()

    return None, raw_response.strip()


# ---------------------------------------------------------------------------
# Aggregation and reporting  (copied verbatim from judge_qwen3omni_emotional.py)
# ---------------------------------------------------------------------------

def aggregate_judgments(judgments: list, conditions: list) -> dict:
    """Compute mean/std per condition overall and per emotion."""
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
        ratings_excl = [r for emo, vals in by_cond_emo[cond].items()
                        if emo.lower() != "neutral" for r in vals]
        mean_excl = round(sum(ratings_excl) / len(ratings_excl), 3) if ratings_excl else None
        summary[cond] = {"mean": mean, "std": std, "n": len(ratings),
                         "mean_excl_neutral": mean_excl, "per_emotion": per_emotion}
    return summary


def print_summary_table(summary: dict, conditions: list, title: str):
    """Print a human-readable comparison table."""
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")
    header = f"{'Condition':<20} {'Mean':>6} {'ExclN':>6} {'Std':>6} {'N':>5}"
    print(header)
    print("-" * 46)
    for cond in conditions:
        s = summary.get(cond, {})
        mean = f"{s['mean']:.3f}" if s.get("mean") is not None else "N/A"
        mean_excl = f"{s['mean_excl_neutral']:.3f}" if s.get("mean_excl_neutral") is not None else "N/A"
        std  = f"{s['std']:.3f}"  if s.get("std")  is not None else "N/A"
        n    = str(s.get("n", 0))
        print(f"{cond:<20} {mean:>6} {mean_excl:>6} {std:>6} {n:>5}")

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
# Output-path derivation  (judge-tagged so it never clobbers Qwen outputs)
# ---------------------------------------------------------------------------

def derive_output_paths(manifest_path: Path, output_dir: Path, judge_tag: str):
    """Map a manifest to judge-tagged output filenames.

    manifest.jsonl           -> judgments_<tag>.jsonl / summary_<tag>.json
    manifest_qwen3omni.jsonl -> judgments_<tag>_qwen3omni.jsonl / summary_<tag>_qwen3omni.json
    """
    stem = manifest_path.stem
    if stem == "manifest":
        suffix = ""
    elif stem.startswith("manifest"):
        suffix = stem[len("manifest"):]  # e.g. "_qwen3omni"
    else:
        suffix = "_" + stem
    judgments_path = output_dir / f"judgments_{judge_tag}{suffix}.jsonl"
    summary_path = output_dir / f"summary_{judge_tag}{suffix}.json"
    return judgments_path, summary_path


def detect_conditions(manifest_path: Path, all_conditions: list) -> list:
    """Auto-detect which conditions are present in the manifest."""
    all_keys = set()
    with open(manifest_path) as f:
        for line in f:
            all_keys.update(json.loads(line).keys())
    detected = [c for c in all_conditions if f"{c}_response" in all_keys]
    return detected if detected else all_conditions[:3]


# ---------------------------------------------------------------------------
# Backend-agnostic judging driver
# ---------------------------------------------------------------------------

def run_judge_over_manifest(
    *,
    manifest_path: Path,
    output_dir: Path,
    conditions: list,
    all_conditions: list,
    build_prompt,        # (emotion, valence, arousal) -> (system_prompt, user_text)
    judge_fn,            # (system_prompt, user_text, audio_path) -> (raw_text, usage_dict)
    judge_tag: str,
    title: str,
    skip_existing: bool = False,
    limit: int | None = None,
):
    """Shared main loop: load manifest, resume/no-clobber, judge, write outputs.

    Returns the aggregated summary dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    judgments_path, summary_path = derive_output_paths(manifest_path, output_dir, judge_tag)

    # --- Hard no-clobber guard --------------------------------------------
    # Never touch an existing output file unless the user explicitly asked to
    # resume (--skip-existing appends only the missing (id, condition) pairs).
    if judgments_path.exists() and not skip_existing:
        print(
            f"ERROR: output already exists: {judgments_path}\n"
            f"       Refusing to overwrite. Pass --skip-existing to resume "
            f"(append only), or move/remove the file first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Manifest    : {manifest_path}")
    print(f"Output dir  : {output_dir}")
    print(f"Judgments   : {judgments_path}")
    print(f"Summary     : {summary_path}")
    print(f"Conditions  : {conditions}")

    records = []
    with open(manifest_path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} manifest records")

    existing_keys = set()      # (id, condition) already judged
    all_judgments = []
    if skip_existing and judgments_path.exists():
        with open(judgments_path) as f:
            for line in f:
                j = json.loads(line)
                all_judgments.append(j)
                existing_keys.add((j["id"], j["condition"]))
        print(f"Resuming: {len(existing_keys)} judgments already done")

    # Build the work list.
    todo = []
    for rec in records:
        for cond in conditions:
            audio_path = rec.get(f"{cond}_response")
            if audio_path is None:
                continue
            if not Path(audio_path).exists():
                print(f"  SKIP {rec['id']} / {cond}: audio not found: {audio_path}")
                continue
            if skip_existing and (rec["id"], cond) in existing_keys:
                continue
            todo.append((rec, cond, audio_path))

    if limit is not None:
        todo = todo[:limit]
    print(f"Judgments to compute: {len(todo)}")

    if not todo:
        print("Nothing to do.")
        if all_judgments:
            summary = aggregate_judgments(all_judgments, conditions)
            print_summary_table(summary, conditions, title)
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
        return None

    parse_failures = 0
    n_written = 0
    usage_totals = defaultdict(int)
    preexisting = judgments_path.exists()
    judgments_file = open(judgments_path, "a")
    try:
        for idx, (rec, cond, audio_path) in enumerate(todo):
            sample_id = rec["id"]
            emotion = rec.get("emotion") or rec.get("emotion_gt", "unknown")
            valence = rec.get("valence", 0.0)
            arousal = rec.get("arousal", 0.0)

            system_prompt, user_text = build_prompt(emotion, valence, arousal)

            t0 = time.time()
            try:
                raw_response, usage = judge_fn(system_prompt, user_text, audio_path)
            except FatalJudgeError:
                raise  # abort the whole run; cleanup happens in finally
            except Exception as e:
                print(f"  ERROR during judging {sample_id}/{cond}: {e}")
                raw_response, usage = "", {}
            elapsed = time.time() - t0

            rating, justification = parse_rating(raw_response)
            if rating is None:
                parse_failures += 1
            for k, v in (usage or {}).items():
                if isinstance(v, (int, float)):
                    usage_totals[k] += v

            status = f"rating={rating}" if rating is not None else "PARSE_FAILED"
            print(f"[{idx+1}/{len(todo)}] {sample_id} / {cond} ({emotion})  "
                  f"{status}  ({elapsed:.1f}s)")
            if rating is None:
                print(f"    Raw: {raw_response[:160]!r}")

            judgment = {
                "id": sample_id,
                "emotion": emotion,
                "condition": cond,
                "rating": rating,
                "justification": justification,
                "raw_response": raw_response,
                "judge": judge_tag,
                "usage": usage,
            }
            all_judgments.append(judgment)
            judgments_file.write(json.dumps(judgment) + "\n")
            judgments_file.flush()
            n_written += 1
    finally:
        judgments_file.close()
        # If we newly created the file but wrote nothing (e.g. fatal abort on the
        # very first call), remove the empty artifact so a clean re-run doesn't
        # trip the no-clobber guard.
        if not preexisting and n_written == 0 and judgments_path.exists():
            try:
                judgments_path.unlink()
            except OSError:
                pass

    summary = aggregate_judgments(all_judgments, conditions)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary  : {summary_path}")
    print(f"Saved judgments: {judgments_path}")
    print(f"Parse failures : {parse_failures}/{len(todo)}")
    if usage_totals:
        print(f"Token usage    : {dict(usage_totals)}")

    print_summary_table(summary, conditions, title)
    return summary
