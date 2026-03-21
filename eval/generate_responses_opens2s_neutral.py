#!/usr/bin/env python3
"""
Generate audio responses from OpenS2S for neutral-input emotion evaluation.

Unlike generate_responses_opens2s.py (which uses a static system prompt),
this script injects the emotion label into the system prompt per sample so
OpenS2S knows the user's stated emotion — even though the audio is neutral.

Reads manifest.jsonl produced by eval/generate_responses_neutral.py.
Delegates inference to Models/OpenS2S/OpenS2S/batch_infer_eval_neutral.py
(which accepts per-job system_prompt).

Usage:
    conda run -n opens2s2 python -m eval.generate_responses_opens2s_neutral \\
        --manifest results/eval_neutral/manifest.jsonl

    # Resume:
    conda run -n opens2s2 python -m eval.generate_responses_opens2s_neutral \\
        --manifest results/eval_neutral/manifest.jsonl \\
        --skip-existing
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OPENS2S_DIR = "/home/sd3705/emo_recog_2025s/Models/OpenS2S/OpenS2S"
DEFAULT_MODEL_PATH  = "/home/sd3705/emo_recog_2025s/Models/OpenS2S/weights"
DEFAULT_FLOW_PATH   = "/home/sd3705/emo_recog_2025s/Models/OpenS2S/weights/glm-4-voice-decoder"


def get_system_prompt(emotion: str) -> str:
    """Build an emotion-conditioned system prompt for OpenS2S."""
    return (
        f"You are a helpful assistant. The user is currently feeling {emotion.lower()}. "
        f"Respond with appropriate empathy and support for their emotional state."
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate OpenS2S responses for neutral-input emotion evaluation"
    )
    parser.add_argument(
        "--manifest", type=str, required=True,
        help="Path to manifest.jsonl from eval/generate_responses_neutral.py "
             "(used to select the same query audio files)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory. Defaults to the same directory as --manifest.",
    )
    parser.add_argument(
        "--opens2s-dir", type=str, default=DEFAULT_OPENS2S_DIR,
        help=f"OpenS2S source directory (must contain batch_infer_eval_neutral.py). "
             f"Default: {DEFAULT_OPENS2S_DIR}",
    )
    parser.add_argument(
        "--model-path", type=str, default=DEFAULT_MODEL_PATH,
        help=f"OpenS2S model weights directory. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--flow-path", type=str, default=DEFAULT_FLOW_PATH,
        help=f"OpenS2S audio decoder directory. Default: {DEFAULT_FLOW_PATH}",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip samples whose output audio already exists (enables resuming)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    opens2s_dir = Path(args.opens2s_dir)
    batch_script = opens2s_dir / "batch_infer_eval_neutral.py"
    if not batch_script.exists():
        print(f"ERROR: batch_infer_eval_neutral.py not found at {batch_script}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else manifest_path.parent
    audio_dir  = output_dir / "audio" / "opens2s"
    audio_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = output_dir / "manifest_opens2s.jsonl"

    print(f"Manifest (input)  : {manifest_path}")
    print(f"Output dir        : {output_dir}")
    print(f"Audio output dir  : {audio_dir}")
    print(f"Out manifest      : {out_manifest}")
    print(f"OpenS2S dir       : {opens2s_dir}")
    print(f"Skip existing     : {args.skip_existing}\n")

    # Load source manifest
    records = []
    with open(manifest_path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records from manifest")

    # Load existing output manifest for resume
    existing_ids = set()
    out_records_map = {}
    if args.skip_existing and out_manifest.exists():
        with open(out_manifest) as f:
            for line in f:
                r = json.loads(line)
                existing_ids.add(r["id"])
                out_records_map[r["id"]] = r
        print(f"Resuming: {len(existing_ids)} already done\n")

    # Determine which samples need generation
    todo = []
    for rec in records:
        sample_id = rec["id"]
        out_wav = audio_dir / f"{sample_id}.wav"
        if args.skip_existing and sample_id in existing_ids and out_wav.exists():
            continue
        emotion = rec.get("emotion", "Neutral")
        todo.append({
            "id":            sample_id,
            "query_audio":   rec["query_audio"],
            "output_audio":  str(out_wav.resolve()),
            "system_prompt": get_system_prompt(emotion),
        })

    print(f"Samples to generate: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        # Still write/update manifest from existing records
    else:
        # Write jobs file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_jobs.jsonl", delete=False, dir=output_dir
        ) as jf:
            jobs_file = jf.name
            for job in todo:
                jf.write(json.dumps(job) + "\n")

        results_file = jobs_file.replace("_jobs.jsonl", "_results.jsonl")

        print(f"\nLaunching batch_infer_eval_neutral.py  (cwd: {opens2s_dir})")
        print(f"  jobs    : {jobs_file}")
        print(f"  results : {results_file}\n")

        try:
            subprocess.run(
                [
                    sys.executable,
                    str(batch_script),
                    "--jobs",       jobs_file,
                    "--results",    results_file,
                    "--model-path", args.model_path,
                    "--flow-path",  args.flow_path,
                ],
                check=True,
                cwd=str(opens2s_dir),   # Run from OpenS2S dir so src/ imports work
            )
        except subprocess.CalledProcessError as e:
            print(f"\nERROR: batch_infer_eval_neutral.py exited with code {e.returncode}", file=sys.stderr)
            sys.exit(1)
        finally:
            # Clean up jobs file
            try:
                os.unlink(jobs_file)
            except OSError:
                pass

        # Read results
        if Path(results_file).exists():
            with open(results_file) as rf:
                for line in rf:
                    r = json.loads(line)
                    if r.get("ok"):
                        out_records_map[r["id"]] = r
            try:
                os.unlink(results_file)
            except OSError:
                pass

    # Write output manifest (all records, both existing and new)
    print(f"\nWriting manifest: {out_manifest}")
    with open(out_manifest, "w") as f:
        for rec in records:
            sample_id = rec["id"]
            out_wav = audio_dir / f"{sample_id}.wav"
            result  = out_records_map.get(sample_id, {})
            out_rec = {
                "id":               sample_id,
                "emotion":          rec["emotion"],
                "valence":          rec["valence"],
                "arousal":          rec["arousal"],
                "query_audio":      rec["query_audio"],
                "opens2s_response": str(out_wav.resolve()) if out_wav.exists() else None,
                "opens2s_text":     result.get("text"),
            }
            f.write(json.dumps(out_rec) + "\n")

    ok    = sum(1 for rec in records if (audio_dir / f"{rec['id']}.wav").exists())
    total = len(records)
    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Generated : {ok}/{total} samples")
    print(f"  Manifest  : {out_manifest}")
    print(f"\nNext step:")
    print(f"  conda run -n qwen3omni python -m eval.judge_qwen3omni_neutral \\")
    print(f"      --manifest {out_manifest.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
