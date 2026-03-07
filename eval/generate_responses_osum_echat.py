#!/usr/bin/env python3
"""
Generate audio responses from OSUM-EChat for empathy evaluation.

Reads the manifest.jsonl produced by eval/generate_responses.py to use the
exact same query audio files (ensuring a fair comparison). Generates responses
for two OSUM-EChat conditions:
  osum_no_think — S2S without chain-of-thought reasoning
  osum_think    — S2S with linguistic-paralinguistic dual thinking

Saves responses to audio/osum_no_think/ and audio/osum_think/ and writes
manifest_osum.jsonl — nothing in the original manifest or audio directories
is touched.

Inference is delegated to Models/OSUM-EChat/OSUM/OSUM-EChat/batch_infer_eval_osum.py
via subprocess (with that directory as CWD), which avoids sys.path conflicts
between this project's imports and OSUM-EChat's wenet/gxl_ai_utils imports.

Usage:
    conda run -n OSUM-EChat python -m eval.generate_responses_osum_echat \\
        --manifest /engram/naplab/users/sd3705/emo_recog_2025s/eval/manifest.jsonl

    # Resume:
    conda run -n OSUM-EChat python -m eval.generate_responses_osum_echat \\
        --manifest /engram/naplab/users/sd3705/emo_recog_2025s/eval/manifest.jsonl \\
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

DEFAULT_OSUM_DIR = "/home/sd3705/emo_recog_2025s/Models/OSUM-EChat/OSUM/OSUM-EChat"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate OSUM-EChat responses (no_think + think) for empathy evaluation"
    )
    parser.add_argument(
        "--manifest", type=str, required=True,
        help="Path to manifest.jsonl from eval/generate_responses.py "
             "(used to select the same query audio files)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory. Defaults to the same directory as --manifest.",
    )
    parser.add_argument(
        "--osum-dir", type=str, default=DEFAULT_OSUM_DIR,
        help=f"OSUM-EChat source directory (must contain batch_infer_eval_osum.py). "
             f"Default: {DEFAULT_OSUM_DIR}",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip samples whose output audio already exists (enables resuming)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_mode(mode, todo, audio_dir, osum_dir, output_dir):
    """Run batch inference for one mode (no_think or think) via subprocess."""
    if not todo:
        print(f"\n  All {mode} responses already exist — skipping")
        return {}

    batch_script = Path(osum_dir) / "batch_infer_eval_osum.py"
    if not batch_script.exists():
        print(f"ERROR: batch_infer_eval_osum.py not found at {batch_script}", file=sys.stderr)
        sys.exit(1)

    print(f"\nLaunching batch_infer_eval_osum.py  mode={mode}  (cwd: {osum_dir})")
    print(f"  samples: {len(todo)}")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=f"_{mode}_jobs.jsonl", delete=False, dir=output_dir
    ) as jf:
        jobs_file = jf.name
        for job in todo:
            jf.write(json.dumps(job) + "\n")

    results_file = jobs_file.replace(f"_{mode}_jobs.jsonl", f"_{mode}_results.jsonl")

    try:
        subprocess.run(
            [
                sys.executable,
                str(batch_script),
                "--jobs",    jobs_file,
                "--results", results_file,
                "--mode",    mode.removeprefix("osum_"),
            ],
            check=True,
            cwd=str(osum_dir),
        )
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: batch_infer_eval_osum.py [{mode}] exited with code {e.returncode}",
              file=sys.stderr)
        sys.exit(1)
    finally:
        try:
            os.unlink(jobs_file)
        except OSError:
            pass

    results_map = {}
    if Path(results_file).exists():
        with open(results_file) as rf:
            for line in rf:
                r = json.loads(line)
                if r.get("ok"):
                    results_map[r["id"]] = r
        try:
            os.unlink(results_file)
        except OSError:
            pass

    return results_map


def main():
    args = parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    osum_dir = Path(args.osum_dir)

    output_dir = Path(args.output_dir) if args.output_dir else manifest_path.parent
    audio_dirs = {
        "osum_no_think": output_dir / "audio" / "osum_no_think",
        "osum_think":    output_dir / "audio" / "osum_think",
    }
    for d in audio_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    out_manifest = output_dir / "manifest_osum.jsonl"

    print(f"Manifest (input)  : {manifest_path}")
    print(f"Output dir        : {output_dir}")
    print(f"Out manifest      : {out_manifest}")
    print(f"OSUM-EChat dir    : {osum_dir}")
    print(f"Skip existing     : {args.skip_existing}\n")

    # Load source manifest
    records = []
    with open(manifest_path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records from manifest")

    # Load existing output manifest for resume
    existing_ids = {mode: set() for mode in ["osum_no_think", "osum_think"]}
    out_records_map = {}
    if args.skip_existing and out_manifest.exists():
        with open(out_manifest) as f:
            for line in f:
                r = json.loads(line)
                out_records_map[r["id"]] = r
                if r.get("osum_no_think_response") and Path(r["osum_no_think_response"]).exists():
                    existing_ids["osum_no_think"].add(r["id"])
                if r.get("osum_think_response") and Path(r["osum_think_response"]).exists():
                    existing_ids["osum_think"].add(r["id"])
        done_no_think = len(existing_ids["osum_no_think"])
        done_think    = len(existing_ids["osum_think"])
        print(f"Resuming: {done_no_think} no_think, {done_think} think already done\n")

    # Build job lists per mode
    todos = {}
    for mode in ["osum_no_think", "osum_think"]:
        todo = []
        for rec in records:
            sample_id = rec["id"]
            out_wav   = audio_dirs[mode] / f"{sample_id}.wav"
            if args.skip_existing and sample_id in existing_ids[mode] and out_wav.exists():
                continue
            todo.append({
                "id":           sample_id,
                "query_audio":  rec["query_audio"],
                "output_audio": str(out_wav.resolve()),
            })
        todos[mode] = todo
        print(f"Samples to generate [{mode}]: {len(todo)}")

    # Run each mode via subprocess
    results_maps = {}
    for mode in ["osum_no_think", "osum_think"]:
        results_maps[mode] = run_mode(
            mode, todos[mode], audio_dirs[mode], osum_dir, output_dir
        )
        # Merge new results into out_records_map
        for sample_id, r in results_maps[mode].items():
            if sample_id not in out_records_map:
                # Find original record to get emotion/valence/arousal/query_audio
                orig = next((rec for rec in records if rec["id"] == sample_id), {})
                out_records_map[sample_id] = {
                    "id":          sample_id,
                    "emotion":     orig.get("emotion"),
                    "valence":     orig.get("valence"),
                    "arousal":     orig.get("arousal"),
                    "query_audio": orig.get("query_audio"),
                }

    # Write output manifest (all records)
    print(f"\nWriting manifest: {out_manifest}")
    with open(out_manifest, "w") as f:
        for rec in records:
            sample_id = rec["id"]
            no_think_wav = audio_dirs["osum_no_think"] / f"{sample_id}.wav"
            think_wav    = audio_dirs["osum_think"]    / f"{sample_id}.wav"

            out_rec = out_records_map.get(sample_id, {})
            # Start from original fields
            record_out = {
                "id":          sample_id,
                "emotion":     rec.get("emotion"),
                "valence":     rec.get("valence"),
                "arousal":     rec.get("arousal"),
                "query_audio": rec.get("query_audio"),
                "osum_no_think_response": str(no_think_wav.resolve()) if no_think_wav.exists() else None,
                "osum_think_response":    str(think_wav.resolve())    if think_wav.exists()    else None,
            }
            # Add text outputs if available
            no_think_result = results_maps["osum_no_think"].get(sample_id, {})
            think_result    = results_maps["osum_think"].get(sample_id, {})
            if no_think_result.get("text"):
                record_out["osum_no_think_text"] = no_think_result["text"]
            if think_result.get("text"):
                record_out["osum_think_text"] = think_result["text"]

            f.write(json.dumps(record_out) + "\n")

    no_think_ok = sum(1 for rec in records
                      if (audio_dirs["osum_no_think"] / f"{rec['id']}.wav").exists())
    think_ok    = sum(1 for rec in records
                      if (audio_dirs["osum_think"]    / f"{rec['id']}.wav").exists())
    total = len(records)

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  osum_no_think : {no_think_ok}/{total}")
    print(f"  osum_think    : {think_ok}/{total}")
    print(f"  Manifest      : {out_manifest}")
    print(f"\nNext step:")
    print(f"  conda run -n qwen3omni python -m eval.judge_qwen3omni \\")
    print(f"      --manifest {out_manifest.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
