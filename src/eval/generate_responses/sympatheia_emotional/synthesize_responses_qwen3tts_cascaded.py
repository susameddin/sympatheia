#!/usr/bin/env python3
"""
Stage 2 of the cascaded baseline: text + talk style → Qwen3-TTS → output audio.

Reads manifest_qwen3omni_cascade_text.jsonl produced by Stage 1
(generate_responses_emotional_qwen3omni_cascade_text.py), synthesizes speech for each sample
using the cascade_style field as the Qwen3-TTS instruct parameter.

Usage:
    conda run -n qwen3-tts4 --no-capture-output \\
        python -m eval.generate_responses.sympatheia_emotional.synthesize_responses_qwen3tts_cascaded \\
        --manifest <path/to/manifest_qwen3omni_cascade_text.jsonl> \\
        --output-dir <path/to/output/>

    # Resume:
    ... --skip-existing
"""

import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_SPEAKER    = "Vivian"
FALLBACK_STYLE     = "Neutral, clear, friendly."


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2 cascaded baseline: Qwen3-TTS synthesis from Qwen3-Omni text+style"
    )
    parser.add_argument("--manifest", type=str, required=True,
                        help="manifest_qwen3omni_cascade_text.jsonl from Stage 1")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory. Defaults to same dir as --manifest.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME,
                        help=f"Qwen3-TTS model name or local path. Default: {DEFAULT_MODEL_NAME}")
    parser.add_argument("--speaker", type=str, default=DEFAULT_SPEAKER,
                        help=f"TTS speaker voice. Default: {DEFAULT_SPEAKER}")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip samples whose output WAV already exists (resume)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model(model_name: str):
    qwen3_tts_path = Path(__file__).resolve().parents[4] / "Models" / "TTS" / "Qwen3-TTS" / "Qwen3-TTS"
    if str(qwen3_tts_path) not in sys.path:
        sys.path.insert(0, str(qwen3_tts_path))

    from qwen_tts import Qwen3TTSModel

    print(f"Loading Qwen3-TTS from: {model_name}")
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        model_name,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    print(f"Model loaded in {time.time() - t0:.1f}s\n")
    return model


def synthesize(model, text: str, speaker: str, instruct: str) -> tuple:
    """Call Qwen3-TTS and return (wav_array, sample_rate)."""
    wavs, sr = model.generate_custom_voice(
        text=[text],
        language=["English"],
        speaker=[speaker],
        instruct=[instruct],
        max_new_tokens=2048,
    )
    return wavs[0], sr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    output_dir   = Path(args.output_dir) if args.output_dir else manifest_path.parent
    audio_dir    = output_dir / "audio" / "qwen3tts_cascaded"
    out_manifest = output_dir / "manifest_qwen3tts_cascaded.jsonl"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"Manifest (input)  : {manifest_path}")
    print(f"Output dir        : {output_dir}")
    print(f"Audio output dir  : {audio_dir}")
    print(f"Out manifest      : {out_manifest}")
    print(f"Model             : {args.model_name}")
    print(f"Speaker           : {args.speaker}")
    print(f"Skip existing     : {args.skip_existing}\n")

    records = []
    with open(manifest_path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records from manifest")

    existing_ids: set = set()
    out_records_map: dict = {}
    if args.skip_existing and out_manifest.exists():
        with open(out_manifest) as f:
            for line in f:
                r = json.loads(line)
                existing_ids.add(r["id"])
                out_records_map[r["id"]] = r
        print(f"Resuming: {len(existing_ids)} already done\n")

    todo = []
    for rec in records:
        sid     = rec["id"]
        out_wav = audio_dir / f"{sid}.wav"
        if args.skip_existing and sid in existing_ids and out_wav.exists():
            continue
        if not rec.get("cascade_text"):
            print(f"  SKIP {sid}: no cascade_text in manifest (Stage 1 may have failed)")
            continue
        todo.append(rec)

    print(f"Samples to generate: {len(todo)}")

    if todo:
        print(f"\nCUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  {torch.cuda.device_count()} GPU(s): {torch.cuda.get_device_name(0)}")
        model = load_model(args.model_name)

        for idx, rec in enumerate(todo):
            sid     = rec["id"]
            text    = rec["cascade_text"]
            style   = rec.get("cascade_style") or FALLBACK_STYLE
            out_wav = audio_dir / f"{sid}.wav"

            print(f"\n[{idx+1}/{len(todo)}] {sid}  ({rec.get('emotion', '?')})")
            print(f"  Style   : {style!r}")
            print(f"  Text    : {text[:100]!r}")

            t0 = time.time()
            try:
                wav, sr = synthesize(model, text, args.speaker, style)
            except Exception as e:
                print(f"  ERROR during synthesis: {e}")
                continue

            sf.write(str(out_wav), wav, sr)
            print(f"  Saved   : {out_wav}  ({time.time() - t0:.1f}s)")

            out_records_map[sid] = {
                "id":                        sid,
                "emotion":                   rec.get("emotion"),
                "valence":                   rec.get("valence"),
                "arousal":                   rec.get("arousal"),
                "query_audio":               rec.get("query_audio"),
                "cascade_text":              text,
                "cascade_style":             style,
                "qwen3tts_cascaded_response": str(out_wav.resolve()),
            }

    print(f"\nWriting manifest: {out_manifest}")
    with open(out_manifest, "w") as f:
        for rec in records:
            sid     = rec["id"]
            out_wav = audio_dir / f"{sid}.wav"
            existing = out_records_map.get(sid, {})
            out_rec = {
                "id":                        sid,
                "emotion":                   rec.get("emotion"),
                "valence":                   rec.get("valence"),
                "arousal":                   rec.get("arousal"),
                "query_audio":               rec.get("query_audio"),
                "cascade_text":              rec.get("cascade_text"),
                "cascade_style":             rec.get("cascade_style"),
                "qwen3tts_cascaded_response": str(out_wav.resolve()) if out_wav.exists() else None,
            }
            f.write(json.dumps(out_rec) + "\n")

    ok    = sum(1 for r in records if (audio_dir / f"{r['id']}.wav").exists())
    total = len(records)
    print(f"\n{'='*60}")
    print(f"DONE  {ok}/{total} audio files generated")
    print(f"  Manifest: {out_manifest}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
