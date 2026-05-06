#!/usr/bin/env python3
"""
Generate audio responses from Kimi-Audio for neutral-input emotion evaluation.

Unlike generate_responses_emotional_kimiaudio.py (which passes audio only), this script
prepends a text message describing the user's emotion before the audio query. This is
the only available mechanism for KimiAudio — system-role messages are not supported
(blocked by assertion in prompt_manager.py), but message_type="text" is valid.

Reads the manifest.jsonl produced by generate_responses_neutral_sympatheia.py.
Saves responses to audio/kimiaudio/ and writes manifest_kimiaudio.jsonl.

Usage:
    # Note: run this in the Kimi-Audio environment (see https://github.com/MoonshotAI/Kimi-Audio)
    python -m eval.generate_responses.sympatheia_neutral.generate_responses_neutral_kimiaudio \\
        --manifest /path/to/manifest.jsonl

    # Resume:
    python -m eval.generate_responses.sympatheia_neutral.generate_responses_neutral_kimiaudio \\
        --manifest /path/to/manifest.jsonl \\
        --skip-existing
"""

import argparse
import json
import sys
import time
from pathlib import Path

KIMI_AUDIO_DIR = "/path/to/Kimi-Audio"
DEFAULT_MODEL  = "moonshotai/Kimi-Audio-7B-Instruct"
SAMPLE_RATE    = 24000

SAMPLING_PARAMS = {
    "audio_temperature": 0.8,
    "audio_top_k": 10,
    "text_temperature": 0.0,
    "text_top_k": 5,
    "audio_repetition_penalty": 1.0,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Kimi-Audio responses for neutral-input emotion evaluation"
    )
    parser.add_argument(
        "--manifest", type=str, required=True,
        help="Path to manifest.jsonl from generate_responses_neutral_sympatheia.py "
             "(used to select the same query audio files)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory. Defaults to the same directory as --manifest.",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Kimi-Audio model path. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip samples whose output audio already exists (enables resuming)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model(model_path: str):
    kimi_dir = Path(KIMI_AUDIO_DIR)
    if str(kimi_dir) not in sys.path:
        sys.path.insert(0, str(kimi_dir))

    from kimia_infer.api.kimia import KimiAudio

    print(f"Loading Kimi-Audio from: {model_path}")
    t0 = time.time()
    model = KimiAudio(model_path=model_path, load_detokenizer=True)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")
    return model


def generate_response(
    model, query_audio_path: str, emotion: str, valence: float, arousal: float
) -> tuple[object, str]:
    """Run Kimi-Audio on one query with emotion context injected as a text message."""
    messages = [
        {
            "role": "user",
            "message_type": "text",
            "content": (
                f"I want you to know that I am currently feeling {emotion.lower()} "
                f"(valence={valence:.2f}, arousal={arousal:.2f}). "
                f"Please respond warmly and acknowledge my emotional state. "
                f"Please respond in English."
            ),
        },
        {"role": "user", "message_type": "audio", "content": query_audio_path},
    ]
    wav_output, text_output = model.generate(messages, **SAMPLING_PARAMS, output_type="both")
    return wav_output, text_output or ""


def save_audio(wav_tensor, out_path: Path):
    import soundfile as sf
    sf.write(str(out_path), wav_tensor.detach().cpu().view(-1).numpy(), SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    output_dir  = Path(args.output_dir) if args.output_dir else manifest_path.parent
    audio_dir   = output_dir / "audio" / "kimiaudio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = output_dir / "manifest_kimiaudio.jsonl"

    print(f"Manifest (input)  : {manifest_path}")
    print(f"Output dir        : {output_dir}")
    print(f"Audio output dir  : {audio_dir}")
    print(f"Out manifest      : {out_manifest}")
    print(f"Model             : {args.model}")
    print(f"Skip existing     : {args.skip_existing}\n")

    # Load source manifest
    records = []
    with open(manifest_path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records from manifest")

    # Load existing output manifest for resume
    existing_ids: set = set()
    out_records_map: dict = {}
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
        out_wav   = audio_dir / f"{sample_id}.wav"
        if args.skip_existing and sample_id in existing_ids and out_wav.exists():
            continue
        todo.append(rec)

    print(f"Samples to generate: {len(todo)}")

    if todo:
        import torch
        print(f"\nCUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  {torch.cuda.device_count()} GPU(s): {torch.cuda.get_device_name(0)}")
        model = load_model(args.model)

        for idx, rec in enumerate(todo):
            sample_id   = rec["id"]
            query_audio = rec["query_audio"]
            emotion     = rec.get("emotion", "Neutral")
            valence     = rec.get("valence", 0.0)
            arousal     = rec.get("arousal", 0.0)
            out_wav     = audio_dir / f"{sample_id}.wav"

            print(f"\n[{idx+1}/{len(todo)}] {sample_id}  ({emotion}, V={valence:+.2f}, A={arousal:+.2f})")

            if not Path(query_audio).exists():
                print(f"  SKIP: query audio not found: {query_audio}")
                continue

            t0 = time.time()
            try:
                wav_tensor, text_out = generate_response(model, query_audio, emotion, valence, arousal)
            except Exception as e:
                print(f"  ERROR during generation: {e}")
                continue

            elapsed = time.time() - t0
            save_audio(wav_tensor, out_wav)
            print(f"  Saved audio: {out_wav}  ({elapsed:.1f}s)")
            if text_out:
                print(f"  Text: {text_out[:120]!r}")

            out_records_map[sample_id] = {
                "id":                 sample_id,
                "emotion":            emotion,
                "valence":            valence,
                "arousal":            arousal,
                "query_audio":        query_audio,
                "kimiaudio_response": str(out_wav.resolve()),
                "kimiaudio_text":     text_out or None,
            }

    # Write output manifest (all records, both existing and new)
    print(f"\nWriting manifest: {out_manifest}")
    with open(out_manifest, "w") as f:
        for rec in records:
            sample_id = rec["id"]
            out_wav   = audio_dir / f"{sample_id}.wav"
            existing  = out_records_map.get(sample_id, {})
            out_rec = {
                "id":                 sample_id,
                "emotion":            rec.get("emotion"),
                "valence":            rec.get("valence"),
                "arousal":            rec.get("arousal"),
                "query_audio":        rec.get("query_audio"),
                "kimiaudio_response": str(out_wav.resolve()) if out_wav.exists() else None,
                "kimiaudio_text":     existing.get("kimiaudio_text"),
            }
            f.write(json.dumps(out_rec) + "\n")

    ok    = sum(1 for rec in records if (audio_dir / f"{rec['id']}.wav").exists())
    total = len(records)
    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Generated : {ok}/{total} samples")
    print(f"  Manifest  : {out_manifest}")
    print(f"\nNext step:")
    print(f"  python -m eval.judge.judge_qwen3omni_neutral \\")
    print(f"      --manifest {out_manifest.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
