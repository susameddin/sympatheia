#!/usr/bin/env python3
"""
Generate audio responses from Qwen2.5-Omni for empathy evaluation.

Reads the manifest.jsonl produced by generate_responses_emotional_sympatheia.py to use the
exact same query audio files (ensuring a fair comparison). Saves responses to
audio/qwen2_5omni/ and writes manifest_qwen2_5omni.jsonl — nothing in the original
manifest or audio directories is touched.

Usage:
    conda run -n qwen2_5omni --no-capture-output python -m eval.generate_responses.sympatheia_emotional.generate_responses_emotional_qwen2_5omni \\
        --manifest results/eval_judge/manifest.jsonl

    # Resume:
    conda run -n qwen2_5omni --no-capture-output python -m eval.generate_responses.sympatheia_emotional.generate_responses_emotional_qwen2_5omni \\
        --manifest results/eval_judge/manifest.jsonl \\
        --skip-existing

    # Smoke test (no manifest needed):
    conda run -n qwen2_5omni --no-capture-output python -m eval.generate_responses.sympatheia_emotional.generate_responses_emotional_qwen2_5omni \\
        --smoke-test
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL   = "/engram/naplab/users/sd3705/emo_recog_2025s/Models/Qwen2.5-Omni-7B"
DEFAULT_SPEAKER = "Chelsie"
SAMPLE_RATE     = 24000
USE_AUDIO_IN_VIDEO = True

SMOKE_AUDIO = str(
    Path(__file__).resolve().parents[4]
    / "Models" / "Qwen2_5-Omni" / "example_prompt" / "rough_day.wav"
)

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Qwen2.5-Omni responses for empathy evaluation"
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to manifest.jsonl from generate_responses_emotional_sympatheia.py "
             "(used to select the same query audio files)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory. Defaults to the same directory as --manifest.",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Qwen2.5-Omni model path or HF repo ID. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--speaker", type=str, default=DEFAULT_SPEAKER,
        help=f"Voice speaker for Qwen2.5-Omni audio output. Default: {DEFAULT_SPEAKER}",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip samples whose output audio already exists (enables resuming)",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Load the model, run one inference on a built-in audio file, and exit",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model(model_path: str):
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    print(f"Loading Qwen2.5-Omni from: {model_path}")
    t0 = time.time()
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path, dtype="auto", device_map="auto"
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")
    return model, processor


def build_conversation(query_audio_path: str) -> list:
    """Build the conversation dict for Qwen2.5-Omni with query audio as input."""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": query_audio_path},
            ],
        },
    ]


def generate_response(model, processor, conversation: list, speaker: str) -> tuple[str, object]:
    """Run Qwen2.5-Omni on the conversation and return (text, audio_tensor)."""
    from qwen_omni_utils import process_mm_info

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
        text_ids, audio = model.generate(
            **inputs,
            speaker=speaker,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
        )

    full_text = processor.batch_decode(
        text_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    # Extract only the assistant turn (text_ids includes input tokens)
    if "\nassistant\n" in full_text:
        text_out = full_text.split("\nassistant\n")[-1].strip()
    else:
        text_out = full_text.split("\n")[-1]

    return text_out, audio


def save_audio(audio_tensor, out_path: Path):
    """Decode Qwen2.5-Omni audio tensor and save as 24 kHz WAV."""
    audio_np = np.array(
        audio_tensor.reshape(-1).float().detach().cpu().numpy() * 32767,
        dtype=np.int16,
    )
    sf.write(str(out_path), audio_np, samplerate=SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke_test(model, processor, speaker: str):
    audio_path = SMOKE_AUDIO
    if not Path(audio_path).exists():
        print(f"[SMOKE TEST] WARNING: built-in audio not found at {audio_path}")
        print("[SMOKE TEST] Skipping audio input — using text-only fallback")
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "I had such a rough day today."}],
            },
        ]
    else:
        conversation = build_conversation(audio_path)

    print("[SMOKE TEST] Running single inference...")
    t0 = time.time()
    text_out, audio_tensor = generate_response(model, processor, conversation, speaker)
    elapsed = time.time() - t0

    out_path = Path(__file__).parent / "smoke_test_out.wav"
    if audio_tensor is not None:
        save_audio(audio_tensor, out_path)
        n_samples = audio_tensor.reshape(-1).shape[0]
        print(f"[SMOKE TEST] Text: {text_out[:120]!r}")
        print(f"[SMOKE TEST] Audio saved: {out_path} ({n_samples} samples @ {SAMPLE_RATE} Hz)  [{elapsed:.1f}s]")
        print("[SMOKE TEST] PASS")
    else:
        print(f"[SMOKE TEST] Text: {text_out[:120]!r}")
        print("[SMOKE TEST] WARNING: no audio tensor returned")
        print("[SMOKE TEST] FAIL")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.smoke_test:
        print(f"CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  {torch.cuda.device_count()} GPU(s): {torch.cuda.get_device_name(0)}")
        model, processor = load_model(args.model)
        run_smoke_test(model, processor, args.speaker)
        return

    if not args.manifest:
        print("ERROR: --manifest is required (or use --smoke-test)", file=sys.stderr)
        sys.exit(1)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else manifest_path.parent
    audio_dir  = output_dir / "audio" / "qwen2_5omni"
    audio_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = output_dir / "manifest_qwen2_5omni.jsonl"

    print(f"Manifest (input)  : {manifest_path}")
    print(f"Output dir        : {output_dir}")
    print(f"Audio output dir  : {audio_dir}")
    print(f"Out manifest      : {out_manifest}")
    print(f"Model             : {args.model}")
    print(f"Speaker           : {args.speaker}")
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
        print(f"\nCUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  {torch.cuda.device_count()} GPU(s): {torch.cuda.get_device_name(0)}")
        model, processor = load_model(args.model)

        for idx, rec in enumerate(todo):
            sample_id   = rec["id"]
            query_audio = rec["query_audio"]
            out_wav     = audio_dir / f"{sample_id}.wav"

            print(f"\n[{idx+1}/{len(todo)}] {sample_id}  ({rec.get('emotion', '?')})")

            if not Path(query_audio).exists():
                print(f"  SKIP: query audio not found: {query_audio}")
                continue

            conversation = build_conversation(query_audio)

            t0 = time.time()
            try:
                text_out, audio_tensor = generate_response(model, processor, conversation, args.speaker)
            except Exception as e:
                print(f"  ERROR during generation: {e}")
                continue

            elapsed = time.time() - t0

            if audio_tensor is not None:
                save_audio(audio_tensor, out_wav)
                print(f"  Saved audio: {out_wav}  ({elapsed:.1f}s)")
            else:
                print(f"  WARNING: no audio output returned ({elapsed:.1f}s)")

            if text_out:
                print(f"  Text: {text_out[:120]!r}")

            out_records_map[sample_id] = {
                "id":                   sample_id,
                "emotion":              rec.get("emotion"),
                "valence":              rec.get("valence"),
                "arousal":              rec.get("arousal"),
                "query_audio":          query_audio,
                "qwen2_5omni_response": str(out_wav.resolve()) if out_wav.exists() else None,
                "qwen2_5omni_text":     text_out or None,
            }

    # Write output manifest (all records, both existing and new)
    print(f"\nWriting manifest: {out_manifest}")
    with open(out_manifest, "w") as f:
        for rec in records:
            sample_id = rec["id"]
            out_wav   = audio_dir / f"{sample_id}.wav"
            existing  = out_records_map.get(sample_id, {})
            out_rec = {
                "id":                   sample_id,
                "emotion":              rec.get("emotion"),
                "valence":              rec.get("valence"),
                "arousal":              rec.get("arousal"),
                "query_audio":          rec.get("query_audio"),
                "qwen2_5omni_response": str(out_wav.resolve()) if out_wav.exists() else None,
                "qwen2_5omni_text":     existing.get("qwen2_5omni_text"),
            }
            f.write(json.dumps(out_rec) + "\n")

    ok    = sum(1 for rec in records if (audio_dir / f"{rec['id']}.wav").exists())
    total = len(records)
    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Generated : {ok}/{total} samples")
    print(f"  Manifest  : {out_manifest}")
    print(f"\nNext step:")
    print(f"  conda run -n qwen3omni --no-capture-output python -m eval.judge.judge_qwen3omni_emotional \\")
    print(f"      --manifest {out_manifest.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
