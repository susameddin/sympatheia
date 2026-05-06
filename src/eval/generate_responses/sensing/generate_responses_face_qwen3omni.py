#!/usr/bin/env python3
"""
Generate audio responses from Qwen3-Omni using face expression image + query audio.

Reads the face sensing manifest (produced by generate_responses_sensing.py --modality face),
which already contains both image_path and query_audio per sample. Feeds the face image and
neutral query audio directly to Qwen3-Omni so the model can see the user's emotional state
and respond accordingly — no intermediate VA extraction step.

Condition added: face_image_qwen3omni

Prereqs:
    # 1. Precompute HSEmotion face predictions ():
    python -m integration.face_module.precompute_hsemotion --n-per-class 150

    # 2. Generate face_va / no_va Sympatheia responses:
    python -m eval.generate_responses.sensing.generate_responses_sensing --modality face

Usage:
    # Note: run this in the Qwen3-Omni environment (see https://github.com/QwenLM/Qwen3)
    python -m eval.generate_responses.sensing.generate_responses_face_qwen3omni \\
        --manifest eval/eval_face_hsemo/manifest.jsonl

    # Resume:
    # Note: run this in the Qwen3-Omni environment (see https://github.com/QwenLM/Qwen3)
    python -m eval.generate_responses.sensing.generate_responses_face_qwen3omni \\
        --manifest eval/eval_face_hsemo/manifest.jsonl \\
        --skip-existing
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

DEFAULT_MODEL   = "Qwen/Qwen3-Omni"
DEFAULT_MANIFEST = "eval/eval_face_hsemo/manifest.jsonl"
DEFAULT_SPEAKER  = "Chelsie"
SAMPLE_RATE      = 24000
USE_AUDIO_IN_VIDEO = True
CONDITION        = "face_image_qwen3omni"

SYSTEM_PROMPT = (
    "You are a warm, empathetic voice assistant. "
    "You will receive an image showing the user's facial expression alongside their spoken message. "
    "Your most important job is to acknowledge and directly address the emotion you see on their face — "
    "do not ignore it or give a generic response. "
    "If they look sad or are crying, lead with heartfelt consolation and let them know you care. "
    "If they look angry or frustrated, respond calmly, validate their feelings, and show understanding. "
    "If they look fearful or anxious, be gentle and reassuring. "
    "If they look happy or excited, match their energy with warmth and enthusiasm. "
    "Always name or reflect their emotional state early in your response so they feel truly heard. "
    "Keep your response concise and conversational."
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Qwen3-Omni responses conditioned on face expression image + audio"
    )
    parser.add_argument(
        "--manifest", type=str, default=DEFAULT_MANIFEST,
        help=f"Path to face sensing manifest.jsonl (contains image_path and query_audio). "
             f"Default: {DEFAULT_MANIFEST}",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory. Defaults to the same directory as --manifest.",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Qwen3-Omni model path or HF repo ID. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--speaker", type=str, default=DEFAULT_SPEAKER,
        help=f"Voice speaker for Qwen3-Omni audio output. Default: {DEFAULT_SPEAKER}",
    )
    parser.add_argument(
        "--emotions", type=str, default=None,
        help="Comma-separated list of emotions to process, e.g. 'Happy,Sad,Fear'. "
             "Case-insensitive. Defaults to all emotions.",
    )
    parser.add_argument(
        "--one-per-emotion", action="store_true",
        help="Run exactly one sample per emotion class (first occurrence). "
             "Useful for a quick sanity check across all emotions.",
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
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    print(f"Loading Qwen3-Omni from: {model_path}")
    t0 = time.time()
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path, dtype="auto", device_map="auto"
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")
    return model, processor


def build_conversation(image_path: str, query_audio_path: str) -> list:
    """Build conversation with face image + audio in user turn; emotion context in system."""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "audio", "audio": query_audio_path},
            ],
        },
    ]


def generate_response(model, processor, conversation: list, speaker: str) -> tuple[str, object]:
    """Run Qwen3-Omni on the conversation and return (text, audio_tensor)."""
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
            thinker_return_dict_in_generate=True,
            thinker_max_new_tokens=2048,
            speaker=speaker,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
        )

    decoded = processor.batch_decode(
        text_ids.sequences[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    text_out = decoded[0] if decoded else ""
    return text_out, audio


def save_audio(audio_tensor, out_path: Path):
    """Decode Qwen3-Omni audio tensor and save as 24 kHz WAV."""
    audio_np = np.array(
        audio_tensor.reshape(-1).float().detach().cpu().numpy() * 32767,
        dtype=np.int16,
    )
    sf.write(str(out_path), audio_np, samplerate=SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        print(
            "Run first:\n"
            "  "
            "python -m eval.generate_responses.sensing.generate_responses_sensing --modality face",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else manifest_path.parent
    audio_dir  = output_dir / "audio" / CONDITION
    audio_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = output_dir / f"manifest_{CONDITION}.jsonl"

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
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records from manifest")

    # Optionally filter to specific emotions
    if args.emotions:
        allowed = {e.strip().lower() for e in args.emotions.split(",")}
        records = [r for r in records if r.get("emotion", "").lower() in allowed]
        print(f"Filtered to {len(records)} records matching emotions: {args.emotions}")

    # Optionally keep only the first sample per emotion
    if args.one_per_emotion:
        seen: set = set()
        one_each = []
        for r in records:
            emo = r.get("emotion", "").lower()
            if emo not in seen:
                seen.add(emo)
                one_each.append(r)
        records = one_each
        emotions_found = [r.get("emotion") for r in records]
        print(f"One-per-emotion: {len(records)} samples — {emotions_found}")

    # Load existing output manifest for resume
    existing_texts: dict = {}
    if args.skip_existing and out_manifest.exists():
        n_existing = 0
        with open(out_manifest) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    existing_texts[r["id"]] = r.get(f"{CONDITION}_text")
                    n_existing += 1
        print(f"Resuming: {n_existing} already in output manifest\n")

    # Determine which samples need generation
    todo = []
    for rec in records:
        out_wav = audio_dir / f"{rec['id']}.wav"
        if args.skip_existing and out_wav.exists():
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
            image_path  = rec.get("image_path", "")
            query_audio = rec.get("query_audio", "")
            emotion     = rec.get("emotion", "Unknown")
            valence     = rec.get("valence", 0.0)
            arousal     = rec.get("arousal", 0.0)
            out_wav     = audio_dir / f"{sample_id}.wav"

            print(
                f"\n[{idx+1}/{len(todo)}] {sample_id}  "
                f"({emotion}, V={valence:+.2f}, A={arousal:+.2f})"
            )

            if not image_path or not Path(image_path).exists():
                print(f"  SKIP: image not found: {image_path!r}")
                continue
            if not query_audio or not Path(query_audio).exists():
                print(f"  SKIP: query audio not found: {query_audio!r}")
                continue

            conversation = build_conversation(image_path, query_audio)

            t0 = time.time()
            try:
                text_out, audio_tensor = generate_response(
                    model, processor, conversation, args.speaker
                )
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
                existing_texts[sample_id] = text_out

    # Write output manifest (all records, preserving all input fields)
    print(f"\nWriting manifest: {out_manifest}")
    with open(out_manifest, "w") as f:
        for rec in records:
            sample_id = rec["id"]
            out_wav   = audio_dir / f"{sample_id}.wav"
            out_rec   = dict(rec)
            out_rec[f"{CONDITION}_response"] = str(out_wav.resolve()) if out_wav.exists() else None
            out_rec[f"{CONDITION}_text"]     = existing_texts.get(sample_id)
            f.write(json.dumps(out_rec) + "\n")

    ok    = sum(1 for rec in records if (audio_dir / f"{rec['id']}.wav").exists())
    total = len(records)
    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Generated : {ok}/{total} samples")
    print(f"  Manifest  : {out_manifest}")
    print(f"\nNext step (judge — requires Qwen3-Omni environment):")
    print(
        f"  "
        f"python -m eval.judge.judge_qwen3omni_emotional \\\n"
        f"      --manifest {out_manifest.resolve()} \\\n"
        f"      --conditions {CONDITION}"
    )
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
