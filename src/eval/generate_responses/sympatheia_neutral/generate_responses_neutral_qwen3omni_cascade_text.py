#!/usr/bin/env python3
"""
Stage 1 of the cascaded baseline (neutral-input variant): input audio → Qwen3-Omni → response text + talk style.

Like generate_responses_emotional_qwen3omni_cascade_text.py but for neutral-audio manifests:
the emotion label and VA values are injected into the system prompt so Qwen3-Omni
knows the user's stated emotion even though the audio sounds neutral.

Qwen3-Omni is prompted to output a structured reply:
    STYLE: <short prosody phrase>
    RESPONSE: <spoken response text>

The parsed style and text are saved to manifest_qwen3omni_cascade_text.jsonl and fed
into Stage 2 (synthesize_responses_qwen3tts_cascaded.py) which synthesizes audio via
Qwen3-TTS using the style as the TTS instruct parameter.

Stage 2 script lives at:
  ../sympatheia_emotional/synthesize_responses_qwen3tts_cascaded.py

Usage:
    # Note: run this in the Qwen3-Omni environment (see https://github.com/QwenLM/Qwen3)
    python -m eval.generate_responses.sympatheia_neutral.generate_responses_neutral_qwen3omni_cascade_text \\
    --manifest <path/to/sample_neutral_manifest.jsonl> \\
    --output-dir <path/to/output/>

    # Resume:
    ... --skip-existing
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "Qwen/Qwen3-Omni"
USE_AUDIO_IN_VIDEO = True

FALLBACK_STYLE = "Neutral, clear, friendly."


def get_system_prompt(emotion: str, valence: float, arousal: float) -> str:
    return (
        f"You are a warm, empathetic voice assistant. "
        f"The user is currently feeling {emotion.lower()} "
        f"(emotional valence={valence:.2f}, arousal={arousal:.2f}). "
        f"Respond supportively, acknowledging their {emotion.lower()} emotional state with genuine care.\n"
        "Format your reply EXACTLY as two lines:\n"
        "STYLE: <a short phrase describing the speaking tone>\n"
        "RESPONSE: <your spoken response>"
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1 cascaded baseline (neutral-input): Qwen3-Omni → text + talk style"
    )
    parser.add_argument("--manifest", type=str, required=True,
                        help="Input manifest.jsonl with query_audio, emotion, valence, arousal fields")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory. Defaults to same dir as --manifest.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Qwen3-Omni model path or HF repo ID. Default: {DEFAULT_MODEL}")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip samples already present in output manifest (resume)")
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


def build_conversation(query_audio_path: str, emotion: str, valence: float, arousal: float) -> list:
    return [
        {"role": "system", "content": [{"type": "text", "text": get_system_prompt(emotion, valence, arousal)}]},
        {"role": "user",   "content": [{"type": "audio", "audio": query_audio_path}]},
    ]


def generate_text(model, processor, conversation: list) -> str:
    """Run Qwen3-Omni and return text output only (no audio decoding)."""
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
        result = model.generate(
            **inputs,
            thinker_return_dict_in_generate=True,
            thinker_max_new_tokens=2048,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
        )

    # model.generate() returns (GenerateDecoderOnlyOutput, audio_or_None) as a tuple,
    # or just GenerateDecoderOnlyOutput when no audio is generated.
    text_ids_obj = result[0] if isinstance(result, tuple) else result
    sequences = text_ids_obj.sequences if hasattr(text_ids_obj, "sequences") else text_ids_obj

    decoded = processor.batch_decode(
        sequences[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0] if decoded else ""


def parse_style_response(raw: str) -> tuple[str, str]:
    """Extract STYLE: and RESPONSE: lines from Qwen3-Omni output."""
    style = FALLBACK_STYLE
    response = raw.strip()

    for line in raw.splitlines():
        if line.startswith("STYLE:"):
            style = line[len("STYLE:"):].strip() or FALLBACK_STYLE
        elif line.startswith("RESPONSE:"):
            rest = raw[raw.index("RESPONSE:") + len("RESPONSE:"):].strip()
            response = rest if rest else raw.strip()
            break

    return style, response


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
    out_manifest = output_dir / "manifest_qwen3omni_cascade_text.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Manifest (input)  : {manifest_path}")
    print(f"Output dir        : {output_dir}")
    print(f"Out manifest      : {out_manifest}")
    print(f"Model             : {args.model}")
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

    todo = [r for r in records if not (args.skip_existing and r["id"] in existing_ids)]
    print(f"Samples to generate: {len(todo)}")

    if todo:
        print(f"\nCUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  {torch.cuda.device_count()} GPU(s): {torch.cuda.get_device_name(0)}")
        model, processor = load_model(args.model)

        for idx, rec in enumerate(todo):
            sample_id = rec["id"]
            query_audio = rec["query_audio"]
            emotion   = rec.get("emotion", "Neutral")
            valence   = rec.get("valence", 0.0)
            arousal   = rec.get("arousal", 0.0)

            print(f"\n[{idx+1}/{len(todo)}] {sample_id}  ({emotion}, V={valence:+.2f}, A={arousal:+.2f})")

            if not Path(query_audio).exists():
                print(f"  SKIP: query audio not found: {query_audio}")
                continue

            conversation = build_conversation(query_audio, emotion, valence, arousal)

            t0 = time.time()
            try:
                raw_text = generate_text(model, processor, conversation)
            except Exception as e:
                print(f"  ERROR during generation: {e}")
                continue

            elapsed = time.time() - t0
            style, response = parse_style_response(raw_text)

            print(f"  Style   : {style!r}  ({elapsed:.1f}s)")
            print(f"  Response: {response[:120]!r}")

            out_records_map[sample_id] = {
                "id":            sample_id,
                "emotion":       emotion,
                "valence":       valence,
                "arousal":       arousal,
                "query_audio":   query_audio,
                "cascade_style": style,
                "cascade_text":  response,
            }

    print(f"\nWriting manifest: {out_manifest}")
    with open(out_manifest, "w") as f:
        for rec in records:
            sid = rec["id"]
            out_rec = out_records_map.get(sid, {
                "id":            sid,
                "emotion":       rec.get("emotion"),
                "valence":       rec.get("valence"),
                "arousal":       rec.get("arousal"),
                "query_audio":   rec.get("query_audio"),
                "cascade_style": None,
                "cascade_text":  None,
            })
            f.write(json.dumps(out_rec) + "\n")

    ok    = sum(1 for r in records if out_records_map.get(r["id"], {}).get("cascade_text"))
    total = len(records)
    print(f"\n{'='*60}")
    print(f"DONE  {ok}/{total} samples have text output")
    print(f"  Manifest: {out_manifest}")
    print(f"\nNext step (Stage 2):")
    print(f"  \\")
    print(f"      python -m eval.generate_responses.sympatheia_emotional.synthesize_responses_qwen3tts_cascaded \\")
    print(f"      --manifest {out_manifest.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
