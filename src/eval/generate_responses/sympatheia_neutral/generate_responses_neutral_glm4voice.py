#!/usr/bin/env python3
"""
Generate audio responses from base GLM-4-Voice using NEUTRAL audio input.

Reads the manifest.jsonl produced by generate_responses_neutral_sympatheia.py to use the
exact same query audio files (ensuring a fair comparison). Saves responses to
audio/base/ and writes manifest_base.jsonl — nothing in the original
manifest or audio directories is touched.

The emotion label IS injected into the system prompt so the base model receives
the same information as the fine-tuned model and other baselines (Qwen3-Omni,
OpenS2S) — this makes the comparison fair.

Usage:
    python -m eval.generate_responses.sympatheia_neutral.generate_responses_neutral_glm4voice \\
        --manifest results/eval_neutral_.../manifest.jsonl

    # Resume:
    python -m eval.generate_responses.sympatheia_neutral.generate_responses_neutral_glm4voice \\
        --manifest results/eval_neutral_.../manifest.jsonl \\
        --skip-existing
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoModel, AutoTokenizer

# PROJECT_ROOT is the src/ dir, 4 levels up from eval/generate_responses/sympatheia_neutral/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from src.vocoder_src import GLM4CodecEncoder, GLM4CodecDecoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "THUDM/glm-4-voice-9b"
DECODER_SAMPLE_RATE = 22050


def get_system_prompt(emotion: str) -> str:
    """Build an emotion-conditioned system prompt for the base model."""
    return f"Please respond in English. The user is feeling {emotion.lower()}."


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate base GLM-4-Voice responses for neutral-input emotion evaluation"
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
        "--skip-existing", action="store_true",
        help="Skip samples whose output audio already exists (enables resuming)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Audio / inference helpers
# ---------------------------------------------------------------------------

def encode_audio(wav_path: Path, encoder) -> str:
    """Encode a WAV file to a string of <|audio_X|> tokens."""
    audio_tokens = encoder([str(wav_path)])[0]
    return "".join([f"<|audio_{x}|>" for x in audio_tokens])


def build_prompt(user_tokens: str, system_prompt: str) -> str:
    return f"<|system|>\n{system_prompt}\n<|user|>\n{user_tokens}\n<|assistant|>\n"


def generate_one(prompt: str, model, tokenizer, decoder, audio_0_id: int):
    """Run generation. Returns (text_output: str, waveform: np.ndarray | None)."""
    model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            temperature=0.2,
            top_p=0.8,
            max_new_tokens=2000,
        )
    generated = outputs[0][model_inputs["input_ids"].shape[1]:]

    audio_toks, text_toks = [], []
    for tok in generated:
        if tok.item() >= audio_0_id:
            audio_toks.append(tok)
        else:
            text_toks.append(tok)

    text_output = tokenizer.decode(text_toks, skip_special_tokens=True)

    if not audio_toks:
        return text_output, None

    ids_shifted = torch.tensor(
        [[t.item() - audio_0_id for t in audio_toks]], dtype=torch.long
    )
    waveform = decoder(ids_shifted).squeeze().cpu().numpy()
    return text_output, waveform


def unload_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else manifest_path.parent
    audio_dir  = output_dir / "audio" / "base"
    audio_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = output_dir / "manifest_base.jsonl"

    print(f"Manifest (input)  : {manifest_path}")
    print(f"Output dir        : {output_dir}")
    print(f"Audio output dir  : {audio_dir}")
    print(f"Out manifest      : {out_manifest}")
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

        print("Loading tokenizer and speech codec components...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
        audio_0_id = tokenizer.convert_tokens_to_ids("<|audio_0|>")
        encoder = GLM4CodecEncoder()
        decoder_path = str(PROJECT_ROOT / "glm-4-voice-decoder")
        decoder = GLM4CodecDecoder(decoder_path)
        print(f"audio_0_id = {audio_0_id}\n")

        print(f"Loading base model: {BASE_MODEL_ID}")
        t0 = time.time()
        model = AutoModel.from_pretrained(BASE_MODEL_ID, trust_remote_code=True, device_map="auto")
        model.eval()
        print(f"Model loaded in {time.time() - t0:.1f}s\n")

        for idx, rec in enumerate(todo):
            sample_id   = rec["id"]
            query_audio = rec["query_audio"]
            emotion     = rec.get("emotion", "Neutral")
            out_wav     = audio_dir / f"{sample_id}.wav"

            print(f"\n[{idx+1}/{len(todo)}] {sample_id}  ({emotion})")

            if not Path(query_audio).exists():
                print(f"  SKIP: query audio not found: {query_audio}")
                continue

            user_tokens = encode_audio(Path(query_audio), encoder)
            prompt = build_prompt(user_tokens, get_system_prompt(emotion))

            t1 = time.time()
            try:
                text, waveform = generate_one(prompt, model, tokenizer, decoder, audio_0_id)
            except Exception as e:
                print(f"  ERROR during generation: {e}")
                continue

            elapsed = time.time() - t1

            if waveform is not None:
                sf.write(str(out_wav), waveform, DECODER_SAMPLE_RATE)
                print(f"  Saved audio: {out_wav.name}  ({elapsed:.1f}s)")
            else:
                print(f"  WARNING: no audio generated ({elapsed:.1f}s)")

            if text:
                print(f"  Text: {text[:120]!r}")

            out_records_map[sample_id] = {
                "id":            sample_id,
                "emotion":       emotion,
                "valence":       rec.get("valence"),
                "arousal":       rec.get("arousal"),
                "query_audio":   query_audio,
                "base_response": str(out_wav.resolve()) if out_wav.exists() else None,
                "base_text":     text or None,
            }

        print(f"\nUnloading model...")
        unload_model(model)

    # Write output manifest (all records, both existing and new)
    print(f"\nWriting manifest: {out_manifest}")
    with open(out_manifest, "w") as f:
        for rec in records:
            sample_id = rec["id"]
            out_wav   = audio_dir / f"{sample_id}.wav"
            existing  = out_records_map.get(sample_id, {})
            out_rec = {
                "id":            sample_id,
                "emotion":       rec.get("emotion"),
                "valence":       rec.get("valence"),
                "arousal":       rec.get("arousal"),
                "query_audio":   rec.get("query_audio"),
                "base_response": str(out_wav.resolve()) if out_wav.exists() else None,
                "base_text":     existing.get("base_text"),
            }
            f.write(json.dumps(out_rec) + "\n")

    ok    = sum(1 for rec in records if (audio_dir / f"{rec['id']}.wav").exists())
    total = len(records)
    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Generated : {ok}/{total} samples")
    print(f"  Manifest  : {out_manifest}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
