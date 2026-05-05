#!/usr/bin/env python3
"""
Generate fine-tuned GLM-4-Voice responses on VoiceAssistant-400K queries.

Provides an unbiased evaluation of the fine-tuned model on neutral voice
assistant queries that are NOT part of the Sympatheia training data. Queries
are streamed from HuggingFace so only N samples are downloaded.

Conditions:
  finetuned_va  — fine-tuned LoRA checkpoint + valence/arousal (0.00, 0.00) in system prompt
  finetuned_na  — fine-tuned LoRA checkpoint + "User emotion N/A" prompt

Outputs:
  <output-dir>/audio/query/          — N WAVs downloaded from VoiceAssistant-400K
  <output-dir>/audio/finetuned_va/   — fine-tuned model responses (VA=0,0)
  <output-dir>/audio/finetuned_na/   — fine-tuned model responses (no VA info)
  <output-dir>/manifest.jsonl        — evaluation manifest
  <output-dir>/eval.jsonl            — reference answers for coherence eval

Usage:
    conda run -n glm4voice3 --no-capture-output python -m eval.generate_responses.sympatheia_neutral.voiceassistant400k.generate_responses_va400k_sympatheia \\
        --finetuned-experiment experiments/my-experiment \\
        --checkpoint-step 2000

    # Smoke test:
    conda run -n glm4voice3 --no-capture-output python -m eval.generate_responses.sympatheia_neutral.voiceassistant400k.generate_responses_va400k_sympatheia \\
        --finetuned-experiment experiments/my-experiment \\
        --checkpoint-step 2000 \\
        --num-samples 3 \\
        --skip-existing
"""

import argparse
import gc
import io
import json
import sys
import time
from itertools import islice
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vocoder_src import GLM4CodecEncoder, GLM4CodecDecoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "THUDM/glm-4-voice-9b"
DEFAULT_FINETUNED_EXPERIMENT = "experiments/sympatheia-12emo-20260312-100309"
DEFAULT_CHECKPOINT_STEP = 2000
DEFAULT_OUTPUT_DIR = "/engram/naplab/users/sd3705/emo_recog_2025s/eval/eval_va400k"
DECODER_SAMPLE_RATE = 22050
QUERY_SAMPLE_RATE = 16000

NA_SYSTEM_PROMPT = "Please respond in English. User emotion N/A"
VA_SYSTEM_PROMPT = "Please respond in English. User emotion (valence=0.00, arousal=0.00)"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate fine-tuned GLM-4-Voice responses on VoiceAssistant-400K queries"
    )
    parser.add_argument(
        "--finetuned-experiment", type=str, default=DEFAULT_FINETUNED_EXPERIMENT,
        help=f"Path to fine-tuned experiment dir (relative to project root or absolute). "
             f"Default: {DEFAULT_FINETUNED_EXPERIMENT}",
    )
    parser.add_argument(
        "--checkpoint-step", type=int, default=DEFAULT_CHECKPOINT_STEP,
        help=f"Checkpoint step within the experiment dir. Default: {DEFAULT_CHECKPOINT_STEP}",
    )
    parser.add_argument(
        "--num-samples", type=int, default=100,
        help="Number of queries to sample from VoiceAssistant-400K (default: 100)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--skip-rows", type=int, default=2000,
        help="Skip the first N rows of the stream before sampling (default: 2000, "
             "skips the ~1615 identity samples at the start).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip generation if output audio already exists (enables resuming)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_va400k_samples(num_samples: int, skip_rows: int = 2000) -> list:
    """Stream VoiceAssistant-400K, skip the first skip_rows rows, then take num_samples.

    Uses Audio(decode=False) to skip torchcodec decoding; audio bytes are
    decoded manually with soundfile later.
    """
    try:
        from collections import Counter
        from datasets import load_dataset, Audio as HFAudio
    except ImportError:
        print("ERROR: 'datasets' package not installed.", file=sys.stderr)
        sys.exit(1)

    print(f"Streaming gpt-omni/VoiceAssistant-400K (skip_rows={skip_rows})...")
    ds = load_dataset("gpt-omni/VoiceAssistant-400K", split="train", streaming=True)
    # Disable automatic audio decoding to avoid torchcodec version issues.
    # question_audio will be {"bytes": b"...", "path": ...} instead of a decoded array.
    ds = ds.cast_column("question_audio", HFAudio(decode=False))

    if skip_rows > 0:
        print(f"  Skipping first {skip_rows} rows...")
        ds = ds.skip(skip_rows)

    chosen = list(islice(ds, num_samples))
    split_dist = Counter(x.get("split_name", "N/A") for x in chosen)
    print(f"  Selected {len(chosen)} samples — split_name distribution: {dict(split_dist)}\n")

    return chosen


def decode_hf_audio(audio_field: dict) -> tuple[np.ndarray, int]:
    """Decode a raw HuggingFace audio field (bytes or path) into (array, sample_rate)."""
    raw_bytes = audio_field.get("bytes")
    if raw_bytes:
        arr, sr = sf.read(io.BytesIO(raw_bytes))
    else:
        arr, sr = sf.read(audio_field["path"])
    return np.array(arr, dtype=np.float32), sr


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def resample_audio(array: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return array
    try:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(target_sr, orig_sr)
        return resample_poly(array, target_sr // g, orig_sr // g).astype(np.float32)
    except ImportError:
        target_len = int(len(array) * target_sr / orig_sr)
        return np.interp(
            np.linspace(0, len(array) - 1, target_len),
            np.arange(len(array)),
            array,
        ).astype(np.float32)


def encode_audio(wav_path: Path, encoder) -> str:
    audio_tokens = encoder([str(wav_path)])[0]
    return "".join([f"<|audio_{x}|>" for x in audio_tokens])


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def build_prompt(user_tokens: str, system_prompt: str) -> str:
    return f"<|system|>\n{system_prompt}\n<|user|>\n{user_tokens}\n<|assistant|>\n"


def generate_one(prompt: str, model, tokenizer, decoder, audio_0_id: int):
    """Returns (text_output: str, waveform: np.ndarray | None)."""
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

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  {torch.cuda.device_count()} GPU(s): {torch.cuda.get_device_name(0)}")
    print()

    # Resolve checkpoint
    finetuned_exp = Path(args.finetuned_experiment)
    if not finetuned_exp.is_absolute():
        finetuned_exp = PROJECT_ROOT / finetuned_exp
    finetuned_ckpt = finetuned_exp / f"checkpoint-{args.checkpoint_step}"
    if not finetuned_ckpt.exists():
        print(f"ERROR: Fine-tuned checkpoint not found: {finetuned_ckpt}", file=sys.stderr)
        sys.exit(1)

    # Set up output directories
    output_dir = Path(args.output_dir)
    audio_dirs = {
        "query":        output_dir / "audio" / "query",
        "finetuned_va": output_dir / "audio" / "finetuned_va",
        "finetuned_na": output_dir / "audio" / "finetuned_na",
    }
    for d in audio_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    manifest_path   = output_dir / "manifest.jsonl"
    eval_jsonl_path = output_dir / "eval.jsonl"

    print(f"Fine-tuned checkpoint : {finetuned_ckpt}")
    print(f"Output dir            : {output_dir}")
    print(f"Num samples           : {args.num_samples}")
    print(f"Skip rows             : {args.skip_rows}")
    print(f"Skip existing         : {args.skip_existing}\n")

    # --- Stream and sample from HuggingFace ---
    hf_samples = load_va400k_samples(args.num_samples, args.skip_rows)

    # --- Save query WAVs to disk ---
    print("Saving query audio files...")
    samples = []
    for i, item in enumerate(hf_samples):
        sample_id = f"neutral_{i:03d}"
        query_wav = audio_dirs["query"] / f"{sample_id}.wav"

        if not (args.skip_existing and query_wav.exists()):
            arr, sr = decode_hf_audio(item["question_audio"])
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            arr = resample_audio(arr, sr, QUERY_SAMPLE_RATE)
            sf.write(str(query_wav), arr, QUERY_SAMPLE_RATE)

        samples.append({
            "id":     sample_id,
            "wav":    query_wav,
            "answer": item.get("answer", ""),
        })
        print(f"  {sample_id}: saved")

    print(f"\nSaved {len(samples)} query WAVs\n")

    # --- Load codec + tokenizer ---
    print("Loading tokenizer and speech codec...")
    tokenizer  = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    audio_0_id = tokenizer.convert_tokens_to_ids("<|audio_0|>")
    encoder    = GLM4CodecEncoder()
    decoder    = GLM4CodecDecoder(str(PROJECT_ROOT / "glm-4-voice-decoder"))
    print(f"audio_0_id = {audio_0_id}\n")

    # --- Encode all query WAVs ---
    print("Encoding query audio files...")
    for s in samples:
        s["user_tokens"] = encode_audio(s["wav"], encoder)
        print(f"  Encoded {s['id']}")

    # --- Fine-tuned model (VA + NA conditions) ---
    ft_todo = [
        s for s in samples
        if not (
            args.skip_existing
            and (audio_dirs["finetuned_va"] / f"{s['id']}.wav").exists()
            and (audio_dirs["finetuned_na"] / f"{s['id']}.wav").exists()
        )
    ]

    if ft_todo:
        print(f"\n{'='*60}")
        print(f"Fine-tuned model  ({len(ft_todo)} samples × 2 conditions)")
        print(f"  Checkpoint: {finetuned_ckpt.name}")
        print(f"{'='*60}")
        t0 = time.time()
        ft_model = AutoPeftModelForCausalLM.from_pretrained(
            str(finetuned_ckpt), device_map="auto", trust_remote_code=True
        )
        ft_model.eval()
        print(f"Fine-tuned model loaded in {time.time() - t0:.1f}s\n")

        for i, s in enumerate(ft_todo):
            print(f"\n  [{i+1}/{len(ft_todo)}] {s['id']}")

            # finetuned_va
            out_va = audio_dirs["finetuned_va"] / f"{s['id']}.wav"
            if not (args.skip_existing and out_va.exists()):
                prompt = build_prompt(s["user_tokens"], VA_SYSTEM_PROMPT)
                t1 = time.time()
                text, waveform = generate_one(prompt, ft_model, tokenizer, decoder, audio_0_id)
                elapsed = time.time() - t1
                if waveform is not None:
                    sf.write(str(out_va), waveform, DECODER_SAMPLE_RATE)
                    print(f"    [finetuned_va] {out_va.name}  ({elapsed:.1f}s)")
                    print(f"    Text: {text[:100]!r}")
                else:
                    print(f"    [finetuned_va] WARNING: no audio generated ({elapsed:.1f}s)")
                s["finetuned_va_text"] = text
            else:
                print(f"    [finetuned_va] skipped (exists)")

            # finetuned_na
            out_na = audio_dirs["finetuned_na"] / f"{s['id']}.wav"
            if not (args.skip_existing and out_na.exists()):
                prompt = build_prompt(s["user_tokens"], NA_SYSTEM_PROMPT)
                t1 = time.time()
                text, waveform = generate_one(prompt, ft_model, tokenizer, decoder, audio_0_id)
                elapsed = time.time() - t1
                if waveform is not None:
                    sf.write(str(out_na), waveform, DECODER_SAMPLE_RATE)
                    print(f"    [finetuned_na] {out_na.name}  ({elapsed:.1f}s)")
                    print(f"    Text: {text[:100]!r}")
                else:
                    print(f"    [finetuned_na] WARNING: no audio generated ({elapsed:.1f}s)")
                s["finetuned_na_text"] = text
            else:
                print(f"    [finetuned_na] skipped (exists)")

        print(f"\nUnloading fine-tuned model...")
        unload_model(ft_model)
    else:
        print("\nAll fine-tuned responses already exist — skipping")

    # --- Write manifest.jsonl ---
    print(f"\nWriting manifest: {manifest_path}")
    with open(manifest_path, "w") as f:
        for s in samples:
            def abs_if_exists(p: Path):
                return str(p.resolve()) if p.exists() else None

            rec = {
                "id":                    s["id"],
                "emotion":               "Neutral",
                "valence":               0.0,
                "arousal":               0.0,
                "query_audio":           str(s["wav"].resolve()),
                "finetuned_va_response": abs_if_exists(audio_dirs["finetuned_va"] / f"{s['id']}.wav"),
                "finetuned_na_response": abs_if_exists(audio_dirs["finetuned_na"] / f"{s['id']}.wav"),
            }
            if "finetuned_va_text" in s:
                rec["finetuned_va_text"] = s["finetuned_va_text"]
            if "finetuned_na_text" in s:
                rec["finetuned_na_text"] = s["finetuned_na_text"]
            f.write(json.dumps(rec) + "\n")

    # --- Write eval.jsonl (reference answers for coherence eval) ---
    print(f"Writing eval.jsonl: {eval_jsonl_path}")
    with open(eval_jsonl_path, "w") as f:
        for s in samples:
            ref_text = (
                f"<|system|>\nPlease respond in English.\n"
                f"<|user|>\n{s['user_tokens']}\n"
                f"<|assistant|>\n{s['answer']}\n"
            )
            entry = {
                "id":      s["id"],
                "valence": 0.0,
                "arousal": 0.0,
                "text":    ref_text,
            }
            f.write(json.dumps(entry) + "\n")

    va_ok = sum(1 for s in samples if (audio_dirs["finetuned_va"] / f"{s['id']}.wav").exists())
    na_ok = sum(1 for s in samples if (audio_dirs["finetuned_na"] / f"{s['id']}.wav").exists())
    total = len(samples)

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Samples:        {total}")
    print(f"  finetuned_va:   {va_ok}/{total}")
    print(f"  finetuned_na:   {na_ok}/{total}")
    print(f"  Manifest:       {manifest_path}")
    print(f"  eval.jsonl:     {eval_jsonl_path}")
    print(f"\nNext step (base model):")
    print(f"  conda run -n glm4voice3 --no-capture-output python -m eval.generate_responses.sympatheia_neutral.voiceassistant400k.generate_responses_va400k_glm4voice \\")
    print(f"      --manifest {manifest_path.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
