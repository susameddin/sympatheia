#!/usr/bin/env python3
"""
Generate audio responses from model conditions using NEUTRAL audio input.

Unlike generate_responses.py (which uses emotionally expressive query audio),
this script uses neutral audio from Part2 and injects emotion labels via system
prompt only.  This isolates how well models adapt to externally stated emotions
(not perceived from audio).

Conditions:
  base          — base GLM-4-Voice (THUDM/glm-4-voice-9b), no emotion conditioning
  finetuned_va  — fine-tuned LoRA checkpoint + valence/arousal in system prompt
  finetuned_na  — fine-tuned LoRA checkpoint + "User emotion N/A" prompt

Outputs:
  <output-dir>/audio/finetuned_va/  — fine-tuned with VA
  <output-dir>/audio/finetuned_na/  — fine-tuned without VA
  <output-dir>/manifest.jsonl       — metadata for judge script
  (optionally) <output-dir>/audio/base/ — if --skip-base is NOT used

The output dir is auto-constructed from the experiment name + checkpoint step
when --output-dir is not explicitly provided:
  <engram>/eval_neutral_<experiment>_ckpt<step>/

Usage:
    # Auto-versioned output dir, full eval (including base):
    conda run -n glm4voice3_eval python -m eval.generate_responses_neutral \\
        --finetuned-experiment experiments/my-experiment \\
        --checkpoint-step 1400 \\
        --num-samples 10

    # Skip base model, reuse from shared dir:
    conda run -n glm4voice3_eval python -m eval.generate_responses_neutral \\
        --finetuned-experiment experiments/my-experiment \\
        --checkpoint-step 1400 \\
        --shared-dir results/eval_neutral_shared \\
        --skip-base

    # Quick test with 2 emotions:
    conda run -n glm4voice3_eval python -m eval.generate_responses_neutral \\
        --emotions happy sad \\
        --num-samples 2 \\
        --output-dir /tmp/eval_neutral_test/
"""

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoModel, AutoTokenizer
from peft import AutoPeftModelForCausalLM

# Project root is one level up from this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vocoder import GLM4CodecEncoder, GLM4CodecDecoder
from constants import EMOTION_VA_MAPPING, ALL_EMOTIONS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "THUDM/glm-4-voice-9b"
DEFAULT_FINETUNED_EXPERIMENT = (
    "experiments/sympatheia-12emo-20260312-100309"
)
DEFAULT_CHECKPOINT_STEP = 2000
DEFAULT_NEUTRAL_AUDIO_DIR = (
    "/engram/naplab/users/sd3705/Datasets/Sympatheia_12Emo_Neutral"
    "/audio/eval/query/neutral"
)
DEFAULT_ENGRAM_BASE = "/engram/naplab/users/sd3705/emo_recog_2025s"
DECODER_SAMPLE_RATE = 22050

PLAIN_SYSTEM_PROMPT = "Please respond in English."
NA_SYSTEM_PROMPT = "Please respond in English. User emotion N/A"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate model responses for neutral-input emotion evaluation (3 conditions)"
    )
    parser.add_argument(
        "--finetuned-experiment", type=str, default=DEFAULT_FINETUNED_EXPERIMENT,
        help=f"Path to fine-tuned experiment dir (relative to project root or absolute). "
             f"Default: {DEFAULT_FINETUNED_EXPERIMENT}",
    )
    parser.add_argument(
        "--checkpoint-step", type=int, default=DEFAULT_CHECKPOINT_STEP,
        help=f"Checkpoint step to use within the fine-tuned experiment dir. "
             f"Default: {DEFAULT_CHECKPOINT_STEP}",
    )
    parser.add_argument(
        "--neutral-audio-dir", type=str, default=DEFAULT_NEUTRAL_AUDIO_DIR,
        help=f"Directory containing neutral query .wav files. Default: {DEFAULT_NEUTRAL_AUDIO_DIR}",
    )
    parser.add_argument(
        "--num-samples", type=int, default=10,
        help="Number of neutral audio files to use (same files reused across all emotions, default: 10)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for audio files and manifest.jsonl. "
             "Default: auto-constructed as <engram>/eval_neutral_<experiment>_ckpt<step>/",
    )
    parser.add_argument(
        "--shared-dir", type=str, default=None,
        help="Path to shared eval dir containing base model responses. "
             "Used with --skip-base to reuse existing base audio.",
    )
    parser.add_argument(
        "--skip-base", action="store_true",
        help="Skip base model generation. Requires --shared-dir to reference "
             "existing base responses in the manifest.",
    )
    parser.add_argument(
        "--emotions", type=str, nargs="+", default=None,
        help="Subset of emotions to evaluate (default: all 11). "
             "E.g.: --emotions happy sad angry",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip generation if output audio file already exists (enables resuming)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def sample_queries_neutral(neutral_dir: str, emotions: list, num_samples: int, seed: int) -> list:
    """Sample N neutral WAV files and replicate across all emotions.

    The same N files are reused for every emotion so that any difference in
    model responses must come from the emotion conditioning, not audio content.
    """
    rng = random.Random(seed)
    all_wavs = sorted(Path(neutral_dir).glob("*.wav"))
    if not all_wavs:
        return []
    chosen = rng.sample(all_wavs, min(num_samples, len(all_wavs)))
    chosen = sorted(chosen)

    samples = []
    print(f"\nSampling {len(chosen)} neutral queries from: {neutral_dir}")
    for emotion in sorted(emotions):
        for i, wav in enumerate(chosen):
            samples.append({
                "id": f"{emotion.lower()}_{i:02d}",
                "emotion": emotion,
                "wav": wav,
            })
        print(f"  {emotion:<12}: {len(chosen)} queries (neutral audio)")
    print(f"Total: {len(samples)} queries ({len(chosen)} unique files × {len(emotions)} emotions)\n")
    return samples


def encode_audio(wav_path: Path, encoder) -> str:
    """Encode a WAV file to a string of <|audio_X|> tokens."""
    audio_tokens = encoder([str(wav_path)])[0]
    return "".join([f"<|audio_{x}|>" for x in audio_tokens])


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

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

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  {torch.cuda.device_count()} GPU(s): {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}\n")

    # Resolve paths
    finetuned_exp = Path(args.finetuned_experiment)
    if not finetuned_exp.is_absolute():
        finetuned_exp = PROJECT_ROOT / finetuned_exp
    finetuned_ckpt = finetuned_exp / f"checkpoint-{args.checkpoint_step}"
    if not finetuned_ckpt.exists():
        print(f"ERROR: Fine-tuned checkpoint not found: {finetuned_ckpt}", file=sys.stderr)
        sys.exit(1)

    # Validate --skip-base requires --shared-dir
    if args.skip_base and not args.shared_dir:
        print("ERROR: --skip-base requires --shared-dir to reference existing base responses.",
              file=sys.stderr)
        sys.exit(1)

    # Auto-construct output dir from experiment + checkpoint if not provided
    if args.output_dir is None:
        exp_name = finetuned_exp.name
        output_dir = Path(DEFAULT_ENGRAM_BASE) / f"eval_neutral_{exp_name}_ckpt{args.checkpoint_step}"
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir

    shared_dir = Path(args.shared_dir) if args.shared_dir else None
    if shared_dir and not shared_dir.is_absolute():
        shared_dir = PROJECT_ROOT / shared_dir

    # Set up audio dirs
    if args.skip_base and shared_dir:
        audio_dirs = {
            "base":         shared_dir / "audio" / "base",
            "finetuned_va": output_dir / "audio" / "finetuned_va",
            "finetuned_na": output_dir / "audio" / "finetuned_na",
        }
    else:
        audio_dirs = {
            "base":         output_dir / "audio" / "base",
            "finetuned_va": output_dir / "audio" / "finetuned_va",
            "finetuned_na": output_dir / "audio" / "finetuned_na",
        }

    # Create output dirs (don't create base dir if using shared)
    for key, d in audio_dirs.items():
        if key == "base" and args.skip_base:
            continue
        d.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    print(f"Fine-tuned checkpoint : {finetuned_ckpt}")
    print(f"Neutral audio dir     : {args.neutral_audio_dir}")
    print(f"Output dir            : {output_dir}")
    if shared_dir:
        print(f"Shared dir            : {shared_dir}")
    print(f"Skip base             : {args.skip_base}")
    print(f"Samples per emotion   : {args.num_samples}")
    print(f"Skip existing         : {args.skip_existing}")

    # Determine emotions to evaluate
    emotions = args.emotions if args.emotions else ALL_EMOTIONS
    # Normalize capitalization
    cap_map = {e.lower(): e for e in ALL_EMOTIONS}
    emotions = [cap_map.get(e.lower(), e) for e in emotions]

    # Sample queries
    samples = sample_queries_neutral(args.neutral_audio_dir, emotions, args.num_samples, args.seed)
    if not samples:
        print("No samples found. Check --neutral-audio-dir and --emotions.", file=sys.stderr)
        sys.exit(1)

    # Load shared components (tokenizer + encoder + decoder — same for all models)
    print("Loading tokenizer and speech codec components...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    audio_0_id = tokenizer.convert_tokens_to_ids("<|audio_0|>")
    encoder = GLM4CodecEncoder()
    decoder_path = str(PROJECT_ROOT / "glm-4-voice-decoder")
    decoder = GLM4CodecDecoder(decoder_path)
    print(f"audio_0_id = {audio_0_id}\n")

    # Pre-encode all query audio files (once, reused across model conditions)
    # Note: same neutral file may be encoded multiple times (once per emotion),
    # but encoding is fast and this keeps the code simple.
    print("Encoding query audio files...")
    for s in samples:
        s["user_tokens"] = encode_audio(s["wav"], encoder)
        print(f"  Encoded {s['id']}")

    # Helper: check if a condition output exists for a sample
    def output_exists(sample, condition):
        return (audio_dirs[condition] / f"{sample['id']}.wav").exists()

    # -----------------------------------------------------------------------
    # PASS 1: Base model
    # -----------------------------------------------------------------------
    if args.skip_base:
        print(f"\nPASS 1: Skipped (--skip-base). Base audio from: {audio_dirs['base']}")
    else:
        base_todo = [s for s in samples if not (args.skip_existing and output_exists(s, "base"))]

        if base_todo:
            print(f"\n{'='*60}")
            print(f"PASS 1: Base model  ({len(base_todo)} samples)")
            print(f"{'='*60}")
            print(f"Loading base model: {BASE_MODEL_ID}")
            t0 = time.time()
            base_model = AutoModel.from_pretrained(
                BASE_MODEL_ID, trust_remote_code=True, device_map="auto"
            )
            base_model.eval()
            print(f"Base model loaded in {time.time() - t0:.1f}s")

            for i, s in enumerate(base_todo):
                out_path = audio_dirs["base"] / f"{s['id']}.wav"
                print(f"\n  [{i+1}/{len(base_todo)}] {s['id']} ({s['emotion']})")
                prompt = build_prompt(s["user_tokens"], PLAIN_SYSTEM_PROMPT)
                t1 = time.time()
                text, waveform = generate_one(prompt, base_model, tokenizer, decoder, audio_0_id)
                elapsed = time.time() - t1
                if waveform is None:
                    print(f"    WARNING: no audio generated ({elapsed:.1f}s)")
                else:
                    sf.write(str(out_path), waveform, DECODER_SAMPLE_RATE)
                    print(f"    Saved: {out_path.name}  ({elapsed:.1f}s)")
                    print(f"    Text: {text[:100]!r}")
                s["base_text"] = text

            print(f"\nUnloading base model...")
            unload_model(base_model)
        else:
            print(f"\nPASS 1: All base responses already exist — skipping")

    # -----------------------------------------------------------------------
    # PASS 2: Fine-tuned model (VA + NA conditions)
    # -----------------------------------------------------------------------
    ft_todo = [
        s for s in samples
        if not (args.skip_existing and output_exists(s, "finetuned_va") and output_exists(s, "finetuned_na"))
    ]

    if ft_todo:
        print(f"\n{'='*60}")
        print(f"PASS 2: Fine-tuned model  ({len(ft_todo)} samples × 2 conditions)")
        print(f"  Checkpoint: {finetuned_ckpt.name}")
        print(f"{'='*60}")
        t0 = time.time()
        ft_model = AutoPeftModelForCausalLM.from_pretrained(
            str(finetuned_ckpt), device_map="auto", trust_remote_code=True
        )
        ft_model.eval()
        print(f"Fine-tuned model loaded in {time.time() - t0:.1f}s")

        for i, s in enumerate(ft_todo):
            emotion = s["emotion"]
            v, a = EMOTION_VA_MAPPING.get(emotion, (0.0, 0.0))
            print(f"\n  [{i+1}/{len(ft_todo)}] {s['id']} ({emotion}, V={v:+.2f}, A={a:+.2f})")

            # finetuned_va — with emotion values
            out_va = audio_dirs["finetuned_va"] / f"{s['id']}.wav"
            if not (args.skip_existing and out_va.exists()):
                va_prompt = (
                    f"Please respond in English. "
                    f"User emotion (valence={v:.2f}, arousal={a:.2f})"
                )
                prompt = build_prompt(s["user_tokens"], va_prompt)
                t1 = time.time()
                text, waveform = generate_one(prompt, ft_model, tokenizer, decoder, audio_0_id)
                elapsed = time.time() - t1
                if waveform is None:
                    print(f"    [finetuned_va] WARNING: no audio ({elapsed:.1f}s)")
                else:
                    sf.write(str(out_va), waveform, DECODER_SAMPLE_RATE)
                    print(f"    [finetuned_va] Saved: {out_va.name}  ({elapsed:.1f}s)")
                    print(f"    Text: {text[:100]!r}")
                s["finetuned_va_text"] = text
            else:
                print(f"    [finetuned_va] Already exists, skipping")

            # finetuned_na — "User emotion N/A" prompt
            out_na = audio_dirs["finetuned_na"] / f"{s['id']}.wav"
            if not (args.skip_existing and out_na.exists()):
                prompt = build_prompt(s["user_tokens"], NA_SYSTEM_PROMPT)
                t1 = time.time()
                text, waveform = generate_one(prompt, ft_model, tokenizer, decoder, audio_0_id)
                elapsed = time.time() - t1
                if waveform is None:
                    print(f"    [finetuned_na] WARNING: no audio ({elapsed:.1f}s)")
                else:
                    sf.write(str(out_na), waveform, DECODER_SAMPLE_RATE)
                    print(f"    [finetuned_na] Saved: {out_na.name}  ({elapsed:.1f}s)")
                    print(f"    Text: {text[:100]!r}")
                s["finetuned_na_text"] = text
            else:
                print(f"    [finetuned_na] Already exists, skipping")

        print(f"\nUnloading fine-tuned model...")
        unload_model(ft_model)
    else:
        print(f"\nPASS 2: All fine-tuned responses already exist — skipping")

    # -----------------------------------------------------------------------
    # Write manifest (scan disk for what was actually produced)
    # -----------------------------------------------------------------------
    print(f"\nWriting manifest: {manifest_path}")
    with open(manifest_path, "w") as f:
        for s in samples:
            v, a = EMOTION_VA_MAPPING.get(s["emotion"], (0.0, 0.0))

            def abs_path_if_exists(p: Path):
                return str(p.resolve()) if p.exists() else None

            rec = {
                "id": s["id"],
                "emotion": s["emotion"],
                "valence": v,
                "arousal": a,
                "query_audio": str(Path(s["wav"]).resolve()),
                "base_response":         abs_path_if_exists(audio_dirs["base"]         / f"{s['id']}.wav"),
                "finetuned_va_response": abs_path_if_exists(audio_dirs["finetuned_va"] / f"{s['id']}.wav"),
                "finetuned_na_response": abs_path_if_exists(audio_dirs["finetuned_na"] / f"{s['id']}.wav"),
            }
            if "base_text" in s:
                rec["base_text"] = s["base_text"]
            if "finetuned_va_text" in s:
                rec["finetuned_va_text"] = s["finetuned_va_text"]
            if "finetuned_na_text" in s:
                rec["finetuned_na_text"] = s["finetuned_na_text"]
            f.write(json.dumps(rec) + "\n")

    total = len(samples)
    base_ok = sum(1 for s in samples if (audio_dirs["base"] / f"{s['id']}.wav").exists())
    va_ok   = sum(1 for s in samples if (audio_dirs["finetuned_va"] / f"{s['id']}.wav").exists())
    na_ok   = sum(1 for s in samples if (audio_dirs["finetuned_na"] / f"{s['id']}.wav").exists())

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Samples:        {total}")
    print(f"  base:           {base_ok}/{total}")
    print(f"  finetuned_va:   {va_ok}/{total}")
    print(f"  finetuned_na:   {na_ok}/{total}")
    print(f"  Manifest:       {manifest_path}")
    print(f"\nNext step:")
    print(f"  conda run -n qwen3omni python -m eval.judge_qwen3omni_neutral \\")
    print(f"      --manifest {manifest_path.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
