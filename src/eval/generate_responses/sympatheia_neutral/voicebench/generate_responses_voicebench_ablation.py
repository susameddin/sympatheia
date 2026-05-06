#!/usr/bin/env python3
"""
QA quality ablation using VoiceBench CommonEval audio.

Tests whether fine-tuning degrades general question-answering ability by
comparing three conditions on real human voices from VoiceBench CommonEval
(hlt-lab/voicebench on HuggingFace):

  base               — base GLM-4-Voice (THUDM/glm-4-voice-9b)
  finetuned_neutral_va — fine-tuned + neutral VA=(0,0) in system prompt
  finetuned_na       — fine-tuned + "User emotion N/A" prompt

Each VoiceBench audio is run once per condition (no emotion replication).
All manifest rows use emotion="Neutral" and VA=(0,0) so judge_qwen3omni_neutral
applies the neutral quality rubric (helpful, clear, friendly) rather than the
empathy rubric.

See generate_responses_voicebench.py for the companion script that tests all
12 emotions injected into the same VoiceBench audio.

Outputs:
  <output-dir>/audio_cache/                — cached query WAVs from VoiceBench
  <output-dir>/audio/base/                 — base model responses
  <output-dir>/audio/finetuned_neutral_va/ — fine-tuned with neutral VA=(0,0)
  <output-dir>/audio/finetuned_na/         — fine-tuned without VA
  <output-dir>/manifest.jsonl              — one row per audio for judge

Usage:
    # First, download VoiceBench once:
    #   python eval/retired/download_voicebench.py

    python -m eval.generate_responses.sympatheia_neutral.voicebench.generate_responses_voicebench_ablation \\
        --num-samples 50

    # Reuse audio_cache from a previous voicebench run:
    python -m eval.generate_responses.sympatheia_neutral.voicebench.generate_responses_voicebench_ablation \\
        --shared-dir /path/to/voicebench/run --skip-base
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
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vocoder_src import GLM4CodecEncoder, GLM4CodecDecoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "THUDM/glm-4-voice-9b"
DEFAULT_FINETUNED_EXPERIMENT = (
    "experiments/sympatheia-12emo-v2-20260320-100225"
)
DEFAULT_CHECKPOINT_STEP = 2200
DEFAULT_ENGRAM_BASE = "eval"
DEFAULT_VOICEBENCH_DIR = "/path/to/VoiceBench/commoneval"
DECODER_SAMPLE_RATE = 22050

PLAIN_SYSTEM_PROMPT   = "Please respond in English."
NEUTRAL_VA_PROMPT     = "Please respond in English. User emotion (valence=0.00, arousal=0.00)"
NA_SYSTEM_PROMPT      = "Please respond in English. User emotion N/A"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="QA quality ablation: does fine-tuning hurt general answering on real voices?"
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
        "--num-samples", type=int, default=50,
        help="Number of VoiceBench audio files to sample. Default: 50",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for audio files and manifest.jsonl. "
             "Default: auto-constructed as <engram>/eval_voicebench_ablation_<experiment>_ckpt<step>/",
    )
    parser.add_argument(
        "--shared-dir", type=str, default=None,
        help="Path to a previous run's output dir to reuse audio_cache and/or base responses. "
             "Used with --skip-base.",
    )
    parser.add_argument(
        "--skip-base", action="store_true",
        help="Skip base model generation. Requires --shared-dir with existing base responses.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling VoiceBench items (default: 42)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip generation if output audio file already exists (enables resuming)",
    )
    parser.add_argument(
        "--voicebench-dir", type=str, default=DEFAULT_VOICEBENCH_DIR,
        help=f"Path to local VoiceBench CommonEval directory (from download_voicebench.py). "
             f"Default: {DEFAULT_VOICEBENCH_DIR}",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def load_voicebench(num_samples: int, seed: int, voicebench_dir: str) -> list:
    """Load VoiceBench CommonEval from a local directory (created by download_voicebench.py).

    Returns list of dicts with keys: audio_id, source_index, audio_array,
    audio_sr, wav (Path, set after cache_audio).
    """
    vb_dir = Path(voicebench_dir)
    meta_path = vb_dir / "metadata.jsonl"
    if not meta_path.exists():
        print(f"ERROR: metadata.jsonl not found in {vb_dir}", file=sys.stderr)
        print(f"  Run: python eval/retired/download_voicebench.py --out-dir {vb_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading VoiceBench CommonEval from: {vb_dir}")
    with open(meta_path) as f:
        all_meta = [json.loads(line) for line in f]
    print(f"  {len(all_meta)} samples found")

    rng = random.Random(seed)
    all_indices = list(range(len(all_meta)))
    chosen = sorted(rng.sample(all_indices, min(num_samples, len(all_meta))))
    print(f"  Sampled {len(chosen)} items (seed={seed})\n")

    items = []
    for i, idx in enumerate(chosen):
        meta = all_meta[idx]
        wav_path = vb_dir / meta["audio"]
        audio_array, audio_sr = sf.read(str(wav_path))
        items.append({
            "audio_id":     f"voicebench_{i:04d}",
            "source_index": meta["index"],
            "audio_array":  audio_array,
            "audio_sr":     audio_sr,
        })
    return items


def cache_audio(items: list, cache_dir: Path, skip_existing: bool) -> None:
    """Save VoiceBench audio arrays as WAV files into cache_dir.

    Adds 'wav' (Path) to each item in-place.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        wav_path = cache_dir / f"{item['audio_id']}.wav"
        item["wav"] = wav_path
        if skip_existing and wav_path.exists():
            continue
        sf.write(str(wav_path), item["audio_array"], item["audio_sr"])
    print(f"Cached {len(items)} audio files → {cache_dir}")


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

    if args.skip_base and not args.shared_dir:
        print("ERROR: --skip-base requires --shared-dir to reference existing base responses.",
              file=sys.stderr)
        sys.exit(1)

    # Auto-construct output dir
    if args.output_dir is None:
        exp_name = finetuned_exp.name
        output_dir = Path(DEFAULT_ENGRAM_BASE) / f"eval_voicebench_ablation_{exp_name}_ckpt{args.checkpoint_step}"
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir

    shared_dir = Path(args.shared_dir) if args.shared_dir else None
    if shared_dir and not shared_dir.is_absolute():
        shared_dir = PROJECT_ROOT / shared_dir

    # Use shared audio_cache if it exists (avoids re-downloading)
    if shared_dir and (shared_dir / "audio_cache").exists():
        audio_cache_dir = shared_dir / "audio_cache"
    else:
        audio_cache_dir = output_dir / "audio_cache"

    if args.skip_base and shared_dir:
        base_audio_dir = shared_dir / "audio" / "base"
    else:
        base_audio_dir = output_dir / "audio" / "base"

    neutral_va_dir = output_dir / "audio" / "finetuned_neutral_va"
    na_audio_dir   = output_dir / "audio" / "finetuned_na"
    manifest_path  = output_dir / "manifest.jsonl"

    for d in ([neutral_va_dir, na_audio_dir] + ([] if args.skip_base else [base_audio_dir])):
        d.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fine-tuned checkpoint : {finetuned_ckpt}")
    print(f"Output dir            : {output_dir}")
    if shared_dir:
        print(f"Shared dir            : {shared_dir}")
    print(f"Skip base             : {args.skip_base}")
    print(f"Num samples           : {args.num_samples}")
    print(f"Skip existing         : {args.skip_existing}")
    print(f"VoiceBench dir        : {args.voicebench_dir}")

    # Load VoiceBench data and cache audio to disk
    samples = load_voicebench(args.num_samples, args.seed, args.voicebench_dir)
    cache_audio(samples, audio_cache_dir, args.skip_existing)
    print(f"\nTotal samples: {len(samples)}\n")

    # Shared components
    print("Loading tokenizer and speech codec components...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    audio_0_id = tokenizer.convert_tokens_to_ids("<|audio_0|>")
    encoder = GLM4CodecEncoder()
    decoder_path = str(PROJECT_ROOT / "glm-4-voice-decoder")
    decoder = GLM4CodecDecoder(decoder_path)
    print(f"audio_0_id = {audio_0_id}\n")

    # Encode all audio files
    print("Encoding query audio files...")
    for s in samples:
        s["user_tokens"] = encode_audio(s["wav"], encoder)
        print(f"  Encoded {s['audio_id']}")

    # -----------------------------------------------------------------------
    # PASS 1: Base model
    # -----------------------------------------------------------------------
    if args.skip_base:
        print(f"\nPASS 1: Skipped (--skip-base). Base audio from: {base_audio_dir}")
    else:
        base_todo = [
            s for s in samples
            if not (args.skip_existing and (base_audio_dir / f"{s['audio_id']}.wav").exists())
        ]

        if base_todo:
            print(f"\n{'='*60}")
            print(f"PASS 1: Base model  ({len(base_todo)}/{len(samples)} samples)")
            print(f"{'='*60}")
            print(f"Loading base model: {BASE_MODEL_ID}")
            t0 = time.time()
            base_model = AutoModel.from_pretrained(
                BASE_MODEL_ID, trust_remote_code=True, device_map="auto"
            )
            base_model.eval()
            print(f"Base model loaded in {time.time() - t0:.1f}s")

            for i, s in enumerate(base_todo):
                out_path = base_audio_dir / f"{s['audio_id']}.wav"
                print(f"\n  [{i+1}/{len(base_todo)}] {s['audio_id']}")
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
    # PASS 2: Fine-tuned model (finetuned_neutral_va + finetuned_na)
    # -----------------------------------------------------------------------
    ft_todo_nva = [
        s for s in samples
        if not (args.skip_existing and (neutral_va_dir / f"{s['audio_id']}.wav").exists())
    ]
    ft_todo_na = [
        s for s in samples
        if not (args.skip_existing and (na_audio_dir / f"{s['audio_id']}.wav").exists())
    ]

    if ft_todo_nva or ft_todo_na:
        print(f"\n{'='*60}")
        print(f"PASS 2: Fine-tuned model")
        print(f"  finetuned_neutral_va : {len(ft_todo_nva)}/{len(samples)} samples")
        print(f"  finetuned_na         : {len(ft_todo_na)}/{len(samples)} samples")
        print(f"  Checkpoint           : {finetuned_ckpt.name}")
        print(f"{'='*60}")
        t0 = time.time()
        ft_model = AutoPeftModelForCausalLM.from_pretrained(
            str(finetuned_ckpt), device_map="auto", trust_remote_code=True
        )
        ft_model.eval()
        print(f"Fine-tuned model loaded in {time.time() - t0:.1f}s")

        # finetuned_neutral_va — VA=(0,0)
        for i, s in enumerate(ft_todo_nva):
            out_nva = neutral_va_dir / f"{s['audio_id']}.wav"
            print(f"\n  [NVA {i+1}/{len(ft_todo_nva)}] {s['audio_id']}")
            prompt = build_prompt(s["user_tokens"], NEUTRAL_VA_PROMPT)
            t1 = time.time()
            text, waveform = generate_one(prompt, ft_model, tokenizer, decoder, audio_0_id)
            elapsed = time.time() - t1
            if waveform is None:
                print(f"    WARNING: no audio generated ({elapsed:.1f}s)")
            else:
                sf.write(str(out_nva), waveform, DECODER_SAMPLE_RATE)
                print(f"    Saved: {out_nva.name}  ({elapsed:.1f}s)")
                print(f"    Text: {text[:100]!r}")
            s["finetuned_neutral_va_text"] = text

        # finetuned_na
        for i, s in enumerate(ft_todo_na):
            out_na = na_audio_dir / f"{s['audio_id']}.wav"
            print(f"\n  [NA {i+1}/{len(ft_todo_na)}] {s['audio_id']}")
            prompt = build_prompt(s["user_tokens"], NA_SYSTEM_PROMPT)
            t1 = time.time()
            text, waveform = generate_one(prompt, ft_model, tokenizer, decoder, audio_0_id)
            elapsed = time.time() - t1
            if waveform is None:
                print(f"    WARNING: no audio generated ({elapsed:.1f}s)")
            else:
                sf.write(str(out_na), waveform, DECODER_SAMPLE_RATE)
                print(f"    Saved: {out_na.name}  ({elapsed:.1f}s)")
                print(f"    Text: {text[:100]!r}")
            s["finetuned_na_text"] = text

        print(f"\nUnloading fine-tuned model...")
        unload_model(ft_model)
    else:
        print(f"\nPASS 2: All fine-tuned responses already exist — skipping")

    # -----------------------------------------------------------------------
    # Write manifest (one row per audio; emotion=Neutral so judge uses neutral rubric)
    # -----------------------------------------------------------------------
    print(f"\nWriting manifest: {manifest_path}")

    def abs_path_if_exists(p: Path):
        return str(p.resolve()) if p.exists() else None

    with open(manifest_path, "w") as f:
        for s in samples:
            aid = s["audio_id"]
            rec = {
                "id":           aid,
                "source_index": s["source_index"],
                # emotion=Neutral so judge_qwen3omni_neutral uses the neutral quality rubric
                "emotion":      "Neutral",
                "valence":      0.0,
                "arousal":      0.0,
                "query_audio":  str((audio_cache_dir / f"{aid}.wav").resolve()),
                "base_response":                 abs_path_if_exists(base_audio_dir / f"{aid}.wav"),
                "finetuned_neutral_va_response": abs_path_if_exists(neutral_va_dir / f"{aid}.wav"),
                "finetuned_na_response":         abs_path_if_exists(na_audio_dir   / f"{aid}.wav"),
            }
            if "base_text" in s:
                rec["base_text"] = s["base_text"]
            if "finetuned_neutral_va_text" in s:
                rec["finetuned_neutral_va_text"] = s["finetuned_neutral_va_text"]
            if "finetuned_na_text" in s:
                rec["finetuned_na_text"] = s["finetuned_na_text"]
            f.write(json.dumps(rec) + "\n")

    total    = len(samples)
    base_ok  = sum(1 for s in samples if (base_audio_dir / f"{s['audio_id']}.wav").exists())
    nva_ok   = sum(1 for s in samples if (neutral_va_dir / f"{s['audio_id']}.wav").exists())
    na_ok    = sum(1 for s in samples if (na_audio_dir   / f"{s['audio_id']}.wav").exists())

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Samples               : {total}")
    print(f"  base                  : {base_ok}/{total}")
    print(f"  finetuned_neutral_va  : {nva_ok}/{total}")
    print(f"  finetuned_na          : {na_ok}/{total}")
    print(f"  Manifest              : {manifest_path}")
    print(f"\nNext step:")
    print(f"  python -m eval.judge.judge_qwen3omni_neutral \\")
    print(f"      --manifest {manifest_path.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
