#!/usr/bin/env python3
"""
Generate audio responses from fine-tuned GLM-4-Voice for empathy evaluation.

Conditions:
  finetuned_va  — fine-tuned LoRA checkpoint + valence/arousal in system prompt
  finetuned_na  — fine-tuned LoRA checkpoint + no emotion values (plain prompt)

Outputs:
  <output-dir>/audio/finetuned_va/  — fine-tuned with VA
  <output-dir>/audio/finetuned_na/  — fine-tuned without VA
  <output-dir>/manifest.jsonl       — metadata for judge script

The output dir is auto-constructed from the experiment name + checkpoint step
when --output-dir is not explicitly provided:
  eval_emotional_<experiment>_ckpt<step>/

For base GLM-4-Voice responses, use generate_responses_emotional_glm4voice_base.py.

Usage:
    python -m eval.generate_responses.sympatheia_emotional.generate_responses_emotional_sympatheia \\
        --finetuned-experiment experiments/my-experiment \\
        --checkpoint-step 1400 \\
        --num-samples 10

    # Resume:
    python -m eval.generate_responses.sympatheia_emotional.generate_responses_emotional_sympatheia \\
        --finetuned-experiment experiments/my-experiment \\
        --checkpoint-step 1400 \\
        --skip-existing

    # Quick test with 2 emotions:
    python -m eval.generate_responses.sympatheia_emotional.generate_responses_emotional_sympatheia \\
        --emotions happy sad \\
        --num-samples 2 \\
        --output-dir /tmp/eval_test/
"""

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

# Project root is 4 levels up from this file (eval/generate_responses/sympatheia_emotional/ -> sympatheia/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vocoder_src import GLM4CodecEncoder, GLM4CodecDecoder
from src.constants import EMOTION_VA_MAPPING, ALL_EMOTIONS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "THUDM/glm-4-voice-9b"
DEFAULT_FINETUNED_EXPERIMENT = (
    "experiments/sympatheia-12emo-YYYYMMDD-HHMMSS"
)
DEFAULT_CHECKPOINT_STEP = 2000
DEFAULT_EVAL_AUDIO_DIR = "/path/to/Sympatheia_12Emo_Emotional_v2/audio/eval"
DEFAULT_ENGRAM_BASE = "eval"
DECODER_SAMPLE_RATE = 22050

PLAIN_SYSTEM_PROMPT = "Please respond in English."


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate fine-tuned GLM-4-Voice responses for empathy evaluation (VA + NA conditions)"
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
        "--eval-audio-dir", type=str, default=DEFAULT_EVAL_AUDIO_DIR,
        help=f"Root dir containing {{emotion}}_query/ subdirs (input queries). Default: {DEFAULT_EVAL_AUDIO_DIR}",
    )
    parser.add_argument(
        "--num-samples", type=int, default=10,
        help="Number of query audio files to sample per emotion (default: 10)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for audio files and manifest.jsonl. "
             "Default: auto-constructed as <engram>/eval_emotional_<experiment>_ckpt<step>/",
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

def sample_queries(eval_audio_dir: str, emotions: list, num_samples: int, seed: int) -> list:
    """Sample up to num_samples WAV files per emotion from {eval_audio_dir}/{emotion}_query/."""
    rng = random.Random(seed)
    samples = []
    print(f"\nSampling queries from: {eval_audio_dir}")
    for emotion in sorted(emotions):
        query_dir = Path(eval_audio_dir) / f"{emotion.lower()}_query"
        if not query_dir.exists():
            print(f"  WARNING: {query_dir} not found — skipping {emotion}")
            continue
        wavs = sorted(query_dir.glob("*.wav"))
        if not wavs:
            print(f"  WARNING: no .wav files in {query_dir} — skipping {emotion}")
            continue
        chosen = rng.sample(wavs, min(num_samples, len(wavs)))
        for i, wav in enumerate(sorted(chosen)):
            samples.append({
                "id": f"{emotion.lower()}_{i:02d}",
                "emotion": emotion,
                "wav": wav,
            })
        print(f"  {emotion:<12}: {len(chosen)} queries")
    print(f"Total: {len(samples)} queries\n")
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

    # Auto-construct output dir from experiment + checkpoint if not provided
    if args.output_dir is None:
        exp_name = finetuned_exp.name
        output_dir = Path(DEFAULT_ENGRAM_BASE) / f"eval_emotional_{exp_name}_ckpt{args.checkpoint_step}"
    else:
        output_dir = Path(args.output_dir)

    audio_dirs = {
        "finetuned_va": output_dir / "audio" / "finetuned_va",
        "finetuned_na": output_dir / "audio" / "finetuned_na",
    }
    for d in audio_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    print(f"Fine-tuned checkpoint : {finetuned_ckpt}")
    print(f"Eval audio dir        : {args.eval_audio_dir}")
    print(f"Output dir            : {output_dir}")
    print(f"Samples per emotion   : {args.num_samples}")
    print(f"Skip existing         : {args.skip_existing}")

    # Determine emotions to evaluate
    emotions = args.emotions if args.emotions else ALL_EMOTIONS
    cap_map = {e.lower(): e for e in ALL_EMOTIONS}
    emotions = [cap_map.get(e.lower(), e) for e in emotions]

    # Sample queries
    samples = sample_queries(args.eval_audio_dir, emotions, args.num_samples, args.seed)
    if not samples:
        print("No samples found. Check --eval-audio-dir and --emotions.", file=sys.stderr)
        sys.exit(1)

    # Load shared components (tokenizer + encoder + decoder)
    print("Loading tokenizer and speech codec components...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    audio_0_id = tokenizer.convert_tokens_to_ids("<|audio_0|>")
    encoder = GLM4CodecEncoder()
    decoder_path = str(PROJECT_ROOT / "glm-4-voice-decoder")
    decoder = GLM4CodecDecoder(decoder_path)
    print(f"audio_0_id = {audio_0_id}\n")

    # Pre-encode all query audio files
    print("Encoding query audio files...")
    for s in samples:
        s["user_tokens"] = encode_audio(s["wav"], encoder)
        print(f"  Encoded {s['id']}")

    def output_exists(sample, condition):
        return (audio_dirs[condition] / f"{sample['id']}.wav").exists()

    # -----------------------------------------------------------------------
    # Fine-tuned model (VA + NA conditions)
    # -----------------------------------------------------------------------
    ft_todo = [
        s for s in samples
        if not (args.skip_existing and output_exists(s, "finetuned_va") and output_exists(s, "finetuned_na"))
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

            # finetuned_na — no emotion values (plain prompt)
            out_na = audio_dirs["finetuned_na"] / f"{s['id']}.wav"
            if not (args.skip_existing and out_na.exists()):
                prompt = build_prompt(s["user_tokens"], PLAIN_SYSTEM_PROMPT)
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
        print(f"\nAll fine-tuned responses already exist — skipping")

    # -----------------------------------------------------------------------
    # Write manifest
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
                "finetuned_va_response": abs_path_if_exists(audio_dirs["finetuned_va"] / f"{s['id']}.wav"),
                "finetuned_na_response": abs_path_if_exists(audio_dirs["finetuned_na"] / f"{s['id']}.wav"),
            }
            if "finetuned_va_text" in s:
                rec["finetuned_va_text"] = s["finetuned_va_text"]
            if "finetuned_na_text" in s:
                rec["finetuned_na_text"] = s["finetuned_na_text"]
            f.write(json.dumps(rec) + "\n")

    total = len(samples)
    va_ok = sum(1 for s in samples if (audio_dirs["finetuned_va"] / f"{s['id']}.wav").exists())
    na_ok = sum(1 for s in samples if (audio_dirs["finetuned_na"] / f"{s['id']}.wav").exists())

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Samples:        {total}")
    print(f"  finetuned_va:   {va_ok}/{total}")
    print(f"  finetuned_na:   {na_ok}/{total}")
    print(f"  Manifest:       {manifest_path}")
    print(f"\nNext step:")
    print(f"  python -m eval.judge.judge_qwen3omni_emotional \\")
    print(f"      --manifest {manifest_path.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
