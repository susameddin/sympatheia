"""Sensitivity analysis: how much VA noise can the speech model tolerate?

For each of 12 emotion anchors, adds Gaussian noise at σ = {0.0, 0.1, 0.2, 0.3, 0.5}
and generates speech responses.  Downstream metrics (UTMOS, WER, LLM judge)
quantify quality degradation vs. noise level.

Usage:
    conda run -n glm4voice3 --no-capture-output python -m integration.sensitivity_analysis \
        --checkpoint experiments/sympatheia-12emo-20260312-100309/checkpoint-2000

    # Then judge:
    conda run -n qwen3omni --no-capture-output python -m eval.judge.judge_sensitivity \
        --manifest <output-dir>/manifest.jsonl
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vocoder_src import GLM4CodecEncoder, GLM4CodecDecoder
from src.constants import EMOTION_VA_MAPPING

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = str(PROJECT_ROOT / "experiments" / "sympatheia-12emo-v2-20260320-100225" / "checkpoint-2200")
DEFAULT_NEUTRAL_QUERY_DIR = os.environ.get("NEUTRAL_QUERY_DIR", "/path/to/eval/query/neutral")
DEFAULT_OUTPUT_DIR = os.environ.get("EVAL_OUTPUT_DIR", "./eval/eval_sensitivity")
DECODER_SAMPLE_RATE = 22050

NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5]
N_REPEATS = 3  # repeats per (anchor, noise_level) for averaging


def build_prompt(user_tokens, valence, arousal):
    system = (
        f"Please respond in English. "
        f"User emotion (valence={valence:.2f}, arousal={arousal:.2f})"
    )
    return f"<|system|>\n{system}\n<|user|>\n{user_tokens}\n<|assistant|>\n"


def generate_one(prompt, model, tokenizer, decoder, audio_0_id):
    model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **model_inputs, temperature=0.2, top_p=0.8, max_new_tokens=2000,
        )
    generated = outputs[0][model_inputs["input_ids"].shape[1]:]

    audio_toks, text_toks = [], []
    for tok in generated:
        if tok.item() >= audio_0_id:
            audio_toks.append(tok)
        else:
            text_toks.append(tok)

    text = tokenizer.decode(text_toks, skip_special_tokens=True)
    if not audio_toks:
        return text, None

    ids_shifted = torch.tensor(
        [[t.item() - audio_0_id for t in audio_toks]], dtype=torch.long
    )
    waveform = decoder(ids_shifted).squeeze().cpu().numpy()
    return text, waveform


def parse_args():
    parser = argparse.ArgumentParser(description="VA noise sensitivity analysis")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--neutral-query-dir", default=DEFAULT_NEUTRAL_QUERY_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=NOISE_LEVELS)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    rng = np.random.RandomState(args.seed)

    # Sample n_repeats neutral queries per emotion (without replacement where possible)
    print("Sampling neutral query audio files…")
    query_map = {}
    py_rng = random.Random(args.seed)
    neutral_wavs = sorted(Path(args.neutral_query_dir).glob("*.wav"))
    if not neutral_wavs:
        print(f"ERROR: no .wav files found in {args.neutral_query_dir}", file=sys.stderr)
        sys.exit(1)
    for emotion in EMOTION_VA_MAPPING:
        if args.n_repeats <= len(neutral_wavs):
            queries = py_rng.sample(neutral_wavs, args.n_repeats)
        else:
            queries = py_rng.choices(neutral_wavs, k=args.n_repeats)
        query_map[emotion] = queries
        print(f"  {emotion}: {[q.name for q in queries]}")
    print(f"  {len(query_map)} emotions mapped to {args.n_repeats} neutral queries each\n")

    # Load model
    print("Loading tokenizer and speech codec…")
    tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-voice-9b", trust_remote_code=True)
    audio_0_id = tokenizer.convert_tokens_to_ids("<|audio_0|>")
    encoder = GLM4CodecEncoder()
    decoder = GLM4CodecDecoder(str(PROJECT_ROOT / "glm-4-voice-decoder"))

    print(f"Loading fine-tuned model: {args.checkpoint}")
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.checkpoint, device_map="auto", trust_remote_code=True
    )
    model.eval()

    # Encode queries
    print("Encoding queries…")
    encoded_queries = {}
    for emotion, wav_paths in query_map.items():
        encoded_queries[emotion] = []
        for wav_path in wav_paths:
            tokens = encoder([str(wav_path)])[0]
            encoded_queries[emotion].append("".join([f"<|audio_{x}|>" for x in tokens]))

    # Generate responses
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_records = []
    total = 0

    for emotion, (v_anchor, a_anchor) in EMOTION_VA_MAPPING.items():
        if emotion not in encoded_queries:
            continue

        for sigma in args.noise_levels:
            for rep in range(args.n_repeats):
                user_tokens = encoded_queries[emotion][rep]
                query_path = query_map[emotion][rep]

                # Add noise
                if sigma > 0:
                    noise_v = rng.normal(0, sigma)
                    noise_a = rng.normal(0, sigma)
                    v = float(np.clip(v_anchor + noise_v, -1, 1))
                    a = float(np.clip(a_anchor + noise_a, -1, 1))
                else:
                    v, a = v_anchor, a_anchor

                sample_id = f"{emotion.lower()}_s{sigma:.1f}_r{rep}"
                out_path = audio_dir / f"{sample_id}.wav"

                if args.skip_existing and out_path.exists():
                    manifest_records.append({
                        "id": sample_id, "emotion": emotion,
                        "anchor_v": v_anchor, "anchor_a": a_anchor,
                        "noisy_v": v, "noisy_a": a,
                        "sigma": sigma, "repeat": rep,
                        "query_audio": str(query_path.resolve()),
                        "response_audio": str(out_path.resolve()),
                    })
                    continue

                prompt = build_prompt(user_tokens, v, a)
                t0 = time.time()
                text, waveform = generate_one(prompt, model, tokenizer, decoder, audio_0_id)
                elapsed = time.time() - t0
                total += 1

                if waveform is not None:
                    sf.write(str(out_path), waveform, DECODER_SAMPLE_RATE)

                status = "OK" if waveform is not None else "NO_AUDIO"
                print(
                    f"  {sample_id:<30} V=({v_anchor:+.2f}→{v:+.2f}) "
                    f"A=({a_anchor:+.2f}→{a:+.2f}) σ={sigma:.1f} {status} ({elapsed:.1f}s)"
                )

                rec = {
                    "id": sample_id, "emotion": emotion,
                    "anchor_v": v_anchor, "anchor_a": a_anchor,
                    "noisy_v": v, "noisy_a": a,
                    "sigma": sigma, "repeat": rep,
                    "text": text,
                    "query_audio": str(query_path.resolve()),
                }
                if waveform is not None:
                    rec["response_audio"] = str(out_path.resolve())
                manifest_records.append(rec)

    # Write manifest
    manifest_path = output_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for rec in manifest_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\n{'='*60}")
    print(f"DONE — {total} responses generated")
    print(f"  Manifest: {manifest_path}")
    n_audio = len(list(audio_dir.glob("*.wav")))
    print(f"  Audio files: {n_audio}")
    print(f"\nExpected total: {len(query_map)} emotions × {len(args.noise_levels)} noise levels × {args.n_repeats} repeats = {len(query_map) * len(args.noise_levels) * args.n_repeats}")
    print(f"\nNext: conda run -n qwen3omni --no-capture-output python -m eval.judge.judge_sensitivity --manifest {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
