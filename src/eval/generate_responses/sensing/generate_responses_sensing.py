#!/usr/bin/env python3
"""Unified sensing generate-responses script (face | eeg | text).

Loads pre-computed predictions for the specified sensing modality and generates
Sympatheia speech model responses under the modality's conditions.

Precompute steps (run first):
    python -m integration.face_module.precompute_hsemotion   --n-per-class 150
    python -m integration.seed_module.precompute_maet        --n-per-class 6
    python -m integration.text_module.precompute_text        --n-per-class 200

Generate:
    python -m eval.generate_responses.sensing.generate_responses_sensing --modality face
    python -m eval.generate_responses.sensing.generate_responses_sensing --modality eeg
    python -m eval.generate_responses.sensing.generate_responses_sensing --modality text

Judge:
    python -m eval.judge.judge_qwen3omni_emotional --manifest <output-dir>/manifest.jsonl
"""

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from src.vocoder_src import GLM4CodecEncoder, GLM4CodecDecoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "THUDM/glm-4-voice-9b"
DEFAULT_CHECKPOINT = str(
    PROJECT_ROOT / "experiments" / "sympatheia-12emo-YYYYMMDD-HHMMSS" / "checkpoint-N"
)
DECODER_SAMPLE_RATE = 22050
DEFAULT_EVAL_AUDIO = (
    "/path/to/Sympatheia_12Emo_Neutral_v2/audio/eval/query/neutral"
)

_MODALITY_DEFAULTS = {
    "face": {
        "predictions": str(
            PROJECT_ROOT / "integration" / "face_module" / "cache" / "face_predictions.json"
        ),
        "output_dir": "eval/eval_face_hsemo",
        "conditions": ["face_va", "no_va"],
    },
    "eeg": {
        "predictions": str(
            PROJECT_ROOT / "integration" / "seed_module" / "cache" / "eeg_predictions.json"
        ),
        "output_dir": "eval/eval_eeg_maet",
        "conditions": ["eeg_only", "eye_only", "combined", "no_va"],
    },
    "text": {
        "predictions": str(
            PROJECT_ROOT / "integration" / "text_module" / "cache" / "text_predictions.json"
        ),
        "output_dir": "eval/eval_text_e2e",
        "conditions": ["text_va", "no_va"],
    },
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def assign_query_wavs(eval_audio_dir: str, samples: list, seed: int) -> list:
    rng = random.Random(seed)
    wavs = sorted(Path(eval_audio_dir).glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No .wav files found in: {eval_audio_dir}")
    print(f"Query pool: {len(wavs)} WAV(s) in {eval_audio_dir}")
    return [rng.choice(wavs) for _ in samples]


def encode_audio(wav_path, encoder) -> str:
    tokens = encoder([str(wav_path)])[0]
    return "".join(f"<|audio_{x}|>" for x in tokens)


def build_prompt(user_tokens: str, valence=None, arousal=None) -> str:
    if valence is None:
        system = "Please respond in English."
    else:
        system = (
            f"Please respond in English. "
            f"User emotion (valence={valence:.2f}, arousal={arousal:.2f})"
        )
    return f"<|system|>\n{system}\n<|user|>\n{user_tokens}\n<|assistant|>\n"


def generate_one(prompt, model, tokenizer, decoder, audio_0_id):
    """Run generation. Returns (text: str, waveform: np.ndarray | None)."""
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


# ---------------------------------------------------------------------------
# Modality-specific VA extraction
# ---------------------------------------------------------------------------

def _get_va_face(sample, cond):
    return sample["predicted_valence"], sample["predicted_arousal"]


def _get_va_eeg(sample, cond):
    return sample[cond]["predicted_valence"], sample[cond]["predicted_arousal"]


def _get_va_text(sample, cond):
    return sample["predicted_valence"], sample["predicted_arousal"]


_GET_VA = {"face": _get_va_face, "eeg": _get_va_eeg, "text": _get_va_text}


# ---------------------------------------------------------------------------
# Modality-specific base manifest record
# ---------------------------------------------------------------------------

def _base_rec_face(sample, query_wav):
    return {
        "id": sample["id"],
        "emotion": sample["emotion"],
        "valence": sample["gt_valence"],
        "arousal": sample["gt_arousal"],
        "predicted_valence": sample["predicted_valence"],
        "predicted_arousal": sample["predicted_arousal"],
        "top_emotion": sample["top_emotion"],
        "image_path": sample["image_path"],
        "query_audio": str(query_wav.resolve()),
    }


def _base_rec_eeg(sample, query_wav):
    rec = {
        "id": sample["id"],
        "emotion": sample["emotion"],
        "valence": sample["gt_valence"],
        "arousal": sample["gt_arousal"],
        "subject_id": sample["subject_id"],
        "video_id": sample["video_id"],
        "query_audio": str(query_wav.resolve()),
    }
    for mod in ("eeg_only", "eye_only", "combined"):
        rec[f"{mod}_predicted_valence"] = sample[mod]["predicted_valence"]
        rec[f"{mod}_predicted_arousal"] = sample[mod]["predicted_arousal"]
        rec[f"{mod}_top_emotion"] = sample[mod]["top_emotion"]
    return rec


def _base_rec_text(sample, query_wav):
    return {
        "id": sample["id"],
        "emotion": sample["emotion"],
        "valence": sample["gt_valence"],
        "arousal": sample["gt_arousal"],
        "predicted_valence": sample["predicted_valence"],
        "predicted_arousal": sample["predicted_arousal"],
        "top_emotion": sample["top_emotion"],
        "input_text": sample["input_text"],
        "query_audio": str(query_wav.resolve()),
    }


_BASE_REC = {"face": _base_rec_face, "eeg": _base_rec_eeg, "text": _base_rec_text}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified sensing generate-responses: face | eeg | text"
    )
    parser.add_argument(
        "--modality", required=True, choices=["face", "eeg", "text"],
        help="Sensing modality to run",
    )
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT,
        help=f"Path to fine-tuned LoRA checkpoint. Default: {DEFAULT_CHECKPOINT}",
    )
    parser.add_argument(
        "--predictions", default=None,
        help="Path to precomputed predictions JSON. Defaults to modality-specific path.",
    )
    parser.add_argument(
        "--eval-audio-dir", default=DEFAULT_EVAL_AUDIO,
        help=f"Directory with neutral query WAV files. Default: {DEFAULT_EVAL_AUDIO}",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for audio and manifest. Defaults to modality-specific path.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip generation if output audio already exists (enables resuming)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    modality = args.modality
    cfg = _MODALITY_DEFAULTS[modality]

    predictions_path = args.predictions or cfg["predictions"]
    output_dir = Path(args.output_dir or cfg["output_dir"])
    conditions = cfg["conditions"]
    get_va = _GET_VA[modality]
    get_base_rec = _BASE_REC[modality]

    print(f"Modality:       {modality}")
    print(f"Predictions:    {predictions_path}")
    print(f"Output dir:     {output_dir}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  {torch.cuda.device_count()} GPU(s): {torch.cuda.get_device_name(0)}")

    # Resolve checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    # Load predictions
    if not os.path.exists(predictions_path):
        print(f"ERROR: predictions not found: {predictions_path}", file=sys.stderr)
        print(
            f"Run first:\n"
            f"  "
            f"python -m integration.{modality}_module.precompute_{'hsemotion' if modality == 'face' else ('maet' if modality == 'eeg' else 'text')}",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(predictions_path) as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples\n")

    # Create output dirs
    for cond in conditions:
        (output_dir / "audio" / cond).mkdir(parents=True, exist_ok=True)

    # Assign query WAVs
    assigned_wavs = assign_query_wavs(args.eval_audio_dir, samples, args.seed)

    # Load tokenizer + codec
    print("\nLoading tokenizer and speech codec...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    audio_0_id = tokenizer.convert_tokens_to_ids("<|audio_0|>")
    encoder = GLM4CodecEncoder()
    decoder = GLM4CodecDecoder(str(PROJECT_ROOT / "glm-4-voice-decoder"))
    print(f"audio_0_id = {audio_0_id}")

    # Encode unique query WAVs
    print("Encoding neutral query audio files...")
    wav_token_cache = {}
    for wav in sorted(set(assigned_wavs)):
        wav_token_cache[wav] = encode_audio(wav, encoder)
        print(f"  Encoded {wav.name}")

    # Load model
    print(f"\nLoading fine-tuned model: {ckpt_path.name}")
    t0 = time.time()
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(ckpt_path), device_map="auto", trust_remote_code=True
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    # Generate responses
    manifest_records = []
    total_generated = 0

    for sample, query_wav in zip(samples, assigned_wavs):
        tok_str = wav_token_cache[query_wav]
        rec = get_base_rec(sample, query_wav)

        for cond in conditions:
            out_path = output_dir / "audio" / cond / f"{sample['id']}.wav"

            if args.skip_existing and out_path.exists():
                rec[f"{cond}_response"] = str(out_path.resolve())
                print(f"  {sample['id']} [{cond:>12}] already exists, skipping")
                continue

            v, a = (None, None) if cond == "no_va" else get_va(sample, cond)
            prompt = build_prompt(tok_str, v, a)

            t1 = time.time()
            text_out, waveform = generate_one(prompt, model, tokenizer, decoder, audio_0_id)
            elapsed = time.time() - t1
            total_generated += 1

            if waveform is not None:
                sf.write(str(out_path), waveform, DECODER_SAMPLE_RATE)
                rec[f"{cond}_response"] = str(out_path.resolve())
                status = "OK"
            else:
                status = "NO_AUDIO"

            va_str = f"VA=({v:+.2f},{a:+.2f})" if v is not None else "VA=none"
            print(
                f"  {sample['id']} [{cond:>12}] {sample['emotion']:<10} "
                f"{va_str} {status} ({elapsed:.1f}s)"
            )
            if text_out:
                print(f"    Text: {text_out[:100]!r}")

        manifest_records.append(rec)

    # Write manifest
    manifest_path = output_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for r in manifest_records:
            f.write(json.dumps(r) + "\n")

    print(f"\n{'='*60}")
    print(f"DONE — {modality} modality")
    print(f"  Samples:        {len(samples)}")
    print(f"  Unique queries: {len(set(assigned_wavs))}")
    print(f"  Generated:      {total_generated} responses")
    for cond in conditions:
        n = len(list((output_dir / "audio" / cond).glob("*.wav")))
        print(f"  {cond:>14}: {n} audio files")
    print(f"  Manifest:       {manifest_path}")
    print(
        f"\nNext:\n  "
        f"python -m eval.judge.judge_qwen3omni_emotional "
        f"--manifest {manifest_path.resolve()}"
    )
    print(f"{'='*60}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
