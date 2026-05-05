"""End-to-end evaluation: SEED-VII EEG/Eye → VA → Speech → LLM Judge.

Loads precomputed MAET predictions (from precompute_maet.py), generates
speech responses under selected conditions, and writes a manifest for
downstream LLM-judge evaluation.

Conditions (one VA-conditioned per modality + shared no_va baseline):
  eeg_only  — fine-tuned model with EEG-only predicted VA
  eye_only  — fine-tuned model with Eye-only predicted VA
  combined  — fine-tuned model with EEG+Eye combined predicted VA
  no_va     — fine-tuned model with no emotion context (baseline)

Usage:
    # All three modalities (default)
    conda run -n glm4voice3 --no-capture-output python -m integration.seed_module.eval_seed_end_to_end \\
        --checkpoint experiments/sympatheia-12emo-v2-20260320-100225/checkpoint-3930

    # Subset of modalities
    conda run -n glm4voice3 --no-capture-output python -m integration.seed_module.eval_seed_end_to_end \\
        --checkpoint experiments/sympatheia-12emo-v2-20260320-100225/checkpoint-3930 \\
        --modality eeg_only combined

    # Then judge (neutral rubric):
    conda run -n qwen3omni --no-capture-output python -m eval.judge.judge_qwen3omni_neutral \\
        --manifest <output-dir>/manifest.jsonl \\
        --conditions eeg_only eye_only combined no_va
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vocoder_src import GLM4CodecEncoder, GLM4CodecDecoder
from src.constants import SPEECH_ANCHORS

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_CHECKPOINT = str(
    PROJECT_ROOT / "experiments"
    / "sympatheia-12emo-v2-20260320-100225" / "checkpoint-2200"
)
_DEFAULT_EVAL_AUDIO = os.environ.get("NEUTRAL_QUERY_DIR", "/path/to/eval/query/neutral")
_DEFAULT_OUTPUT_BASE = os.environ.get("EVAL_OUTPUT_DIR", "./eval")
_DECODER_SAMPLE_RATE = 22050

_MODULE_DIR = Path(__file__).resolve().parent
_DEFAULT_PREDICTIONS = str(_MODULE_DIR / "cache" / "eeg_predictions.json")

ALL_MODALITIES = ["eeg_only", "eye_only", "combined"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nearest_anchor_name(v: float, a: float) -> str:
    best_dist, best_name = float("inf"), "neutral"
    for name, (av, aa) in SPEECH_ANCHORS.items():
        d = (v - av) ** 2 + (a - aa) ** 2
        if d < best_dist:
            best_dist, best_name = d, name
    return best_name


def build_prompt(user_tokens: str, valence, arousal) -> str:
    if valence is None:
        system = "Please respond in English. User emotion N/A"
    else:
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SEED-VII end-to-end evaluation")
    parser.add_argument("--checkpoint", default=_DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--modality", choices=ALL_MODALITIES, nargs="+", default=ALL_MODALITIES,
        help="Which MAET modalities to run. Default: all three",
    )
    parser.add_argument(
        "--predictions", default=_DEFAULT_PREDICTIONS,
        help="Path to precomputed predictions JSON",
    )
    parser.add_argument("--eval-audio-dir", default=_DEFAULT_EVAL_AUDIO)
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory. Defaults to <eval_base>/eval_seed_<modalities>",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--device",        default="cuda")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    modalities = args.modality
    mod_tag = "_".join(modalities)
    output_dir = Path(
        args.output_dir or f"{_DEFAULT_OUTPUT_BASE}/eval_seed_{mod_tag}"
    )

    # Load precomputed predictions
    if not os.path.exists(args.predictions):
        print(f"ERROR: predictions not found: {args.predictions}", file=sys.stderr)
        print("Run:  conda run -n s --no-capture-output "
              "python -m integration.seed_module.precompute_maet", file=sys.stderr)
        sys.exit(1)

    with open(args.predictions) as f:
        predictions = json.load(f)
    print(f"Loaded {len(predictions)} predictions from {args.predictions}")

    # Load query pool — one random WAV assigned per sample (text-module style)
    rng = random.Random(args.seed)
    query_wavs = sorted(Path(args.eval_audio_dir).glob("*.wav"))
    if not query_wavs:
        print(f"ERROR: No WAV files in {args.eval_audio_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Query pool: {len(query_wavs)} WAVs, 1 randomly assigned per sample")
    assigned_wavs = {pred["id"]: rng.choice(query_wavs) for pred in predictions}

    all_conds = modalities + ["no_va"]
    output_dir.mkdir(parents=True, exist_ok=True)
    for cond in all_conds:
        (output_dir / "audio" / cond).mkdir(parents=True, exist_ok=True)

    # Load speech model
    print(f"\nLoading tokenizer and speech codec…")
    tokenizer = AutoTokenizer.from_pretrained(
        "THUDM/glm-4-voice-9b", trust_remote_code=True
    )
    audio_0_id = tokenizer.convert_tokens_to_ids("<|audio_0|>")
    encoder = GLM4CodecEncoder()
    decoder = GLM4CodecDecoder(str(PROJECT_ROOT / "glm-4-voice-decoder"))

    print(f"Loading fine-tuned model: {args.checkpoint}")
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.checkpoint, device_map="auto", trust_remote_code=True
    )
    model.eval()

    # Encode queries lazily (cache to avoid re-encoding the same WAV)
    wav_token_cache: dict[Path, str] = {}

    def get_query_tokens(wav: Path) -> str:
        if wav not in wav_token_cache:
            tokens = encoder([str(wav)])[0]
            wav_token_cache[wav] = "".join([f"<|audio_{x}|>" for x in tokens])
        return wav_token_cache[wav]

    grouped: dict[str, dict] = {}
    total = 0

    for pred in predictions:
        sample_id  = pred["id"]
        emotion_gt = pred["emotion"]
        query_wav  = assigned_wavs[sample_id]
        user_tok_str = get_query_tokens(query_wav)

        if sample_id not in grouped:
            grouped[sample_id] = {
                "id":          sample_id,
                "emotion":     emotion_gt,
                "valence":     pred.get("gt_valence", 0.0),
                "arousal":     pred.get("gt_arousal", 0.0),
                "subject_id":  pred.get("subject_id"),
                "query_audio": str(query_wav.resolve()),
            }

            # VA-conditioned responses for each modality
            for mod in modalities:
                mod_data = pred[mod]
                v_pred = mod_data["predicted_valence"]
                a_pred = mod_data["predicted_arousal"]
                top_emotion = mod_data["top_emotion"]
                nearest = nearest_anchor_name(v_pred, a_pred)

                grouped[sample_id][f"{mod}_top_emotion"] = top_emotion
                grouped[sample_id][f"{mod}_valence"] = v_pred
                grouped[sample_id][f"{mod}_arousal"] = a_pred

                out_path = output_dir / "audio" / mod / f"{sample_id}.wav"
                if not (args.skip_existing and out_path.exists()):
                    prompt = build_prompt(user_tok_str, v_pred, a_pred)
                    t0 = time.time()
                    text, waveform = generate_one(prompt, model, tokenizer, decoder, audio_0_id)
                    elapsed = time.time() - t0
                    total += 1
                    if waveform is not None:
                        sf.write(str(out_path), waveform, _DECODER_SAMPLE_RATE)
                    status = "OK" if waveform is not None else "NO_AUDIO"
                    print(
                        f"  {sample_id} [{mod:>10}] GT={emotion_gt:<10} "
                        f"pred={top_emotion:<10} → {nearest:<12} {status} ({elapsed:.1f}s)"
                    )
                if out_path.exists():
                    grouped[sample_id][f"{mod}_response"] = str(out_path.resolve())

            # no_va baseline — generate once, shared across modalities
            no_va_path = output_dir / "audio" / "no_va" / f"{sample_id}.wav"
            if "no_va_response" not in grouped[sample_id]:
                if not (args.skip_existing and no_va_path.exists()):
                    prompt = build_prompt(user_tok_str, None, None)
                    t0 = time.time()
                    text, waveform = generate_one(prompt, model, tokenizer, decoder, audio_0_id)
                    elapsed = time.time() - t0
                    total += 1
                    if waveform is not None:
                        sf.write(str(no_va_path), waveform, _DECODER_SAMPLE_RATE)
                    status = "OK" if waveform is not None else "NO_AUDIO"
                    print(
                        f"  {sample_id} [    no_va] GT={emotion_gt:<10} "
                        f"{'baseline':<10}   {'N/A':<12} {status} ({elapsed:.1f}s)"
                    )
                if no_va_path.exists():
                    grouped[sample_id]["no_va_response"] = str(no_va_path.resolve())

    # Write manifest
    manifest_path = output_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for rec in grouped.values():
            f.write(json.dumps(rec) + "\n")

    conditions_str = " ".join(modalities + ["no_va"])
    print(f"\n{'='*60}")
    print(f"DONE — {total} responses generated")
    print(f"  Modalities: {modalities}")
    print(f"  Manifest  : {manifest_path}")
    for cond in all_conds:
        n = len(list((output_dir / "audio" / cond).glob("*.wav")))
        print(f"  {cond:>12}: {n} audio files")
    print(
        f"\nNext step (neutral judge):\n"
        f"  conda run -n qwen3omni --no-capture-output "
        f"python -m eval.judge.judge_qwen3omni_neutral \\\n"
        f"    --manifest {manifest_path.resolve()} \\\n"
        f"    --conditions {conditions_str}"
    )


if __name__ == "__main__":
    main()
