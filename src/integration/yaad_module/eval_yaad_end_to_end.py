"""End-to-end evaluation: YAAD ECG/GSR → VA → Speech → LLM Judge.

Loads precomputed YAAD predictions (from precompute_yaad.py), generates
speech responses for each sample under two conditions per modality, and
writes a manifest for downstream LLM-judge evaluation.

Conditions written per modality (e.g. for ecg):
  ecg_va   — fine-tuned model with predicted VA from ECG
  no_va    — fine-tuned model with no emotion context (baseline)

Multiple modalities can be run in one pass; they share the no_va baseline
and write into a combined manifest.

Usage:
    # Single modality
    python -m integration.yaad_module.eval_yaad_end_to_end \\
        --checkpoint experiments/sympatheia-12emo-YYYYMMDD-HHMMSS/checkpoint-N \\
        --modality ecg

    # Multiple modalities in one pass
    python -m integration.yaad_module.eval_yaad_end_to_end \\
        --checkpoint experiments/sympatheia-12emo-YYYYMMDD-HHMMSS/checkpoint-N \\
        --modality ecg gsr

    # Then judge (neutral rubric):
    # Note: run this in the Qwen3-Omni environment (see https://github.com/QwenLM/Qwen3)
    python -m eval.judge.judge_qwen3omni_neutral \\
        --manifest <output-dir>/manifest.jsonl \\
        --conditions ecg_va gsr_va no_va
"""

from __future__ import annotations

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from src.vocoder_src import GLM4CodecEncoder, GLM4CodecDecoder
from src.constants import SPEECH_ANCHORS

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_EXPERIMENTS = PROJECT_ROOT / "experiments"
_DEFAULT_CHECKPOINT = str(
    _EXPERIMENTS / "sympatheia-12emo-YYYYMMDD-HHMMSS" / "checkpoint-N"
)
_DEFAULT_EVAL_AUDIO = os.environ.get("NEUTRAL_QUERY_DIR", "/path/to/eval/query/neutral")
_DEFAULT_OUTPUT_BASE = os.environ.get("EVAL_OUTPUT_DIR", "./eval")
_DECODER_SAMPLE_RATE = 22050

_MODULE_DIR = Path(__file__).resolve().parent


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
    parser = argparse.ArgumentParser(description="YAAD end-to-end evaluation")
    parser.add_argument("--checkpoint", default=_DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--modality", choices=["ecg", "gsr", "fusion"], nargs="+", default=["ecg"],
        help="One or more modalities to evaluate. E.g. --modality ecg gsr",
    )
    parser.add_argument("--eval-audio-dir", default=_DEFAULT_EVAL_AUDIO,
                        help="Directory with neutral query WAV files")
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory. Defaults to <eval_base>/eval_yaad_<modalities>",
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

    modalities = args.modality  # list, e.g. ["ecg", "gsr"]
    mod_tag = "_".join(modalities)
    output_dir = Path(
        args.output_dir or f"{_DEFAULT_OUTPUT_BASE}/eval_yaad_{mod_tag}"
    )

    # Load precomputed predictions for each modality; verify they exist first
    all_preds: dict[str, list] = {}
    for mod in modalities:
        pred_path = _MODULE_DIR / "cache" / f"yaad_predictions_{mod}.json"
        if not pred_path.exists():
            print(f"ERROR: predictions not found: {pred_path}", file=sys.stderr)
            print(f"Run:  python -m integration.yaad_module.precompute_yaad --modality {mod}",
                  file=sys.stderr)
            sys.exit(1)
        with open(pred_path) as f:
            all_preds[mod] = json.load(f)
        print(f"Loaded {len(all_preds[mod])} {mod.upper()} predictions")

    # Load query pool — one random WAV assigned per sample (text-module style)
    rng = random.Random(args.seed)
    query_wavs = sorted(Path(args.eval_audio_dir).glob("*.wav"))
    if not query_wavs:
        print(f"ERROR: No WAV files in {args.eval_audio_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Query pool: {len(query_wavs)} WAVs, 1 randomly assigned per sample")

    # Conditions: one va-condition per modality + shared no_va baseline
    va_conds = [f"{mod}_va" for mod in modalities]
    all_conds = va_conds + ["no_va"]

    # Create output dirs
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

    # Assign one random query WAV per sample (consistent across modalities via seed)
    # Build sample_id list from the first modality to assign queries
    first_mod_preds = all_preds[modalities[0]]
    assigned_wavs = {pred["id"]: rng.choice(query_wavs) for pred in first_mod_preds}
    # For modalities with different record sets, assign independently
    for mod in modalities[1:]:
        for pred in all_preds[mod]:
            if pred["id"] not in assigned_wavs:
                assigned_wavs[pred["id"]] = rng.choice(query_wavs)

    grouped: dict[str, dict] = {}
    total = 0

    for mod in modalities:
        va_cond = f"{mod}_va"
        for pred in all_preds[mod]:
            sample_id = pred["id"]
            emotion_gt  = pred["emotion"]
            v_pred      = pred["predicted_valence"]
            a_pred      = pred["predicted_arousal"]
            top_emotion = pred["top_emotion"]
            nearest     = nearest_anchor_name(v_pred, a_pred)
            query_wav   = assigned_wavs[sample_id]
            user_tok_str = get_query_tokens(query_wav)

            if sample_id not in grouped:
                grouped[sample_id] = {
                    "id":          sample_id,
                    "emotion":     emotion_gt,
                    "valence":     pred.get("gt_valence", v_pred),
                    "arousal":     pred.get("gt_arousal", a_pred),
                    "query_audio": str(query_wav.resolve()),
                }

            grouped[sample_id][f"{mod}_top_emotion"] = top_emotion
            grouped[sample_id][f"{mod}_valence"]     = v_pred
            grouped[sample_id][f"{mod}_arousal"]     = a_pred

            # VA condition for this modality
            out_path = output_dir / "audio" / va_cond / f"{sample_id}.wav"
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
                    f"  {sample_id} [{va_cond:>10}] GT={emotion_gt:<10} "
                    f"pred={top_emotion:<10} → {nearest:<12} {status} ({elapsed:.1f}s)"
                )
            if out_path.exists():
                grouped[sample_id][f"{va_cond}_response"] = str(out_path.resolve())

            # no_va baseline (generate once per sample, shared across modalities)
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

    conditions_str = " ".join(va_conds + ["no_va"])
    print(f"\n{'='*60}")
    print(f"DONE — {total} responses generated")
    print(f"  Modalities: {modalities}")
    print(f"  Manifest  : {manifest_path}")
    for cond in all_conds:
        n = len(list((output_dir / "audio" / cond).glob("*.wav")))
        print(f"  {cond:>12}: {n} audio files")
    print(
        f"\nNext step (neutral judge):\n"
        f"  "
        f"python -m eval.judge.judge_qwen3omni_neutral \\\n"
        f"    --manifest {manifest_path.resolve()} \\\n"
        f"    --conditions {conditions_str}"
    )


if __name__ == "__main__":
    main()
