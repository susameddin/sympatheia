#!/usr/bin/env python3
"""Generate face_va and no_va responses for human-subjects experiment manifests.

Reads the manifest.jsonl produced by the experiment app, generates two responses
per query using the fine-tuned GLM-4-Voice model:
  - face_va : conditioned on the detected valence/arousal from the recording
  - no_va   : no VA conditioning (control condition)

Deduplicates rerecords by keeping the last entry per query ID.
Writes response WAVs to {session_dir}/audio/{condition}/{id}.wav and updates
manifest.jsonl with the filled-in response paths.

Usage:
    conda run -n glm4voice3 --no-capture-output python \\
        sympatheia/eval/generate_responses/sensing/generate_responses_experiment.py \\
        --manifest /engram/naplab/users/sd3705/emo_recog_2025s/experiment_face/s02/manifest.jsonl

Then run the judge:
    conda run -n qwen3omni --no-capture-output python \\
        -m eval.judge.judge_qwen3omni_neutral \\
        --manifest /engram/naplab/users/sd3705/emo_recog_2025s/experiment_face/s02/manifest.jsonl \\
        --conditions face_va no_va
"""

import argparse
import json
import os
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

from src.vocoder_src import GLM4CodecEncoder, GLM4CodecDecoder

BASE_MODEL_ID      = "THUDM/glm-4-voice-9b"
DECODER_SAMPLE_RATE = 22050
DEFAULT_CHECKPOINT = str(
    PROJECT_ROOT / "experiments" / "sympatheia-12emo-v2-20260320-100225" / "checkpoint-2800"
)


def parse_args():
    p = argparse.ArgumentParser(description="Generate experiment responses (face_va + no_va)")
    p.add_argument("--manifest", required=True, help="Path to experiment manifest.jsonl")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="LoRA checkpoint path")
    p.add_argument("--skip-existing", action="store_true", help="Skip already-generated responses")
    return p.parse_args()


def encode_audio(wav_path, encoder):
    tokens = encoder([str(wav_path)])[0]
    return "".join(f"<|audio_{x}|>" for x in tokens)


def build_prompt(user_tokens, valence=None, arousal=None):
    if valence is None:
        system = "Please respond in English."
    else:
        system = (
            f"Please respond in English. "
            f"User emotion (valence={valence:.2f}, arousal={arousal:.2f})"
        )
    return f"<|system|>\n{system}\n<|user|>\n{user_tokens}\n<|assistant|>\n"


def generate_one(prompt, model, tokenizer, decoder, audio_0_id):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, temperature=0.2, top_p=0.8, max_new_tokens=2000)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    audio_toks, text_toks = [], []
    for tok in generated:
        if tok.item() >= audio_0_id:
            audio_toks.append(tok)
        else:
            text_toks.append(tok)
    text = tokenizer.decode(text_toks, skip_special_tokens=True)
    if not audio_toks:
        return text, None
    ids_shifted = torch.tensor([[t.item() - audio_0_id for t in audio_toks]], dtype=torch.long)
    waveform = decoder(ids_shifted).squeeze().cpu().numpy()
    return text, waveform


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    session_dir = manifest_path.parent

    # Read and deduplicate: for repeated IDs keep the last occurrence (most recent rerecord)
    raw_records = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))

    seen, deduped = set(), []
    for rec in reversed(raw_records):
        if rec["id"] not in seen:
            seen.add(rec["id"])
            deduped.append(rec)
    records = list(reversed(deduped))

    print(f"Manifest : {manifest_path}")
    print(f"Records  : {len(raw_records)} raw → {len(records)} after dedup")

    # Create output dirs
    for cond in ("face_va", "no_va"):
        (session_dir / "audio" / cond).mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoading tokenizer and speech codec...")
    tokenizer  = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    audio_0_id = tokenizer.convert_tokens_to_ids("<|audio_0|>")
    encoder    = GLM4CodecEncoder()
    decoder    = GLM4CodecDecoder(str(PROJECT_ROOT / "glm-4-voice-decoder"))

    print(f"Loading model from {ckpt_path.name} ...")
    t0 = time.time()
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(ckpt_path), device_map="auto", trust_remote_code=True
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    # Generate
    for i, rec in enumerate(records):
        sample_id  = rec["id"]
        query_audio = rec["query_audio"]
        valence    = rec["valence"]
        arousal    = rec["arousal"]

        if not Path(query_audio).exists():
            print(f"[{i+1}/{len(records)}] {sample_id}: query audio not found, skipping")
            continue

        print(f"[{i+1}/{len(records)}] {sample_id}  ({rec['emotion']})  V={valence:+.2f} A={arousal:+.2f}")

        # Encode query audio once
        user_tokens = encode_audio(query_audio, encoder)

        for cond in ("face_va", "no_va"):
            out_path = session_dir / "audio" / cond / f"{sample_id}.wav"
            if args.skip_existing and out_path.exists():
                rec[f"{cond}_response"] = str(out_path)
                print(f"  [{cond}] already exists, skipping")
                continue

            v, a = (valence, arousal) if cond == "face_va" else (None, None)
            prompt = build_prompt(user_tokens, v, a)

            t1 = time.time()
            text_out, waveform = generate_one(prompt, model, tokenizer, decoder, audio_0_id)
            elapsed = time.time() - t1

            if waveform is not None:
                sf.write(str(out_path), waveform, DECODER_SAMPLE_RATE)
                rec[f"{cond}_response"] = str(out_path)
                status = "OK"
            else:
                rec[f"{cond}_response"] = None
                status = "NO_AUDIO"

            print(f"  [{cond}] {status} ({elapsed:.1f}s)  text: {text_out[:80]!r}")

    # Write updated manifest (deduped, response paths filled in)
    with open(manifest_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nUpdated manifest: {manifest_path}")
    print(f"\nNext — run judge:")
    print(
        f"  conda run -n qwen3omni --no-capture-output python \\\n"
        f"      -m eval.judge.judge_qwen3omni_neutral \\\n"
        f"      --manifest {manifest_path} \\\n"
        f"      --conditions face_va no_va"
    )


if __name__ == "__main__":
    main()
