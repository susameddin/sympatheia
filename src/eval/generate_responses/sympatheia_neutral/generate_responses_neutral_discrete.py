#!/usr/bin/env python3
"""
Generate audio responses from the DISCRETE-LABEL ablation model using NEUTRAL audio input.

Ablation counterpart of generate_responses_neutral_sympatheia.py, following the
SAME competitor-model convention as generate_responses_neutral_glm4voice.py /
_opens2s.py / _kimiaudio.py: it READS the original VA manifest.jsonl (produced by
generate_responses_neutral_sympatheia.py), reuses the exact same query audio for a
fair comparison, and writes a SEPARATE manifest_discrete.jsonl. Nothing in the
original manifest or audio directories is touched.

The only conditioning difference vs. the VA model is the system prompt: this
injects the discrete emotion *word* ("User emotion: Sad") instead of continuous
VA values. Point --finetuned-experiment at a checkpoint fine-tuned with
train_sympatheia.py on the discrete-label dataset variant produced by
dataset_creation/create_discrete_variant.py — the training recipe is otherwise
identical to the VA model's, so the conditioning representation is the only
controlled difference.

Conditions produced:
  finetuned_discrete    — discrete-label checkpoint + "User emotion: <Emotion>" prompt
  finetuned_discrete_na — discrete-label checkpoint + "User emotion N/A" prompt
                          (the discrete model's own no-cue baseline)

Outputs (written next to the input manifest, or into --output-dir):
  <output-dir>/audio/finetuned_discrete/      — *.wav
  <output-dir>/audio/finetuned_discrete_na/   — *.wav
  <output-dir>/manifest_discrete.jsonl        — metadata for judge script

Usage:
    python -m eval.generate_responses.sympatheia_neutral.generate_responses_neutral_discrete \\
        --manifest /path/to/eval_neutral_.../manifest.jsonl \\
        --finetuned-experiment experiments/sympatheia-discrete-YYYYMMDD-HHMMSS \\
        --checkpoint-step 1400

    # Resume:
    python -m eval.generate_responses.sympatheia_neutral.generate_responses_neutral_discrete \\
        --manifest /path/to/eval_neutral_.../manifest.jsonl \\
        --finetuned-experiment experiments/sympatheia-discrete-YYYYMMDD-HHMMSS \\
        --checkpoint-step 1400 \\
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
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

# PROJECT_ROOT is the src/ dir (4 levels up). Add both src/ and its parent
# (sympatheia/) to sys.path so `from src.vocoder_src ...` resolves no matter which
# directory the script is launched from (e.g. `-m eval...` run from src/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from src.vocoder_src import GLM4CodecEncoder, GLM4CodecDecoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "THUDM/glm-4-voice-9b"
DECODER_SAMPLE_RATE = 22050

NA_SYSTEM_PROMPT = "Please respond in English. User emotion N/A"

# The two conditions this script produces: manifest key prefix -> audio subdir.
CONDITIONS = ["finetuned_discrete", "finetuned_discrete_na"]


def discrete_system_prompt(emotion: str) -> str:
    """Discrete-label system prompt (the ablation of the VA prompt)."""
    return f"Please respond in English. User emotion: {emotion}"


def remap_path(path: str, root_maps) -> str:
    """Apply OLD=NEW prefix substitutions to a query_audio path (no-op if root_maps is None).

    Useful when the input manifest's audio paths were moved/renamed since it was
    written (e.g. a dataset dir renamed). Substitutions are applied in order.
    """
    if root_maps:
        for m in root_maps:
            old, _, new = m.partition("=")
            path = path.replace(old, new)
    return path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate discrete-label GLM-4-Voice responses for neutral-input evaluation "
                    "(reads the original VA manifest, writes manifest_discrete.jsonl)"
    )
    parser.add_argument(
        "--manifest", type=str, required=True,
        help="Path to the ORIGINAL VA manifest.jsonl from "
             "generate_responses_neutral_sympatheia.py (used to select the same query audio files). "
             "This file is read-only and never modified.",
    )
    parser.add_argument(
        "--finetuned-experiment", type=str, required=True,
        help="Path to the DISCRETE-trained experiment dir (relative to project root or absolute).",
    )
    parser.add_argument(
        "--checkpoint-step", type=int, required=True,
        help="Checkpoint step to use within the discrete experiment dir.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory. Defaults to the same directory as --manifest.",
    )
    parser.add_argument(
        "--audio-root-map", action="append", default=None, metavar="OLD=NEW",
        help="Remap query_audio path prefixes (repeatable), e.g. "
             "--audio-root-map /old/root=/new/root. Use when the manifest's audio "
             "paths were moved/renamed. Applied to each query_audio before use; "
             "the manifest file itself is not modified.",
    )
    parser.add_argument(
        "--skip-na", action="store_true",
        help="Generate ONLY the finetuned_discrete condition; skip the "
             "finetuned_discrete_na baseline entirely.",
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

    # Resolve discrete checkpoint
    finetuned_exp = Path(args.finetuned_experiment)
    if not finetuned_exp.is_absolute():
        finetuned_exp = PROJECT_ROOT / finetuned_exp
    finetuned_ckpt = finetuned_exp / f"checkpoint-{args.checkpoint_step}"
    if not finetuned_ckpt.exists():
        print(f"ERROR: discrete checkpoint not found: {finetuned_ckpt}", file=sys.stderr)
        sys.exit(1)

    # Which conditions to actually generate (--skip-na drops the NA baseline).
    active_conditions = ["finetuned_discrete"] if args.skip_na else list(CONDITIONS)

    output_dir = Path(args.output_dir) if args.output_dir else manifest_path.parent
    audio_dirs = {c: output_dir / "audio" / c for c in CONDITIONS}
    for cond in active_conditions:
        audio_dirs[cond].mkdir(parents=True, exist_ok=True)
    out_manifest = output_dir / "manifest_discrete.jsonl"

    print(f"Manifest (input)      : {manifest_path}")
    print(f"Discrete checkpoint   : {finetuned_ckpt}")
    print(f"Output dir            : {output_dir}")
    print(f"Out manifest          : {out_manifest}")
    print(f"Skip existing         : {args.skip_existing}\n")

    # Load source (VA) manifest — READ ONLY
    records = []
    with open(manifest_path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records from manifest")

    # Load existing output manifest for resume (text carry-over)
    out_records_map: dict = {}
    if args.skip_existing and out_manifest.exists():
        with open(out_manifest) as f:
            for line in f:
                r = json.loads(line)
                out_records_map[r["id"]] = r
        print(f"Resuming: {len(out_records_map)} records already in {out_manifest.name}\n")

    def all_conditions_done(sample_id):
        return all((audio_dirs[c] / f"{sample_id}.wav").exists() for c in active_conditions)

    # Determine which samples need generation
    todo = [rec for rec in records if not (args.skip_existing and all_conditions_done(rec["id"]))]
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

        print(f"Loading discrete model: {finetuned_ckpt}")
        t0 = time.time()
        model = AutoPeftModelForCausalLM.from_pretrained(
            str(finetuned_ckpt), device_map="auto", trust_remote_code=True
        )
        model.eval()
        print(f"Model loaded in {time.time() - t0:.1f}s\n")

        for idx, rec in enumerate(todo):
            sample_id   = rec["id"]
            query_audio = remap_path(rec["query_audio"], args.audio_root_map)
            emotion     = rec.get("emotion", "Neutral")
            print(f"\n[{idx+1}/{len(todo)}] {sample_id}  ({emotion})")

            if not Path(query_audio).exists():
                print(f"  SKIP: query audio not found: {query_audio}")
                continue

            user_tokens = encode_audio(Path(query_audio), encoder)

            # Prompts per condition (same query audio, different system prompt)
            prompts = {
                "finetuned_discrete":    discrete_system_prompt(emotion),
                "finetuned_discrete_na": NA_SYSTEM_PROMPT,
            }

            texts = {}
            for cond in active_conditions:
                out_wav = audio_dirs[cond] / f"{sample_id}.wav"
                if args.skip_existing and out_wav.exists():
                    print(f"  [{cond}] already exists, skipping")
                    continue
                prompt = build_prompt(user_tokens, prompts[cond])
                t1 = time.time()
                try:
                    text, waveform = generate_one(prompt, model, tokenizer, decoder, audio_0_id)
                except Exception as e:
                    print(f"  [{cond}] ERROR during generation: {e}")
                    continue
                elapsed = time.time() - t1
                if waveform is not None:
                    sf.write(str(out_wav), waveform, DECODER_SAMPLE_RATE)
                    print(f"  [{cond}] Saved: {out_wav.name}  ({elapsed:.1f}s)")
                else:
                    print(f"  [{cond}] WARNING: no audio ({elapsed:.1f}s)")
                if text:
                    print(f"  [{cond}] Text: {text[:100]!r}")
                texts[cond] = text or None

            # Merge with any carried-over text from a previous resume run
            prev = out_records_map.get(sample_id, {})
            out_records_map[sample_id] = {
                "id":          sample_id,
                "emotion":     emotion,
                "valence":     rec.get("valence"),
                "arousal":     rec.get("arousal"),
                "query_audio": query_audio,
                "finetuned_discrete_text":    texts.get("finetuned_discrete", prev.get("finetuned_discrete_text")),
                "finetuned_discrete_na_text": texts.get("finetuned_discrete_na", prev.get("finetuned_discrete_na_text")),
            }

        print(f"\nUnloading model...")
        unload_model(model)

    # Write output manifest (all records, both existing and new). Original manifest untouched.
    print(f"\nWriting manifest: {out_manifest}")

    def abs_if_exists(p: Path):
        return str(p.resolve()) if p.exists() else None

    with open(out_manifest, "w") as f:
        for rec in records:
            sample_id = rec["id"]
            existing = out_records_map.get(sample_id, {})
            out_rec = {
                "id":          sample_id,
                "emotion":     rec.get("emotion"),
                "valence":     rec.get("valence"),
                "arousal":     rec.get("arousal"),
                "query_audio": remap_path(rec.get("query_audio"), args.audio_root_map),
                "finetuned_discrete_response":    abs_if_exists(audio_dirs["finetuned_discrete"] / f"{sample_id}.wav"),
                "finetuned_discrete_text":        existing.get("finetuned_discrete_text"),
                "finetuned_discrete_na_response": abs_if_exists(audio_dirs["finetuned_discrete_na"] / f"{sample_id}.wav"),
                "finetuned_discrete_na_text":     existing.get("finetuned_discrete_na_text"),
            }
            f.write(json.dumps(out_rec) + "\n")

    total = len(records)

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Samples:               {total}")
    for cond in active_conditions:
        n_ok = sum(1 for rec in records if (audio_dirs[cond] / f"{rec['id']}.wav").exists())
        print(f"  {cond + ':':<22} {n_ok}/{total}")
    print(f"  Manifest:              {out_manifest}")
    print(f"\nNext step:")
    print(f"  python -m eval.judge.judge_qwen3omni_neutral \\")
    print(f"      --manifest {out_manifest.resolve()} \\")
    print(f"      --conditions {' '.join(active_conditions)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
