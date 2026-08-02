#!/usr/bin/env python3
"""
DISCRETE-LABEL ablation on VoiceBench CommonEval (real human voices).

Discrete counterpart of generate_responses_voicebench.py, following the same
competitor-model convention as the other _discrete.py scripts: it READS the
original VA VoiceBench manifest.jsonl (produced by generate_responses_voicebench.py),
reuses the exact same cached query audio for a fair comparison, and writes a
SEPARATE manifest_discrete.jsonl. Nothing in the original manifest or audio
directories is touched.

The only conditioning difference vs. the VA model is the system prompt: this
injects the discrete emotion *word* ("User emotion: Sad") instead of continuous
VA values. Point --finetuned-experiment at a checkpoint fine-tuned with
train_sympatheia.py on the discrete-label dataset variant produced by
dataset_creation/create_discrete_variant.py — the training recipe is otherwise
identical to the VA model's, so the conditioning representation is the only
controlled difference.

Conditions produced:
  finetuned_discrete    — discrete-label checkpoint + "User emotion: <Emotion>" prompt,
                          generated once per (emotion, audio) row.
  finetuned_discrete_na — discrete-label checkpoint + "User emotion N/A" prompt,
                          generated ONCE per unique audio (the prompt is identical
                          across emotions), then referenced by all 12 emotion rows —
                          mirrors the ~11x saving in generate_responses_voicebench.py.

The manifest keeps emotion per row (Angry, Happy, ...) so judge_qwen3omni_neutral
applies its emotion-adaptation rubric on out-of-domain real speech.

Outputs (written next to the input manifest, or into --output-dir):
  <output-dir>/audio/finetuned_discrete/      — {id}.wav       (per emotion×audio)
  <output-dir>/audio/finetuned_discrete_na/   — {audio_id}.wav (once per audio)
  <output-dir>/manifest_discrete.jsonl        — metadata for judge script

Usage:
    python -m eval.generate_responses.sympatheia_neutral.voicebench.generate_responses_voicebench_discrete \\
        --manifest /path/to/eval_voicebench_.../manifest.jsonl \\
        --finetuned-experiment experiments/sympatheia-discrete-YYYYMMDD-HHMMSS \\
        --checkpoint-step 2200

    # Resume:
    python -m eval.generate_responses.sympatheia_neutral.voicebench.generate_responses_voicebench_discrete \\
        --manifest /path/to/eval_voicebench_.../manifest.jsonl \\
        --finetuned-experiment experiments/sympatheia-discrete-YYYYMMDD-HHMMSS \\
        --checkpoint-step 2200 \\
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

# This file is at src/eval/generate_responses/sympatheia_neutral/voicebench/ — one
# level deeper than the other _discrete scripts. Add both src/ and its parent
# (sympatheia/) to sys.path so `from src.vocoder_src ...` resolves no matter which
# directory the script is launched from.
_HERE = Path(__file__).resolve()
SRC_DIR = _HERE.parents[4]        # .../sympatheia/src
PROJECT_DIR = _HERE.parents[5]    # .../sympatheia
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from src.vocoder_src import GLM4CodecEncoder, GLM4CodecDecoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "THUDM/glm-4-voice-9b"
DECODER_SAMPLE_RATE = 22050

NA_SYSTEM_PROMPT = "Please respond in English. User emotion N/A"


def discrete_system_prompt(emotion: str) -> str:
    """Discrete-label system prompt (the ablation of the VA prompt)."""
    return f"Please respond in English. User emotion: {emotion}"


def remap_path(path: str, root_maps) -> str:
    """Apply OLD=NEW prefix substitutions to a query_audio path (no-op if root_maps is None)."""
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
        description="Discrete-label ablation on VoiceBench CommonEval "
                    "(reads the original VA manifest, writes manifest_discrete.jsonl)"
    )
    parser.add_argument(
        "--manifest", type=str, required=True,
        help="Path to the ORIGINAL VA VoiceBench manifest.jsonl from "
             "generate_responses_voicebench.py (used to select the same cached query audio). "
             "This file is read-only and never modified.",
    )
    parser.add_argument(
        "--finetuned-experiment", type=str, required=True,
        help="Path to the DISCRETE-trained experiment dir (relative to src/ or absolute).",
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
        help="Remap query_audio path prefixes (repeatable). Usually unnecessary for "
             "VoiceBench since query audio is cached locally alongside the manifest.",
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
        finetuned_exp = SRC_DIR / finetuned_exp
    finetuned_ckpt = finetuned_exp / f"checkpoint-{args.checkpoint_step}"
    if not finetuned_ckpt.exists():
        print(f"ERROR: discrete checkpoint not found: {finetuned_ckpt}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else manifest_path.parent
    discrete_dir = output_dir / "audio" / "finetuned_discrete"
    na_dir       = output_dir / "audio" / "finetuned_discrete_na"
    discrete_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_na:
        na_dir.mkdir(parents=True, exist_ok=True)
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
    print(f"Loaded {len(records)} rows from manifest "
          f"({len(set(r.get('audio_id') for r in records))} unique audios)")

    # audio_id -> its query_audio path (remapped); same wav across a row's 12 emotions
    audio_query = {}
    for r in records:
        aid = r.get("audio_id") or r["id"]
        audio_query.setdefault(aid, remap_path(r["query_audio"], args.audio_root_map))

    # Load existing output manifest for resume (text carry-over)
    out_text_discrete = {}   # keyed by row id
    out_text_na = {}         # keyed by audio_id
    if args.skip_existing and out_manifest.exists():
        with open(out_manifest) as f:
            for line in f:
                r = json.loads(line)
                if r.get("finetuned_discrete_text") is not None:
                    out_text_discrete[r["id"]] = r["finetuned_discrete_text"]
                aid = r.get("audio_id") or r["id"]
                if r.get("finetuned_discrete_na_text") is not None:
                    out_text_na[aid] = r["finetuned_discrete_na_text"]
        print(f"Resuming from existing {out_manifest.name}\n")

    # Work lists
    discrete_todo = [
        r for r in records
        if not (args.skip_existing and (discrete_dir / f"{r['id']}.wav").exists())
    ]
    na_audio_ids = []
    seen = set()
    for r in records:
        aid = r.get("audio_id") or r["id"]
        if aid in seen:
            continue
        seen.add(aid)
        if not args.skip_na and not (args.skip_existing and (na_dir / f"{aid}.wav").exists()):
            na_audio_ids.append(aid)

    print(f"finetuned_discrete    to generate: {len(discrete_todo)} rows")
    if args.skip_na:
        print(f"finetuned_discrete_na : SKIPPED (--skip-na)")
    else:
        print(f"finetuned_discrete_na to generate: {len(na_audio_ids)} unique audios")

    if discrete_todo or na_audio_ids:
        print(f"\nCUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  {torch.cuda.device_count()} GPU(s): {torch.cuda.get_device_name(0)}")

        print("Loading tokenizer and speech codec components...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
        audio_0_id = tokenizer.convert_tokens_to_ids("<|audio_0|>")
        encoder = GLM4CodecEncoder()
        decoder = GLM4CodecDecoder(str(SRC_DIR / "glm-4-voice-decoder"))
        print(f"audio_0_id = {audio_0_id}\n")

        print(f"Loading discrete model: {finetuned_ckpt}")
        t0 = time.time()
        model = AutoPeftModelForCausalLM.from_pretrained(
            str(finetuned_ckpt), device_map="auto", trust_remote_code=True
        )
        model.eval()
        print(f"Model loaded in {time.time() - t0:.1f}s\n")

        # Encode each unique audio once (reused across the audio's 12 emotion prompts)
        token_cache = {}

        def tokens_for(aid):
            if aid not in token_cache:
                wav = Path(audio_query[aid])
                if not wav.exists():
                    token_cache[aid] = None
                else:
                    token_cache[aid] = encode_audio(wav, encoder)
            return token_cache[aid]

        # PASS 1: finetuned_discrete (per emotion×audio row)
        print(f"{'='*60}\nPASS 1: finetuned_discrete  ({len(discrete_todo)} rows)\n{'='*60}")
        for i, r in enumerate(discrete_todo):
            rid = r["id"]
            aid = r.get("audio_id") or rid
            emotion = r.get("emotion", "Neutral")
            toks = tokens_for(aid)
            print(f"\n  [{i+1}/{len(discrete_todo)}] {rid}  ({emotion})")
            if toks is None:
                print(f"    SKIP: query audio not found: {audio_query[aid]}")
                continue
            prompt = build_prompt(toks, discrete_system_prompt(emotion))
            t1 = time.time()
            try:
                text, waveform = generate_one(prompt, model, tokenizer, decoder, audio_0_id)
            except Exception as e:
                print(f"    ERROR: {e}")
                continue
            elapsed = time.time() - t1
            if waveform is not None:
                sf.write(str(discrete_dir / f"{rid}.wav"), waveform, DECODER_SAMPLE_RATE)
                print(f"    Saved: {rid}.wav  ({elapsed:.1f}s)")
            else:
                print(f"    WARNING: no audio ({elapsed:.1f}s)")
            if text:
                print(f"    Text: {text[:100]!r}")
            out_text_discrete[rid] = text or None

        # PASS 2: finetuned_discrete_na (once per unique audio)
        print(f"\n{'='*60}\nPASS 2: finetuned_discrete_na  ({len(na_audio_ids)} audios)\n{'='*60}")
        for i, aid in enumerate(na_audio_ids):
            toks = tokens_for(aid)
            print(f"\n  [{i+1}/{len(na_audio_ids)}] {aid}")
            if toks is None:
                print(f"    SKIP: query audio not found: {audio_query[aid]}")
                continue
            prompt = build_prompt(toks, NA_SYSTEM_PROMPT)
            t1 = time.time()
            try:
                text, waveform = generate_one(prompt, model, tokenizer, decoder, audio_0_id)
            except Exception as e:
                print(f"    ERROR: {e}")
                continue
            elapsed = time.time() - t1
            if waveform is not None:
                sf.write(str(na_dir / f"{aid}.wav"), waveform, DECODER_SAMPLE_RATE)
                print(f"    Saved: {aid}.wav  ({elapsed:.1f}s)")
            else:
                print(f"    WARNING: no audio ({elapsed:.1f}s)")
            if text:
                print(f"    Text: {text[:100]!r}")
            out_text_na[aid] = text or None

        print(f"\nUnloading model...")
        unload_model(model)

    # Write output manifest (all rows). Original manifest untouched.
    print(f"\nWriting manifest: {out_manifest}")

    def abs_if_exists(p: Path):
        return str(p.resolve()) if p.exists() else None

    with open(out_manifest, "w") as f:
        for r in records:
            rid = r["id"]
            aid = r.get("audio_id") or rid
            out_rec = {
                "id":           rid,
                "audio_id":     r.get("audio_id"),
                "source_index": r.get("source_index"),
                "emotion":      r.get("emotion"),
                "valence":      r.get("valence"),
                "arousal":      r.get("arousal"),
                "query_audio":  audio_query.get(aid, r.get("query_audio")),
                "finetuned_discrete_response":    abs_if_exists(discrete_dir / f"{rid}.wav"),
                "finetuned_discrete_text":        out_text_discrete.get(rid),
                # NA is shared across all 12 emotion rows of the same audio
                "finetuned_discrete_na_response": abs_if_exists(na_dir / f"{aid}.wav"),
                "finetuned_discrete_na_text":     out_text_na.get(aid),
            }
            f.write(json.dumps(out_rec) + "\n")

    total = len(records)
    discrete_ok = sum(1 for r in records if (discrete_dir / f"{r['id']}.wav").exists())
    conditions = ["finetuned_discrete"] if args.skip_na else ["finetuned_discrete",
                                                              "finetuned_discrete_na"]

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Rows:                  {total}")
    print(f"  finetuned_discrete:    {discrete_ok}/{total}")
    if not args.skip_na:
        na_ok = sum(1 for aid in seen if (na_dir / f"{aid}.wav").exists())
        print(f"  finetuned_discrete_na: {na_ok}/{len(seen)} unique audios")
    print(f"  Manifest:              {out_manifest}")
    print(f"\nNext step:")
    print(f"  python -m eval.judge.judge_qwen3omni_neutral \\")
    print(f"      --manifest {out_manifest.resolve()} \\")
    print(f"      --conditions {' '.join(conditions)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
