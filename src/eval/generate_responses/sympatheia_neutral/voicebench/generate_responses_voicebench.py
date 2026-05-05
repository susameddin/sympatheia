#!/usr/bin/env python3
"""
Generate audio responses using VoiceBench CommonEval as a second neutral dataset.

Unlike generate_responses_neutral.py (which uses synthetic neutral audio),
this script uses real human voices from VoiceBench CommonEval
(hlt-lab/voicebench on HuggingFace).  The evaluation structure mirrors
generate_responses_neutral.py exactly: all 12 emotions are injected via
system prompt into each audio, isolating how well models adapt to externally
stated emotions on out-of-domain real speech.

Conditions:
  finetuned_va  — fine-tuned LoRA checkpoint + valence/arousal in system prompt
  finetuned_na  — fine-tuned LoRA checkpoint + "User emotion N/A" prompt

Key difference from generate_responses_neutral.py:
  - finetuned_na is generated ONCE per unique audio (not once per emotion×audio)
    since the prompt is identical regardless of emotion label.  The manifest
    records point all 12 emotion rows of the same audio to the same
    finetuned_na file, saving ~11x generation cost.
  - finetuned_va is generated once per (emotion, audio) pair.

Outputs:
  <output-dir>/audio_cache/          — cached query WAVs from VoiceBench
  <output-dir>/audio/finetuned_va/   — fine-tuned with VA (one per emotion×audio)
  <output-dir>/audio/finetuned_na/   — fine-tuned without VA (one per audio)
  <output-dir>/manifest.jsonl        — 12×N rows for judge_qwen3omni_neutral

Usage:
    # First, download VoiceBench once:
    #   python eval/retired/download_voicebench.py

    conda run -n glm4voice3 --no-capture-output python -m eval.generate_responses.sympatheia_neutral.voicebench.generate_responses_voicebench \\
        --num-samples 100

    # Quick test with 2 samples and 2 emotions:
    conda run -n glm4voice3 --no-capture-output python -m eval.generate_responses.sympatheia_neutral.voicebench.generate_responses_voicebench \\
        --num-samples 2 --emotions happy sad --output-dir /tmp/vb_test/
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
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

# Project root is one level up from this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vocoder_src import GLM4CodecEncoder, GLM4CodecDecoder
from src.constants import EMOTION_VA_MAPPING, ALL_EMOTIONS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "THUDM/glm-4-voice-9b"
DEFAULT_FINETUNED_EXPERIMENT = (
    "experiments/sympatheia-12emo-v2-20260320-100225"
)
DEFAULT_CHECKPOINT_STEP = 2200
DEFAULT_ENGRAM_BASE = "/engram/naplab/users/sd3705/emo_recog_2025s/eval"
DEFAULT_VOICEBENCH_DIR = "/engram/naplab/users/sd3705/Datasets/VoiceBench/commoneval"
DECODER_SAMPLE_RATE = 22050

NA_SYSTEM_PROMPT = "Please respond in English. User emotion N/A"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate model responses using VoiceBench CommonEval as a neutral dataset"
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
        help="Number of VoiceBench audio files to sample (reused across all emotions). "
             "Default: 50  (→ 600 manifest rows with 12 emotions)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for audio files and manifest.jsonl. "
             "Default: auto-constructed as <engram>/eval_voicebench_<experiment>_ckpt<step>/",
    )
    parser.add_argument(
        "--emotions", type=str, nargs="+", default=None,
        help="Subset of emotions to evaluate (default: all 12). "
             "E.g.: --emotions happy sad angry",
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

    Returns list of dicts with keys: source_index, audio_array, audio_sr.
    Items are sampled with a fixed seed and returned in index order.
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
    for idx in chosen:
        meta = all_meta[idx]
        wav_path = vb_dir / meta["audio"]
        audio_array, audio_sr = sf.read(str(wav_path))
        items.append({
            "source_index": meta["index"],
            "audio_array":  audio_array,
            "audio_sr":     audio_sr,
        })
    return items


def cache_audio(items: list, cache_dir: Path, skip_existing: bool) -> None:
    """Save VoiceBench audio arrays as WAV files into cache_dir.

    Adds 'wav' (Path) and 'audio_id' (str) to each item in-place.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(items):
        audio_id = f"voicebench_{i:04d}"
        wav_path = cache_dir / f"{audio_id}.wav"
        item["audio_id"] = audio_id
        item["wav"] = wav_path
        if skip_existing and wav_path.exists():
            continue
        sf.write(str(wav_path), item["audio_array"], item["audio_sr"])
    print(f"Cached {len(items)} audio files → {cache_dir}")


def build_all_samples(unique_audios: list, emotions: list) -> list:
    """Create the full cross-product sample list (N audios × M emotions).

    Each element shares the wav/audio_id from its parent unique_audio.
    """
    samples = []
    for emotion in sorted(emotions):
        v, a = EMOTION_VA_MAPPING.get(emotion, (0.0, 0.0))
        for audio in unique_audios:
            samples.append({
                "id":           f"{audio['audio_id']}_{emotion.lower()}",
                "audio_id":     audio["audio_id"],
                "source_index": audio["source_index"],
                "emotion":      emotion,
                "valence":      v,
                "arousal":      a,
                "wav":          audio["wav"],
            })
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

    # Auto-construct output dir
    if args.output_dir is None:
        exp_name = finetuned_exp.name
        output_dir = Path(DEFAULT_ENGRAM_BASE) / f"eval_voicebench_{exp_name}_ckpt{args.checkpoint_step}"
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir

    audio_cache_dir = output_dir / "audio_cache"
    va_audio_dir    = output_dir / "audio" / "finetuned_va"
    na_audio_dir    = output_dir / "audio" / "finetuned_na"
    manifest_path   = output_dir / "manifest.jsonl"

    for d in [audio_cache_dir, va_audio_dir, na_audio_dir]:
        d.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fine-tuned checkpoint : {finetuned_ckpt}")
    print(f"Output dir            : {output_dir}")
    print(f"Num audio samples     : {args.num_samples}")
    print(f"Skip existing         : {args.skip_existing}")
    print(f"VoiceBench dir        : {args.voicebench_dir}")

    # Determine emotions to evaluate
    emotions = args.emotions if args.emotions else ALL_EMOTIONS
    cap_map = {e.lower(): e for e in ALL_EMOTIONS}
    emotions = [cap_map.get(e.lower(), e) for e in emotions]

    # Load VoiceBench data
    unique_audios = load_voicebench(args.num_samples, args.seed, args.voicebench_dir)

    # Cache audio to disk (needed by encoder and judge)
    cache_audio(unique_audios, audio_cache_dir, args.skip_existing)

    # Build full sample list (N audios × M emotions)
    all_samples = build_all_samples(unique_audios, emotions)
    print(f"\nSample counts:")
    print(f"  Unique audios : {len(unique_audios)}")
    print(f"  Emotions      : {len(emotions)}  ({', '.join(sorted(emotions))})")
    print(f"  Total rows    : {len(all_samples)}\n")

    # Shared components
    print("Loading tokenizer and speech codec components...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    audio_0_id = tokenizer.convert_tokens_to_ids("<|audio_0|>")
    encoder = GLM4CodecEncoder()
    decoder_path = str(PROJECT_ROOT / "glm-4-voice-decoder")
    decoder = GLM4CodecDecoder(decoder_path)
    print(f"audio_0_id = {audio_0_id}\n")

    # Encode each unique audio once
    print("Encoding query audio files (one per unique audio)...")
    for audio in unique_audios:
        audio["user_tokens"] = encode_audio(audio["wav"], encoder)
        print(f"  Encoded {audio['audio_id']}")

    # Propagate tokens to all_samples (by matching audio_id)
    tokens_by_id = {a["audio_id"]: a["user_tokens"] for a in unique_audios}
    for s in all_samples:
        s["user_tokens"] = tokens_by_id[s["audio_id"]]

    # -----------------------------------------------------------------------
    # Fine-tuned model (finetuned_va per sample + finetuned_na per audio)
    # -----------------------------------------------------------------------
    ft_todo_va = [
        s for s in all_samples
        if not (args.skip_existing and (va_audio_dir / f"{s['id']}.wav").exists())
    ]
    ft_todo_na = [
        a for a in unique_audios
        if not (args.skip_existing and (na_audio_dir / f"{a['audio_id']}.wav").exists())
    ]

    if ft_todo_va or ft_todo_na:
        print(f"\n{'='*60}")
        print(f"PASS 2: Fine-tuned model")
        print(f"  finetuned_va : {len(ft_todo_va)}/{len(all_samples)} samples")
        print(f"  finetuned_na : {len(ft_todo_na)}/{len(unique_audios)} unique audios")
        print(f"  Checkpoint   : {finetuned_ckpt.name}")
        print(f"{'='*60}")
        t0 = time.time()
        ft_model = AutoPeftModelForCausalLM.from_pretrained(
            str(finetuned_ckpt), device_map="auto", trust_remote_code=True
        )
        ft_model.eval()
        print(f"Fine-tuned model loaded in {time.time() - t0:.1f}s")

        # finetuned_va — one run per (emotion, audio)
        for i, s in enumerate(ft_todo_va):
            v, a = s["valence"], s["arousal"]
            out_va = va_audio_dir / f"{s['id']}.wav"
            print(f"\n  [VA {i+1}/{len(ft_todo_va)}] {s['id']}  (V={v:+.2f}, A={a:+.2f})")
            va_prompt = (
                f"Please respond in English. "
                f"User emotion (valence={v:.2f}, arousal={a:.2f})"
            )
            prompt = build_prompt(s["user_tokens"], va_prompt)
            t1 = time.time()
            text, waveform = generate_one(prompt, ft_model, tokenizer, decoder, audio_0_id)
            elapsed = time.time() - t1
            if waveform is None:
                print(f"    WARNING: no audio generated ({elapsed:.1f}s)")
            else:
                sf.write(str(out_va), waveform, DECODER_SAMPLE_RATE)
                print(f"    Saved: {out_va.name}  ({elapsed:.1f}s)")
                print(f"    Text: {text[:100]!r}")
            s["finetuned_va_text"] = text

        # finetuned_na — one run per unique audio
        for i, audio in enumerate(ft_todo_na):
            out_na = na_audio_dir / f"{audio['audio_id']}.wav"
            print(f"\n  [NA {i+1}/{len(ft_todo_na)}] {audio['audio_id']}")
            prompt = build_prompt(audio["user_tokens"], NA_SYSTEM_PROMPT)
            t1 = time.time()
            text, waveform = generate_one(prompt, ft_model, tokenizer, decoder, audio_0_id)
            elapsed = time.time() - t1
            if waveform is None:
                print(f"    WARNING: no audio generated ({elapsed:.1f}s)")
            else:
                sf.write(str(out_na), waveform, DECODER_SAMPLE_RATE)
                print(f"    Saved: {out_na.name}  ({elapsed:.1f}s)")
                print(f"    Text: {text[:100]!r}")
            audio["finetuned_na_text"] = text

        print(f"\nUnloading fine-tuned model...")
        unload_model(ft_model)
    else:
        print(f"\nPASS 2: All fine-tuned responses already exist — skipping")

    # -----------------------------------------------------------------------
    # Write manifest (scan disk for what was actually produced)
    # -----------------------------------------------------------------------
    # Build lookup dicts for per-audio text fields
    na_text_by_id  = {a["audio_id"]: a.get("finetuned_na_text") for a in unique_audios}
    va_text_by_sid = {s["id"]: s.get("finetuned_va_text") for s in all_samples}

    print(f"\nWriting manifest: {manifest_path}")

    def abs_path_if_exists(p: Path):
        return str(p.resolve()) if p.exists() else None

    with open(manifest_path, "w") as f:
        for s in all_samples:
            aid = s["audio_id"]
            rec = {
                "id":           s["id"],
                "audio_id":     aid,
                "source_index": s["source_index"],
                "emotion":      s["emotion"],
                "valence":      s["valence"],
                "arousal":      s["arousal"],
                "query_audio":  str((audio_cache_dir / f"{aid}.wav").resolve()),
                "finetuned_va_response": abs_path_if_exists(va_audio_dir / f"{s['id']}.wav"),
                "finetuned_na_response": abs_path_if_exists(na_audio_dir / f"{aid}.wav"),
            }
            vt = va_text_by_sid.get(s["id"])
            if vt is not None:
                rec["finetuned_va_text"] = vt
            nt = na_text_by_id.get(aid)
            if nt is not None:
                rec["finetuned_na_text"] = nt
            f.write(json.dumps(rec) + "\n")

    total  = len(all_samples)
    va_ok  = sum(1 for s in all_samples if (va_audio_dir / f"{s['id']}.wav").exists())
    na_ok  = sum(1 for a in unique_audios if (na_audio_dir / f"{a['audio_id']}.wav").exists())

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Unique audios   : {len(unique_audios)}")
    print(f"  Manifest rows   : {total}")
    print(f"  finetuned_va    : {va_ok}/{total} samples")
    print(f"  finetuned_na    : {na_ok}/{len(unique_audios)} unique audios")
    print(f"  Manifest        : {manifest_path}")
    print(f"\nNext step:")
    print(f"  conda run -n qwen3omni --no-capture-output python -m eval.judge.judge_qwen3omni_neutral \\")
    print(f"      --manifest {manifest_path.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
