#!/usr/bin/env python3
"""
Generate Part 2 audio using Qwen3-TTS with multi-GPU support.

Two-pass generation:
  Pass 1 — Query audio (1,500 unique queries):
    - Source: unique_queries_{split}.jsonl
    - Speaker: "Ryan"
    - Style: EMOTION_INSTRUCTIONS[query_emotion]["query"]
    - Output: audio/{split}/query/{query_emotion.lower()}/{query_index}.wav

  Pass 2 — Response audio (16,500 pairs):
    - Source: sampled_{split}.jsonl
    - Speaker: "Vivian"
    - Style: EMOTION_INSTRUCTIONS[response_emotion]["response"]
    - Output: audio/{split}/response/{response_emotion.lower()}/{pair_index}.wav

Run:
  conda run -n qwen3-tts4 python dataset_creation/generate_part2_audio_multigpu.py \\
      --metadata-dir /engram/naplab/users/sd3705/Datasets/Sympatheia_11Emo_17k_Part2/metadata/ \\
      --output-audio-dir /engram/naplab/users/sd3705/Datasets/Sympatheia_11Emo_17k_Part2/audio/ \\
      --num-gpus 4 --batch-size 16 --resume
"""

import argparse
import json
import queue
import sys
import time
from multiprocessing import Manager, Process, Queue
from pathlib import Path
from typing import Any, Dict, List, Optional

import soundfile as sf
import torch
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Emotion instructions (mirrors generate_qwen3tts_audio_11emo_multigpu.py)
# ─────────────────────────────────────────────────────────────────────────────
EMOTION_INSTRUCTIONS = {
    "Sad":        {"query": "Very sad.",       "response": "Warm, gentle, reassuring."},
    "Excited":    {"query": "Very excited.",   "response": "Upbeat, bright, lively."},
    "Frustrated": {"query": "Very frustrated.","response": "Calm, patient, steady."},
    "Neutral":    {"query": "Neutral.",        "response": "Neutral, clear, friendly."},
    "Happy":      {"query": "Very happy.",     "response": "Cheerful, warm, upbeat."},
    "Angry":      {"query": "Very angry.",     "response": "Calm, firm, controlled."},
    "Fear":       {"query": "Very scared.",    "response": "Soft, soothing, steady."},
    "Relaxed":    {"query": "Very relaxed.",   "response": "Calm, chill, soothing."},
    "Surprised":  {"query": "Very surprised.", "response": "Curious, bright, attentive."},
    "Disgusted":  {"query": "Very disgusted.", "response": "Calm, brief, slightly distanced."},
    "Tired":      {"query": "Very tired.",     "response": "Low energy, slow, gentle."},
}

SPEAKER_CONFIG = {"query": "Ryan", "response": "Vivian"}

ALL_EMOTIONS = list(EMOTION_INSTRUCTIONS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Part 2 audio (query + 11 response emotions) with multi-GPU"
    )
    parser.add_argument("--metadata-dir", type=Path, required=True,
                        help="Directory with unique_queries_{split}.jsonl and sampled_{split}.jsonl")
    parser.add_argument("--output-audio-dir", type=Path, required=True,
                        help="Root directory for audio output")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size per GPU (default: 16)")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Number of GPUs (default: auto-detect)")
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="Checkpoint interval in samples (default: 100)")
    parser.add_argument("--model-name", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                        help="Qwen3-TTS model name")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max retries per failed batch (default: 3)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint if available")
    parser.add_argument("--pass", dest="run_pass", choices=["query", "response", "both"],
                        default="both",
                        help="Which pass to run (default: both)")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model init
# ─────────────────────────────────────────────────────────────────────────────
def init_model(model_name: str, device: str):
    qwen3_tts_path = (
        Path(__file__).resolve().parent.parent.parent
        / "Models" / "TTS" / "Qwen3-TTS" / "Qwen3-TTS"
    )
    if str(qwen3_tts_path) not in sys.path:
        sys.path.insert(0, str(qwen3_tts_path))

    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(
        model_name,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Batch TTS generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_batch_audio(
    model,
    texts: List[str],
    speaker: str,
    instruct: str,
    max_retries: int = 3,
) -> Optional[tuple]:
    for attempt in range(max_retries):
        try:
            wavs, sr = model.generate_custom_voice(
                text=texts,
                language=["English"] * len(texts),
                speaker=[speaker] * len(texts),
                instruct=[instruct] * len(texts),
                max_new_tokens=2048,
            )
            return wavs, sr
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries} after error: {e}")
                time.sleep(2)
            else:
                print(f"  Failed after {max_retries} attempts: {e}", file=sys.stderr)
                return None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Worker process (shared by both passes)
# ─────────────────────────────────────────────────────────────────────────────
def worker_process(
    gpu_id: int,
    model_name: str,
    work_queue: Queue,
    results_queue: Queue,
    batch_size: int,
    max_retries: int,
    output_audio_dir: Path,
):
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Initializing model on {device}...")
    try:
        model = init_model(model_name, device)
        print(f"[GPU {gpu_id}] Model loaded")
    except Exception as e:
        print(f"[GPU {gpu_id}] Failed to load model: {e}", file=sys.stderr)
        return

    processed_count = 0

    while True:
        try:
            work_item = work_queue.get(timeout=1)
            if work_item is None:
                print(f"[GPU {gpu_id}] Done. Total batches: {processed_count}")
                break

            # work_item = (audio_type, emotion_key, output_subdir, batch_samples, text_field)
            audio_type, emotion_key, output_subdir, batch_samples, text_field = work_item

            speaker = SPEAKER_CONFIG[audio_type]
            instruct = EMOTION_INSTRUCTIONS[emotion_key][audio_type]

            out_dir = output_audio_dir / output_subdir
            out_dir.mkdir(parents=True, exist_ok=True)

            texts = [s[text_field] for s in batch_samples]
            result = generate_batch_audio(model, texts, speaker, instruct, max_retries)

            if result is None:
                for sample in batch_samples:
                    results_queue.put(("failure", {
                        "audio_type": audio_type,
                        "emotion_key": emotion_key,
                        "index": sample["_out_index"],
                        "gpu_id": gpu_id,
                    }))
            else:
                wavs, sr = result
                for wav, sample in zip(wavs, batch_samples):
                    out_path = out_dir / f"{sample['_out_index']}.wav"
                    try:
                        sf.write(str(out_path), wav, sr)
                        results_queue.put(("success", 1))
                    except Exception as e:
                        results_queue.put(("failure", {
                            "audio_type": audio_type,
                            "emotion_key": emotion_key,
                            "index": sample["_out_index"],
                            "error": str(e),
                            "gpu_id": gpu_id,
                        }))

            processed_count += 1
            if processed_count % 10 == 0:
                torch.cuda.empty_cache()

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[GPU {gpu_id}] Unexpected error: {e}", file=sys.stderr)
            continue


# ─────────────────────────────────────────────────────────────────────────────
# Run a pass with multi-GPU workers
# ─────────────────────────────────────────────────────────────────────────────
def run_multigpu_pass(
    work_items: List[tuple],
    label: str,
    output_audio_dir: Path,
    model_name: str,
    batch_size: int,
    num_gpus: int,
    max_retries: int,
    checkpoint_path: Path,
    failures_path: Path,
    resume: bool,
):
    """
    work_items: list of (audio_type, emotion_key, output_subdir, [samples], text_field)
    where each sample has '_out_index' and text_field (the key for text).
    """
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")

    # Load checkpoint
    if resume and checkpoint_path.exists():
        with checkpoint_path.open() as f:
            ckpt = json.load(f)
        processed_set = set(ckpt.get("processed_indices", []))
        failed = ckpt.get("failed_samples", [])
        print(f"Resuming: {len(processed_set)} indices already processed")
    else:
        processed_set = set()
        failed = []

    # Flatten work_items into batches
    manager = Manager()
    work_queue = manager.Queue()
    results_queue = manager.Queue()

    total_batches = 0
    for audio_type, emotion_key, output_subdir, samples, text_field in work_items:
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            # Filter already processed
            batch = [s for s in batch if s["_out_index"] not in processed_set]
            if not batch:
                continue
            work_queue.put((audio_type, emotion_key, output_subdir, batch, text_field))
            total_batches += 1

    print(f"Batches to process: {total_batches}")

    for _ in range(num_gpus):
        work_queue.put(None)

    workers = []
    for gpu_id in range(num_gpus):
        p = Process(
            target=worker_process,
            args=(gpu_id, model_name, work_queue, results_queue, batch_size, max_retries, output_audio_dir),
        )
        p.start()
        workers.append(p)
        print(f"Started worker GPU {gpu_id}")

    total_ok = 0
    total_fail = 0
    with tqdm(total=total_batches, desc=label) as pbar:
        while any(w.is_alive() for w in workers) or not results_queue.empty():
            try:
                rtype, rdata = results_queue.get(timeout=0.5)
                if rtype == "success":
                    total_ok += rdata
                    pbar.update(1)
                elif rtype == "failure":
                    failed.append(rdata)
                    total_fail += 1
            except queue.Empty:
                continue

    for w in workers:
        w.join()

    # Save checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("w") as f:
        json.dump({"processed_indices": list(processed_set), "failed_samples": failed}, f, indent=2)

    if failed:
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        with failures_path.open("w") as f:
            for item in failed:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Failures logged → {failures_path}")

    print(f"Done: {total_ok} ok, {total_fail} failed")


# ─────────────────────────────────────────────────────────────────────────────
# Load metadata helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> List[Dict]:
    samples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.num_gpus is None:
        num_gpus = torch.cuda.device_count()
        print(f"Auto-detected {num_gpus} GPUs")
    else:
        num_gpus = args.num_gpus
    if num_gpus == 0:
        print("Error: No GPUs available!", file=sys.stderr)
        sys.exit(1)

    args.output_audio_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_audio_dir.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "eval"]:
        # ── Pass 1: Query audio ──────────────────────────────────────────────
        if args.run_pass in ("query", "both"):
            uq_path = args.metadata_dir / f"unique_queries_{split}.jsonl"
            if not uq_path.exists():
                print(f"Skipping query pass for {split}: {uq_path} not found")
            else:
                unique_queries = load_jsonl(uq_path)
                print(f"\nLoaded {len(unique_queries)} unique queries for {split}")

                # Group by query_emotion for batch efficiency (same emotion → same TTS instruct)
                emotion_groups: Dict[str, List[Dict]] = {}
                for q in unique_queries:
                    emo = q["query_emotion"]
                    q["_out_index"] = q["query_index"]  # output filename = query_index
                    emotion_groups.setdefault(emo, []).append(q)

                work_items = []
                for emo, samples in emotion_groups.items():
                    output_subdir = f"{split}/query/{emo.lower()}"
                    work_items.append(("query", emo, output_subdir, samples, "query_text"))

                run_multigpu_pass(
                    work_items=work_items,
                    label=f"Pass 1 — Query audio [{split}]",
                    output_audio_dir=args.output_audio_dir,
                    model_name=args.model_name,
                    batch_size=args.batch_size,
                    num_gpus=num_gpus,
                    max_retries=args.max_retries,
                    checkpoint_path=logs_dir / f"query_progress_{split}.json",
                    failures_path=logs_dir / f"query_failures_{split}.jsonl",
                    resume=args.resume,
                )

        # ── Pass 2: Response audio ───────────────────────────────────────────
        if args.run_pass in ("response", "both"):
            pairs_path = args.metadata_dir / f"sampled_{split}.jsonl"
            if not pairs_path.exists():
                print(f"Skipping response pass for {split}: {pairs_path} not found")
                continue

            pairs = load_jsonl(pairs_path)
            print(f"\nLoaded {len(pairs)} pairs for {split}")

            # Group by response_emotion for batch efficiency
            resp_emo_groups: Dict[str, List[Dict]] = {}
            for p in pairs:
                resp_emo = p["response_emotion"]
                p["_out_index"] = p["index"]  # output filename = pair index (e.g. p2_Sad_00001_Happy)
                resp_emo_groups.setdefault(resp_emo, []).append(p)

            work_items = []
            for resp_emo, samples in resp_emo_groups.items():
                output_subdir = f"{split}/response/{resp_emo.lower()}"
                work_items.append(("response", resp_emo, output_subdir, samples, "response_text"))

            run_multigpu_pass(
                work_items=work_items,
                label=f"Pass 2 — Response audio [{split}]",
                output_audio_dir=args.output_audio_dir,
                model_name=args.model_name,
                batch_size=args.batch_size,
                num_gpus=num_gpus,
                max_retries=args.max_retries,
                checkpoint_path=logs_dir / f"response_progress_{split}.json",
                failures_path=logs_dir / f"response_failures_{split}.jsonl",
                resume=args.resume,
            )

    print("\n" + "=" * 60)
    print("Audio generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
