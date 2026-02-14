#!/usr/bin/env python3
"""
Generate emotion-conditioned audio using Qwen3-TTS with multi-GPU support (11 emotions).

Reads sampled OpenS2S metadata and generates query/response audio files
with emotion-specific instructions using Qwen3-TTS model across multiple GPUs.
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys
from multiprocessing import Process, Queue, Manager
import queue

import torch
import soundfile as sf
from tqdm import tqdm


# Emotion-specific instructions for 11 emotions
EMOTION_INSTRUCTIONS = {
    "Sad": {"query": "Very sad.", "response": "Warm, gentle, reassuring."},
    "Excited": {"query": "Very excited.", "response": "Upbeat, bright, lively."},
    "Frustrated": {"query": "Very frustrated.", "response": "Calm, patient, steady."},
    "Neutral": {"query": "Neutral.", "response": "Neutral, clear, friendly."},
    "Happy": {"query": "Very happy.", "response": "Cheerful, warm, upbeat."},
    "Angry": {"query": "Very angry.", "response": "Calm, firm, controlled."},
    "Fear": {"query": "Very scared.", "response": "Soft, soothing, steady."},
    "Relaxed": {"query": "Very relaxed.", "response": "Calm, chill, soothing."},
    "Surprised": {"query": "Very surprised.", "response": "Curious, bright, attentive."},
    "Disgusted": {"query": "Very disgusted.", "response": "Calm, brief, slightly distanced."},
    "Tired": {"query": "Very tired.", "response": "Low energy, slow, gentle."},
}

# All 11 emotions
ALL_EMOTIONS = ["Sad", "Excited", "Frustrated", "Neutral", "Happy", "Angry", "Fear", "Relaxed", "Surprised", "Disgusted", "Tired"]

# Speaker configuration
SPEAKER_CONFIG = {"query": "Ryan", "response": "Vivian"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate emotion-conditioned audio using Qwen3-TTS (Multi-GPU, 11 emotions)"
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        required=True,
        help="Directory containing sampled_train.jsonl and sampled_eval.jsonl",
    )
    parser.add_argument(
        "--output-audio-dir",
        type=Path,
        required=True,
        help="Output directory for generated audio files",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for TTS generation per GPU (default: 16)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: auto-detect all available)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save checkpoint every N samples (default: 100)",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        help="Qwen3-TTS model name (default: Qwen3-TTS-12Hz-1.7B-CustomVoice)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries for failed generations (default: 3)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )
    return parser.parse_args()


def load_metadata(metadata_dir: Path, split: str) -> List[Dict[str, Any]]:
    """Load metadata JSONL file."""
    metadata_path = metadata_dir / f"sampled_{split}.jsonl"
    print(f"Loading {split} metadata from: {metadata_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    samples = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    print(f"Loaded {len(samples)} {split} samples")
    return samples


def init_model(model_name: str, device: str):
    """Initialize Qwen3-TTS model."""
    # Add Qwen3-TTS to Python path
    qwen3_tts_path = Path(__file__).resolve().parent.parent.parent / "Models" / "TTS" / "Qwen3-TTS" / "Qwen3-TTS"
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


def generate_batch(
    model,
    texts: List[str],
    speaker: str,
    instruct: str,
    max_retries: int = 3,
) -> Optional[tuple]:
    """Generate audio for a batch with retry logic."""
    batch_size = len(texts)

    for attempt in range(max_retries):
        try:
            wavs, sr = model.generate_custom_voice(
                text=texts,
                language=["English"] * batch_size,
                speaker=[speaker] * batch_size,
                instruct=[instruct] * batch_size,
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


def worker_process(
    gpu_id: int,
    model_name: str,
    work_queue: Queue,
    results_queue: Queue,
    batch_size: int,
    max_retries: int,
    output_audio_dir: Path,
):
    """Worker process for generating audio on a specific GPU."""
    device = f"cuda:{gpu_id}"

    print(f"[GPU {gpu_id}] Initializing model on {device}...")
    try:
        model = init_model(model_name, device)
        print(f"[GPU {gpu_id}] Model loaded successfully")
    except Exception as e:
        print(f"[GPU {gpu_id}] Failed to load model: {e}", file=sys.stderr)
        return

    processed_count = 0

    while True:
        try:
            # Get work item (with timeout to check if done)
            work_item = work_queue.get(timeout=1)

            if work_item is None:  # Poison pill to stop worker
                print(f"[GPU {gpu_id}] Finished processing. Total: {processed_count} batches")
                break

            split, emotion, audio_type, batch_samples = work_item

            # Get speaker and instruction
            speaker = SPEAKER_CONFIG[audio_type]
            instruct = EMOTION_INSTRUCTIONS[emotion][audio_type]

            # Create output directory
            output_dir = output_audio_dir / split / f"{emotion.lower()}_{audio_type}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Extract texts
            if audio_type == "query":
                texts = [s["query_text"] for s in batch_samples]
            else:
                texts = [s["response_text"] for s in batch_samples]

            # Generate audio
            result = generate_batch(model, texts, speaker, instruct, max_retries)

            if result is None:
                # Report failures
                for sample in batch_samples:
                    failure_info = {
                        "split": split,
                        "emotion": emotion,
                        "type": audio_type,
                        "index": sample["index"],
                        "gpu_id": gpu_id,
                    }
                    results_queue.put(("failure", failure_info))
            else:
                wavs, sr = result

                # Save WAV files
                success_count = 0
                for wav, sample in zip(wavs, batch_samples):
                    output_path = output_dir / f"{sample['index']}.wav"
                    try:
                        sf.write(str(output_path), wav, sr)
                        success_count += 1
                    except Exception as e:
                        failure_info = {
                            "split": split,
                            "emotion": emotion,
                            "type": audio_type,
                            "index": sample["index"],
                            "error": str(e),
                            "gpu_id": gpu_id,
                        }
                        results_queue.put(("failure", failure_info))

                # Report success
                results_queue.put(("success", success_count))

            processed_count += 1

            # Periodic GPU cache clear
            if processed_count % 10 == 0:
                torch.cuda.empty_cache()

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[GPU {gpu_id}] Unexpected error: {e}", file=sys.stderr)
            continue


def process_split_multigpu(
    samples: List[Dict[str, Any]],
    split: str,
    output_audio_dir: Path,
    model_name: str,
    batch_size: int,
    num_gpus: int,
    max_retries: int,
    checkpoint_every: int,
    checkpoint_path: Path,
    failures_path: Path,
    resume: bool,
):
    """Process one split using multiple GPUs."""
    print(f"\n{'=' * 60}")
    print(f"Processing {split} split ({len(samples)} samples) on {num_gpus} GPUs")
    print(f"{'=' * 60}")

    # Load checkpoint if resuming
    if resume and checkpoint_path.exists():
        with checkpoint_path.open("r") as f:
            checkpoint_data = json.load(f)
        processed_samples = set(checkpoint_data.get("processed_samples", []))
        failed_samples = checkpoint_data.get("failed_samples", [])
        print(f"Resuming: {len(processed_samples)} samples already processed")
    else:
        processed_samples = set()
        failed_samples = []

    # Group samples by emotion (all 11 emotions)
    emotion_samples = {emotion: [] for emotion in ALL_EMOTIONS}
    for sample in samples:
        emotion = sample["query_emotion"]
        if emotion in emotion_samples:
            emotion_samples[emotion].append(sample)

    # Create work queue and results queue
    manager = Manager()
    work_queue = manager.Queue()
    results_queue = manager.Queue()

    # Prepare work items (batches to process)
    total_batches = 0
    for emotion, emotion_sample_list in emotion_samples.items():
        if not emotion_sample_list:
            continue

        for audio_type in ["query", "response"]:
            # Create batches
            for i in range(0, len(emotion_sample_list), batch_size):
                batch = emotion_sample_list[i : i + batch_size]

                # Check if already processed
                batch_indices = [s["index"] for s in batch]
                batch_key = f"{split}_{emotion}_{audio_type}_" + "_".join(
                    str(idx) for idx in batch_indices
                )

                if batch_key not in processed_samples:
                    work_queue.put((split, emotion, audio_type, batch))
                    total_batches += 1

    print(f"Total batches to process: {total_batches}")

    # Add poison pills (one per worker)
    for _ in range(num_gpus):
        work_queue.put(None)

    # Start worker processes
    workers = []
    for gpu_id in range(num_gpus):
        p = Process(
            target=worker_process,
            args=(
                gpu_id,
                model_name,
                work_queue,
                results_queue,
                batch_size,
                max_retries,
                output_audio_dir,
            ),
        )
        p.start()
        workers.append(p)
        print(f"Started worker on GPU {gpu_id}")

    # Monitor progress
    total_processed = 0
    total_failed = 0

    with tqdm(total=total_batches, desc=f"Processing {split}") as pbar:
        while any(w.is_alive() for w in workers) or not results_queue.empty():
            try:
                result_type, result_data = results_queue.get(timeout=0.5)

                if result_type == "success":
                    total_processed += result_data
                    pbar.update(1)
                elif result_type == "failure":
                    failed_samples.append(result_data)
                    total_failed += 1

                # Save checkpoint periodically
                if total_processed % checkpoint_every == 0 and total_processed > 0:
                    checkpoint_data = {
                        "processed_samples": list(processed_samples),
                        "failed_samples": failed_samples,
                    }
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    with checkpoint_path.open("w") as f:
                        json.dump(checkpoint_data, f, indent=2)

            except queue.Empty:
                continue

    # Wait for all workers to finish
    for w in workers:
        w.join()

    # Final checkpoint save
    checkpoint_data = {
        "processed_samples": list(processed_samples),
        "failed_samples": failed_samples,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("w") as f:
        json.dump(checkpoint_data, f, indent=2)

    # Save failures log
    if failed_samples:
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        with failures_path.open("w") as f:
            for failure in failed_samples:
                f.write(json.dumps(failure, ensure_ascii=False) + "\n")
        print(f"\nFailed samples logged to: {failures_path}")

    print(f"\n{split.capitalize()} split complete:")
    print(f"  Total processed: {total_processed}")
    print(f"  Total failed: {total_failed}")


def main():
    args = parse_args()

    # Determine number of GPUs
    if args.num_gpus is None:
        num_gpus = torch.cuda.device_count()
        print(f"Auto-detected {num_gpus} GPUs")
    else:
        num_gpus = args.num_gpus
        print(f"Using {num_gpus} GPUs")

    if num_gpus == 0:
        print("Error: No GPUs available!", file=sys.stderr)
        sys.exit(1)

    # Create output directories
    args.output_audio_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_audio_dir.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Process train and eval splits
    for split in ["train", "eval"]:
        # Load metadata
        samples = load_metadata(args.metadata_dir, split)

        # Setup checkpoint paths
        checkpoint_path = logs_dir / f"generation_progress_{split}.json"
        failures_path = logs_dir / f"generation_failures_{split}.jsonl"

        # Process split with multi-GPU
        process_split_multigpu(
            samples=samples,
            split=split,
            output_audio_dir=args.output_audio_dir,
            model_name=args.model_name,
            batch_size=args.batch_size,
            num_gpus=num_gpus,
            max_retries=args.max_retries,
            checkpoint_every=args.checkpoint_every,
            checkpoint_path=checkpoint_path,
            failures_path=failures_path,
            resume=args.resume,
        )

    print("\n" + "=" * 60)
    print("Audio generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
