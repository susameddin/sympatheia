#!/usr/bin/env python3
"""
Generate brand-new emotion-conditioned instruction-response text pairs using Qwen3-32B-Instruct.

Two-stage pipeline (following OpenS2S §3.2):
  Stage 1: Generate user instructions expressing the target emotion (thinking OFF)
  Stage 2: Generate empathetic assistant responses with thinking mode ON

Output format is identical to sampled_train.jsonl / sampled_eval.jsonl so the
existing TTS and GLM-4-Voice encoding steps work without modification.

Run with:
  conda run -n qwen3-tts4 python dataset_creation/generate_new_text_pairs.py \
      --llm-model Qwen/Qwen3-32B-Instruct \
      --output-dir /path/to/new_metadata/ \
      --samples-per-emotion 1000
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Emotion style descriptions (mirrors EMOTION_INSTRUCTIONS in the TTS script)
# ─────────────────────────────────────────────────────────────────────────────
EMOTION_STYLE = {
    "Sad":        {"query": "Very sad",        "response": "Warm, gentle, reassuring"},
    "Excited":    {"query": "Very excited",    "response": "Upbeat, bright, lively"},
    "Frustrated": {"query": "Very frustrated", "response": "Calm, patient, steady"},
    "Neutral":    {"query": "Neutral",         "response": "Neutral, clear, friendly"},
    "Happy":      {"query": "Very happy",      "response": "Cheerful, warm, upbeat"},
    "Angry":      {"query": "Very angry",      "response": "Calm, firm, controlled"},
    "Fear":       {"query": "Very scared",     "response": "Soft, soothing, steady"},
    "Relaxed":    {"query": "Very relaxed",    "response": "Calm, chill, soothing"},
    "Surprised":  {"query": "Very surprised",  "response": "Curious, bright, attentive"},
    "Disgusted":  {"query": "Very disgusted",  "response": "Calm, brief, slightly distanced"},
    "Tired":      {"query": "Very tired",      "response": "Low energy, slow, gentle"},
}

ALL_EMOTIONS = list(EMOTION_STYLE.keys())

TOPIC_POOL = [
    "daily routine",
    "work or study stress",
    "family dynamics",
    "friendship",
    "health and wellness",
    "weather",
    "food and dining",
    "travel plans",
    "movies or TV shows",
    "hobbies",
    "money and finances",
    "technology",
    "current news",
    "sports",
    "learning something new",
    "home chores",
    "pets",
    "shopping",
    "social media",
    "future plans",
]


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate brand-new emotion text pairs with Qwen3-32B-Instruct"
    )
    parser.add_argument(
        "--llm-model",
        default="Qwen/Qwen3-32B",
        help="Path or HuggingFace ID of the LLM (default: Qwen/Qwen3-32B)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save sampled_train.jsonl and sampled_eval.jsonl",
    )
    parser.add_argument(
        "--samples-per-emotion",
        type=int,
        default=1000,
        help="Number of pairs to generate per emotion (default: 1000)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of samples for training set (default: 0.7)",
    )
    parser.add_argument(
        "--emotions",
        nargs="+",
        default=ALL_EMOTIONS,
        choices=ALL_EMOTIONS,
        help="Emotions to generate (default: all 11)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Generation batch size (default: 8)",
    )
    parser.add_argument(
        "--stage1-temperature",
        type=float,
        default=0.85,
        help="Sampling temperature for Stage 1 instruction generation (default: 0.85)",
    )
    parser.add_argument(
        "--stage2-temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for Stage 2 response generation (default: 0.7)",
    )
    parser.add_argument(
        "--stage1-max-new-tokens",
        type=int,
        default=128,
        help="Max new tokens for Stage 1 (default: 128)",
    )
    parser.add_argument(
        "--stage2-max-new-tokens",
        type=int,
        default=4096,
        help="Max new tokens for Stage 2 (includes thinking block, default: 4096)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for empty/garbled outputs per sample (default: 3)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--nshard",
        type=int,
        default=1,
        help="Total number of shards for multi-node generation (default: 1)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="This shard's rank index (default: 0)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from partial output if sampled_stage1.jsonl exists in output-dir",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Shard helpers (mirrors OpenS2S text_generation.py)
# ─────────────────────────────────────────────────────────────────────────────
def get_shard_range(total: int, nshard: int, rank: int):
    start = round(total / nshard * rank)
    end = round(total / nshard * (rank + 1))
    return start, end


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────
def build_stage1_messages(emotion: str, topic: str) -> List[Dict[str, str]]:
    """Chat messages for Stage 1 (user instruction generation)."""
    query_style = EMOTION_STYLE[emotion]["query"]
    return [
        {
            "role": "system",
            "content": "You are generating training data for an empathetic speech dialogue system.",
        },
        {
            "role": "user",
            "content": (
                f"Generate a natural, conversational, spoken-style instruction or question "
                f"(1\u20132 sentences) that someone who is {query_style} might say about: {topic}.\n\n"
                "Requirements:\n"
                "- Spoken English only (as if said aloud, not written)\n"
                "- The emotional state should naturally show in the words and phrasing\n"
                "- Output ONLY the instruction text \u2014 no quotes, no explanation"
            ),
        },
    ]


def build_stage2_messages(emotion: str, instruction: str) -> List[Dict[str, str]]:
    """Chat messages for Stage 2 (response generation, thinking ON)."""
    query_style = EMOTION_STYLE[emotion]["query"]
    response_style = EMOTION_STYLE[emotion]["response"]
    return [
        {
            "role": "system",
            "content": (
                "You are an empathetic AI assistant. "
                "Think carefully about the user's emotional state and what would be most helpful before responding."
            ),
        },
        {
            "role": "user",
            "content": (
                f'The user (who is {query_style}) said: "{instruction}"\n\n'
                f"Generate a concise, natural spoken response (2\u20133 sentences) that is {response_style}.\n"
                "It should be dialogue-appropriate, empathetic, and sound natural when spoken aloud.\n\n"
                "Output ONLY the response text \u2014 no quotes, no explanation."
            ),
        },
    ]


def apply_template(tokenizer, messages: List[Dict], enable_thinking: bool) -> str:
    """Apply Qwen3 chat template with thinking mode control."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Output parsing
# ─────────────────────────────────────────────────────────────────────────────
def strip_thinking(text: str) -> str:
    """Remove Qwen3 <think>...</think> block."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def clean_output(text: str) -> str:
    """Strip thinking block and remove accidental wrapper quotes."""
    text = strip_thinking(text)
    # Remove surrounding quotes that models sometimes add
    text = text.strip().strip('"').strip("'").strip()
    return text


def is_valid(text: str) -> bool:
    """Reject empty, too-short, or clearly meta/garbled outputs."""
    if not text or len(text.split()) < 4:
        return False
    low = text.lower()
    bad_starts = [
        "here is", "here's", "certainly!", "sure!", "of course!",
        "output:", "instruction:", "response:",
    ]
    for bad in bad_starts:
        if low.startswith(bad):
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Batch generation with transformers
# ─────────────────────────────────────────────────────────────────────────────
def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
) -> List[str]:
    """Tokenize and generate responses for a batch of prompts."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the prompt)
    input_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[:, input_len:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# Slot building
# ─────────────────────────────────────────────────────────────────────────────
def build_slots(
    emotions: List[str],
    samples_per_emotion: int,
    nshard: int,
    rank: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Build the full list of (emotion, topic, local_index) slots for this shard.
    Topics are cycled evenly across samples for diversity.
    """
    all_slots = []
    rng = random.Random(seed)

    for emotion in emotions:
        topics: List[str] = []
        while len(topics) < samples_per_emotion:
            shuffled = TOPIC_POOL[:]
            rng.shuffle(shuffled)
            topics.extend(shuffled)
        topics = topics[:samples_per_emotion]

        for i, topic in enumerate(topics):
            all_slots.append({"emotion": emotion, "topic": topic, "local_idx": i})

    # Apply sharding
    start, end = get_shard_range(len(all_slots), nshard, rank)
    return all_slots[start:end]


# ─────────────────────────────────────────────────────────────────────────────
# Stage runners
# ─────────────────────────────────────────────────────────────────────────────
def run_stage1(
    model,
    tokenizer,
    slots: List[Dict[str, Any]],
    batch_size: int,
    temperature: float,
    max_new_tokens: int,
    max_retries: int,
) -> List[Optional[str]]:
    """Generate user instructions for all slots (thinking mode OFF)."""
    results: List[Optional[str]] = [None] * len(slots)
    pending = list(range(len(slots)))

    for attempt in range(max_retries):
        if not pending:
            break
        print(
            f"\n[Stage 1] Attempt {attempt + 1}/{max_retries} — "
            f"{len(pending)} slots remaining"
        )

        prompts = [
            apply_template(
                tokenizer,
                build_stage1_messages(slots[i]["emotion"], slots[i]["topic"]),
                enable_thinking=False,
            )
            for i in pending
        ]

        outputs: List[str] = []
        for start in tqdm(range(0, len(prompts), batch_size), desc="Stage 1 batches"):
            outputs.extend(
                generate_batch(
                    model,
                    tokenizer,
                    prompts[start : start + batch_size],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            )

        still_pending = []
        for idx, raw in zip(pending, outputs):
            text = clean_output(raw)
            if is_valid(text):
                results[idx] = text
            else:
                still_pending.append(idx)

        pending = still_pending

    if pending:
        print(
            f"[Stage 1] Warning: {len(pending)} slots failed after {max_retries} retries — skipping.",
            file=sys.stderr,
        )
    return results


def run_stage2(
    model,
    tokenizer,
    slots: List[Dict[str, Any]],
    instructions: List[Optional[str]],
    batch_size: int,
    temperature: float,
    max_new_tokens: int,
    max_retries: int,
) -> List[Optional[str]]:
    """Generate assistant responses (thinking mode ON)."""
    results: List[Optional[str]] = [None] * len(slots)
    # Only slots that succeeded in Stage 1
    pending = [i for i, inst in enumerate(instructions) if inst is not None]

    for attempt in range(max_retries):
        if not pending:
            break
        print(
            f"\n[Stage 2] Attempt {attempt + 1}/{max_retries} — "
            f"{len(pending)} slots remaining"
        )

        prompts = [
            apply_template(
                tokenizer,
                build_stage2_messages(slots[i]["emotion"], instructions[i]),
                enable_thinking=True,
            )
            for i in pending
        ]

        outputs: List[str] = []
        for start in tqdm(range(0, len(prompts), batch_size), desc="Stage 2 batches"):
            outputs.extend(
                generate_batch(
                    model,
                    tokenizer,
                    prompts[start : start + batch_size],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            )

        still_pending = []
        for idx, raw in zip(pending, outputs):
            text = clean_output(raw)
            if is_valid(text):
                results[idx] = text
            else:
                still_pending.append(idx)

        pending = still_pending

    if pending:
        print(
            f"[Stage 2] Warning: {len(pending)} slots failed after {max_retries} retries — skipping.",
            file=sys.stderr,
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────
def save_results(
    slots: List[Dict[str, Any]],
    instructions: List[Optional[str]],
    responses: List[Optional[str]],
    output_dir: Path,
    train_ratio: float,
    seed: int,
):
    """Split successful samples into train/eval and save as JSONL."""
    output_dir.mkdir(parents=True, exist_ok=True)

    emotion_samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    skipped = 0

    for slot, inst, resp in zip(slots, instructions, responses):
        if inst is None or resp is None:
            skipped += 1
            continue
        emotion = slot["emotion"]
        emotion_samples[emotion].append(
            {
                "index": f"new_{emotion}_{slot['local_idx']:05d}",
                "query_text": inst,
                "query_emotion": emotion,
                "source_emotion": emotion,
                "response_text": resp,
            }
        )

    if skipped:
        print(f"\nSkipped {skipped} samples due to generation failures.")

    rng = random.Random(seed + 1)
    train_data: List[Dict] = []
    eval_data: List[Dict] = []

    print("\nEmotion split breakdown:")
    for emotion in ALL_EMOTIONS:
        samples = emotion_samples.get(emotion, [])
        if not samples:
            continue
        rng.shuffle(samples)
        n_train = int(len(samples) * train_ratio)
        train_data.extend(samples[:n_train])
        eval_data.extend(samples[n_train:])
        print(f"  {emotion}: {n_train} train + {len(samples) - n_train} eval")

    rng.shuffle(train_data)
    rng.shuffle(eval_data)

    train_path = output_dir / "sampled_train.jsonl"
    eval_path = output_dir / "sampled_eval.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for sample in train_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with eval_path.open("w", encoding="utf-8") as f:
        for sample in eval_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(train_data)} train samples → {train_path}")
    print(f"Saved {len(eval_data)} eval  samples → {eval_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    random.seed(args.random_seed)

    n_gpus = torch.cuda.device_count()
    print(f"\nDetected {n_gpus} GPU(s)")
    print(f"Configuration:")
    print(f"  LLM model:            {args.llm_model}")
    print(f"  Emotions:             {args.emotions}")
    print(f"  Samples per emotion:  {args.samples_per_emotion}")
    print(f"  Train ratio:          {args.train_ratio}")
    print(f"  Shard:                {args.rank + 1}/{args.nshard}")
    print(f"  Batch size:           {args.batch_size}")
    print(f"  Stage 1 temperature:  {args.stage1_temperature}")
    print(f"  Stage 2 temperature:  {args.stage2_temperature}")

    # ── Build generation slots ──────────────────────────────────────────────
    print("\nBuilding generation slots...")
    slots = build_slots(
        args.emotions,
        args.samples_per_emotion,
        args.nshard,
        args.rank,
        args.random_seed,
    )
    print(f"Slots for this shard: {len(slots)}")

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"\nLoading model: {args.llm_model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_model, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.llm_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded on: {model.device}")

    # ── Stage 1: generate user instructions ────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 1: Generating user instructions (thinking OFF)")
    print("=" * 60)
    instructions = run_stage1(
        model=model,
        tokenizer=tokenizer,
        slots=slots,
        batch_size=args.batch_size,
        temperature=args.stage1_temperature,
        max_new_tokens=args.stage1_max_new_tokens,
        max_retries=args.max_retries,
    )
    n_ok1 = sum(1 for x in instructions if x is not None)
    print(f"\nStage 1 complete: {n_ok1}/{len(slots)} succeeded")

    # ── Save Stage 1 checkpoint ─────────────────────────────────────────────
    stage1_ckpt = args.output_dir / "stage1_checkpoint.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with stage1_ckpt.open("w", encoding="utf-8") as f:
        for slot, inst in zip(slots, instructions):
            f.write(
                json.dumps(
                    {
                        "emotion": slot["emotion"],
                        "topic": slot["topic"],
                        "local_idx": slot["local_idx"],
                        "instruction": inst,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Stage 1 checkpoint saved → {stage1_ckpt}")

    # ── Stage 2: generate assistant responses ──────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 2: Generating assistant responses (thinking ON)")
    print("=" * 60)
    responses = run_stage2(
        model=model,
        tokenizer=tokenizer,
        slots=slots,
        instructions=instructions,
        batch_size=args.batch_size,
        temperature=args.stage2_temperature,
        max_new_tokens=args.stage2_max_new_tokens,
        max_retries=args.max_retries,
    )
    n_ok2 = sum(1 for x in responses if x is not None)
    print(f"\nStage 2 complete: {n_ok2}/{len(slots)} succeeded")

    # ── Save final JSONL files ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Saving results")
    print("=" * 60)
    save_results(
        slots=slots,
        instructions=instructions,
        responses=responses,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.random_seed,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
