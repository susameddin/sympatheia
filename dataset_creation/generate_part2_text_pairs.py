#!/usr/bin/env python3
"""
Generate Part 2 of the Sympatheia dataset: 11 response emotions per query.

For each unique query (in emotion X), generate responses for all 11 target
response emotions. The VA label during training will come from the response
emotion — teaching the model to prioritize the label over the audio.

Dataset counts:
  - Non-neutral emotions (10): 100 queries each = 1,000 unique queries
  - Neutral: 500 unique queries
  - Total: 1,500 unique queries × 11 response emotions = 16,500 pairs

Two-stage pipeline:
  Stage 1: Generate user query in the query emotion's style (thinking OFF)
  Stage 2: For each of 11 response emotions, generate a response in that
           emotional register (thinking ON)

Run (preview mode — inspect a few samples before full run):
  conda run -n qwen3-tts4 python dataset_creation/generate_part2_text_pairs.py \\
      --llm-model Qwen/Qwen3-32B-Instruct \\
      --output-dir /engram/naplab/users/sd3705/Datasets/Sympatheia_11Emo_17k_Part2/metadata/ \\
      --preview 3

Run (full):
  conda run -n qwen3-tts4 python dataset_creation/generate_part2_text_pairs.py \\
      --llm-model Qwen/Qwen3-32B-Instruct \\
      --output-dir /engram/naplab/users/sd3705/Datasets/Sympatheia_11Emo_17k_Part2/metadata/ \\
      --resume
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
# Emotion style descriptions (mirrors generate_new_text_pairs.py)
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

# Queries per emotion — Neutral gets 5× more as it is most common in practice
SAMPLES_PER_EMOTION: Dict[str, int] = {e: 100 for e in ALL_EMOTIONS}
SAMPLES_PER_EMOTION["Neutral"] = 500

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
        description="Generate Part 2 text pairs: 11 response emotions per query"
    )
    parser.add_argument(
        "--llm-model",
        default="Qwen/Qwen3-32B",
        help="Path or HuggingFace ID of the LLM",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save metadata JSONL files",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of queries for training set (default: 0.7)",
    )
    parser.add_argument(
        "--emotions",
        nargs="+",
        default=ALL_EMOTIONS,
        choices=ALL_EMOTIONS,
        help="Query emotions to generate (default: all 11)",
    )
    parser.add_argument(
        "--response-emotions",
        nargs="+",
        default=ALL_EMOTIONS,
        choices=ALL_EMOTIONS,
        help="Response emotions to generate per query (default: all 11)",
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
        help="Stage 1 temperature (default: 0.85)",
    )
    parser.add_argument(
        "--stage2-temperature",
        type=float,
        default=0.7,
        help="Stage 2 temperature (default: 0.7)",
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
        help="Max new tokens for Stage 2 (default: 4096)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per failed sample (default: 3)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from stage1_checkpoint_part2.jsonl if it exists",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Preview mode: generate N query groups (N×11 pairs) and print "
            "them to stdout, then exit without writing the full dataset. "
            "Samples 1 query per emotion for diversity."
        ),
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────
def build_stage1_messages(emotion: str, topic: str) -> List[Dict[str, str]]:
    """Chat messages for Stage 1 — user query in the given emotion."""
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


def build_stage2_messages(
    query_emotion: str, response_emotion: str, instruction: str
) -> List[Dict[str, str]]:
    """
    Chat messages for Stage 2 — response in the response_emotion's style.

    The prompt explicitly states:
      1. The user's emotional state (from query_emotion)
      2. The target response style (from response_emotion)
    This mirrors the EMOTION_STYLE mappings used in Part 1.
    """
    query_style = EMOTION_STYLE[query_emotion]["query"]
    response_style = EMOTION_STYLE[response_emotion]["response"]
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
                "It should be dialogue-appropriate and sound natural when spoken aloud.\n\n"
                "Output ONLY the response text \u2014 no quotes, no explanation."
            ),
        },
    ]


def apply_template(tokenizer, messages: List[Dict], enable_thinking: bool) -> str:
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
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def clean_output(text: str) -> str:
    text = strip_thinking(text)
    text = text.strip().strip('"').strip("'").strip()
    return text


def is_valid(text: str) -> bool:
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
# Batch generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
) -> List[str]:
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

    input_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[:, input_len:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# Slot building
# ─────────────────────────────────────────────────────────────────────────────
def build_query_slots(
    emotions: List[str],
    preview_n: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Build the list of unique query slots: (emotion, topic, local_index).

    In preview mode, sample 1 query per emotion up to preview_n total.
    In full mode, use SAMPLES_PER_EMOTION counts.
    """
    rng = random.Random(seed)
    slots = []

    if preview_n > 0:
        # 1 query per emotion, cycling through emotions until we have preview_n
        for i, emotion in enumerate(emotions[:preview_n]):
            topic = rng.choice(TOPIC_POOL)
            slots.append({"emotion": emotion, "topic": topic, "local_idx": 0})
    else:
        for emotion in emotions:
            n = SAMPLES_PER_EMOTION[emotion]
            topics: List[str] = []
            while len(topics) < n:
                shuffled = TOPIC_POOL[:]
                rng.shuffle(shuffled)
                topics.extend(shuffled)
            topics = topics[:n]
            for i, topic in enumerate(topics):
                slots.append({"emotion": emotion, "topic": topic, "local_idx": i})

    return slots


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: generate query texts
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
    results: List[Optional[str]] = [None] * len(slots)
    pending = list(range(len(slots)))

    for attempt in range(max_retries):
        if not pending:
            break
        print(f"\n[Stage 1] Attempt {attempt + 1}/{max_retries} — {len(pending)} slots remaining")

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
                    model, tokenizer, prompts[start:start + batch_size],
                    max_new_tokens=max_new_tokens, temperature=temperature,
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
        print(f"[Stage 1] Warning: {len(pending)} slots failed — skipping.", file=sys.stderr)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: generate 11 responses per query
# ─────────────────────────────────────────────────────────────────────────────
def run_stage2(
    model,
    tokenizer,
    slots: List[Dict[str, Any]],
    instructions: List[Optional[str]],
    response_emotions: List[str],
    batch_size: int,
    temperature: float,
    max_new_tokens: int,
    max_retries: int,
) -> Dict[int, Dict[str, Optional[str]]]:
    """
    Returns a dict: slot_idx → {response_emotion: response_text}.
    Each successful query slot gets one response per response_emotion.
    """
    # Flatten into (slot_idx, response_emotion) jobs
    jobs = [
        (i, resp_emo)
        for i, inst in enumerate(instructions)
        if inst is not None
        for resp_emo in response_emotions
    ]

    job_results: Dict[tuple, Optional[str]] = {job: None for job in jobs}
    pending = list(jobs)

    for attempt in range(max_retries):
        if not pending:
            break
        print(
            f"\n[Stage 2] Attempt {attempt + 1}/{max_retries} — "
            f"{len(pending)} (slot, resp_emo) pairs remaining"
        )

        prompts = [
            apply_template(
                tokenizer,
                build_stage2_messages(
                    query_emotion=slots[i]["emotion"],
                    response_emotion=resp_emo,
                    instruction=instructions[i],
                ),
                enable_thinking=True,
            )
            for (i, resp_emo) in pending
        ]

        outputs: List[str] = []
        for start in tqdm(range(0, len(prompts), batch_size), desc="Stage 2 batches"):
            outputs.extend(
                generate_batch(
                    model, tokenizer, prompts[start:start + batch_size],
                    max_new_tokens=max_new_tokens, temperature=temperature,
                )
            )

        still_pending = []
        for job, raw in zip(pending, outputs):
            text = clean_output(raw)
            if is_valid(text):
                job_results[job] = text
            else:
                still_pending.append(job)
        pending = still_pending

    if pending:
        print(f"[Stage 2] Warning: {len(pending)} jobs failed — skipping.", file=sys.stderr)

    # Reorganize: slot_idx → {resp_emo: text}
    organized: Dict[int, Dict[str, Optional[str]]] = defaultdict(dict)
    for (i, resp_emo), text in job_results.items():
        organized[i][resp_emo] = text
    return dict(organized)


# ─────────────────────────────────────────────────────────────────────────────
# Preview: print samples to stdout
# ─────────────────────────────────────────────────────────────────────────────
def print_preview(
    slots: List[Dict[str, Any]],
    instructions: List[Optional[str]],
    response_map: Dict[int, Dict[str, Optional[str]]],
    response_emotions: List[str],
):
    print("\n" + "=" * 70)
    print("PREVIEW — sample query-response groups")
    print("=" * 70)
    for i, (slot, inst) in enumerate(zip(slots, instructions)):
        if inst is None:
            continue
        print(f"\nQUERY [{slot['emotion']}] (topic: {slot['topic']}):")
        print(f"  \"{inst}\"")
        resps = response_map.get(i, {})
        for resp_emo in response_emotions:
            text = resps.get(resp_emo)
            status = text if text else "[FAILED]"
            print(f"  → [{resp_emo:<11}] {status}")
    print("\n" + "=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────
def save_results(
    slots: List[Dict[str, Any]],
    instructions: List[Optional[str]],
    response_map: Dict[int, Dict[str, Optional[str]]],
    response_emotions: List[str],
    output_dir: Path,
    train_ratio: float,
    seed: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all pairs (query + response_emotion combinations)
    # Group by query emotion for balanced train/eval split
    emotion_pairs: Dict[str, List[Dict]] = defaultdict(list)
    emotion_unique_queries: Dict[str, List[Dict]] = defaultdict(list)

    for i, (slot, inst) in enumerate(zip(slots, instructions)):
        if inst is None:
            continue
        query_emotion = slot["emotion"]
        query_index = f"p2_{query_emotion}_{slot['local_idx']:05d}"
        resps = response_map.get(i, {})

        # Track unique query (used for unique_queries files)
        unique_q = {
            "query_index": query_index,
            "query_text": inst,
            "query_emotion": query_emotion,
        }
        emotion_unique_queries[query_emotion].append(unique_q)

        # Create one pair per response emotion
        for resp_emo in response_emotions:
            text = resps.get(resp_emo)
            if text is None:
                continue
            pair_index = f"{query_index}_{resp_emo}"
            emotion_pairs[query_emotion].append(
                {
                    "index": pair_index,
                    "query_index": query_index,
                    "query_text": inst,
                    "query_emotion": query_emotion,
                    "response_emotion": resp_emo,
                    "response_text": text,
                }
            )

    # Split: train/eval split is done AT THE QUERY LEVEL so the same query
    # never appears in both splits (prevents data leakage)
    rng = random.Random(seed + 1)
    train_pairs: List[Dict] = []
    eval_pairs: List[Dict] = []
    train_queries: List[Dict] = []
    eval_queries: List[Dict] = []

    print("\nEmotion breakdown:")
    for emotion in ALL_EMOTIONS:
        unique_qs = emotion_unique_queries.get(emotion, [])
        if not unique_qs:
            continue

        rng.shuffle(unique_qs)
        n_train_q = int(len(unique_qs) * train_ratio)
        train_q_set = {q["query_index"] for q in unique_qs[:n_train_q]}

        train_queries.extend(unique_qs[:n_train_q])
        eval_queries.extend(unique_qs[n_train_q:])

        t_pairs = [p for p in emotion_pairs.get(emotion, []) if p["query_index"] in train_q_set]
        e_pairs = [p for p in emotion_pairs.get(emotion, []) if p["query_index"] not in train_q_set]
        train_pairs.extend(t_pairs)
        eval_pairs.extend(e_pairs)

        n_q = len(unique_qs)
        print(
            f"  {emotion}: {n_q} queries → "
            f"{len(t_pairs)} train pairs + {len(e_pairs)} eval pairs"
        )

    # Shuffle
    rng.shuffle(train_pairs)
    rng.shuffle(eval_pairs)
    rng.shuffle(train_queries)
    rng.shuffle(eval_queries)

    # Write files
    for name, data in [
        ("sampled_train.jsonl", train_pairs),
        ("sampled_eval.jsonl", eval_pairs),
        ("unique_queries_train.jsonl", train_queries),
        ("unique_queries_eval.jsonl", eval_queries),
    ]:
        path = output_dir / name
        with path.open("w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved {len(data):>6} rows → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    random.seed(args.random_seed)

    preview_mode = args.preview > 0

    n_gpus = torch.cuda.device_count()
    print(f"\nDetected {n_gpus} GPU(s)")
    print(f"LLM model:          {args.llm_model}")
    print(f"Query emotions:     {args.emotions}")
    print(f"Response emotions:  {args.response_emotions}")
    if preview_mode:
        print(f"Mode:               PREVIEW ({args.preview} query groups)")
    else:
        total_queries = sum(SAMPLES_PER_EMOTION[e] for e in args.emotions)
        total_pairs = total_queries * len(args.response_emotions)
        print(f"Mode:               FULL ({total_queries} queries × {len(args.response_emotions)} resp emotions = {total_pairs} pairs)")

    # ── Build query slots ────────────────────────────────────────────────────
    print("\nBuilding query slots...")
    slots = build_query_slots(
        emotions=args.emotions,
        preview_n=args.preview,
        seed=args.random_seed,
    )
    print(f"Total query slots: {len(slots)}")

    # ── Load checkpoint if resuming ──────────────────────────────────────────
    stage1_ckpt = args.output_dir / "stage1_checkpoint_part2.jsonl"
    instructions: List[Optional[str]] = [None] * len(slots)

    if not preview_mode and args.resume and stage1_ckpt.exists():
        print(f"\nResuming from Stage 1 checkpoint: {stage1_ckpt}")
        ckpt_map: Dict[str, str] = {}
        with stage1_ckpt.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                key = f"{row['emotion']}_{row['local_idx']}"
                if row.get("instruction"):
                    ckpt_map[key] = row["instruction"]
        for i, slot in enumerate(slots):
            key = f"{slot['emotion']}_{slot['local_idx']}"
            if key in ckpt_map:
                instructions[i] = ckpt_map[key]
        n_loaded = sum(1 for x in instructions if x is not None)
        print(f"Loaded {n_loaded}/{len(slots)} Stage 1 results from checkpoint")
        # Re-run Stage 1 only for still-None slots
        still_none = [i for i, x in enumerate(instructions) if x is None]
        if still_none:
            print(f"Running Stage 1 for {len(still_none)} remaining slots...")
    else:
        still_none = list(range(len(slots)))

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"\nLoading LLM: {args.llm_model}")
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

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    if still_none:
        print("\n" + "=" * 60)
        print("STAGE 1: Generating user queries (thinking OFF)")
        print("=" * 60)
        partial_slots = [slots[i] for i in still_none]
        partial_results = run_stage1(
            model=model,
            tokenizer=tokenizer,
            slots=partial_slots,
            batch_size=args.batch_size,
            temperature=args.stage1_temperature,
            max_new_tokens=args.stage1_max_new_tokens,
            max_retries=args.max_retries,
        )
        for i, res in zip(still_none, partial_results):
            instructions[i] = res

        n_ok1 = sum(1 for x in instructions if x is not None)
        print(f"\nStage 1 complete: {n_ok1}/{len(slots)} succeeded")

        # Save checkpoint
        if not preview_mode:
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

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 2: Generating responses for all response emotions (thinking ON)")
    print("=" * 60)
    response_map = run_stage2(
        model=model,
        tokenizer=tokenizer,
        slots=slots,
        instructions=instructions,
        response_emotions=args.response_emotions,
        batch_size=args.batch_size,
        temperature=args.stage2_temperature,
        max_new_tokens=args.stage2_max_new_tokens,
        max_retries=args.max_retries,
    )
    total_generated = sum(
        sum(1 for v in resps.values() if v is not None)
        for resps in response_map.values()
    )
    print(f"\nStage 2 complete: {total_generated} responses generated")

    # ── Preview or save ──────────────────────────────────────────────────────
    if preview_mode:
        print_preview(slots, instructions, response_map, args.response_emotions)
        # Also save preview JSONL
        args.output_dir.mkdir(parents=True, exist_ok=True)
        preview_path = args.output_dir / "preview_pairs.jsonl"
        with preview_path.open("w", encoding="utf-8") as f:
            for i, (slot, inst) in enumerate(zip(slots, instructions)):
                if inst is None:
                    continue
                q_idx = f"p2_{slot['emotion']}_preview_{slot['local_idx']:05d}"
                resps = response_map.get(i, {})
                for resp_emo in args.response_emotions:
                    text = resps.get(resp_emo)
                    if text:
                        f.write(
                            json.dumps(
                                {
                                    "index": f"{q_idx}_{resp_emo}",
                                    "query_index": q_idx,
                                    "query_text": inst,
                                    "query_emotion": slot["emotion"],
                                    "response_emotion": resp_emo,
                                    "response_text": text,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
        print(f"\nPreview JSONL saved → {preview_path}")
        print("Review the output above, then run without --preview for the full dataset.")
        return

    # ── Save full dataset ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Saving results")
    print("=" * 60)
    save_results(
        slots=slots,
        instructions=instructions,
        response_map=response_map,
        response_emotions=args.response_emotions,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.random_seed,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
