#!/usr/bin/env python3
"""
Generate Part 2 v2 of the Sympatheia dataset: neutral-only queries with
12 response emotions.

All queries are neutral (no emotional coloring). For each neutral query,
generate responses for all 12 target emotions. The VA label comes from the
response emotion, forcing the model to rely on the VA label rather than
the query audio's emotion.

Dataset counts:
  - 500 neutral queries × 12 response emotions = 6,000 pairs

Two-stage pipeline:
  Stage 1: Generate neutral user queries (thinking OFF)
  Stage 2: For each of 12 response emotions, generate a response that
           clearly addresses the user's emotional state (thinking ON)

Run (preview mode — inspect a few samples before full run):
  conda run -n qwen3-tts4 python dataset_creation/generate_part2_text_pairs.py \
      --llm-model /engram/naplab/users/sd3705/models/Qwen3-32B \
      --output-dir /engram/naplab/users/sd3705/Datasets/Sympatheia_12Emo_Neutral/metadata/ \
      --preview 3

Run (full):
  conda run -n qwen3-tts4 python dataset_creation/generate_part2_text_pairs.py \
      --llm-model /engram/naplab/users/sd3705/models/Qwen3-32B \
      --output-dir /engram/naplab/users/sd3705/Datasets/Sympatheia_12Emo_Neutral/metadata/ \
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

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Response emotion context — rich descriptions for Stage 2 prompt
# ─────────────────────────────────────────────────────────────────────────────
RESPONSE_EMOTION_CONTEXT = {
    "Sad": {
        "user_feeling": "deeply sad, emotionally hurt, and feeling down",
        "response_goal": "comfort them and validate their sadness while answering their question — connect the answer to their emotional wellbeing where natural",
        "example_cues": "acknowledge their pain warmly, answer their actual question, then tie the answer back to gentle encouragement or comfort; show you care about both their feelings and their needs",
        "avoid": "Don't forget to answer their actual question. Don't spend the entire response on emotional validation without addressing what they asked. Don't minimize their sadness either — do both.",
    },
    "Excited": {
        "user_feeling": "bursting with excitement, thrilled, and full of energy",
        "response_goal": "match their high energy and celebrate with them while answering their question enthusiastically",
        "example_cues": "mirror their excitement, answer their question with genuine enthusiasm, amplify what they're excited about by connecting it to real content and information",
        "avoid": "Don't be merely polite or lukewarm. Don't replace the answer with pure cheerleading — celebrate AND answer. Don't tone down their energy.",
    },
    "Frustrated": {
        "user_feeling": "very frustrated, stuck, and losing patience",
        "response_goal": "validate their frustration and show understanding, then answer their question with practical, useful content",
        "example_cues": "acknowledge that the situation is annoying, show you understand why they're frustrated, then provide a real answer — frustrated people need both empathy and solutions",
        "avoid": "Don't be dismissive of their frustration. Don't tell them to 'just relax'. Don't skip answering their question to only talk about their frustration.",
    },
    "Neutral": {
        "user_feeling": "neutral — not expressing any strong emotion",
        "response_goal": "answer their question helpfully and naturally, without commenting on or acknowledging their emotional state",
        "example_cues": "just be a warm, friendly, helpful assistant; respond to the content of their message without any emotional framing or acknowledgment",
        "avoid": "Don't comment on their calmness or neutrality (e.g. 'It's great that you're in a calm place'). Don't project emotions. Don't be therapeutic. Just answer the question like a normal friendly assistant.",
    },
    "Happy": {
        "user_feeling": "genuinely happy, joyful, and in a wonderful mood",
        "response_goal": "share in their happiness warmly and celebrate what makes them happy, while answering their question with that same positive energy",
        "example_cues": "express genuine joy for them, answer their question with warm positive energy, reference what makes them happy and connect it to the actual content of your answer",
        "avoid": "Don't be generic. Don't just say 'that's great' without answering the question. Share their joy AND give them a real, useful response.",
    },
    "Angry": {
        "user_feeling": "very angry, upset, and possibly feeling wronged",
        "response_goal": "validate their anger and show you understand why they're upset, then answer their question calmly and helpfully",
        "example_cues": "acknowledge their right to be angry, stay calm and measured, then provide a real answer to their question; show you're on their side while being helpful",
        "avoid": "Don't ignore their anger. Don't be preachy or lecture them. Don't tell them to calm down. Don't forget to answer their actual question.",
    },
    "Anxious": {
        "user_feeling": "anxious, worried, and feeling unsafe or nervous",
        "response_goal": "reassure them and acknowledge their anxiety, then answer their question with clear, concrete information that naturally helps reduce their worry",
        "example_cues": "validate that their anxiety is understandable, provide a clear and specific answer — concrete information naturally reduces anxiety; connect the answer to reassurance",
        "avoid": "Don't dismiss their anxiety as irrational. Don't say 'don't worry about it'. Don't add unnecessary caveats that increase worry. Answer clearly while being reassuring.",
    },
    "Relaxed": {
        "user_feeling": "very relaxed, at ease, and content",
        "response_goal": "match their calm energy while answering their question at a leisurely, unhurried pace",
        "example_cues": "keep the vibe mellow and easy, answer their question without introducing urgency, enjoy the conversational moment together",
        "avoid": "Don't introduce stress or urgency. Don't be overly energetic. Answer the question while matching their relaxed vibe.",
    },
    "Surprised": {
        "user_feeling": "genuinely surprised and taken aback (in a curious way)",
        "response_goal": "engage with their surprise and share in the wonder, while answering their question with genuine curiosity",
        "example_cues": "share in the surprise, answer their question while exploring the unexpected together, build on their sense of discovery with real information",
        "avoid": "Don't be indifferent to what surprised them. Show genuine curiosity and engagement while still answering the question.",
    },
    "Disgusted": {
        "user_feeling": "disgusted, repulsed, or revolted by something",
        "response_goal": "validate their disgust directly and show you understand why it's repulsive, then answer their question without lingering unnecessarily",
        "example_cues": "use words like 'disgusting', 'gross', or 'revolting' to show you truly understand; acknowledge their reaction makes complete sense; then answer their actual question",
        "avoid": "Don't philosophize or try to normalize what disgusted them. Don't use generic words like 'frustrating' instead of naming the disgust. Don't forget to answer the question.",
    },
    "Tired": {
        "user_feeling": "exhausted, drained, running on empty, and worn out",
        "response_goal": "acknowledge how exhausted they are and validate their need for rest, while answering their question gently and concisely",
        "example_cues": "recognize their exhaustion, answer their question briefly and gently — don't overload them; keep it warm but concise so you don't add to their mental load",
        "avoid": "Don't skip acknowledging their tiredness. Don't give an overly long response. Don't forget to answer their question. Keep it gentle and manageable.",
    },
    "Content": {
        "user_feeling": "content, satisfied, and at peace",
        "response_goal": "appreciate the moment with them and reinforce their contentment, while answering their question warmly",
        "example_cues": "acknowledge their peaceful state, answer their question with warm gentle energy, reflect on what's making them feel good while providing real content",
        "avoid": "Don't introduce new problems or urgency. Don't be overly energetic. Answer the question while keeping the contented, peaceful vibe.",
    },
}

ALL_EMOTIONS = list(RESPONSE_EMOTION_CONTEXT.keys())

NUM_NEUTRAL_QUERIES = 500

TOPIC_POOL = [
    # Daily life
    "daily routine",
    "morning habits",
    "evening wind-down routine",
    "commuting to work or school",
    "running errands",
    "organizing your living space",
    "home chores",
    "laundry and cleaning",
    "meal prepping for the week",
    "grocery shopping",
    # Work & study
    "work or study stress",
    "job interviews",
    "workplace relationships with colleagues",
    "switching careers",
    "working from home",
    "balancing work and personal life",
    "learning something new",
    "taking an online course",
    "preparing for an exam",
    "giving a presentation",
    # Relationships & social
    "family dynamics",
    "friendship",
    "dating and romantic relationships",
    "dealing with difficult neighbors",
    "reconnecting with old friends",
    "hosting a dinner party",
    "attending a wedding",
    "helping a friend move",
    "meeting new people",
    "family traditions and holidays",
    # Health & wellness
    "health and wellness",
    "starting a new exercise routine",
    "getting enough sleep",
    "dealing with a minor illness",
    "visiting the dentist or doctor",
    "mental health and self-care",
    "trying meditation or yoga",
    "managing allergies",
    "staying hydrated throughout the day",
    "recovering from a sports injury",
    # Food & cooking
    "food and dining",
    "trying a new restaurant",
    "cooking a complicated recipe",
    "baking desserts",
    "dietary restrictions and food choices",
    "coffee and tea preferences",
    "ordering takeout",
    "farmer's markets and local produce",
    "meal planning on a budget",
    "kitchen gadgets and tools",
    # Travel & outdoors
    "travel plans",
    "booking flights and hotels",
    "road trip planning",
    "camping and hiking",
    "visiting a national park",
    "navigating public transportation",
    "packing for a trip",
    "jet lag and time zones",
    "travel photography",
    "exploring a new city",
    # Entertainment & media
    "movies or TV shows",
    "podcasts and audiobooks",
    "live music and concerts",
    "board games and card games",
    "video games",
    "reading books and book clubs",
    "stand-up comedy shows",
    "museum and art gallery visits",
    "streaming services and recommendations",
    "theater and live performances",
    # Hobbies & creativity
    "hobbies",
    "painting or drawing",
    "playing a musical instrument",
    "gardening and plant care",
    "DIY home improvement projects",
    "photography as a hobby",
    "knitting or crafting",
    "writing a journal or blog",
    "collecting things as a hobby",
    "learning a new language",
    # Finance & practical
    "money and finances",
    "budgeting and saving money",
    "investing for beginners",
    "paying off student loans or debt",
    "renting versus buying a home",
    "shopping",
    "online shopping habits",
    "comparing insurance plans",
    "tax season preparation",
    "negotiating a salary raise",
    # Technology & digital life
    "technology",
    "smartphone tips and tricks",
    "managing email and notifications",
    "home automation and smart devices",
    "computer troubleshooting",
    "social media",
    "online privacy and security",
    "backing up important files",
    "choosing a new laptop or phone",
    "using productivity apps",
    "setting up a home Wi-Fi network",
    # Nature & weather
    "weather",
    "seasonal changes and preparation",
    "dealing with extreme heat or cold",
    "rainy day activities",
    "gardening with the seasons",
    "weather forecasts and planning outdoor events",
    "stargazing and astronomy",
    "birdwatching or wildlife spotting",
    # Sports & fitness
    "sports",
    "following a favorite sports team",
    "learning to swim or a new sport",
    "running a marathon or 5K",
    "joining a gym or fitness class",
    "watching the Olympics or World Cup",
    "cycling or biking",
    "yoga and stretching routines",
    # Life transitions & future
    "future plans",
    "moving to a new city",
    "starting a side project",
    "adopting or fostering a pet",
    "pets",
    "planning a big life milestone",
    "retirement planning",
    "volunteering and community service",
    "setting personal goals for the year",
    "current news",
    "discussing a recent documentary",
]


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Part 2 v2 text pairs: neutral queries × 11 response emotions"
    )
    parser.add_argument(
        "--llm-model",
        default="/engram/naplab/users/sd3705/models/Qwen3-32B",
        help="Path or HuggingFace ID of the LLM",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save metadata JSONL files",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=NUM_NEUTRAL_QUERIES,
        help=f"Number of unique neutral queries to generate (default: {NUM_NEUTRAL_QUERIES})",
    )
    parser.add_argument(
        "--overgenerate-factor",
        type=float,
        default=2.5,
        help="Over-generate by this factor to compensate for dedup filtering (default: 2.5)",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.85,
        help="Cosine similarity threshold for dedup (default: 0.85)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of queries for training set (default: 0.7)",
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
        help="Resume from stage1_checkpoint_part2v2.jsonl if it exists",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Preview mode: generate N neutral queries and their 11 response "
            "variants, then print them to stdout and exit."
        ),
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────
def build_stage1_messages(topic: str) -> List[Dict[str, str]]:
    """Chat messages for Stage 1 — neutral user query."""
    return [
        {
            "role": "system",
            "content": "You are generating training data for a speech dialogue system.",
        },
        {
            "role": "user",
            "content": (
                f"Generate a natural, conversational, spoken-style question or request "
                f"(1\u20132 sentences) about: {topic}.\n\n"
                "Requirements:\n"
                "- The tone must be completely NEUTRAL \u2014 no emotional words, no excitement, "
                "no sadness, no frustration, no strong feelings of any kind\n"
                "- It should sound like a calm, matter-of-fact person asking a straightforward question\n"
                "- Spoken English only (as if said aloud, not written)\n"
                "- Do NOT include emotional adjectives, exclamation marks, or sentiment words\n"
                "- Output ONLY the question/request text \u2014 no quotes, no explanation"
            ),
        },
    ]


def build_stage2_messages(
    response_emotion: str, instruction: str
) -> List[Dict[str, str]]:
    """
    Chat messages for Stage 2 — response addressing a user in a specific
    emotional state. The instruction is always a neutral query.
    """
    ctx = RESPONSE_EMOTION_CONTEXT[response_emotion]
    return [
        {
            "role": "system",
            "content": (
                "You are a deeply empathetic AI assistant. You always acknowledge and "
                "address the user's emotions explicitly and warmly. At the same time, "
                "you always answer their actual question or request \u2014 weaving emotional "
                "support and the topic together naturally. Never ignore the user's "
                "emotions, and never ignore their question."
            ),
        },
        {
            "role": "user",
            "content": (
                f"The user is feeling {ctx['user_feeling']}.\n"
                f"They said: \"{instruction}\"\n\n"
                f"Your goal: {ctx['response_goal']}.\n"
                f"Guidelines: {ctx['example_cues']}.\n"
                f"Avoid: {ctx['avoid']}\n\n"
                "Generate a natural spoken response (3\u20135 sentences) that:\n"
                "1. Answers their question AND addresses their emotional state, woven together naturally throughout\n"
                "2. The emotional awareness should flow into the answer itself \u2014 not be a separate block\n"
                "3. Do NOT open with an explicit emotion label like 'I'm sorry you're sad' or 'I can tell you're frustrated' \u2014 "
                "instead, let the empathy come through in HOW you talk about the topic and connect it to their feelings\n"
                "4. A listener could tell WHAT EMOTION you're responding to AND what the user asked about\n\n"
                "IMPORTANT: Your response must contain a real answer to the user's question. "
                "The empathy and the answer should feel like one flowing conversation, not two separate parts.\n\n"
                "BAD example (emotional but ignores the question):\n"
                "  User (sad): 'How is the weather today?'\n"
                "  'I'm so sorry you're feeling so down \u2014 that must be really hard. "
                "It's completely okay to feel this way. Would you like to talk about what's on your mind?'\n"
                "BAD example (answer and emotion feel disconnected):\n"
                "  User (sad): 'How is the weather today?'\n"
                "  'The weather today is mild and partly cloudy. I can hear you're going through "
                "a really tough time, and I'm sorry you're carrying that weight.'\n"
                "GOOD example (empathy woven naturally into the answer):\n"
                "  User (sad): 'How is the weather today?'\n"
                "  'It's actually a pretty gentle day out there \u2014 mild and partly cloudy, the kind of "
                "weather that's easy on you when everything feels heavy. Sometimes just stepping outside "
                "for a quiet moment can take a bit of that weight off. You don't have to push through "
                "everything at once.'\n\n"
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
# Embedding-based deduplication
# ─────────────────────────────────────────────────────────────────────────────
def build_dedup_filter(threshold: float = 0.85):
    """Create a dedup checker using sentence embeddings.

    Returns a callable ``is_duplicate(text) -> bool`` that returns True when
    *text* is too similar (cosine similarity >= *threshold*) to any previously
    accepted text.
    """
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings: List[np.ndarray] = []

    def is_duplicate(text: str) -> bool:
        emb = emb_model.encode([text], normalize_embeddings=True)[0]
        if embeddings:
            sims = np.dot(np.stack(embeddings), emb)
            if float(np.max(sims)) >= threshold:
                return True
        embeddings.append(emb)
        return False

    return is_duplicate


def dedup_and_trim(
    instructions: List[Optional[str]],
    target_count: int,
    threshold: float = 0.85,
) -> int:
    """Post-generation dedup + trim.  Modifies *instructions* in-place.

    Returns the number of unique queries kept.
    """
    is_dup = build_dedup_filter(threshold)
    kept = 0
    for i, inst in enumerate(instructions):
        if inst is None:
            continue
        if is_dup(inst):
            instructions[i] = None
        else:
            kept += 1
            if kept >= target_count:
                # Null out remaining to trim to target
                for j in range(i + 1, len(instructions)):
                    instructions[j] = None
                break
    return kept


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
# Slot building — all neutral queries
# ─────────────────────────────────────────────────────────────────────────────
def build_query_slots(
    num_queries: int,
    preview_n: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    slots = []

    n = preview_n if preview_n > 0 else num_queries

    topics: List[str] = []
    while len(topics) < n:
        shuffled = TOPIC_POOL[:]
        rng.shuffle(shuffled)
        topics.extend(shuffled)
    topics = topics[:n]

    for i, topic in enumerate(topics):
        slots.append({"emotion": "Neutral", "topic": topic, "local_idx": i})

    return slots


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: generate neutral query texts
# ─────────────────────────────────────────────────────────────────────────────
def run_stage1(
    model,
    tokenizer,
    slots: List[Dict[str, Any]],
    batch_size: int,
    temperature: float,
    max_new_tokens: int,
    max_retries: int,
    is_duplicate=None,
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
                build_stage1_messages(slots[i]["topic"]),
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
            if is_valid(text) and (is_duplicate is None or not is_duplicate(text)):
                results[idx] = text
            else:
                still_pending.append(idx)
        pending = still_pending

    if pending:
        print(f"[Stage 1] Warning: {len(pending)} slots failed/duplicate — skipping.", file=sys.stderr)
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
    print("PREVIEW — sample query-response groups (all queries are NEUTRAL)")
    print("=" * 70)
    for i, (slot, inst) in enumerate(zip(slots, instructions)):
        if inst is None:
            continue
        print(f"\nQUERY [Neutral] (topic: {slot['topic']}):")
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
    dedup_threshold: float = 0.85,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pairs: List[Dict] = []
    all_unique_queries: List[Dict] = []

    # Re-index sequentially so indices are contiguous after dedup trimming
    seq_idx = 0
    for i, (slot, inst) in enumerate(zip(slots, instructions)):
        if inst is None:
            continue
        query_index = f"p2v2_Neutral_{seq_idx:05d}"
        seq_idx += 1
        resps = response_map.get(i, {})

        unique_q = {
            "query_index": query_index,
            "query_text": inst,
            "query_emotion": "Neutral",
        }
        all_unique_queries.append(unique_q)

        for resp_emo in response_emotions:
            text = resps.get(resp_emo)
            if text is None:
                continue
            pair_index = f"{query_index}_{resp_emo}"
            all_pairs.append(
                {
                    "index": pair_index,
                    "query_index": query_index,
                    "query_text": inst,
                    "query_emotion": "Neutral",
                    "response_emotion": resp_emo,
                    "response_text": text,
                }
            )

    # ── Embedding-clustered train/eval split ─────────────────────────────────
    # Cluster semantically similar queries so they stay in the same split
    query_texts = [q["query_text"] for q in all_unique_queries]
    print(f"\nEncoding {len(query_texts)} queries for cluster-based split...")
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embs = emb_model.encode(query_texts, normalize_embeddings=True)

    # Greedy clustering: group queries with cosine sim >= threshold
    clusters: List[List[int]] = []
    assigned: set = set()
    for i in range(len(query_texts)):
        if i in assigned:
            continue
        cluster = [i]
        assigned.add(i)
        for j in range(i + 1, len(query_texts)):
            if j in assigned:
                continue
            if float(np.dot(query_embs[i], query_embs[j])) >= dedup_threshold:
                cluster.append(j)
                assigned.add(j)
        clusters.append(cluster)

    print(f"  {len(query_texts)} queries → {len(clusters)} clusters")

    # Split clusters into train/eval
    rng = random.Random(seed + 1)
    rng.shuffle(clusters)
    n_train_clusters = int(len(clusters) * train_ratio)

    train_indices: set = set()
    for c in clusters[:n_train_clusters]:
        train_indices.update(c)

    train_q_set = {all_unique_queries[i]["query_index"] for i in train_indices}

    train_queries = [all_unique_queries[i] for i in sorted(train_indices)]
    eval_queries = [all_unique_queries[i] for i in range(len(all_unique_queries)) if i not in train_indices]

    train_pairs = [p for p in all_pairs if p["query_index"] in train_q_set]
    eval_pairs = [p for p in all_pairs if p["query_index"] not in train_q_set]

    rng.shuffle(train_pairs)
    rng.shuffle(eval_pairs)
    rng.shuffle(train_queries)
    rng.shuffle(eval_queries)

    print(f"\nDataset split (cluster-based, threshold={dedup_threshold}):")
    print(f"  Train: {len(train_queries)} queries → {len(train_pairs)} pairs")
    print(f"  Eval:  {len(eval_queries)} queries → {len(eval_pairs)} pairs")

    # Response emotion distribution
    print("\nResponse emotion distribution (train):")
    r_emo_count: Dict[str, int] = defaultdict(int)
    for p in train_pairs:
        r_emo_count[p["response_emotion"]] += 1
    for emo in ALL_EMOTIONS:
        print(f"  {emo:<12}: {r_emo_count.get(emo, 0)}")

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
    print(f"Response emotions:  {args.response_emotions}")
    if preview_mode:
        print(f"Mode:               PREVIEW ({args.preview} neutral queries × {len(args.response_emotions)} resp emotions)")
    else:
        total_pairs = args.num_queries * len(args.response_emotions)
        print(f"Mode:               FULL ({args.num_queries} neutral queries × {len(args.response_emotions)} resp emotions = {total_pairs} pairs)")

    # ── Build query slots (over-generate to compensate for dedup) ───────────
    raw_count = int(args.num_queries * args.overgenerate_factor) if not preview_mode else args.preview
    print(f"\nBuilding query slots (all neutral)...")
    print(f"  Target unique queries: {args.num_queries}")
    if not preview_mode:
        print(f"  Over-generate factor:  {args.overgenerate_factor}x → {raw_count} raw slots")
        print(f"  Dedup threshold:       {args.dedup_threshold}")
    slots = build_query_slots(
        num_queries=raw_count,
        preview_n=args.preview,
        seed=args.random_seed,
    )
    print(f"Total query slots: {len(slots)}")

    # ── Create dedup filter ──────────────────────────────────────────────────
    is_dup = build_dedup_filter(threshold=args.dedup_threshold)

    # ── Load checkpoint if resuming ──────────────────────────────────────────
    stage1_ckpt = args.output_dir / "stage1_checkpoint_part2v2.jsonl"
    instructions: List[Optional[str]] = [None] * len(slots)

    if not preview_mode and args.resume and stage1_ckpt.exists():
        print(f"\nResuming from Stage 1 checkpoint: {stage1_ckpt}")
        ckpt_map: Dict[int, str] = {}
        with stage1_ckpt.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("instruction"):
                    ckpt_map[row["local_idx"]] = row["instruction"]
        for i, slot in enumerate(slots):
            if slot["local_idx"] in ckpt_map:
                inst = ckpt_map[slot["local_idx"]]
                instructions[i] = inst
                # Populate dedup filter with checkpoint texts
                is_dup(inst)
        n_loaded = sum(1 for x in instructions if x is not None)
        print(f"Loaded {n_loaded}/{len(slots)} Stage 1 results from checkpoint")
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
        print("STAGE 1: Generating NEUTRAL user queries (thinking OFF)")
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
            is_duplicate=is_dup,
        )
        for i, res in zip(still_none, partial_results):
            instructions[i] = res

        n_ok1 = sum(1 for x in instructions if x is not None)
        print(f"\nStage 1 complete: {n_ok1}/{len(slots)} succeeded (before dedup)")

        # Save checkpoint
        if not preview_mode:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            with stage1_ckpt.open("w", encoding="utf-8") as f:
                for slot, inst in zip(slots, instructions):
                    f.write(
                        json.dumps(
                            {
                                "emotion": "Neutral",
                                "topic": slot["topic"],
                                "local_idx": slot["local_idx"],
                                "instruction": inst,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            print(f"Stage 1 checkpoint saved → {stage1_ckpt}")

    # ── Post-generation dedup + trim ────────────────────────────────────────
    if not preview_mode:
        target = args.num_queries
        n_unique = dedup_and_trim(instructions, target, threshold=args.dedup_threshold)
        print(f"\nAfter dedup + trim: {n_unique} unique queries (target: {target})")
        if n_unique < target:
            print(
                f"WARNING: Only {n_unique} unique queries achieved. "
                f"Consider increasing --overgenerate-factor (currently {args.overgenerate_factor}).",
                file=sys.stderr,
            )

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 2: Generating emotion-specific responses (thinking ON)")
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
                q_idx = f"p2v2_Neutral_preview_{slot['local_idx']:05d}"
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
                                    "query_emotion": "Neutral",
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
        dedup_threshold=args.dedup_threshold,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
