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

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
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
    "Anxious":    {"query": "Very anxious",     "response": "Soft, soothing, steady"},
    "Relaxed":    {"query": "Very relaxed",    "response": "Calm, chill, soothing"},
    "Surprised":  {"query": "Very surprised",  "response": "Curious, bright, attentive"},
    "Disgusted":  {"query": "Very disgusted",  "response": "Calm, brief, slightly distanced"},
    "Tired":      {"query": "Very tired",      "response": "Low energy, slow, gentle"},
    "Content":    {"query": "Very content",    "response": "Warm, gentle, satisfied"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Rich emotion context for Stage 2 response generation
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


# ─────────────────────────────────────────────────────────────────────────────
# Emotion-specific keywords for validation (Change C)
# ─────────────────────────────────────────────────────────────────────────────
EMOTION_KEYWORDS = {
    "Disgusted": ["disgust", "gross", "repuls", "revolt", "repel", "sicken", "nasty", "vile", "appall", "off-putting", "stomach-turning", "unpleasant", "awful"],
    "Tired": ["exhaust", "tired", "drain", "worn", "fatigue", "sleep", "rest", "energy", "weary", "burn"],
    "Neutral": [],  # No keyword check for neutral
}

ALL_EMOTIONS = list(EMOTION_STYLE.keys())

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
        description="Generate brand-new emotion text pairs with Qwen3-32B-Instruct"
    )
    parser.add_argument(
        "--llm-model",
        default="/engram/naplab/users/sd3705/models/Qwen3-32B",
        help="Path or HuggingFace ID of the LLM (default: /engram/naplab/users/sd3705/models/Qwen3-32B)",
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
        help="Number of unique pairs to generate per emotion (default: 1000)",
    )
    parser.add_argument(
        "--overgenerate-factor",
        type=float,
        default=1.5,
        help="Over-generate by this factor to compensate for dedup filtering (default: 1.5)",
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
        help="Fraction of samples for training set (default: 0.7)",
    )
    parser.add_argument(
        "--emotions",
        nargs="+",
        default=ALL_EMOTIONS,
        choices=ALL_EMOTIONS,
        help="Emotions to generate (default: all 12)",
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
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Preview mode: generate N samples per emotion and print "
            "query-response pairs to stdout, then exit."
        ),
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
    ctx = RESPONSE_EMOTION_CONTEXT[emotion]
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
                f'They said: "{instruction}"\n\n'
                f"Your goal: {ctx['response_goal']}.\n"
                f"Guidelines: {ctx['example_cues']}.\n"
                f"Avoid: {ctx['avoid']}\n\n"
                "Generate a natural spoken response (3\u20135 sentences) that:\n"
                "1. Acknowledges and addresses the user's emotional state explicitly and warmly\n"
                "2. Answers their actual question or addresses their request with real, useful content\n"
                "3. Weaves the emotional support and the topic together \u2014 connect how the topic "
                "relates to their feelings where natural\n"
                "4. A listener could tell WHAT EMOTION you're responding to AND what the user asked about\n\n"
                "IMPORTANT: Your response must contain a real answer to the user's question. "
                "You should be deeply empathetic, but do it AROUND the topic, not instead of it.\n\n"
                "BAD example (emotional but ignores the question):\n"
                "  User (sad, asking about weather): 'I just feel so down today... what's the weather like?'\n"
                "  'I'm so sorry you're feeling so down \u2014 that must be really hard. "
                "It's completely okay to feel this way. Would you like to talk about what's on your mind?'\n"
                "GOOD example (emotionally rich AND answers the question):\n"
                "  User (sad, asking about weather): 'I just feel so down today... what's the weather like?'\n"
                "  'I hear you, and I'm really sorry you're carrying that heaviness today. "
                "The weather is actually mild and partly cloudy right now \u2014 sometimes just "
                "stepping outside for a bit, even when you're feeling low, can ease that "
                "weight a little. You don't have to push through everything at once.'\n\n"
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


def is_emotion_specific(text: str, emotion: str) -> bool:
    """Check that responses for emotions with known keyword gaps use emotion-specific language."""
    keywords = EMOTION_KEYWORDS.get(emotion)
    if not keywords:  # No keyword check for this emotion
        return True
    low = text.lower()
    return any(kw in low for kw in keywords)


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


def dedup_and_trim_per_emotion(
    slots: List[Dict[str, Any]],
    instructions: List[Optional[str]],
    target_per_emotion: int,
    threshold: float = 0.85,
) -> int:
    """Post-generation dedup + trim per emotion.  Modifies *instructions* in-place.

    Returns total number of unique queries kept.
    """
    # Group indices by emotion
    emotion_indices: Dict[str, List[int]] = defaultdict(list)
    for i, slot in enumerate(slots):
        if instructions[i] is not None:
            emotion_indices[slot["emotion"]].append(i)

    total_kept = 0
    for emotion, indices in emotion_indices.items():
        is_dup = build_dedup_filter(threshold)
        kept = 0
        for i in indices:
            if is_dup(instructions[i]):
                instructions[i] = None
            else:
                kept += 1
                if kept >= target_per_emotion:
                    # Null out remaining for this emotion
                    for j in indices[indices.index(i) + 1:]:
                        instructions[j] = None
                    break
        total_kept += kept
    return total_kept


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
    is_duplicate=None,
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
            if is_valid(text) and (is_duplicate is None or not is_duplicate(text)):
                results[idx] = text
            else:
                still_pending.append(idx)

        pending = still_pending

    if pending:
        print(
            f"[Stage 1] Warning: {len(pending)} slots failed/duplicate after {max_retries} retries — skipping.",
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
            emotion = slots[idx]["emotion"]
            if is_valid(text) and is_emotion_specific(text, emotion):
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
# Preview: print samples to stdout
# ─────────────────────────────────────────────────────────────────────────────
def print_preview(
    slots: List[Dict[str, Any]],
    instructions: List[Optional[str]],
    responses: List[Optional[str]],
):
    print("\n" + "=" * 70)
    print("PREVIEW — sample query-response pairs (emotional queries)")
    print("=" * 70)
    for i, (slot, inst, resp) in enumerate(zip(slots, instructions, responses)):
        if inst is None:
            continue
        emotion = slot["emotion"]
        topic = slot["topic"]
        print(f"\n[{emotion}] (topic: {topic}):")
        print(f"  QUERY:    \"{inst}\"")
        if resp:
            print(f"  RESPONSE: \"{resp}\"")
        else:
            print(f"  RESPONSE: [FAILED]")
    print("\n" + "=" * 70)


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
    dedup_threshold: float = 0.85,
):
    """Split successful samples into train/eval and save as JSONL."""
    output_dir.mkdir(parents=True, exist_ok=True)

    emotion_samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    skipped = 0

    # Re-index sequentially per emotion after dedup trimming
    emotion_counters: Dict[str, int] = defaultdict(int)
    for slot, inst, resp in zip(slots, instructions, responses):
        if inst is None or resp is None:
            skipped += 1
            continue
        emotion = slot["emotion"]
        idx = emotion_counters[emotion]
        emotion_counters[emotion] += 1
        emotion_samples[emotion].append(
            {
                "index": f"new_{emotion}_{idx:05d}",
                "query_text": inst,
                "query_emotion": emotion,
                "source_emotion": emotion,
                "response_text": resp,
            }
        )

    if skipped:
        print(f"\nSkipped {skipped} samples due to generation failures.")

    # ── Embedding-clustered split per emotion ────────────────────────────────
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    rng = random.Random(seed + 1)
    train_data: List[Dict] = []
    eval_data: List[Dict] = []

    print(f"\nEmotion split breakdown (cluster-based, threshold={dedup_threshold}):")
    for emotion in ALL_EMOTIONS:
        samples = emotion_samples.get(emotion, [])
        if not samples:
            continue

        # Encode and cluster
        texts = [s["query_text"] for s in samples]
        embs = emb_model.encode(texts, normalize_embeddings=True)

        clusters: List[List[int]] = []
        assigned: set = set()
        for i in range(len(texts)):
            if i in assigned:
                continue
            cluster = [i]
            assigned.add(i)
            for j in range(i + 1, len(texts)):
                if j in assigned:
                    continue
                if float(np.dot(embs[i], embs[j])) >= dedup_threshold:
                    cluster.append(j)
                    assigned.add(j)
            clusters.append(cluster)

        # Split clusters into train/eval
        rng.shuffle(clusters)
        n_train_clusters = int(len(clusters) * train_ratio)

        train_indices: set = set()
        for c in clusters[:n_train_clusters]:
            train_indices.update(c)

        emo_train = [samples[i] for i in range(len(samples)) if i in train_indices]
        emo_eval = [samples[i] for i in range(len(samples)) if i not in train_indices]

        train_data.extend(emo_train)
        eval_data.extend(emo_eval)
        print(f"  {emotion}: {len(emo_train)} train + {len(emo_eval)} eval ({len(clusters)} clusters from {len(texts)} samples)")

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

    preview_mode = args.preview > 0

    n_gpus = torch.cuda.device_count()
    print(f"\nDetected {n_gpus} GPU(s)")
    print(f"Configuration:")
    print(f"  LLM model:            {args.llm_model}")
    print(f"  Emotions:             {args.emotions}")
    if preview_mode:
        print(f"  Mode:                 PREVIEW ({args.preview} samples/emotion × {len(args.emotions)} emotions)")
    else:
        print(f"  Samples per emotion:  {args.samples_per_emotion}")
        print(f"  Over-generate factor: {args.overgenerate_factor}x")
        print(f"  Dedup threshold:      {args.dedup_threshold}")
        print(f"  Train ratio:          {args.train_ratio}")
        print(f"  Shard:                {args.rank + 1}/{args.nshard}")
    print(f"  Batch size:           {args.batch_size}")
    print(f"  Stage 1 temperature:  {args.stage1_temperature}")
    print(f"  Stage 2 temperature:  {args.stage2_temperature}")

    # ── Build generation slots (over-generate to compensate for dedup) ─────
    if preview_mode:
        raw_per_emotion = args.preview
    else:
        raw_per_emotion = int(args.samples_per_emotion * args.overgenerate_factor)
    print(f"\nBuilding generation slots...")
    if not preview_mode:
        print(f"  Target unique/emotion: {args.samples_per_emotion}")
        print(f"  Raw slots/emotion:     {raw_per_emotion}")
    slots = build_slots(
        args.emotions,
        raw_per_emotion,
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

    # ── Create dedup filter ──────────────────────────────────────────────────
    is_dup = build_dedup_filter(threshold=args.dedup_threshold)

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
        is_duplicate=is_dup,
    )
    n_ok1 = sum(1 for x in instructions if x is not None)
    print(f"\nStage 1 complete: {n_ok1}/{len(slots)} succeeded (before dedup)")

    if not preview_mode:
        # ── Save Stage 1 checkpoint ─────────────────────────────────────────
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

        # ── Post-generation dedup + trim per emotion ─────────────────────────
        n_unique = dedup_and_trim_per_emotion(
            slots, instructions,
            target_per_emotion=args.samples_per_emotion,
            threshold=args.dedup_threshold,
        )
        target_total = args.samples_per_emotion * len(args.emotions)
        print(f"\nAfter dedup + trim: {n_unique} unique queries (target: {target_total})")
        if n_unique < target_total:
            print(
                f"WARNING: Only {n_unique} unique queries achieved. "
                f"Consider increasing --overgenerate-factor (currently {args.overgenerate_factor}).",
                file=sys.stderr,
            )

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

    # ── Preview or save ──────────────────────────────────────────────────────
    if preview_mode:
        print_preview(slots, instructions, responses)
        # Also save preview JSONL
        args.output_dir.mkdir(parents=True, exist_ok=True)
        preview_path = args.output_dir / "preview_pairs.jsonl"
        with preview_path.open("w", encoding="utf-8") as f:
            for slot, inst, resp in zip(slots, instructions, responses):
                if inst is None or resp is None:
                    continue
                f.write(
                    json.dumps(
                        {
                            "index": f"preview_{slot['emotion']}_{slot['local_idx']:05d}",
                            "query_text": inst,
                            "query_emotion": slot["emotion"],
                            "source_emotion": slot["emotion"],
                            "response_text": resp,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"\nPreview JSONL saved → {preview_path}")
        print("Review the output above, then run without --preview for the full dataset.")
        return

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
        dedup_threshold=args.dedup_threshold,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
