"""
Text-to-Valence/Arousal converter.

Uses the already-loaded GLM-4 LLM to extract (valence, arousal) values from a
free-text emotion description (e.g. "I'm feeling really down and exhausted").

Falls back to a keyword-weighted centroid over the 12 emotion anchors if the
LLM response cannot be parsed — requiring no additional imports.
"""

import re
import json
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Emotion anchor data (must stay in sync with EMOTION_ANCHORS in gradio_demo.py)
# ---------------------------------------------------------------------------

_ANCHORS = {
    "sad":        (-0.75, -0.65),
    "excited":    ( 0.75,  0.90),
    "frustrated": (-0.80,  0.35),
    "neutral":    ( 0.00,  0.00),
    "happy":      ( 0.85,  0.35),
    "angry":      (-0.85,  0.85),
    "anxious":    (-0.40,  0.65),
    "relaxed":    ( 0.25, -0.60),
    "surprised":  ( 0.10,  0.80),
    "disgusted":  (-0.82, -0.20),
    "tired":      (-0.15, -0.75),
    "content":    ( 0.60, -0.20),
}

# Synonym lists for the keyword fallback
_SYNONYMS = {
    "sad":        ["sad", "unhappy", "depressed", "down", "gloomy", "miserable",
                   "heartbroken", "sorrowful", "melancholy", "blue", "low"],
    "excited":    ["excited", "thrilled", "elated", "euphoric", "pumped", "energized",
                   "enthusiastic", "hyped", "stoked", "fired up"],
    "frustrated": ["frustrated", "annoyed", "irritated", "aggravated", "fed up",
                   "bothered", "exasperated", "impatient"],
    "neutral":    ["neutral", "okay", "fine", "alright", "indifferent", "normal",
                   "so-so", "meh", "whatever"],
    "happy":      ["happy", "joyful", "pleased", "glad", "cheerful",
                   "delighted", "good", "great", "wonderful", "fantastic"],
    "angry":      ["angry", "furious", "rage", "mad", "livid", "outraged",
                   "irate", "enraged", "fuming"],
    "anxious":    ["anxious", "afraid", "scared", "fearful", "nervous",
                   "worried", "terrified", "dread", "panicked", "uneasy"],
    "relaxed":    ["relaxed", "calm", "peaceful", "serene", "at ease",
                   "tranquil", "chill", "easy"],
    "surprised":  ["surprised", "shocked", "astonished", "amazed", "stunned",
                   "startled", "taken aback"],
    "disgusted":  ["disgusted", "revolted", "repulsed", "sick", "nauseated",
                   "appalled", "grossed out"],
    "tired":      ["tired", "exhausted", "sleepy", "fatigued", "drained",
                   "worn out", "weary", "lethargic", "sluggish"],
    "content":    ["content", "satisfied", "fulfilled", "at peace", "pleased",
                   "gratified", "comfortable"],
}

_INTENSITY_BOOST  = ["very", "super", "extremely", "really", "incredibly",
                     "deeply", "so", "absolutely", "totally"]
_INTENSITY_DAMPEN = ["slightly", "a bit", "a little", "kind of", "somewhat",
                     "mildly", "sort of", "rather", "fairly"]
_NEGATIONS        = ["not", "don't", "doesn't", "didn't", "never", "no longer",
                     "not really", "not very"]

# ---------------------------------------------------------------------------
# LLM prompt constants
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an emotion analysis assistant. Your task is to convert a natural-language \
emotion description into numerical valence and arousal scores.

Valence ranges from -1.0 (very negative/unpleasant) to +1.0 (very positive/pleasant).
Arousal ranges from -1.0 (very calm/low energy) to +1.0 (very energetic/high energy).

Reference emotion anchors:
  sad:        valence=-0.75, arousal=-0.65
  excited:    valence=+0.75, arousal=+0.90
  frustrated: valence=-0.80, arousal=+0.35
  neutral:    valence=+0.00, arousal=+0.00
  happy:      valence=+0.85, arousal=+0.35
  angry:      valence=-0.85, arousal=+0.85
  anxious:    valence=-0.40, arousal=+0.65
  relaxed:    valence=+0.25, arousal=-0.60
  surprised:  valence=+0.10, arousal=+0.80
  disgusted:  valence=-0.82, arousal=-0.20
  tired:      valence=-0.15, arousal=-0.75
  content:    valence=+0.60, arousal=-0.20

Output exactly one JSON object on a single line and nothing else:
{"valence": <float in [-1,1]>, "arousal": <float in [-1,1]>}"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TextToVAConverter:
    """
    Converts free-text emotion descriptions to (valence, arousal) tuples.

    Primary:  GLM-4 LLM prompt → structured JSON parse
    Fallback: Keyword-weighted centroid over 12 emotion anchors
    """

    def __init__(self, glm_model, glm_tokenizer):
        """
        Args:
            glm_model:      The already-loaded GLM-4 model (glm_model global).
            glm_tokenizer:  The corresponding tokenizer (glm_tokenizer global).
        """
        self.model = glm_model
        self.tokenizer = glm_tokenizer

    def convert(self, text: str) -> Tuple[float, float, str]:
        """
        Convert a free-text emotion description to (valence, arousal, info).

        Returns:
            (valence, arousal, info_message)
            - valence, arousal in [-1.0, 1.0]
            - info_message: human-readable description of how VA was derived
        """
        if not text or not text.strip():
            return 0.0, 0.0, "No text provided — defaulting to neutral."

        # Primary: LLM
        v, a, method = self._llm_extract(text.strip())
        if v is not None:
            return v, a, method

        # Fallback: keyword centroid
        return self._keyword_centroid(text.strip())

    # ------------------------------------------------------------------
    # Primary: LLM extraction
    # ------------------------------------------------------------------

    def _llm_extract(self, text: str) -> Tuple[Optional[float], Optional[float], str]:
        """Call GLM-4 with a text-only prompt and parse the JSON response."""
        import torch

        prompt = (
            f"<|system|>\n{_SYSTEM_PROMPT}\n"
            f"<|user|>\nEmotion description: \"{text}\"\n"
            f"<|assistant|>\n"
        )

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=40,   # JSON is ~35 tokens; cap to keep latency low
                    temperature=0.1,     # Near-deterministic structured output
                    top_p=0.9,
                    do_sample=True,
                )

            new_tokens = output[0][inputs["input_ids"].shape[1]:]
            raw_response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            v, a = _parse_va_json(raw_response)
            if v is not None:
                snippet = text[:60] + ("..." if len(text) > 60 else "")
                info = f'LLM extracted: V={v:+.2f}, A={a:+.2f}  (from: "{snippet}")'
                return v, a, info

            logger.warning("LLM response could not be parsed: %r", raw_response)
            return None, None, ""

        except Exception as exc:
            logger.warning("LLM extraction failed (%s), using fallback.", exc)
            return None, None, ""

    # ------------------------------------------------------------------
    # Fallback: keyword-weighted centroid
    # ------------------------------------------------------------------

    def _keyword_centroid(self, text: str) -> Tuple[float, float, str]:
        """
        Soft centroid over anchor emotions based on keyword overlap.

        Handles:
        - Multi-emotion phrases ("excited but nervous")
        - Negation ("not happy")
        - Intensity modifiers ("very", "a bit")
        """
        text_lower = text.lower()

        # Build negation windows: 30-character span after each negation word
        negated_spans = []
        for neg in _NEGATIONS:
            for m in re.finditer(re.escape(neg), text_lower):
                negated_spans.append((m.start(), m.start() + len(neg) + 30))

        # Detect intensity modifier
        boost  = any(w in text_lower for w in _INTENSITY_BOOST)
        dampen = any(w in text_lower for w in _INTENSITY_DAMPEN)
        intensity = 1.3 if boost else (0.7 if dampen else 1.0)

        # Score each emotion
        scores = {}
        for emotion, synonyms in _SYNONYMS.items():
            score = 0.0
            for syn in synonyms:
                for m in re.finditer(re.escape(syn), text_lower):
                    in_neg = any(ns <= m.start() < ne for ns, ne in negated_spans)
                    score += -0.5 if in_neg else 1.0
            scores[emotion] = max(0.0, score)

        total = sum(scores.values())

        if total < 1e-6:
            return (
                0.0, 0.0,
                "No recognisable emotion keywords found — defaulting to neutral.",
            )

        weights = {e: s / total for e, s in scores.items()}

        v_out, a_out = 0.0, 0.0
        for emotion, w in weights.items():
            ev, ea = _ANCHORS[emotion]
            v_out += w * ev
            a_out += w * ea

        v_out = float(np.clip(v_out * intensity, -1.0, 1.0))
        a_out = float(np.clip(a_out * intensity, -1.0, 1.0))

        top = sorted(weights.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{e} ({w:.0%})" for e, w in top if w > 0.05)
        info = f"Keyword match: {top_str}  →  V={v_out:+.2f}, A={a_out:+.2f}"
        return v_out, a_out, info


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def _parse_va_json(text: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract the first valid {"valence": ..., "arousal": ...} JSON from text.

    Three strategies in order:
      1. Full json.loads on the whole string
      2. Regex to find a JSON object containing both keys
      3. Regex to extract raw float values by key name
    """
    text = text.strip()

    # Strategy 1: direct parse
    try:
        obj = json.loads(text)
        return _extract_from_dict(obj)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: find JSON object via regex
    match = re.search(r'\{[^}]*"valence"[^}]*"arousal"[^}]*\}', text, re.DOTALL)
    if not match:
        match = re.search(r'\{[^}]*"arousal"[^}]*"valence"[^}]*\}', text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            return _extract_from_dict(obj)
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: extract float values by key
    v_match = re.search(r'"valence"\s*:\s*([+-]?\d*\.?\d+)', text)
    a_match = re.search(r'"arousal"\s*:\s*([+-]?\d*\.?\d+)', text)
    if v_match and a_match:
        v, a = float(v_match.group(1)), float(a_match.group(1))
        if -1.0 <= v <= 1.0 and -1.0 <= a <= 1.0:
            return v, a

    return None, None


def _extract_from_dict(obj: dict) -> Tuple[Optional[float], Optional[float]]:
    if "valence" in obj and "arousal" in obj:
        v, a = float(obj["valence"]), float(obj["arousal"])
        if -1.0 <= v <= 1.0 and -1.0 <= a <= 1.0:
            return v, a
    return None, None
