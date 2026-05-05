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

# ---------------------------------------------------------------------------
# HuggingFace model constants
# ---------------------------------------------------------------------------

# j-hartmann/emotion-english-distilroberta-base label → _ANCHORS key
_HF_LABEL_TO_ANCHOR = {
    "anger":    "angry",
    "disgust":  "disgusted",
    "fear":     "anxious",
    "joy":      "happy",
    "neutral":  "neutral",
    "sadness":  "sad",
    "surprise": "surprised",
}

# Default HF models for each method
_HF_CLS_MODEL = "j-hartmann/emotion-english-distilroberta-base"
_HF_ZS_MODEL  = "facebook/bart-large-mnli"

_INTENSITY_BOOST  = ["very", "super", "extremely", "really", "incredibly",
                     "deeply", "so", "absolutely", "totally"]
_INTENSITY_DAMPEN = ["slightly", "a bit", "a little", "kind of", "somewhat",
                     "mildly", "sort of", "rather", "fairly"]
_NEGATIONS        = ["not", "don't", "doesn't", "didn't", "never", "no longer",
                     "not really", "not very"]

# ---------------------------------------------------------------------------
# LLM prompt constants
# ---------------------------------------------------------------------------

# Emotion classification prompt — primary LLM method.
# Model may reply in English or Chinese; we scan the response for any known
# emotion keyword (see _CHINESE_TO_ANCHOR and _ANCHORS below).
_CLASSIFY_SYSTEM_PROMPT = """\
你是一个情绪分类助手。请阅读以下文本，选择最能描述其情绪的词语。

只能从以下词语中选择一个：
  幸福、悲伤、愤怒、兴奋、沮丧、平静、焦虑、放松、惊讶、厌恶、疲惫、满足

只输出这一个词，不要输出其他任何内容。"""

# Chinese emotion word → anchor name (for parsing the model's Chinese responses)
_CHINESE_TO_ANCHOR = {
    # happy
    "幸福": "happy", "高兴": "happy", "快乐": "happy", "开心": "happy",
    "愉快": "happy", "喜悦": "happy", "喜": "happy",
    # sad
    "悲伤": "sad", "难过": "sad", "伤心": "sad", "悲痛": "sad", "悲哀": "sad",
    # angry
    "生气": "angry", "愤怒": "angry", "恼怒": "angry", "气愤": "angry", "愤": "angry",
    # excited
    "兴奋": "excited", "激动": "excited", "期待": "excited",
    # frustrated
    "沮丧": "frustrated", "挫败": "frustrated", "烦躁": "frustrated", "郁闷": "frustrated",
    # neutral
    "平静": "neutral", "中立": "neutral", "平淡": "neutral",
    # anxious
    "焦虑": "anxious", "紧张": "anxious", "恐惧": "anxious", "害怕": "anxious",
    "担心": "anxious", "不安": "anxious",
    # relaxed
    "放松": "relaxed", "轻松": "relaxed",
    # surprised
    "惊讶": "surprised", "震惊": "surprised", "惊喜": "surprised",
    # disgusted
    "厌恶": "disgusted", "恶心": "disgusted",
    # tired
    "疲惫": "tired", "疲倦": "tired", "疲劳": "tired",
    # content
    "满足": "content", "知足": "content", "满意": "content",
}

# VA regression prompt — kept for reference / direct VA extraction use.
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

    Primary:  GLM-4 LLM emotion classification → VA anchor lookup
    Fallback: Keyword-weighted centroid over 12 emotion anchors
    """

    def __init__(self, glm_model, glm_tokenizer,
                 hf_model_name: str = _HF_CLS_MODEL,
                 hf_zs_model_name: str = _HF_ZS_MODEL):
        """
        Args:
            glm_model:        The already-loaded GLM-4 model (glm_model global).
            glm_tokenizer:    The corresponding tokenizer (glm_tokenizer global).
            hf_model_name:    HuggingFace text-classification model for method='hf'.
            hf_zs_model_name: HuggingFace zero-shot-classification model for method='hf_zs'.
        """
        self.model = glm_model
        self.tokenizer = glm_tokenizer
        self._hf_model_name    = hf_model_name
        self._hf_zs_model_name = hf_zs_model_name
        self._hf_pipeline    = None   # lazy-loaded on first _hf_classify() call
        self._hf_zs_pipeline = None   # lazy-loaded on first _hf_zs_classify() call

    def convert(self, text: str, method: str = "auto") -> Tuple[float, float, str]:
        """
        Convert a free-text emotion description to (valence, arousal, info).

        Args:
            text:   The emotion description to convert.
            method: Extraction method — one of:
                    "auto"    LLM if loaded, else keyword (default, existing behaviour)
                    "keyword" Always use keyword centroid
                    "llm"     Always use GLM-4 LLM (raises if model not loaded)
                    "hf"      HuggingFace 7-class text classifier (Ekman emotions)
                    "hf_zs"   HuggingFace zero-shot NLI over all 12 anchors

        Returns:
            (valence, arousal, info_message)
            - valence, arousal in [-1.0, 1.0]
            - info_message: human-readable description of how VA was derived
        """
        if not text or not text.strip():
            return 0.0, 0.0, "No text provided — defaulting to neutral."

        text = text.strip()

        if method == "hf":
            return self._hf_classify(text)

        if method == "hf_zs":
            return self._hf_zs_classify(text)

        if method == "keyword":
            return self._keyword_centroid(text)

        if method == "llm":
            if self.model is None:
                raise ValueError("method='llm' requested but no GLM model was provided.")
            v, a, info = self._llm_classify(text)
            if v is not None:
                return v, a, info
            return self._keyword_centroid(text)

        # method == "auto": existing behaviour
        if self.model is not None:
            v, a, info = self._llm_classify(text)
            if v is not None:
                return v, a, info
        return self._keyword_centroid(text)

    # ------------------------------------------------------------------
    # Primary: LLM emotion classification
    # ------------------------------------------------------------------

    def _llm_classify(self, text: str) -> Tuple[Optional[float], Optional[float], str]:
        """Classify text into one of the 12 anchor emotion names.

        Prompts the model in Chinese (its stronger language).  Scans the response
        for the first matching Chinese or English emotion keyword and looks up VA
        from _ANCHORS / _CHINESE_TO_ANCHOR.

        Returns (valence, arousal, info) or (None, None, "") on failure.
        """
        import torch

        prompt = (
            f"<|system|>\n{_CLASSIFY_SYSTEM_PROMPT}\n"
            f"<|user|>\n文本：「{text}」\n情绪词：\n"
            f"<|assistant|>\n"
        )

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.model.device)
            attn_mask = inputs["attention_mask"].to(self.model.device)
            generated = input_ids.clone()
            eos_id = self.tokenizer.eos_token_id
            audio_0_id = self.tokenizer.convert_tokens_to_ids("<|audio_0|>")

            past_kv = None
            is_first = True
            with torch.no_grad():
                for _ in range(40):
                    model_inputs = self.model.prepare_inputs_for_generation(
                        generated,
                        past_key_values=past_kv,
                        attention_mask=attn_mask,
                        is_first_forward=is_first,
                        use_cache=True,
                    )
                    out = self.model(**model_inputs, return_dict=True)
                    past_kv = out.past_key_values
                    is_first = False

                    logits = out.logits[:, -1, :]
                    if audio_0_id > 0:
                        logits[:, audio_0_id:] = float("-inf")
                    next_tok = logits.argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_tok], dim=1)
                    attn_mask = torch.cat(
                        [attn_mask, attn_mask.new_ones((1, 1))], dim=1
                    )
                    if next_tok.item() == eos_id:
                        break

            new_tokens = generated[0, input_ids.shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            emotion = _match_emotion_in_response(response)
            if emotion is None:
                logger.warning("LLM classification: no emotion found in response: %r", response)
                return None, None, ""

            v, a = _ANCHORS[emotion]
            snippet = text[:60] + ("..." if len(text) > 60 else "")
            info = (
                f'LLM classified: "{emotion}"  →  V={v:+.2f}, A={a:+.2f}'
                f'  (from: "{snippet}")'
            )
            return v, a, info

        except Exception as exc:
            logger.warning("LLM classification failed (%s), using fallback.", exc)
            return None, None, ""

    # ------------------------------------------------------------------
    # HuggingFace text-classification (7 Ekman emotions)
    # ------------------------------------------------------------------

    def _hf_classify(self, text: str) -> Tuple[float, float, str]:
        """Classify text using a HuggingFace text-classification pipeline.

        Loads the pipeline lazily on first call.  Uses probability-weighted
        average of VA anchors — same pattern as FaceVASpeechMapper.

        Note: the default model covers only 7 Ekman emotions mapped to 7 of
        the 12 speech anchors.  Five anchors (excited, frustrated, relaxed,
        tired, content) are unreachable by this method.

        Returns (valence, arousal, info_string).
        """
        if self._hf_pipeline is None:
            import torch
            from transformers import pipeline as hf_pipeline
            device = 0 if torch.cuda.is_available() else -1
            logger.info("Loading HF text-classification pipeline: %s (device=%d)", self._hf_model_name, device)
            self._hf_pipeline = hf_pipeline(
                "text-classification",
                model=self._hf_model_name,
                top_k=None,
                device=device,
            )

        raw = self._hf_pipeline(text)
        scores = raw[0]  # list of {label: str, score: float}

        v_out, a_out = 0.0, 0.0
        mapped = []
        for item in scores:
            label  = item["label"].lower()
            prob   = float(item["score"])
            anchor = _HF_LABEL_TO_ANCHOR.get(label)
            if anchor and anchor in _ANCHORS:
                ev, ea = _ANCHORS[anchor]
                v_out += prob * ev
                a_out += prob * ea
                mapped.append((anchor, prob))

        v_out = float(np.clip(v_out, -1.0, 1.0))
        a_out = float(np.clip(a_out, -1.0, 1.0))

        top = sorted(mapped, key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{e} ({p:.0%})" for e, p in top if p > 0.02)
        snippet = text[:60] + ("..." if len(text) > 60 else "")
        info = (
            f'HF-cls [{self._hf_model_name.split("/")[-1]}]: {top_str}'
            f'  →  V={v_out:+.2f}, A={a_out:+.2f}  (from: "{snippet}")'
        )
        return v_out, a_out, info

    # ------------------------------------------------------------------
    # HuggingFace zero-shot NLI (all 12 anchors)
    # ------------------------------------------------------------------

    def _hf_zs_classify(self, text: str) -> Tuple[float, float, str]:
        """Classify text using zero-shot NLI over all 12 speech anchor names.

        Loads the pipeline lazily on first call.  Uses probability-weighted
        average of VA anchors.  All 12 anchors are reachable since the label
        set is passed in as candidate_labels at inference time.

        Returns (valence, arousal, info_string).
        """
        if self._hf_zs_pipeline is None:
            import torch
            from transformers import pipeline as hf_pipeline
            device = 0 if torch.cuda.is_available() else -1
            logger.info("Loading HF zero-shot-classification pipeline: %s (device=%d)", self._hf_zs_model_name, device)
            self._hf_zs_pipeline = hf_pipeline(
                "zero-shot-classification",
                model=self._hf_zs_model_name,
                device=device,
            )

        candidate_labels = list(_ANCHORS.keys())
        result = self._hf_zs_pipeline(text, candidate_labels=candidate_labels, multi_label=False)

        # result["labels"] and result["scores"] are sorted descending by score
        label_to_score = dict(zip(result["labels"], result["scores"]))

        v_out, a_out = 0.0, 0.0
        for anchor, prob in label_to_score.items():
            if anchor in _ANCHORS:
                ev, ea = _ANCHORS[anchor]
                v_out += prob * ev
                a_out += prob * ea

        v_out = float(np.clip(v_out, -1.0, 1.0))
        a_out = float(np.clip(a_out, -1.0, 1.0))

        top = sorted(label_to_score.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{e} ({p:.0%})" for e, p in top if p > 0.02)
        snippet = text[:60] + ("..." if len(text) > 60 else "")
        info = (
            f'HF-zs [{self._hf_zs_model_name.split("/")[-1]}]: {top_str}'
            f'  →  V={v_out:+.2f}, A={a_out:+.2f}  (from: "{snippet}")'
        )
        return v_out, a_out, info

    # ------------------------------------------------------------------
    # LLM VA regression (kept for direct use / comparison)
    # ------------------------------------------------------------------

    def _llm_extract(self, text: str) -> Tuple[Optional[float], Optional[float], str]:
        """Call GLM-4 with a text-only prompt and parse the JSON response.

        Uses a manual token-by-token loop via prepare_inputs_for_generation instead
        of model.generate(), which is incompatible with GLM-4-Voice's KV-cache
        format under transformers ≥4.50 (_extract_past_from_model_output removed).
        """
        import torch

        prompt = (
            f"<|system|>\n{_SYSTEM_PROMPT}\n"
            f"<|user|>\nEmotion description: \"{text}\"\n"
            f"<|assistant|>\n"
        )

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.model.device)
            attn_mask = inputs["attention_mask"].to(self.model.device)
            generated = input_ids.clone()
            eos_id = self.tokenizer.eos_token_id
            # Audio tokens start at this ID; we want text-only output
            audio_0_id = self.tokenizer.convert_tokens_to_ids("<|audio_0|>")

            past_kv = None
            is_first = True
            with torch.no_grad():
                for _ in range(60):  # JSON response is ≤ 35 tokens
                    model_inputs = self.model.prepare_inputs_for_generation(
                        generated,
                        past_key_values=past_kv,
                        attention_mask=attn_mask,
                        is_first_forward=is_first,
                        use_cache=True,
                    )
                    out = self.model(**model_inputs, return_dict=True)
                    past_kv = out.past_key_values
                    is_first = False

                    # Greedy — mask audio tokens so we stay in text space
                    logits = out.logits[:, -1, :]
                    if audio_0_id > 0:
                        logits[:, audio_0_id:] = float("-inf")
                    next_tok = logits.argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_tok], dim=1)
                    attn_mask = torch.cat(
                        [attn_mask, attn_mask.new_ones((1, 1))], dim=1
                    )
                    if next_tok.item() == eos_id:
                        break

            new_tokens = generated[0, input_ids.shape[1]:]
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
# Emotion keyword matcher (Chinese + English)
# ---------------------------------------------------------------------------

def _match_emotion_in_response(response: str) -> Optional[str]:
    """Scan a model response for the first Chinese or English emotion keyword.

    Searches Chinese keywords first (longest match wins to avoid partial
    matches), then English anchor names.  Returns the anchor name string
    (e.g. "happy") or None if nothing is found.
    """
    # Chinese: try longer phrases before shorter ones to avoid, e.g.,
    # matching "焦" inside "焦虑" with a single-char entry.
    for zh, anchor in sorted(_CHINESE_TO_ANCHOR.items(), key=lambda x: -len(x[0])):
        if zh in response:
            return anchor

    # English: word-boundary match, longest first
    response_lower = response.lower()
    for emotion in sorted(_ANCHORS, key=len, reverse=True):
        if re.search(r'\b' + emotion + r'\b', response_lower):
            return emotion

    return None


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def _parse_va_json(text: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract the first valid (valence, arousal) pair from text.

    Four strategies in order:
      1. Full json.loads on the whole string
      2. Regex to find a JSON object containing both keys
      3. Regex to extract raw float values by JSON key name
      4. Plain-text "Valence: X, Arousal: Y" format (fine-tuned model output)
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

    # Strategy 3: extract float values by JSON key name
    v_match = re.search(r'"valence"\s*:\s*([+-]?\d*\.?\d+)', text)
    a_match = re.search(r'"arousal"\s*:\s*([+-]?\d*\.?\d+)', text)
    if v_match and a_match:
        v, a = float(v_match.group(1)), float(a_match.group(1))
        if -1.0 <= v <= 1.0 and -1.0 <= a <= 1.0:
            return v, a

    # Strategy 4: plain-text "Valence: X.XX, Arousal: Y.YY" or
    # "Valence: X.XX\nArousal: Y.YY" — produced by the fine-tuned model
    v_match = re.search(r'[Vv]alence\s*:\s*([+-]?\d*\.?\d+)', text)
    a_match = re.search(r'[Aa]rousal\s*:\s*([+-]?\d*\.?\d+)', text)
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
