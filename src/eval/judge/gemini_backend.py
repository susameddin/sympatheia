#!/usr/bin/env python3
"""
Gemini judge backend (audio input, text output) using the google-genai SDK.

Exposes a ``GeminiJudge`` whose ``.judge(system_prompt, user_text, audio_path)``
matches the ``judge_fn`` contract expected by judge_common.run_judge_over_manifest.

Requires ``google-genai`` (see requirements.txt) and an API key in
``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``):

    GEMINI_API_KEY=... python -m eval.judge.judge_gemini_emotional ...

Get an API key at https://aistudio.google.com/apikey (free tier is rate-limited;
enable Cloud billing on the project for full-speed paid tier).
"""

import mimetypes
import os
import sys
import time
from pathlib import Path

from eval.judge.judge_common import FatalJudgeError

# Repo root, used to locate an optional .env alongside the checked-out tree.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_dotenv_if_needed():
    """Populate GEMINI_API_KEY / GOOGLE_API_KEY from a .env file if not already
    in the environment. No third-party dependency; only sets keys that are unset;
    never prints values. Search order (first file found wins):
        $GEMINI_ENV_FILE, ./.env, <repo root>/.env
    """
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return
    candidates = [
        os.environ.get("GEMINI_ENV_FILE"),
        Path.cwd() / ".env",
        _REPO_ROOT / ".env",
    ]
    for cand in candidates:
        if not cand:
            continue
        p = Path(cand)
        if not p.is_file():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key in ("GEMINI_API_KEY", "GOOGLE_API_KEY") and key not in os.environ:
                os.environ[key] = val
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            print(f"Loaded API key from {p}")
            return


# Indicative Gemini pricing in $ per 1M tokens (paid tier). Used only for the
# end-of-run cost printout, never for control flow, and not kept in sync with
# the vendor's price list — check ai.google.dev/gemini-api/docs/pricing for
# current rates. audio_in is the standalone audio-input rate; on 3.x Flash audio
# is billed at the unified text-input rate, so it matches text_in there.
PRICING = {
    "gemini-3.5-flash":       {"text_in": 1.50, "audio_in": 1.50, "text_out": 9.00},
    "gemini-3.6-flash":       {"text_in": 1.50, "audio_in": 1.50, "text_out": 7.50},
    "gemini-3.5-flash-lite":  {"text_in": 0.30, "audio_in": 0.30, "text_out": 2.50},
    "gemini-3.1-flash-lite":  {"text_in": 0.25, "audio_in": 0.50, "text_out": 1.50},
    "gemini-2.5-flash":       {"text_in": 0.30, "audio_in": 1.00, "text_out": 2.50},
    "gemini-2.5-flash-lite":  {"text_in": 0.10, "audio_in": 0.30, "text_out": 0.40},
}


def get_pricing(model_name: str):
    """Best-effort pricing lookup by prefix; None if unknown."""
    for key, val in PRICING.items():
        if model_name.startswith(key):
            return val
    return None


class GeminiJudge:
    def __init__(
        self,
        model_name: str = "gemini-3.5-flash",
        temperature: float = 0.0,
        max_output_tokens: int = 256,
        disable_thinking: bool = True,
        max_retries: int = 4,
    ):
        try:
            from google import genai            # noqa: F401
            from google.genai import types      # noqa: F401
        except ImportError:
            print(
                "ERROR: google-genai not importable. Install it with:\n"
                "  pip install google-genai",
                file=sys.stderr,
            )
            raise

        _load_dotenv_if_needed()
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("ERROR: set GEMINI_API_KEY (or GOOGLE_API_KEY), or put it in a .env "
                  "file (see gemini_backend._load_dotenv_if_needed for search paths).",
                  file=sys.stderr)
            sys.exit(1)

        self._genai = genai
        self._types = types
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.disable_thinking = disable_thinking
        self.max_retries = max_retries

    # -- config builders ----------------------------------------------------

    def _make_config(self, system_prompt: str, thinking_off: bool):
        types = self._types
        kwargs = dict(
            system_instruction=system_prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        if thinking_off:
            # thinking_budget=0 disables thinking on 2.5-family models; harmless
            # to attempt on 3.x. If a model rejects it we retry without.
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        return types.GenerateContentConfig(**kwargs)

    @staticmethod
    def _mime_for(audio_path: str) -> str:
        guess, _ = mimetypes.guess_type(audio_path)
        if guess and guess.startswith("audio/"):
            return guess
        ext = Path(audio_path).suffix.lower().lstrip(".")
        return {"wav": "audio/wav", "mp3": "audio/mp3", "flac": "audio/flac",
                "ogg": "audio/ogg", "m4a": "audio/mp4"}.get(ext, "audio/wav")

    @staticmethod
    def _usage_to_dict(usage) -> dict:
        if usage is None:
            return {}
        out = {
            "prompt_token_count": getattr(usage, "prompt_token_count", None) or 0,
            "candidates_token_count": getattr(usage, "candidates_token_count", None) or 0,
            "total_token_count": getattr(usage, "total_token_count", None) or 0,
            "thoughts_token_count": getattr(usage, "thoughts_token_count", None) or 0,
        }
        # Per-modality prompt breakdown (AUDIO vs TEXT) when available.
        details = getattr(usage, "prompt_tokens_details", None) or []
        for d in details:
            modality = getattr(d, "modality", None)
            count = getattr(d, "token_count", None) or 0
            if modality is not None:
                out[f"prompt_{str(modality).split('.')[-1].lower()}_tokens"] = count
        return out

    # -- main entry point ---------------------------------------------------

    def judge(self, system_prompt: str, user_text: str, audio_path: str):
        """Return (raw_text, usage_dict). Retries transient errors and one
        empty/parse-failed response with a larger token budget."""
        types = self._types
        audio_bytes = Path(audio_path).read_bytes()
        mime = self._mime_for(audio_path)
        contents = [
            types.Part.from_text(text=user_text),
            types.Part.from_bytes(data=audio_bytes, mime_type=mime),
        ]

        thinking_off = self.disable_thinking
        last_err = None
        for attempt in range(self.max_retries):
            try:
                config = self._make_config(system_prompt, thinking_off)
                resp = self.client.models.generate_content(
                    model=self.model_name, contents=contents, config=config,
                )
                text = (resp.text or "").strip()
                usage = self._usage_to_dict(getattr(resp, "usage_metadata", None))
                if text:
                    return text, usage
                # Empty text: often thinking ate the budget. Retry once with a
                # larger budget (and thinking left as configured).
                self.max_output_tokens = max(self.max_output_tokens, 1024)
                last_err = "empty response text"
            except Exception as e:
                msg = str(e)
                last_err = msg
                # Fatal billing/permission states: retrying can't help. Raise so
                # the run stops immediately instead of silently writing empties.
                if ("credits are depleted" in msg or "billing" in msg.lower()
                        or "PERMISSION_DENIED" in msg or "API key not valid" in msg
                        or "403" in msg):
                    raise FatalJudgeError(
                        f"Fatal Gemini API error (not retryable): {msg}"
                    ) from e
                # If thinking config is the problem, drop it and retry.
                if thinking_off and ("thinking" in msg.lower() or "thinking_config" in msg.lower()):
                    thinking_off = False
                    continue
                # Backoff on transient rate-limit / server errors.
                if any(code in msg for code in ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE")):
                    time.sleep(min(2 ** attempt, 30))
                    continue
                # Unknown error: brief backoff then retry.
                time.sleep(min(2 ** attempt, 10))
        return "", {"error": str(last_err)}
