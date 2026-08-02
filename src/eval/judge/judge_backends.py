#!/usr/bin/env python3
"""
Pluggable judge backends and a resume-safe judgment writer.

judge_common.run_judge_over_manifest drives the manifest-shaped judges, where
each record maps to one audio file and one rubric. Rubrics that ask several
independent questions about the same clip (see judge_interpolation.py) need to
drive the model themselves, so the backend is factored out here.

Both backends expose the same call:

    judge(system_prompt, user_text, audio_paths) -> (raw_text, usage_dict)

``audio_paths`` may be a single path or a list. Rating parsing on an arbitrary
integer scale and a resume-safe JSONL writer live here too, since every caller
of these backends needs them.
"""

import json
import re
import sys
import time
from pathlib import Path

from eval.judge.judge_common import FatalJudgeError

DEFAULT_QWEN_MODEL = "Qwen/Qwen3-Omni"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_rating_scale(raw: str, lo: int = 1, hi: int = 7):
    """Extract (rating, justification) on an arbitrary integer scale.

    ``judge_common.parse_rating`` hardcodes 1-5 in both of its regexes, so a "7"
    would fall through to the fallback and a "7|..." reply would be read as
    unparseable. This is the generalisation used by the wider SAM-style scales;
    judge_common's version is left alone so the published 1-5 numbers keep going
    through exactly the code that produced them.
    """
    digits = "".join(str(d) for d in range(lo, hi + 1)) if hi < 10 else None
    cls = f"[{digits}]" if digits else rf"(?:{'|'.join(str(d) for d in range(lo, hi + 1))})"
    m = re.search(rf"({cls})\s*\|(.+)", raw, re.DOTALL)
    if m:
        return int(m.group(1)), m.group(2).strip()
    m = re.search(rf"\b({cls})\b", raw)
    if m:
        return int(m.group(1)), raw.strip()
    return None, raw.strip()


# ---------------------------------------------------------------------------
# Qwen3-Omni (local)
# ---------------------------------------------------------------------------

class QwenOmniJudge:
    """Local Qwen3-Omni judge.

    Requires the Qwen3-Omni package (https://github.com/QwenLM/Qwen3) and enough
    GPU memory to hold the model; ``device_map="auto"`` will pipeline it across
    however many devices are visible. With too little memory the load can fall
    back to meta tensors and still exit successfully, having produced
    ``rating=None`` for every sample — check the summary's parse-failure count
    before trusting a run.
    """

    tag = "qwen"

    def __init__(self, model_path: str = None, speaker: str = "Chelsie"):
        import torch
        from transformers import (Qwen3OmniMoeForConditionalGeneration,
                                  Qwen3OmniMoeProcessor)
        self._torch = torch
        path = model_path or DEFAULT_QWEN_MODEL
        print(f"Loading Qwen3-Omni judge: {path}")
        t0 = time.time()
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            path, dtype="auto", device_map="auto")
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(path)
        self.speaker = speaker
        print(f"  loaded in {time.time() - t0:.1f}s (device={self.model.device})")

    def judge(self, system_prompt: str, user_text: str, audio_paths):
        from qwen_omni_utils import process_mm_info
        if isinstance(audio_paths, (str, Path)):
            audio_paths = [audio_paths]

        content = [{"type": "text", "text": user_text}]
        for p in audio_paths:
            content.append({"type": "audio", "audio": str(p)})
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]

        use_aiv = True
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_aiv)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos,
                                return_tensors="pt", padding=True,
                                use_audio_in_video=use_aiv)
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        with self._torch.no_grad():
            text_ids, _ = self.model.generate(
                **inputs, speaker=self.speaker,
                thinker_return_dict_in_generate=True, use_audio_in_video=use_aiv)

        decoded = self.processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return (decoded[0] if decoded else ""), {}


# ---------------------------------------------------------------------------
# Gemini (hosted API)
# ---------------------------------------------------------------------------

class GeminiMultiJudge:
    """Multi-audio wrapper around eval.judge.gemini_backend.GeminiJudge.

    Single-clip calls delegate straight to the inner judge, so they share its
    retry and fatal-error policy exactly. Multi-clip calls reuse the same client
    and config with several audio parts.
    """

    tag = "gemini"

    def __init__(self, model_name: str = None):
        from eval.judge.gemini_backend import GeminiJudge
        self._inner = GeminiJudge(model_name=model_name or DEFAULT_GEMINI_MODEL)

    def judge(self, system_prompt: str, user_text: str, audio_paths):
        if isinstance(audio_paths, (str, Path)):
            return self._inner.judge(system_prompt, user_text, str(audio_paths))

        inner, types = self._inner, self._inner._types
        contents = [types.Part.from_text(text=user_text)]
        for p in audio_paths:
            contents.append(types.Part.from_bytes(
                data=Path(p).read_bytes(), mime_type=inner._mime_for(str(p))))

        last_err = None
        for attempt in range(inner.max_retries):
            try:
                cfg = inner._make_config(system_prompt, inner.disable_thinking)
                resp = inner.client.models.generate_content(
                    model=inner.model_name, contents=contents, config=cfg)
                txt = (resp.text or "").strip()
                usage = inner._usage_to_dict(getattr(resp, "usage_metadata", None))
                if txt:
                    return txt, usage
                inner.max_output_tokens = max(inner.max_output_tokens, 1024)
                last_err = "empty response text"
            except Exception as e:
                msg = str(e)
                last_err = msg
                if any(s in msg for s in ("credits are depleted", "PERMISSION_DENIED",
                                          "API key not valid", "403")):
                    raise FatalJudgeError(f"Fatal Gemini API error: {msg}") from e
                time.sleep(min(2 ** attempt, 30))
        return "", {"error": str(last_err)}


def make_judge(backend: str, model_path: str = None):
    if backend == "qwen":
        return QwenOmniJudge(model_path)
    if backend == "gemini":
        return GeminiMultiJudge(model_path)
    raise ValueError(f"unknown backend: {backend!r}")


# ---------------------------------------------------------------------------
# Resume-safe JSONL writer (mirrors judge_common's no-clobber contract)
# ---------------------------------------------------------------------------

class JudgmentWriter:
    def __init__(self, path: Path, key_fields, skip_existing: bool,
                 sibling_glob: str = None):
        """Append-only JSONL writer that refuses to clobber and can resume.

        ``sibling_glob`` makes resume survive a change in shard count. A run
        sharded 4 ways writes ``...shard00of04.jsonl``; re-running it 2 ways
        looks for ``...shard00of02.jsonl``, finds nothing, and redoes work
        already on disk under the other name. Passing a glob that matches every
        shard of the same run lets the writer skip anything already judged,
        whatever shape the previous run had.
        """
        self.path = path
        self.key_fields = key_fields
        self.existing = set()
        self.records = []
        path.parent.mkdir(parents=True, exist_ok=True)

        if sibling_glob:
            n = 0
            for other in sorted(path.parent.glob(sibling_glob)):
                if other == path:
                    continue
                with open(other) as f:
                    for line in f:
                        r = json.loads(line)
                        if r.get("rating") is not None:
                            self.existing.add(self._key(r))
                            n += 1
            if n:
                print(f"Resume: {n} judgment(s) already done in sibling shards")

        if path.exists() and not skip_existing:
            print(f"ERROR: output already exists: {path}\n"
                  f"       Refusing to overwrite. Pass --skip-existing to resume.",
                  file=sys.stderr)
            sys.exit(1)
        if path.exists():
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    self.records.append(r)
                    self.existing.add(self._key(r))
            print(f"Resuming: {len(self.existing)} judgments already done")
        self._fh = open(path, "a")

    def _key(self, rec):
        return tuple(rec.get(k) for k in self.key_fields)

    def done(self, rec) -> bool:
        return self._key(rec) in self.existing

    def write(self, rec):
        self.records.append(rec)
        self.existing.add(self._key(rec))
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()
