"""Shared ASR helpers for affect_eval and differentiation_eval.

Uses openai-whisper (PyTorch-native, no cuDNN dependency) rather than
faster-whisper so it works wherever torch.cuda.is_available() is True.
"""

import json
from pathlib import Path

try:
    import whisper as openai_whisper
    _WHISPER_OK = True
except ImportError:
    _WHISPER_OK = False

DEFAULT_WHISPER_MODEL = "medium"


def load_whisper(model_name: str = DEFAULT_WHISPER_MODEL):
    """Load openai-whisper model. Returns None if unavailable."""
    if not _WHISPER_OK:
        print("WARNING: openai-whisper not installed — pip install openai-whisper")
        return None
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading openai-whisper ({model_name}) on {device}...")
    try:
        model = openai_whisper.load_model(model_name, device=device)
        print("  openai-whisper loaded.")
        return model
    except Exception as e:
        print(f"  openai-whisper load failed: {e}")
        return None


def transcribe(audio_path: str, model) -> str | None:
    """Transcribe audio file. Returns stripped transcript or None on failure."""
    try:
        result = model.transcribe(audio_path, fp16=False)
        return result["text"].strip()
    except Exception as e:
        print(f"  Transcription failed for {audio_path}: {e}")
        return None


def load_asr_cache(cache_path: Path) -> dict[str, str]:
    """Load ASR cache from JSONL. Returns {audio_path: transcript}."""
    cache: dict[str, str] = {}
    if not cache_path.exists():
        return cache
    with open(cache_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
                cache[entry["audio_path"]] = entry["transcript"]
            except Exception:
                pass
    return cache


def append_asr_cache(cache_path: Path, audio_path: str, transcript: str) -> None:
    """Append one transcription to the cache JSONL file."""
    with open(cache_path, "a") as f:
        f.write(json.dumps({"audio_path": audio_path, "transcript": transcript}) + "\n")


def transcribe_manifest(
    records: list[dict],
    conditions: list[str],
    cache_path: Path,
    whisper_model,
    skip_existing: bool = True,
) -> dict[str, dict[str, str]]:
    """Transcribe all (record × condition) audio files, using/updating cache.

    Returns nested dict: {condition: {audio_path: transcript}}
    """
    cache = load_asr_cache(cache_path)
    if skip_existing and cache:
        print(f"  Loaded {len(cache)} cached transcriptions from {cache_path.name}")

    result: dict[str, dict[str, str]] = {c: {} for c in conditions}

    for cond in conditions:
        pending = []
        for rec in records:
            audio_path = rec.get(f"{cond}_response")
            if not audio_path or not Path(audio_path).exists():
                continue
            if skip_existing and audio_path in cache:
                result[cond][audio_path] = cache[audio_path]
            else:
                pending.append(audio_path)

        n_cached = len(result[cond])
        print(f"  [{cond}] {n_cached} cached, {len(pending)} to transcribe")

        if pending and whisper_model is not None:
            for i, audio_path in enumerate(pending):
                transcript = transcribe(audio_path, whisper_model)
                if transcript is None:
                    transcript = ""
                append_asr_cache(cache_path, audio_path, transcript)
                cache[audio_path] = transcript
                result[cond][audio_path] = transcript
                if (i + 1) % 50 == 0:
                    print(f"    transcribed {i+1}/{len(pending)}")
        elif pending:
            print(f"    WARNING: {len(pending)} files need transcription but Whisper not loaded")

    return result
