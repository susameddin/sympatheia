#!/usr/bin/env python3
"""
Evaluation script for GLM-4-Voice emotion-conditioned speech model.

Runs stratified inference over 11 emotions, generates speech responses,
and computes four categories of metrics:
  1. Emotion control  — VA MAE/RMSE, nearest-anchor accuracy, quadrant accuracy
  2. Intelligibility  — WER via Whisper ASR
  3. Naturalness      — UTMOS22 (via torch.hub/SpeechMOS, no extra install needed)
  4. Coherence        — BERTScore F1 and ROUGE-L vs. reference text

Usage (base model):
    conda run -n glm4voice3_eval python evaluate_model.py \\
        --model-path THUDM/glm-4-voice-9b \\
        --is-base \\
        --output-dir results/eval_base \\
        --num-samples 5

Usage (fine-tuned checkpoint):
    conda run -n glm4voice3_eval python evaluate_model.py \\
        --model-path /path/to/checkpoint-700 \\
        --output-dir results/eval_ckpt700 \\
        --num-samples 30

Required (install once in glm4voice3_eval):
    pip install jiwer bert-score rouge-score
UTMOS loads automatically via torch.hub (tarepan/SpeechMOS) — no extra install needed.
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoTokenizer, AutoModel, pipeline as hf_pipeline

# LoRA model loader
from peft import AutoPeftModelForCausalLM

# Project utilities — add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.vocoder import GLM4CodecDecoder
from constants import ALL_EMOTIONS

# ---------------------------------------------------------------------------
# Optional metric packages — graceful degradation if not installed
# ---------------------------------------------------------------------------

try:
    import jiwer
    _JIWER_OK = True
except ImportError:
    _JIWER_OK = False
    print("WARNING: jiwer not installed — WER will be skipped. pip install jiwer")

try:
    from bert_score import score as bert_score_fn
    _BERTSCORE_OK = True
except ImportError:
    _BERTSCORE_OK = False
    print("WARNING: bert_score not installed — BERTScore will be skipped. pip install bert-score")

try:
    from rouge_score import rouge_scorer as rouge_scorer_lib
    _ROUGE_OK = True
except ImportError:
    _ROUGE_OK = False
    print("WARNING: rouge_score not installed — ROUGE-L will be skipped. pip install rouge-score")

# UTMOS loads via torch.hub at model-load time — no separate package needed
_utmos_predictor = None  # set in main() after torch.hub.load

DECODER_SAMPLE_RATE = 22050
DEFAULT_EVAL_JSONL = (
    "/engram/naplab/users/sd3705/Datasets/OpenS2S_11Emo/glm4voice_va_format/eval.jsonl"
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate GLM-4-Voice emotion-conditioned speech model"
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="HuggingFace model ID or local checkpoint path"
    )
    parser.add_argument(
        "--is-base", action="store_true",
        help="Load as base model (AutoModel, no LoRA). "
             "If not set, loads with AutoPeftModelForCausalLM."
    )
    parser.add_argument(
        "--eval-jsonl", type=str, default=DEFAULT_EVAL_JSONL,
        help=f"Path to eval.jsonl (default: {DEFAULT_EVAL_JSONL})"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory for audio output, predictions.jsonl, metrics.json, and plots"
    )
    parser.add_argument(
        "--num-samples", type=int, default=30,
        help="Samples per emotion class (default: 30 → 330 total across 11 emotions)"
    )
    parser.add_argument(
        "--whisper-model", type=str, default="openai/whisper-large-v3",
        help="Whisper model for ASR/WER (default: openai/whisper-large-v3)"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip samples whose audio file already exists (enables resuming)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading and parsing
# ---------------------------------------------------------------------------

def load_eval_samples(eval_jsonl: str, num_samples: int) -> list:
    """Load eval.jsonl and return a stratified list of num_samples per emotion.

    Emotion is extracted from the sample ID: "opens2s_sad_19069" -> "Sad"
    """
    by_emotion = defaultdict(list)
    with open(eval_jsonl) as f:
        for line in f:
            sample = json.loads(line)
            # e.g. "opens2s_sad_19069" -> "sad" -> "Sad"
            emotion = sample["id"].split("_")[1].capitalize()
            sample["emotion"] = emotion
            by_emotion[emotion].append(sample)

    result = []
    print(f"\nStratified eval set ({num_samples} per emotion):")
    for emotion in sorted(by_emotion.keys()):
        selected = by_emotion[emotion][:num_samples]
        result.extend(selected)
        print(f"  {emotion:<12}: {len(selected)} samples")

    missing = [e for e in ALL_EMOTIONS if e not in by_emotion]
    if missing:
        print(f"  WARNING: no samples found for: {missing}")

    print(f"Total: {len(result)} samples\n")
    return result


def extract_user_audio_tokens(text: str) -> str:
    """Extract the user's audio token string from the full conversation text.

    The text field format is:
      <|system|>\\n...\\n<|user|>\\n<|audio_X|>...<|audio_Y|>\\n<|assistant|>\\n...
    """
    match = re.search(r'<\|user\|>\n(.*?)\n<\|assistant\|>', text, re.DOTALL)
    if not match:
        raise ValueError("Could not find user audio tokens in text field")
    return match.group(1)


def extract_reference_text(text: str) -> str:
    """Extract the ground-truth text response from the conversation text.

    The assistant turn has: text response, then a newline, then audio tokens.
    """
    match = re.search(r'<\|assistant\|>\n(.*?)\n<\|audio_', text, re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_language_model(model_path: str, is_base: bool, device: str):
    """Load GLM-4-Voice language model (base or LoRA fine-tuned)."""
    if is_base:
        print(f"Loading base model from {model_path} ...")
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map=device,
        )
    else:
        print(f"Loading fine-tuned LoRA model from {model_path} ...")
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=True,
        )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(audio_tokens_str: str, valence: float, arousal: float, is_base: bool) -> str:
    """Construct the model input string."""
    if is_base:
        system_prompt = "Please respond in English."
    else:
        system_prompt = (
            f"Please respond in English. "
            f"User emotion (valence={valence:.2f}, arousal={arousal:.2f})"
        )
    return f"<|system|>\n{system_prompt}\n<|user|>\n{audio_tokens_str}\n<|assistant|>\n"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_generation(model, tokenizer, prompt: str, audio_0_id: int):
    """Run model.generate() and split output into audio tokens and text.

    Returns:
        audio_token_ids: list of token tensors with id >= audio_0_id
        text_output:     decoded text string
        elapsed:         generation time in seconds
    """
    model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            temperature=0.2,
            top_p=0.8,
            max_new_tokens=2000,
        )
    elapsed = time.time() - t0

    # Slice off the input tokens to get only the newly generated ones
    generated_tokens = outputs[0][model_inputs["input_ids"].shape[1]:]

    audio_token_ids = []
    text_token_ids = []
    for token in generated_tokens:
        if token.item() >= audio_0_id:
            audio_token_ids.append(token)
        else:
            text_token_ids.append(token)

    text_output = tokenizer.decode(text_token_ids, skip_special_tokens=True)
    return audio_token_ids, text_output, elapsed


def decode_audio(audio_token_ids: list, audio_0_id: int, glm_speech_decoder) -> np.ndarray | None:
    """Convert audio token ID list to waveform numpy array.

    Returns None if no audio tokens were generated.
    """
    if len(audio_token_ids) == 0:
        return None
    audio_ids_shifted = torch.tensor(
        [[t.item() - audio_0_id for t in audio_token_ids]], dtype=torch.long
    )
    tts_speech = glm_speech_decoder(audio_ids_shifted)
    return tts_speech.squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# Optional metric helpers
# ---------------------------------------------------------------------------

_WER_TRANSFORM = None

def _get_wer_transform():
    global _WER_TRANSFORM
    if _WER_TRANSFORM is None and _JIWER_OK:
        # jiwer 4.x: chain must end with ReduceToListOfListOfWords
        _WER_TRANSFORM = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.Strip(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.ReduceToListOfListOfWords(),
        ])
    return _WER_TRANSFORM


def compute_wer(reference: str, hypothesis: str) -> float | None:
    """Compute WER between reference and hypothesis. Returns None if jiwer missing."""
    if not _JIWER_OK or not reference.strip() or not hypothesis.strip():
        return None
    transform = _get_wer_transform()
    return jiwer.wer(
        reference, hypothesis,
        reference_transform=transform,   # jiwer 4.x renamed truth_transform -> reference_transform
        hypothesis_transform=transform,
    )


def score_utmos(audio_path: str, predictor) -> float | None:
    """Score audio file with UTMOS22 (tarepan/SpeechMOS). Returns MOS score or None."""
    if predictor is None:
        return None
    try:
        wav, sr = torchaudio.load(audio_path)
        wav_mono = wav.mean(dim=0, keepdim=True)  # (1, T)
        if sr != 16000:
            wav_mono = torchaudio.functional.resample(wav_mono, sr, 16000)
        with torch.no_grad():
            score = predictor(wav_mono, sr=16000)
        return float(score.item())
    except Exception as e:
        print(f"    UTMOS scoring failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(predictions: list) -> dict:
    """Compute all evaluation metrics from the predictions list.

    Each prediction dict must have: emotion, target_valence, target_arousal,
    pred_valence, pred_arousal, wer (or None), utmos (or None).
    """
    metrics = {}
    n = len(predictions)
    metrics["total_samples"] = n

    # --- WER ---
    if _JIWER_OK:
        wer_vals = [p["wer"] for p in predictions if p.get("wer") is not None]
        if wer_vals:
            metrics["wer_mean"] = round(float(np.mean(wer_vals)), 4)
            per_wer = defaultdict(list)
            for p in predictions:
                if p.get("wer") is not None:
                    per_wer[p["emotion"]].append(p["wer"])
            metrics["wer_per_emotion"] = {
                emo: round(float(np.mean(per_wer[emo])), 4) if per_wer[emo] else None
                for emo in ALL_EMOTIONS
            }
        else:
            metrics["wer_mean"] = None
            metrics["wer_per_emotion"] = None
    else:
        metrics["wer_mean"] = None
        metrics["wer_per_emotion"] = None

    # --- UTMOS ---
    utmos_vals = [p["utmos"] for p in predictions if p.get("utmos") is not None]
    if utmos_vals:
        metrics["utmos_mean"] = round(float(np.mean(utmos_vals)), 4)
        per_utmos = defaultdict(list)
        for p in predictions:
            if p.get("utmos") is not None:
                per_utmos[p["emotion"]].append(p["utmos"])
        metrics["utmos_per_emotion"] = {
            emo: round(float(np.mean(per_utmos[emo])), 4) if per_utmos[emo] else None
            for emo in ALL_EMOTIONS
        }
    else:
        metrics["utmos_mean"] = None
        metrics["utmos_per_emotion"] = None

    return metrics


def compute_bertscore_rouge(predictions: list) -> tuple[dict, dict]:
    """Run BERTScore and ROUGE-L on all predictions.

    Returns (bertscore_metrics, rouge_metrics) dicts, or ({}, {}) if packages missing.
    """
    hypotheses = [p.get("text_response", "") for p in predictions]
    references = [p.get("reference_text", "") for p in predictions]

    bertscore_metrics = {}
    rouge_metrics = {}

    if _BERTSCORE_OK:
        print("Computing BERTScore (batch)...")
        try:
            _, _, F = bert_score_fn(
                hypotheses, references,
                lang="en", batch_size=32, verbose=False,
            )
            f1_list = F.tolist()
            # Store per-sample scores back into predictions
            for p, f1 in zip(predictions, f1_list):
                p["bertscore_f1"] = round(float(f1), 4)

            bertscore_metrics["bertscore_f1_mean"] = round(float(np.mean(f1_list)), 4)
            per_bs = defaultdict(list)
            for p in predictions:
                if "bertscore_f1" in p:
                    per_bs[p["emotion"]].append(p["bertscore_f1"])
            bertscore_metrics["bertscore_f1_per_emotion"] = {
                emo: round(float(np.mean(per_bs[emo])), 4) if per_bs[emo] else None
                for emo in ALL_EMOTIONS
            }
        except Exception as e:
            print(f"  BERTScore failed: {e}")
    else:
        bertscore_metrics["bertscore_f1_mean"] = None
        bertscore_metrics["bertscore_f1_per_emotion"] = None

    if _ROUGE_OK:
        print("Computing ROUGE-L...")
        try:
            scorer = rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=True)
            rougel_scores = []
            per_rouge = defaultdict(list)
            for p, hyp, ref in zip(predictions, hypotheses, references):
                if hyp.strip() and ref.strip():
                    result = scorer.score(ref, hyp)
                    f = round(result["rougeL"].fmeasure, 4)
                else:
                    f = 0.0
                rougel_scores.append(f)
                per_rouge[p["emotion"]].append(f)

            rouge_metrics["rougeL_mean"] = round(float(np.mean(rougel_scores)), 4)
            rouge_metrics["rougeL_per_emotion"] = {
                emo: round(float(np.mean(per_rouge[emo])), 4) if per_rouge[emo] else None
                for emo in ALL_EMOTIONS
            }
        except Exception as e:
            print(f"  ROUGE-L failed: {e}")
    else:
        rouge_metrics["rougeL_mean"] = None
        rouge_metrics["rougeL_per_emotion"] = None

    return bertscore_metrics, rouge_metrics


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(metrics: dict):
    """Print a human-readable metrics summary to stdout."""
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total samples:           {metrics['total_samples']}")

    print(f"\n--- Speech Quality ---")
    wer = metrics.get("wer_mean")
    utmos = metrics.get("utmos_mean")
    print(f"WER mean:   {f'{wer:.4f}' if wer is not None else 'N/A (jiwer not installed)'}")
    print(f"UTMOS mean: {f'{utmos:.4f}' if utmos is not None else 'N/A (UTMOS not scored)'}")

    print(f"\n--- Response Coherence ---")
    bs = metrics.get("bertscore_f1_mean")
    rl = metrics.get("rougeL_mean")
    print(f"BERTScore F1: {f'{bs:.4f}' if bs is not None else 'N/A (bert-score not installed)'}")
    print(f"ROUGE-L:      {f'{rl:.4f}' if rl is not None else 'N/A (rouge-score not installed)'}")

    print(f"\n{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Print environment info
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}")
    print(f"\nMetrics enabled: emotion_control=True, wer={_JIWER_OK}, "
          f"utmos=True (torch.hub), bertscore={_BERTSCORE_OK}, rougeL={_ROUGE_OK}\n")

    # Set up output directories
    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"

    # Load eval samples
    samples = load_eval_samples(args.eval_jsonl, args.num_samples)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-voice-9b", trust_remote_code=True)
    audio_0_id = tokenizer.convert_tokens_to_ids('<|audio_0|>')
    print(f"audio_0_id = {audio_0_id}")

    # Load language model (biggest GPU cost — fail fast)
    t0 = time.time()
    model = load_language_model(args.model_path, args.is_base, device="auto")
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Load speech decoder
    print("Loading GLM4CodecDecoder...")
    decoder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "glm-4-voice-decoder")
    glm_speech_decoder = GLM4CodecDecoder(decoder_path)
    print("Decoder loaded.")

    # Load Whisper ASR pipeline (for WER)
    whisper_pipe = None
    if _JIWER_OK:
        print(f"Loading Whisper ASR ({args.whisper_model})...")
        whisper_device = 0 if torch.cuda.is_available() else -1
        whisper_pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=args.whisper_model,
            device=whisper_device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        print("Whisper loaded.")

    # Load UTMOS predictor via torch.hub (downloads model ~392MB on first run)
    global _utmos_predictor
    print("Loading UTMOS22 via torch.hub...")
    try:
        _utmos_predictor = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
        )
        _utmos_predictor.eval()
        print("UTMOS loaded.")
    except Exception as e:
        print(f"  UTMOS load failed: {e} — UTMOS will be skipped")

    # Resume: load already-processed sample IDs
    existing_ids = set()
    predictions = []
    if args.skip_existing and predictions_path.exists():
        with open(predictions_path) as f:
            for line in f:
                p = json.loads(line)
                predictions.append(p)
                existing_ids.add(p["id"])
        print(f"Resuming: {len(existing_ids)} samples already completed\n")

    # -----------------------------------------------------------------------
    # Main generation loop
    # -----------------------------------------------------------------------
    failed_count = 0

    for i, sample in enumerate(samples):
        sample_id = sample["id"]
        emotion = sample["emotion"]
        target_v = float(sample["valence"])
        target_a = float(sample["arousal"])
        audio_path = audio_dir / f"{sample_id}.wav"

        if args.skip_existing and sample_id in existing_ids:
            continue

        print(f"\n[{i+1}/{len(samples)}] {sample_id}  emotion={emotion}  "
              f"VA=({target_v:+.2f}, {target_a:+.2f})")

        # Parse the text field
        try:
            user_audio_tokens = extract_user_audio_tokens(sample["text"])
            reference_text = extract_reference_text(sample["text"])
        except ValueError as e:
            print(f"  ERROR parsing text: {e} — skipping")
            failed_count += 1
            continue

        # Build prompt
        prompt = build_prompt(user_audio_tokens, target_v, target_a, args.is_base)

        # Generate response
        audio_token_ids, text_output, elapsed = run_generation(
            model, tokenizer, prompt, audio_0_id
        )
        print(f"  Generated {len(audio_token_ids)} audio tokens, "
              f"{len(text_output)} chars text in {elapsed:.1f}s")
        if text_output:
            print(f"  Text: {text_output[:120]!r}")

        # Decode audio
        waveform = decode_audio(audio_token_ids, audio_0_id, glm_speech_decoder)
        if waveform is None:
            print(f"  WARNING: No audio tokens generated — skipping")
            failed_count += 1
            continue

        # Save WAV
        sf.write(str(audio_path), waveform, DECODER_SAMPLE_RATE)

        # WER
        wer_score = None
        if whisper_pipe is not None and reference_text:
            try:
                asr_result = whisper_pipe(str(audio_path))
                transcription = asr_result["text"]
                wer_score = compute_wer(reference_text, transcription)
                print(f"  WER: {wer_score:.3f}  transcription: {transcription[:80]!r}")
            except Exception as e:
                print(f"  WER failed: {e}")
                transcription = ""
        else:
            transcription = ""

        # UTMOS
        utmos_score = score_utmos(str(audio_path), _utmos_predictor)
        if utmos_score is not None:
            print(f"  UTMOS: {utmos_score:.3f}")

        # Build record
        record = {
            "id": sample_id,
            "emotion": emotion,
            "target_valence": target_v,
            "target_arousal": target_a,
            "text_response": text_output,
            "reference_text": reference_text,
            "transcription": transcription,
            "wer": wer_score,
            "utmos": utmos_score,
            "audio_path": str(audio_path),
        }
        predictions.append(record)

        # Append-write immediately for resumability
        with open(predictions_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    print(f"\nGeneration complete: {len(predictions)} predictions, {failed_count} failed")

    if not predictions:
        print("No predictions to evaluate!", file=sys.stderr)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # BERTScore and ROUGE-L (batch, after main loop)
    # -----------------------------------------------------------------------
    bertscore_metrics, rouge_metrics = compute_bertscore_rouge(predictions)

    # -----------------------------------------------------------------------
    # Compute all metrics
    # -----------------------------------------------------------------------
    print("Computing metrics...")
    metrics = compute_metrics(predictions)
    metrics["failed_samples"] = failed_count
    metrics.update(bertscore_metrics)
    metrics.update(rouge_metrics)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    print_summary(metrics)
    print(f"All results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
