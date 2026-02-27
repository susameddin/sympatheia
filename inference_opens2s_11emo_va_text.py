"""
GLM-4-Voice inference with 11-emotion valence-arousal values in system prompt (text-based)

Uses continuous VA values for 11 trained emotional states.
Supports interpolation between emotional states.
Only generates outputs from the fine-tuned model.
Supports running inference across multiple checkpoints in one run.

Usage:
    python inference_opens2s_11emo_va_text.py \
        --experiment-dir experiments/glm-model-opens2s-11emo-va-text-lora \
        --checkpoints 100 300 500

    # Or run a single checkpoint (backward compatible):
    python inference_opens2s_11emo_va_text.py \
        --checkpoint experiments/glm-model-opens2s-11emo-va-text-lora/checkpoint-300
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import random
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import soundfile as sf
from src.vocoder import GLM4CodecEncoder, GLM4CodecDecoder
from pathlib import Path
import shutil
import time
import gc


def parse_args():
    parser = argparse.ArgumentParser(
        description="GLM-4-Voice 11-emotion VA inference (supports multiple checkpoints)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--experiment-dir",
        type=str,
        help="Experiment directory containing checkpoint-* folders (use with --checkpoints)",
    )
    group.add_argument(
        "--checkpoint",
        type=str,
        help="Single checkpoint path (e.g., experiments/.../checkpoint-300)",
    )
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        help="Checkpoint step numbers to run (e.g., 100 300 500). Used with --experiment-dir",
    )
    parser.add_argument(
        "--input-audio",
        type=str,
        default=None,
        help="Path to input audio file. If not set, uses a default eval sample",
    )
    parser.add_argument(
        "--compare-mode",
        action="store_true",
        help="Run emotion comparison: for each emotion sample one eval query and generate "
             "correct / N/A / opposite VA outputs. Saves under results_11emo/emotion_comparison/",
    )
    parser.add_argument(
        "--eval-audio-dir",
        type=str,
        default="/engram/naplab/users/sd3705/Datasets/Sympatheia_11Emo_17k/audio/eval",
        help="Root of eval audio dir containing {emotion}_query/ subdirs (used with --compare-mode)",
    )
    parser.add_argument(
        "--compare-seed",
        type=int,
        default=42,
        help="Random seed for sampling eval audio in compare mode (default: 42)",
    )
    return parser.parse_args()


def build_checkpoint_paths(args):
    """Build list of checkpoint paths from args."""
    if args.checkpoint:
        return [Path(args.checkpoint)]

    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.checkpoints:
        # Auto-detect all checkpoint-* directories
        ckpt_dirs = sorted(experiment_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        if not ckpt_dirs:
            print(f"Error: No checkpoint-* directories found in {experiment_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Auto-detected {len(ckpt_dirs)} checkpoints: {[p.name for p in ckpt_dirs]}")
        return ckpt_dirs

    paths = []
    for step in args.checkpoints:
        ckpt_path = experiment_dir / f"checkpoint-{step}"
        if not ckpt_path.exists():
            print(f"Warning: Checkpoint not found: {ckpt_path}, skipping", file=sys.stderr)
        else:
            paths.append(ckpt_path)

    if not paths:
        print("Error: No valid checkpoints found", file=sys.stderr)
        sys.exit(1)

    return paths


# Opposite emotion mapping — partner is the emotion farthest away in VA space
OPPOSITE_EMOTIONS = {
    "sad":        "excited",    # (-0.75,-0.65) ↔ ( 0.75, 0.90)
    "excited":    "sad",        # ( 0.75, 0.90) ↔ (-0.75,-0.65)
    "frustrated": "excited",    # (-0.82,-0.20) → Excited is farthest
    "neutral":    "angry",      # ( 0.00, 0.00) → Angry is farthest
    "happy":      "sad",        # ( 0.85, 0.35) → Sad is farthest
    "angry":      "relaxed",    # (-0.85, 0.85) → Relaxed is farthest
    "fear":       "tired",      # (-0.40, 0.65) → Tired is farthest
    "relaxed":    "angry",      # ( 0.40,-0.45) → Angry is farthest
    "surprised":  "sad",        # ( 0.10, 0.80) → Sad is farthest
    "disgusted":  "happy",      # (-0.80, 0.35) → Happy is farthest
    "tired":      "excited",    # (-0.15,-0.75) → Excited is farthest
}


# Helper function to check if token is audio
def is_audio_token(token_id, audio_0_id):
    return token_id >= audio_0_id


# Helper function for linear interpolation between two VA points
def interpolate_va(va1, va2, t):
    """Interpolate between two (valence, arousal) points. t=0 gives va1, t=1 gives va2."""
    v = va1[0] + t * (va2[0] - va1[0])
    a = va1[1] + t * (va2[1] - va1[1])
    return (v, a)


def build_va_conditions():
    """Build the list of VA conditions to test."""
    # 11 TRAINED EMOTION ANCHORS
    emotion_anchors = {
        "sad":        (-0.75, -0.65),
        "excited":    (0.75, 0.90),
        "frustrated": (-0.82, -0.20),
        "neutral":    (0.00, 0.00),
        "happy":      (0.85, 0.35),
        "angry":      (-0.85, 0.85),
        "fear":       (-0.40, 0.65),
        "relaxed":    (0.40, -0.45),
        "surprised":  (0.10, 0.80),
        "disgusted":  (-0.80, 0.35),
        "tired":      (-0.15, -0.75),
    }

    va_conditions = []

    # 1. All 11 trained emotion anchors
    print("\n=== 11 Trained Emotion Anchors ===")
    for name, (v, a) in emotion_anchors.items():
        va_conditions.append((name, v, a))
        print(f"  {name}: V={v:.2f}, A={a:.2f}")

    # 2. Interpolations between emotion pairs
    print("\n=== Emotion Interpolations ===")

    # Happy <-> Sad (opposite valence)
    for t, label in [(0.25, "happy_75_sad_25"), (0.50, "happy_sad_mid"), (0.75, "happy_25_sad_75")]:
        v, a = interpolate_va(emotion_anchors["happy"], emotion_anchors["sad"], t)
        va_conditions.append((label, v, a))
        print(f"  {label}: V={v:.2f}, A={a:.2f}")

    # Excited <-> Relaxed (high arousal <-> low arousal, both positive)
    for t, label in [(0.25, "excited_75_relaxed_25"), (0.50, "excited_relaxed_mid"), (0.75, "excited_25_relaxed_75")]:
        v, a = interpolate_va(emotion_anchors["excited"], emotion_anchors["relaxed"], t)
        va_conditions.append((label, v, a))
        print(f"  {label}: V={v:.2f}, A={a:.2f}")

    # Angry <-> Neutral (negative high arousal -> center)
    for t, label in [(0.33, "angry_67_neutral_33"), (0.67, "angry_33_neutral_67")]:
        v, a = interpolate_va(emotion_anchors["angry"], emotion_anchors["neutral"], t)
        va_conditions.append((label, v, a))
        print(f"  {label}: V={v:.2f}, A={a:.2f}")

    # Fear <-> Relaxed
    for t, label in [(0.50, "fear_relaxed_mid")]:
        v, a = interpolate_va(emotion_anchors["fear"], emotion_anchors["relaxed"], t)
        va_conditions.append((label, v, a))
        print(f"  {label}: V={v:.2f}, A={a:.2f}")

    # Surprised <-> Neutral
    for t, label in [(0.50, "surprised_neutral_mid")]:
        v, a = interpolate_va(emotion_anchors["surprised"], emotion_anchors["neutral"], t)
        va_conditions.append((label, v, a))
        print(f"  {label}: V={v:.2f}, A={a:.2f}")

    # Frustrated <-> Happy (negative to positive, both mid arousal)
    for t, label in [(0.50, "frustrated_happy_mid")]:
        v, a = interpolate_va(emotion_anchors["frustrated"], emotion_anchors["happy"], t)
        va_conditions.append((label, v, a))
        print(f"  {label}: V={v:.2f}, A={a:.2f}")

    # Disgusted <-> Neutral (negative mid-arousal -> center)
    for t, label in [(0.33, "disgusted_67_neutral_33"), (0.67, "disgusted_33_neutral_67")]:
        v, a = interpolate_va(emotion_anchors["disgusted"], emotion_anchors["neutral"], t)
        va_conditions.append((label, v, a))
        print(f"  {label}: V={v:.2f}, A={a:.2f}")

    # Tired <-> Excited (low energy <-> high energy)
    for t, label in [(0.25, "tired_75_excited_25"), (0.50, "tired_excited_mid"), (0.75, "tired_25_excited_75")]:
        v, a = interpolate_va(emotion_anchors["tired"], emotion_anchors["excited"], t)
        va_conditions.append((label, v, a))
        print(f"  {label}: V={v:.2f}, A={a:.2f}")

    # Disgusted <-> Angry (both very negative, different arousal)
    for t, label in [(0.50, "disgusted_angry_mid")]:
        v, a = interpolate_va(emotion_anchors["disgusted"], emotion_anchors["angry"], t)
        va_conditions.append((label, v, a))
        print(f"  {label}: V={v:.2f}, A={a:.2f}")

    # Tired <-> Relaxed (both low arousal, valence shift)
    for t, label in [(0.50, "tired_relaxed_mid")]:
        v, a = interpolate_va(emotion_anchors["tired"], emotion_anchors["relaxed"], t)
        va_conditions.append((label, v, a))
        print(f"  {label}: V={v:.2f}, A={a:.2f}")

    # Tired <-> Sad (both low arousal and negative, different intensity)
    for t, label in [(0.50, "tired_sad_mid")]:
        v, a = interpolate_va(emotion_anchors["tired"], emotion_anchors["sad"], t)
        va_conditions.append((label, v, a))
        print(f"  {label}: V={v:.2f}, A={a:.2f}")

    # 3. Custom VA space exploration (corners and edges)
    print("\n=== Custom VA Space Points ===")
    custom_points = [
        ("extreme_positive_high", 1.0, 1.0),
        ("extreme_positive_low", 1.0, -1.0),
        ("extreme_negative_high", -1.0, 1.0),
        ("extreme_negative_low", -1.0, -1.0),
        ("positive_calm", 0.50, 0.00),
        ("negative_calm", -0.50, 0.00),
        ("neutral_energetic", 0.00, 0.70),
        ("neutral_lethargic", 0.00, -0.70),
    ]
    for name, v, a in custom_points:
        va_conditions.append((name, v, a))
        print(f"  {name}: V={v:.2f}, A={a:.2f}")

    # 4. N/A condition — no valence/arousal provided, model must infer from audio
    print("\n=== No Emotional Cue (N/A) ===")
    va_conditions.append(("na", None, None))
    print("  na: system prompt = 'User emotion N/A'")

    print(f"\n=== Total conditions: {len(va_conditions)} ===")
    return va_conditions, emotion_anchors, custom_points


def run_inference_for_checkpoint(
    checkpoint_path,
    va_conditions,
    user_input,
    user_audio_path,
    glm_tokenizer,
    glm_speech_decoder,
    audio_0_id,
):
    """Load a checkpoint, run all VA conditions, save results, then unload."""
    ckpt_name = checkpoint_path.name
    print(f"\n{'#'*70}")
    print(f"# CHECKPOINT: {ckpt_name}")
    print(f"# Path: {checkpoint_path}")
    print(f"{'#'*70}")

    # Load model
    load_start = time.time()
    print(f"\nLoading model from {checkpoint_path}...")
    glm_model = AutoPeftModelForCausalLM.from_pretrained(
        str(checkpoint_path),
        device_map="auto",
        trust_remote_code=True,
    )
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.1f}s")

    # Create results folder
    results_dir = checkpoint_path / "results_11emo"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save input audio
    input_audio_save_path = results_dir / "input_audio.wav"
    shutil.copy(user_audio_path, input_audio_save_path)

    # Run inference for all conditions
    ckpt_start_time = time.time()
    output_count = 0
    failed_count = 0

    for name, valence, arousal in va_conditions:
        print(f"\n{'='*60}")
        if valence is None:
            print(f"[{ckpt_name}] Generating: {name} (no emotional cue)")
        else:
            print(f"[{ckpt_name}] Generating: {name} (valence={valence:.2f}, arousal={arousal:.2f})")
        print(f"{'='*60}")

        if valence is None:
            system_prompt = "Please respond in English. User emotion N/A"
        else:
            system_prompt = f"Please respond in English. User emotion (valence={valence:.2f}, arousal={arousal:.2f})"
        inputs = f"<|system|>\n{system_prompt}\n<|user|>\n{user_input}\n<|assistant|>\n"

        inference_start_time = time.time()
        text_output, tts_speech = generate_one(
            inputs, glm_model, glm_tokenizer, glm_speech_decoder, audio_0_id
        )
        inference_time = time.time() - inference_start_time

        print(f"Text output: {text_output[:200]}..." if len(text_output) > 200 else f"Text output: {text_output}")

        if tts_speech is None:
            print(f"WARNING: No audio tokens generated for {name} - skipping")
            failed_count += 1
            continue

        if valence is None:
            out_path = results_dir / f"output_{name}.wav"
        else:
            out_path = results_dir / f"output_{name}_v{valence:.2f}_a{arousal:.2f}.wav"
        sf.write(str(out_path), tts_speech.squeeze(), 22050)
        print(f"Saved: {out_path}")
        print(f"Time: {inference_time:.2f}s")
        output_count += 1

    ckpt_total_time = time.time() - ckpt_start_time

    print(f"\n{'='*60}")
    print(f"[{ckpt_name}] COMPLETE: {output_count} outputs, {failed_count} failed, {ckpt_total_time:.1f}s total")
    print(f"Results: {results_dir}")
    print(f"{'='*60}")

    # Unload model to free GPU memory for next checkpoint
    del glm_model
    gc.collect()
    torch.cuda.empty_cache()

    return output_count, failed_count, ckpt_total_time


def generate_one(
    inputs, glm_model, glm_tokenizer, glm_speech_decoder, audio_0_id
):
    """Shared generation helper — returns (text_output, tts_speech or None)."""
    with torch.no_grad():
        model_inputs = glm_tokenizer(inputs, return_tensors="pt").to(glm_model.device)
        outputs = glm_model.generate(
            **model_inputs,
            temperature=0.2,
            top_p=0.8,
            max_new_tokens=2000,
        )

    generated_tokens = outputs[0][model_inputs["input_ids"].shape[1]:]
    audio_token_ids, text_token_ids = [], []
    for token in generated_tokens:
        if is_audio_token(token, audio_0_id):
            audio_token_ids.append(token)
        else:
            text_token_ids.append(token)

    text_output = glm_tokenizer.decode(text_token_ids, skip_special_tokens=True)

    if not audio_token_ids:
        return text_output, None

    audio_ids_shifted = torch.tensor(
        [[tok.item() - audio_0_id for tok in audio_token_ids]], dtype=torch.long
    )
    tts_speech = glm_speech_decoder(audio_ids_shifted)
    return text_output, tts_speech


def run_emotion_comparison(
    checkpoint_path,
    emotion_anchors,
    eval_audio_dir,
    glm_tokenizer,
    glm_speech_encoder,
    glm_speech_decoder,
    audio_0_id,
    seed=42,
):
    """
    For each of the 11 emotions:
      - Sample one query wav from eval_audio_dir/{emotion}_query/
      - Generate three responses:
          correct  — system prompt with the emotion's own VA values
          na       — system prompt with 'User emotion N/A'
          opposite — system prompt with the farthest-away emotion's VA values
      - Save all outputs under results_11emo/emotion_comparison/{emotion}/
    """
    rng = random.Random(seed)
    ckpt_name = checkpoint_path.name

    print(f"\n{'#'*70}")
    print(f"# COMPARE MODE — CHECKPOINT: {ckpt_name}")
    print(f"# Path: {checkpoint_path}")
    print(f"{'#'*70}")

    load_start = time.time()
    print(f"\nLoading model from {checkpoint_path}...")
    glm_model = AutoPeftModelForCausalLM.from_pretrained(
        str(checkpoint_path),
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Model loaded in {time.time() - load_start:.1f}s")

    results_dir = checkpoint_path / "results_11emo" / "emotion_comparison"
    results_dir.mkdir(parents=True, exist_ok=True)

    ckpt_start = time.time()
    output_count = 0
    failed_count = 0

    for emotion, (valence, arousal) in emotion_anchors.items():
        print(f"\n{'='*60}")
        print(f"[{ckpt_name}] Emotion: {emotion}  (V={valence:.2f}, A={arousal:.2f})")
        print(f"{'='*60}")

        # ── Sample one eval query wav ────────────────────────────────────────
        query_dir = Path(eval_audio_dir) / f"{emotion}_query"
        wav_files = sorted(query_dir.glob("*.wav"))
        if not wav_files:
            print(f"  WARNING: no wav files found in {query_dir}, skipping")
            failed_count += 3
            continue

        chosen_wav = rng.choice(wav_files)
        print(f"  Input: {chosen_wav.name}")

        # Encode audio
        audio_tokens = glm_speech_encoder([str(chosen_wav)])[0]
        user_input = "".join([f"<|audio_{x}|>" for x in audio_tokens])

        # Create per-emotion output dir and save input copy
        emotion_dir = results_dir / emotion
        emotion_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(chosen_wav, emotion_dir / "input_audio.wav")

        # ── Opposite emotion ────────────────────────────────────────────────
        opp_name = OPPOSITE_EMOTIONS[emotion]
        opp_v, opp_a = emotion_anchors[opp_name]

        conditions = [
            (
                f"correct_{emotion}_v{valence:.2f}_a{arousal:.2f}",
                f"Please respond in English. User emotion (valence={valence:.2f}, arousal={arousal:.2f})",
            ),
            (
                "na",
                "Please respond in English. User emotion N/A",
            ),
            (
                f"opposite_{opp_name}_v{opp_v:.2f}_a{opp_a:.2f}",
                f"Please respond in English. User emotion (valence={opp_v:.2f}, arousal={opp_a:.2f})",
            ),
        ]

        for out_stem, system_prompt in conditions:
            inputs = f"<|system|>\n{system_prompt}\n<|user|>\n{user_input}\n<|assistant|>\n"
            t0 = time.time()
            text_out, tts_speech = generate_one(
                inputs, glm_model, glm_tokenizer, glm_speech_decoder, audio_0_id
            )
            elapsed = time.time() - t0

            print(f"  [{out_stem}] {elapsed:.1f}s | text: {text_out[:100]!r}")

            if tts_speech is None:
                print(f"  WARNING: no audio tokens for {out_stem} — skipping")
                failed_count += 1
                continue

            out_path = emotion_dir / f"output_{out_stem}.wav"
            sf.write(str(out_path), tts_speech.squeeze(), 22050)
            print(f"  Saved: {out_path.name}")
            output_count += 1

    ckpt_total = time.time() - ckpt_start
    print(f"\n{'='*60}")
    print(f"[{ckpt_name}] COMPARE DONE: {output_count} outputs, {failed_count} failed, {ckpt_total:.1f}s")
    print(f"Results: {results_dir}")
    print(f"{'='*60}")

    del glm_model
    gc.collect()
    torch.cuda.empty_cache()

    return output_count, failed_count, ckpt_total


def main():
    args = parse_args()

    # Print CUDA info
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA devices:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU 0 name:", torch.cuda.get_device_name(0))
    print("torch version:", torch.__version__)

    # Build checkpoint paths
    checkpoint_paths = build_checkpoint_paths(args)
    print(f"\nCheckpoints to process: {[p.name for p in checkpoint_paths]}")

    # Resolve input audio
    if args.input_audio:
        user_audio_path = Path(args.input_audio)
    else:
        user_audio_path = Path("/engram/naplab/users/sd3705/Datasets/OpenS2S_11Emo/audio/eval/neutral_query/4823.wav")
        if not user_audio_path.exists():
            user_audio_path = Path("/engram/naplab/users/sd3705/Datasets/OpenS2S_9Emo/audio/eval/neutral_query/4823.wav")
    assert user_audio_path.exists(), f"Missing audio: {user_audio_path}"
    print(f"Input audio: {user_audio_path}")

    # Load shared components (tokenizer, encoder, decoder - same for all checkpoints)
    print("\n=== Loading shared components ===")
    glm_tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-voice-9b", trust_remote_code=True)
    glm_speech_encoder = GLM4CodecEncoder()
    glm_speech_decoder = GLM4CodecDecoder("glm-4-voice-decoder")
    audio_0_id = glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
    print("Shared components loaded")

    # Encode user audio (once, reused for all checkpoints)
    audio_tokens = glm_speech_encoder([str(user_audio_path)])[0]
    user_input = "".join([f"<|audio_{x}|>" for x in audio_tokens])

    # Build VA conditions (always needed for emotion_anchors)
    va_conditions, emotion_anchors, custom_points = build_va_conditions()

    # Run inference for each checkpoint
    overall_start = time.time()
    all_results = []

    for ckpt_path in checkpoint_paths:
        if args.compare_mode:
            output_count, failed_count, ckpt_time = run_emotion_comparison(
                checkpoint_path=ckpt_path,
                emotion_anchors=emotion_anchors,
                eval_audio_dir=args.eval_audio_dir,
                glm_tokenizer=glm_tokenizer,
                glm_speech_encoder=glm_speech_encoder,
                glm_speech_decoder=glm_speech_decoder,
                audio_0_id=audio_0_id,
                seed=args.compare_seed,
            )
        else:
            output_count, failed_count, ckpt_time = run_inference_for_checkpoint(
                checkpoint_path=ckpt_path,
                va_conditions=va_conditions,
                user_input=user_input,
                user_audio_path=user_audio_path,
                glm_tokenizer=glm_tokenizer,
                glm_speech_decoder=glm_speech_decoder,
                audio_0_id=audio_0_id,
            )
        all_results.append((ckpt_path.name, output_count, failed_count, ckpt_time))

    overall_time = time.time() - overall_start

    # Final summary across all checkpoints
    print(f"\n{'#'*70}")
    print(f"# OVERALL SUMMARY")
    print(f"{'#'*70}")
    print(f"Total checkpoints processed: {len(all_results)}")
    print(f"Total time: {overall_time:.1f}s")
    print()
    print(f"{'Checkpoint':<25} {'Outputs':>8} {'Failed':>8} {'Time (s)':>10}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*10}")
    for ckpt_name, outputs, failed, t in all_results:
        print(f"{ckpt_name:<25} {outputs:>8} {failed:>8} {t:>10.1f}")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
