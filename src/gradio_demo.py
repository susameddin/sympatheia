"""
Interactive Gradio demo for GLM-4-Voice emotion-conditioned speech-to-speech model.

Records user voice, takes valence/arousal input via sliders or emotion presets,
and generates an emotionally-conditioned speech response.

Usage:
    python gradio_demo.py \
        --checkpoint experiments/sympatheia-12emo-YYYYMMDD-HHMMSS/checkpoint-N \
        --port 7860
"""

import sys
import os

# Ensure a VP8-capable ffmpeg is first on PATH (required for Gradio video preprocessing).
import shutil as _shutil
_ffmpeg_path = _shutil.which("ffmpeg")
if _ffmpeg_path:
    os.environ["PATH"] = os.path.dirname(_ffmpeg_path) + ":" + os.environ.get("PATH", "")
del _shutil, _ffmpeg_path

# Keep the university proxy for outbound internet (needed for share tunnel),
# but bypass it for localhost so Gradio's local health check works.
os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",localhost,127.0.0.1,0.0.0.0"
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",localhost,127.0.0.1,0.0.0.0"

# Ensure both the src dir and its parent (project root) are on the path.
# The parent is needed for `from src.vocoder_src import ...` style imports.
FINETUNE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(FINETUNE_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, FINETUNE_DIR)

import argparse
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gradio as gr
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from src.vocoder_src import GLM4CodecEncoder, GLM4CodecDecoder
from src.integration.text_module.text_to_va import TextToVAConverter
from src.integration.face_module.detect import detect_and_crop_face
from src.integration.face_module.models import HSEmotionFacePredictor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMOTION_ANCHORS = {
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

EMOTION_COLORS = {
    "sad":        (0.30, 0.45, 0.85),   # Steel blue
    "excited":    (1.00, 0.65, 0.00),   # Bright orange
    "frustrated": (0.75, 0.30, 0.30),   # Brick / terracotta
    "neutral":    (0.60, 0.60, 0.60),   # Medium gray
    "happy":      (0.40, 0.80, 0.20),   # Lime green
    "angry":      (0.85, 0.10, 0.10),   # Red
    "anxious":    (0.50, 0.10, 0.70),   # Deep purple
    "relaxed":    (0.20, 0.65, 0.55),   # Teal
    "surprised":  (0.95, 0.50, 0.80),   # Pink
    "disgusted":  (0.55, 0.60, 0.20),   # Olive green
    "tired":      (0.50, 0.50, 0.70),   # Muted lavender
    "content":    (0.80, 0.70, 0.30),   # Warm gold
}

LABEL_OFFSETS = {
    "sad":        ( 0, -14),
    "excited":    (-10, -14),
    "frustrated": ( 12,  10),
    "neutral":    ( 10,   8),
    "happy":      (  0,  10),
    "angry":      (-10,  10),
    "anxious":    ( 14,   4),
    "relaxed":    (  0,  10),
    "surprised":  (  0,  10),
    "disgusted":  ( 14,   4),
    "tired":      (  0, -14),
    "content":    (  0, -14),
}

_HEATMAP_SIGMA = 0.35
_HEATMAP_RES   = 200
_HEATMAP_ALPHA = 0.70

DEFAULT_CHECKPOINT = os.path.join(
    FINETUNE_DIR,
    "experiments/sympatheia-12emo-YYYYMMDD-HHMMSS/checkpoint-N",
)
SAMPLE_RATE = 22050

# Global model references (set in __main__)
glm_tokenizer = None
glm_speech_encoder = None
glm_speech_decoder = None
glm_model = None
audio_0_id = None
text_to_va_converter = None  # Set in __main__ after models are loaded
face_va_predictor = None     # Lazy-loaded on first face detection
face_va_mapper = None        # unused, kept for compatibility

# ---------------------------------------------------------------------------
# VA Plane Visualization
# ---------------------------------------------------------------------------

def _build_heatmap():
    """Precompute Gaussian blob clouds for each emotion (called once at import)."""
    names = list(EMOTION_ANCHORS.keys())
    coords = np.array([EMOTION_ANCHORS[n] for n in names])
    colors = np.array([EMOTION_COLORS[n] for n in names])

    x = np.linspace(-1.15, 1.15, _HEATMAP_RES)
    y = np.linspace(-1.15, 1.15, _HEATMAP_RES)
    X, Y = np.meshgrid(x, y)

    # Start with white background, alpha-blend each emotion cloud on top
    img = np.ones((_HEATMAP_RES, _HEATMAP_RES, 3))

    for i in range(len(names)):
        d2 = (X - coords[i, 0]) ** 2 + (Y - coords[i, 1]) ** 2
        alpha = np.exp(-d2 / (2.0 * _HEATMAP_SIGMA ** 2))
        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1 - alpha * 0.5) + colors[i, c] * alpha * 0.5

    return np.clip(img, 0.0, 1.0)


_VA_HEATMAP = _build_heatmap()


def create_va_plane_figure(valence=0.0, arousal=0.0):
    """Create a matplotlib figure of the valence-arousal emotion space."""
    fig, ax = plt.subplots(figsize=(5, 5))

    # --- Heatmap background ---
    ax.imshow(
        _VA_HEATMAP,
        extent=[-1.15, 1.15, -1.15, 1.15],
        origin="lower",
        aspect="equal",
        alpha=_HEATMAP_ALPHA,
        zorder=0,
    )

    # --- Axes setup ---
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("Valence  (Negative  \u2190  \u2192  Positive)", fontsize=10)
    ax.set_ylabel("Arousal  (Calm  \u2190  \u2192  Energetic)", fontsize=10)
    ax.set_title("Valence\u2013Arousal Emotion Space", fontsize=12)
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--", zorder=1)
    ax.axvline(x=0, color="gray", linewidth=0.5, linestyle="--", zorder=1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, zorder=1)

    # --- Emotion anchor dots and labels ---
    for name, (v, a) in EMOTION_ANCHORS.items():
        color = EMOTION_COLORS[name]

        ax.plot(
            v, a, "o",
            color="white",
            markersize=9,
            markeredgecolor=color,
            markeredgewidth=2.0,
            zorder=5,
        )

        dx, dy = LABEL_OFFSETS[name]
        ha = "center"
        if dx > 5:
            ha = "left"
        elif dx < -5:
            ha = "right"

        ax.annotate(
            name,
            (v, a),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=ha,
            fontsize=8,
            fontweight="bold",
            color="#222222",
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                alpha=0.7,
                edgecolor="none",
            ),
            zorder=6,
        )

    # --- Selection marker: X cross ---
    ax.plot(
        valence, arousal, "x",
        markersize=12,
        markeredgecolor="#222222",
        markeredgewidth=2.0,
        zorder=10,
    )

    ax.annotate(
        f"({valence:.2f}, {arousal:.2f})",
        (valence, arousal),
        textcoords="offset points",
        xytext=(14, -12),
        ha="left",
        fontsize=8,
        fontweight="bold",
        color="#222222",
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            alpha=0.85,
            edgecolor="gray",
            linewidth=0.5,
        ),
        zorder=12,
    )

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_models(checkpoint_path):
    """Load all model components once at startup."""
    decoder_path = os.path.join(FINETUNE_DIR, "glm-4-voice-decoder")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "THUDM/glm-4-voice-9b", trust_remote_code=True
    )

    print("Loading speech encoder (WhisperVQ)...")
    speech_encoder = GLM4CodecEncoder()

    print("Loading speech decoder (Flow + HiFi-T)...")
    speech_decoder = GLM4CodecDecoder(decoder_path)

    a0_id = tokenizer.convert_tokens_to_ids("<|audio_0|>")

    print(f"Loading fine-tuned LoRA model from {checkpoint_path}...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        checkpoint_path,
        device_map="auto",
        trust_remote_code=True,
    )
    print("All models loaded successfully.")

    return tokenizer, speech_encoder, speech_decoder, model, a0_id


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(audio_path, valence, arousal):
    """
    Full inference pipeline:
      1. Encode input audio to tokens
      2. Build prompt (with VA values, or N/A if valence is None)
      3. Generate response
      4. Separate text / audio tokens
      5. Decode audio tokens to waveform

    Pass valence=None to let the model self-detect emotion ("User emotion N/A").
    Returns (output_audio_tuple, text_response, va_figure).
    """
    if audio_path is None:
        raise gr.Error("Please record or upload an audio file first.")

    # 1. Encode input audio
    audio_tokens = glm_speech_encoder([audio_path])[0]
    user_input = "".join([f"<|audio_{x}|>" for x in audio_tokens])

    # 2. Build prompt (matches training format)
    if valence is None:
        system_prompt = "Please respond in English. User emotion N/A"
        va_fig = create_va_plane_figure(0.0, 0.0)
    else:
        valence = max(-1.0, min(1.0, float(valence)))
        arousal = max(-1.0, min(1.0, float(arousal)))
        system_prompt = (
            f"Please respond in English. "
            f"User emotion (valence={valence:.2f}, arousal={arousal:.2f})"
        )
        va_fig = create_va_plane_figure(valence, arousal)
    inputs = f"<|system|>\n{system_prompt}\n<|user|>\n{user_input}\n<|assistant|>\n"

    # 3. Generate
    with torch.no_grad():
        model_inputs = glm_tokenizer(inputs, return_tensors="pt").to(glm_model.device)
        outputs = glm_model.generate(
            **model_inputs,
            temperature=0.2,
            top_p=0.8,
            max_new_tokens=2000,
        )

    # 4. Separate audio and text tokens
    generated_tokens = outputs[0][model_inputs["input_ids"].shape[1]:]

    audio_token_ids = []
    text_token_ids = []
    for token in generated_tokens:
        if token.item() >= audio_0_id:
            audio_token_ids.append(token)
        else:
            text_token_ids.append(token)

    text_output = glm_tokenizer.decode(text_token_ids, skip_special_tokens=True)

    # 5. Decode audio tokens to waveform
    if len(audio_token_ids) == 0:
        return (
            None,
            text_output + "\n\n[WARNING: No audio tokens generated]",
            va_fig,
        )

    audio_ids_shifted = torch.tensor(
        [[t.item() - audio_0_id for t in audio_token_ids]], dtype=torch.long
    )
    tts_speech = glm_speech_decoder(audio_ids_shifted)
    audio_numpy = tts_speech.squeeze().cpu().numpy()

    return (
        (SAMPLE_RATE, audio_numpy),
        text_output,
        va_fig,
    )


# ---------------------------------------------------------------------------
# UI Callbacks
# ---------------------------------------------------------------------------

def on_emotion_preset_change(preset_name):
    """Update sliders and VA plot when user selects an emotion preset."""
    if preset_name is None or preset_name == "Custom":
        return gr.update(), gr.update(), create_va_plane_figure(0, 0)
    v, a = EMOTION_ANCHORS[preset_name]
    return gr.update(value=v), gr.update(value=a), create_va_plane_figure(v, a)


def on_slider_change(valence, arousal):
    """Update the VA plot when sliders move."""
    return create_va_plane_figure(valence, arousal)


def on_mode_change(mode):
    """Toggle visibility of manual / describe / face controls."""
    is_manual   = (mode == "Select Manually")
    is_describe = (mode == "Describe Your Feeling")
    is_face     = (mode == "Detect From Face")
    return (
        gr.update(visible=is_manual),    # manual_controls group
        gr.update(visible=is_describe),  # describe_controls group
        gr.update(visible=is_face),      # face_controls group
    )


def on_describe_emotion(description_text: str):
    """Convert free-text emotion description to VA values and update the plot."""
    if not description_text or not description_text.strip():
        raise gr.Error("Please enter an emotion description first.")

    v, a, info = text_to_va_converter.convert(description_text, method="hf")
    return float(v), float(a), create_va_plane_figure(v, a), info



def on_detect_face(upload_image):
    """Detect face in an uploaded image and map to VA."""
    global face_va_predictor
    if upload_image is None:
        raise gr.Error("Please upload a face image first.")
    if face_va_predictor is None:
        face_va_predictor = HSEmotionFacePredictor()
    cropped = detect_and_crop_face(upload_image) if isinstance(upload_image, np.ndarray) else upload_image
    top_emo, probs = face_va_predictor.predict_emotion(cropped)
    v, a = face_va_predictor.predict_va(cropped)
    top3 = sorted(probs.items(), key=lambda x: -x[1])[:3]
    info_parts = [f"{top_emo} (V={v:+.2f}, A={a:+.2f})"]
    info_parts.append("  " + ", ".join(f"{e}: {p:.1%}" for e, p in top3))
    from PIL import Image as _PILImage
    return float(v), float(a), create_va_plane_figure(v, a), "\n".join(info_parts), _PILImage.fromarray(cropped)


def stream_face_frame(frame, results, sample_crop, last_ts):
    """Per-frame callback for the live face webcam stream."""
    global face_va_predictor
    if frame is None:
        return results, sample_crop, last_ts, ""
    if face_va_predictor is None:
        face_va_predictor = HSEmotionFacePredictor()
    try:
        crop = detect_and_crop_face(frame)
        top_emo, _ = face_va_predictor.predict_emotion(crop)
        v, a = face_va_predictor.predict_va(crop)
        results = results + [(top_emo, v, a)]
        if sample_crop is None:
            sample_crop = crop
        live = f"Frame {len(results)}: {top_emo}  V={v:+.2f} A={a:+.2f}"
    except Exception:
        live = f"Frame {len(results) if results else 0}: no face detected"
    return results, sample_crop, time.time(), live


def process_face_video(video_path):
    """
    Extract audio + face emotion from a recorded webcam video.
    Returns (audio_path, status, face_v, face_a, va_fig, face_info, face_crop_img).
    """
    import cv2, subprocess, tempfile
    global face_va_predictor

    if video_path is None:
        return None, "No video recorded", 0.0, 0.0, create_va_plane_figure(0.0, 0.0), "", None

    # --- Extract audio to a temp wav file ---
    tmp_wav = tempfile.mktemp(suffix=".wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", tmp_wav],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError:
        tmp_wav = None  # no audio track (e.g. browser didn't capture mic)

    # --- Sample frames for face emotion ---
    # webm files (recorded by Chrome) don't store frame count in metadata,
    # so we read all frames sequentially and sample after.
    cap = cv2.VideoCapture(video_path)
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    if face_va_predictor is None:
        face_va_predictor = HSEmotionFacePredictor()

    n_samples = 10
    step = max(1, len(all_frames) // n_samples)
    sampled = all_frames[::step][:n_samples]

    va_list, emotion_list, sample_crop = [], [], None
    for frame in sampled:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            crop = detect_and_crop_face(frame_rgb)
            top_emo, _ = face_va_predictor.predict_emotion(crop)
            v, a = face_va_predictor.predict_va(crop)
            va_list.append((v, a))
            emotion_list.append(top_emo)
            if sample_crop is None:
                sample_crop = crop
        except Exception:
            pass

    if not va_list:
        status = "Audio extracted — no face detected in video frames"
        return tmp_wav, status, 0.0, 0.0, create_va_plane_figure(0.0, 0.0), "No face detected", None

    mean_v = float(np.mean([v for v, _ in va_list]))
    mean_a = float(np.mean([a for _, a in va_list]))
    top_emo = max(set(emotion_list), key=emotion_list.count)
    n = len(va_list)
    counts = {e: emotion_list.count(e) for e in set(emotion_list)}
    top3 = sorted(counts.items(), key=lambda x: -x[1])[:3]
    info_parts = [f"{top_emo} (V={mean_v:+.2f}, A={mean_a:+.2f})"]
    info_parts.append("  " + ", ".join(f"{e}: {c/n:.1%}" for e, c in top3))
    info_parts.append(f"  ({n}/{len(sampled)} frames with face detected)")
    va_fig = create_va_plane_figure(mean_v, mean_a)
    from PIL import Image as _PILImage
    crop_img = _PILImage.fromarray(sample_crop)
    audio_note = "" if tmp_wav else " (no audio — check browser mic permissions)"
    status = f"Audio + face ready: {top_emo}{audio_note}"
    return tmp_wav, status, mean_v, mean_a, va_fig, "\n".join(info_parts), crop_img


def auto_analyze_face(results, crop, last_ts):
    """Fires every 1 s; triggers analysis when stream has been quiet for >1.5 s."""
    no_op = (results, crop, last_ts,
             gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
    if not results or last_ts is None:
        return no_op
    if time.time() - last_ts < 1.5:
        return no_op
    emotions = [r[0] for r in results]
    mean_v = float(np.mean([r[1] for r in results]))
    mean_a = float(np.mean([r[2] for r in results]))
    top_emo = max(set(emotions), key=emotions.count)
    n = len(results)
    counts = {e: emotions.count(e) for e in set(emotions)}
    top3 = sorted(counts.items(), key=lambda x: -x[1])[:3]
    info_parts = [f"{top_emo} (V={mean_v:+.2f}, A={mean_a:+.2f})"]
    info_parts.append("  " + ", ".join(f"{e}: {c/n:.1%}" for e, c in top3))
    info_parts.append(f"  ({n} frames analyzed)")
    va_fig = create_va_plane_figure(mean_v, mean_a)
    from PIL import Image as _PILImage
    crop_img = _PILImage.fromarray(crop) if crop is not None else None
    status = f"Done — {n} frame{'s' if n != 1 else ''} analyzed: {top_emo}"
    return [], None, None, status, mean_v, mean_a, va_fig, "\n".join(info_parts), crop_img


def run_inference_with_mode(audio_path, mode, valence, arousal,
                            describe_v, describe_a,
                            face_v, face_a):
    """Pick VA values based on active mode, then run inference."""
    if mode == "Detect From Audio":
        valence, arousal = None, None
    elif mode == "Describe Your Feeling":
        valence, arousal = describe_v, describe_a
    elif mode == "Detect From Face":
        valence, arousal = face_v, face_a
    return run_inference(audio_path, valence, arousal)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    emotion_choices = ["Custom"] + list(EMOTION_ANCHORS.keys())

    with gr.Blocks(title="Sympatheia: Emotion-Conditioned Speech Generation") as demo:
        gr.Markdown(
            "# Sympatheia: Emotion-Conditioned Speech-to-Speech Generation Demo\n"
            "Record or upload speech, select a target emotion, "
            "and generate an emotionally-conditioned response."
        )

        with gr.Row():
            # ---- Left column: inputs ----
            with gr.Column(scale=1):
                gr.Markdown("### Input Audio")
                audio_input = gr.Audio(
                    label="Record or upload audio",
                    sources=["microphone"],
                    type="filepath",
                )

                gr.Markdown("### Emotion Control")

                emotion_mode = gr.Radio(
                    choices=["Select Manually", "Detect From Audio", "Describe Your Feeling", "Detect From Face"],
                    value="Select Manually",
                    label="Emotion Input Mode",
                )

                # --- Manual controls ---
                with gr.Group(visible=True) as manual_controls:
                    emotion_preset = gr.Dropdown(
                        choices=emotion_choices,
                        value="neutral",
                        label="Emotion Preset",
                    )
                    valence_slider = gr.Slider(
                        minimum=-1.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.01,
                        label="Valence  (Negative \u2190 \u2192 Positive)",
                    )
                    arousal_slider = gr.Slider(
                        minimum=-1.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.01,
                        label="Arousal  (Calm \u2190 \u2192 Energetic)",
                    )

                # --- Describe-your-feeling controls ---
                with gr.Group(visible=False) as describe_controls:
                    feeling_text = gr.Textbox(
                        label="Describe Your Feeling",
                        placeholder='e.g. "I\'m feeling a bit down but also somewhat hopeful"',
                        lines=2,
                        max_lines=4,
                    )
                    describe_btn = gr.Button(
                        "Extract Emotion from Description",
                        variant="secondary",
                    )
                    describe_info = gr.Textbox(
                        label="Extracted Emotion",
                        interactive=False,
                        lines=1,
                    )
                    describe_v = gr.Number(value=0.0, visible=False)
                    describe_a = gr.Number(value=0.0, visible=False)

                # --- Face detection controls ---
                with gr.Group(visible=False) as face_controls:
                    face_poll_timer = gr.Timer(value=1.0)
                    with gr.Tab("Record Video (Audio + Face)"):
                        gr.Markdown(
                            "Record yourself speaking — audio goes to the model, "
                            "face frames are analyzed for emotion automatically."
                        )
                        face_video = gr.Video(
                            sources=["webcam"],
                            label="Record video",
                            include_audio=True,
                            webcam_options=gr.WebcamOptions(mirror=False),
                        )
                        face_video_status = gr.Textbox(
                            label="Status", interactive=False, lines=2
                        )
                    with gr.Tab("Live Webcam (face only)"):
                        face_webcam_stream = gr.Image(
                            sources=["webcam"],
                            streaming=True,
                            type="numpy",
                            label="Start webcam, speak/express — results appear automatically when you stop",
                            height=250,
                        )
                        face_live_status = gr.Textbox(
                            label="Live", interactive=False, lines=1
                        )
                    with gr.Tab("Upload"):
                        face_upload = gr.Image(
                            label="Upload face image",
                            sources=["upload"],
                            type="numpy",
                        )
                        detect_face_btn = gr.Button(
                            "Detect Emotion from Face",
                            variant="secondary",
                        )
                    face_results_state = gr.State([])
                    face_crop_state    = gr.State(None)
                    face_last_ts_state = gr.State(None)
                    face_crop_img = gr.Image(
                        label="Detected Face Crop",
                        interactive=False,
                        height=200,
                    )
                    face_info = gr.Textbox(
                        label="Detected Emotion",
                        interactive=False,
                        lines=3,
                    )
                    face_v = gr.Number(value=0.0, visible=False)
                    face_a = gr.Number(value=0.0, visible=False)

                generate_btn = gr.Button(
                    "Generate Emotional Response", variant="primary"
                )

            # ---- Right column: visualization + outputs ----
            with gr.Column(scale=1):
                gr.Markdown("### Valence\u2013Arousal Plane")
                va_plot = gr.Plot(
                    label="Emotion Space",
                    value=create_va_plane_figure(0.0, 0.0),
                )

                gr.Markdown("### Output")
                output_audio = gr.Audio(
                    label="Generated Speech Response",
                    autoplay=True,
                )
                output_text = gr.Textbox(
                    label="Text Response",
                    interactive=False,
                    lines=3,
                )

        # ---- Event wiring ----

        # Mode toggle → show/hide manual vs describe vs face controls
        emotion_mode.change(
            fn=on_mode_change,
            inputs=[emotion_mode],
            outputs=[manual_controls, describe_controls, face_controls],
        )

        # Describe emotion → extract VA from text, update hidden numbers + plot
        describe_btn.click(
            fn=on_describe_emotion,
            inputs=[feeling_text],
            outputs=[describe_v, describe_a, va_plot, describe_info],
        )

        # Face detection: upload tab → single-image predict
        detect_face_btn.click(
            fn=on_detect_face,
            inputs=[face_upload],
            outputs=[face_v, face_a, va_plot, face_info, face_crop_img],
        )

        # Face + audio video tab → extract audio + face emotion on submit
        face_video.change(
            fn=process_face_video,
            inputs=[face_video],
            outputs=[audio_input, face_video_status, face_v, face_a, va_plot, face_info, face_crop_img],
        )

        # Face detection: live webcam tab → accumulate per-frame results
        face_webcam_stream.stream(
            fn=stream_face_frame,
            inputs=[face_webcam_stream, face_results_state, face_crop_state, face_last_ts_state],
            outputs=[face_results_state, face_crop_state, face_last_ts_state, face_live_status],
            stream_every=0.25,
            time_limit=120,
        )

        # Timer → auto-analyze when stream goes quiet for >1.5 s
        face_poll_timer.tick(
            fn=auto_analyze_face,
            inputs=[face_results_state, face_crop_state, face_last_ts_state],
            outputs=[face_results_state, face_crop_state, face_last_ts_state,
                     face_live_status, face_v, face_a, va_plot, face_info, face_crop_img],
        )

        # Slider release → update plot (register first so we can cancel them)
        val_event = valence_slider.release(
            fn=on_slider_change,
            inputs=[valence_slider, arousal_slider],
            outputs=[va_plot],
        )
        aro_event = arousal_slider.release(
            fn=on_slider_change,
            inputs=[valence_slider, arousal_slider],
            outputs=[va_plot],
        )

        # Preset dropdown → update sliders + plot, cancelling any pending slider events
        emotion_preset.change(
            fn=on_emotion_preset_change,
            inputs=[emotion_preset],
            outputs=[valence_slider, arousal_slider, va_plot],
            cancels=[val_event, aro_event],
        )

        # Generate button → use VA from active mode
        generate_btn.click(
            fn=run_inference_with_mode,
            inputs=[
                audio_input, emotion_mode,
                valence_slider, arousal_slider,
                describe_v, describe_a,
                face_v, face_a,
            ],
            outputs=[output_audio, output_text, va_plot],
        )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLM-4-Voice Emotion Demo")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to the fine-tuned LoRA checkpoint directory",
    )
    parser.add_argument(
        "--ssl", action="store_true",
        help="Enable HTTPS with a self-signed cert (needed for mic access in Safari)",
    )
    args = parser.parse_args()

    # Load models globally
    glm_tokenizer, glm_speech_encoder, glm_speech_decoder, glm_model, audio_0_id = (
        load_models(args.checkpoint)
    )

    # Initialise the text-to-VA converter (reuses already-loaded glm_model)
    text_to_va_converter = TextToVAConverter(glm_model, glm_tokenizer)
    print("Text-to-VA converter ready.")

    # Build and launch
    demo = build_ui()
    demo.queue()

    launch_kwargs = dict(server_name=args.host, server_port=args.port, share=True)
    if args.ssl:
        # Generate a self-signed cert for HTTPS (needed for mic access in Safari)
        import subprocess, tempfile
        cert_dir = tempfile.mkdtemp()
        cert_file = os.path.join(cert_dir, "cert.pem")
        key_file = os.path.join(cert_dir, "key.pem")
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", key_file, "-out", cert_file,
            "-days", "1", "-subj", "/CN=localhost",
        ], check=True, capture_output=True)
        launch_kwargs["ssl_certfile"] = cert_file
        launch_kwargs["ssl_keyfile"] = key_file
        launch_kwargs["ssl_verify"] = False  # self-signed cert; skip Gradio's own health-check verification
        print(f"SSL enabled. You may need to accept the self-signed certificate in your browser.")

    demo.launch(**launch_kwargs)
