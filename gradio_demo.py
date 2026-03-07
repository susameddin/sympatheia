"""
Interactive Gradio demo for GLM-4-Voice emotion-conditioned speech-to-speech model.

Records user voice, takes valence/arousal input via sliders or emotion presets,
and generates an emotionally-conditioned speech response.

Usage:
    python gradio_demo.py \
        --checkpoint ~/emo_recog_2025s/Models/GLM-4-Voice/glm-4-voice-finetune/experiments/glm-model-opens2s-qwen3tts-va-text-lora-20260208-001117/checkpoint-1600 \
        --port 7860
"""

import sys
import os

# Ensure ffmpeg is on PATH (may be missing inside Singularity containers)
_ffmpeg_dir = os.path.expanduser("~/.conda/envs/s/bin")
if os.path.isfile(os.path.join(_ffmpeg_dir, "ffmpeg")):
    os.environ["PATH"] = _ffmpeg_dir + ":" + os.environ.get("PATH", "")

# Keep the university proxy for outbound internet (needed for share tunnel),
# but bypass it for localhost so Gradio's local health check works.
os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",localhost,127.0.0.1,0.0.0.0"
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",localhost,127.0.0.1,0.0.0.0"

# Ensure the finetune directory is on the path for src.vocoder and speech_tokenizer imports
FINETUNE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FINETUNE_DIR)

import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gradio as gr
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from src.vocoder import GLM4CodecEncoder, GLM4CodecDecoder
from text_to_va import TextToVAConverter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMOTION_ANCHORS = {
    "sad":        (-0.75, -0.65),
    "excited":    (0.75,  0.90),
    "frustrated": (-0.82, -0.20),
    "neutral":    (0.00,  0.00),
    "happy":      (0.85,  0.35),
    "angry":      (-0.85, 0.85),
    "fear":       (-0.40, 0.65),
    "relaxed":    (0.40, -0.45),
    "surprised":  (0.10,  0.80),
    "disgusted":  (-0.80, 0.35),
    "tired":      (-0.15, -0.75),
}

EMOTION_COLORS = {
    "sad":        (0.30, 0.45, 0.85),   # Steel blue
    "excited":    (1.00, 0.65, 0.00),   # Bright orange
    "frustrated": (0.75, 0.30, 0.30),   # Brick / terracotta
    "neutral":    (0.60, 0.60, 0.60),   # Medium gray
    "happy":      (0.40, 0.80, 0.20),   # Lime green
    "angry":      (0.85, 0.10, 0.10),   # Red
    "fear":       (0.50, 0.10, 0.70),   # Deep purple
    "relaxed":    (0.20, 0.65, 0.55),   # Teal
    "surprised":  (0.95, 0.50, 0.80),   # Pink
    "disgusted":  (0.55, 0.60, 0.20),   # Olive green
    "tired":      (0.50, 0.50, 0.70),   # Muted lavender
}

LABEL_OFFSETS = {
    "sad":        ( 0, -14),
    "excited":    (-10, -14),
    "frustrated": ( 12,  10),
    "neutral":    ( 10,   8),
    "happy":      (  0,  10),
    "angry":      (-10,  10),
    "fear":       ( 14,   4),
    "relaxed":    (  0,  10),
    "surprised":  (  0,  10),
    "disgusted":  ( 14,   4),
    "tired":      (  0, -14),
}

_HEATMAP_SIGMA = 0.35
_HEATMAP_RES   = 200
_HEATMAP_ALPHA = 0.70

DEFAULT_CHECKPOINT = os.path.join(
    FINETUNE_DIR,
    "experiments/glm-model-opens2s-qwen3tts-va-text-lora-20260205-213752/checkpoint-700",
)
SAMPLE_RATE = 22050

# Global model references (set in __main__)
glm_tokenizer = None
glm_speech_encoder = None
glm_speech_decoder = None
glm_model = None
audio_0_id = None
text_to_va_converter = None  # Set in __main__ after models are loaded

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
    """Toggle visibility of manual vs describe controls."""
    is_manual   = (mode == "Select Manually")
    is_describe = (mode == "Describe Your Feeling")
    return (
        gr.update(visible=is_manual),    # manual_controls group
        gr.update(visible=is_describe),  # describe_controls group
    )


def on_describe_emotion(description_text: str):
    """Convert free-text emotion description to VA values and update the plot."""
    if not description_text or not description_text.strip():
        raise gr.Error("Please enter an emotion description first.")

    v, a, info = text_to_va_converter.convert(description_text)
    return float(v), float(a), create_va_plane_figure(v, a), info



def run_inference_with_mode(audio_path, mode, valence, arousal,
                            describe_v, describe_a):
    """Pick VA values based on active mode, then run inference."""
    if mode == "Detect From Audio":
        valence, arousal = None, None
    elif mode == "Describe Your Feeling":
        valence, arousal = describe_v, describe_a
    return run_inference(audio_path, valence, arousal)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    emotion_choices = ["Custom"] + list(EMOTION_ANCHORS.keys())

    with gr.Blocks(title="GLM-4-Voice Emotion Demo") as demo:
        gr.Markdown(
            "# GLM-4-Voice: Speech-to-Speech Emotion Demo\n"
            "Record or upload speech, select a target emotion, "
            "and generate an emotionally-conditioned response."
        )

        with gr.Row():
            # ---- Left column: inputs ----
            with gr.Column(scale=1):
                gr.Markdown("### Input Audio")
                audio_input = gr.Audio(
                    label="Record or upload audio",
                    source="microphone",
                    type="filepath",
                )

                gr.Markdown("### Emotion Control")

                emotion_mode = gr.Radio(
                    choices=["Select Manually", "Detect From Audio", "Describe Your Feeling"],
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

        # Mode toggle → show/hide manual vs describe controls
        emotion_mode.change(
            fn=on_mode_change,
            inputs=[emotion_mode],
            outputs=[manual_controls, describe_controls],
        )

        # Describe emotion → extract VA from text, update hidden numbers + plot
        describe_btn.click(
            fn=on_describe_emotion,
            inputs=[feeling_text],
            outputs=[describe_v, describe_a, va_plot, describe_info],
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
        print(f"SSL enabled. You may need to accept the self-signed certificate in your browser.")

    demo.launch(**launch_kwargs)
