"""Constants for the audio emotion recognition module."""

# Canonical emotion-to-VA mapping (from dataset_creation/convert_qwen3tts_to_glm4voice_11emo.py)
EMOTION_VA_MAPPING = {
    "Sad": (-0.75, -0.65),
    "Excited": (0.75, 0.90),
    "Frustrated": (-0.82, -0.20),
    "Neutral": (0.00, 0.00),
    "Happy": (0.85, 0.35),
    "Angry": (-0.85, 0.85),
    "Fear": (-0.40, 0.65),
    "Relaxed": (0.40, -0.45),
    "Surprised": (0.10, 0.80),
    "Disgusted": (-0.80, 0.35),
    "Tired": (-0.15, -0.75),
}

ALL_EMOTIONS = [
    "Sad", "Excited", "Frustrated", "Neutral", "Happy",
    "Angry", "Fear", "Relaxed", "Surprised", "Disgusted", "Tired",
]

AUDEERING_MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
SAMPLE_RATE = 16000
