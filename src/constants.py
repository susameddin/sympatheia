"""Shared emotion constants used across evaluation and dataset scripts."""

EMOTION_VA_MAPPING = {
    "Sad":        (-0.75, -0.65),
    "Excited":    ( 0.75,  0.90),
    "Frustrated": (-0.80,  0.35),
    "Neutral":    ( 0.00,  0.00),
    "Happy":      ( 0.85,  0.35),
    "Angry":      (-0.85,  0.85),
    "Anxious":    (-0.40,  0.65),
    "Relaxed":    ( 0.25, -0.60),
    "Surprised":  ( 0.10,  0.80),
    "Disgusted":  (-0.82, -0.20),
    "Tired":      (-0.15, -0.75),
    "Content":    ( 0.60, -0.20),
}

ALL_EMOTIONS = list(EMOTION_VA_MAPPING.keys())

# Lowercase-keyed alias used by integration modules (face, seed, text, yaad)
SPEECH_ANCHORS = {k.lower(): v for k, v in EMOTION_VA_MAPPING.items()}

# Face model (7 classes) → speech anchor name mapping
_FACE_TO_SPEECH = {
    "Anger":    "angry",
    "Disgust":  "disgusted",
    "Fear":     "anxious",
    "Happy":    "happy",
    "Neutral":  "neutral",
    "Sad":      "sad",
    "Surprise": "surprised",
}
