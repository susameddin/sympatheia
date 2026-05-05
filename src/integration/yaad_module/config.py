"""Constants for the YAAD ECG/GSR → VA module."""
import os

_MODULE_DIR = os.path.dirname(__file__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CACHE_DIR   = os.path.join(_MODULE_DIR, "cache")

# ECG model — already trained by ecg_emotion.train
ECG_WEIGHTS = os.path.join(_MODULE_DIR, "ecg_emotion", "cache", "ecg_resnet_deep_model.pt")
# Test indices saved by ecg_emotion.train and yaad_module.train respectively
ECG_TEST_INDICES = os.path.join(_MODULE_DIR, "ecg_emotion", "cache", "ecg_test_indices.npy")
# GSR model — trained by this module's train.py
GSR_WEIGHTS = os.path.join(CACHE_DIR, "gsr_resnet_deep_model.pt")
GSR_TEST_INDICES = os.path.join(CACHE_DIR, "gsr_test_indices.npy")

# ---------------------------------------------------------------------------
# Signal / label constants (mirror ecg_emotion.config)
# ---------------------------------------------------------------------------
ECG_SAMPLES   = 5000
NUM_CLASSES   = 8
EMOTION_NAMES = ["Anger", "Disgust", "Fear", "Happy", "Mixed", "Neutral", "Sad", "Surprise"]

# ---------------------------------------------------------------------------
# YAAD emotion → speech model anchor name
# ---------------------------------------------------------------------------
YAAD_TO_SPEECH = {
    "Anger":    "Angry",
    "Disgust":  "Disgusted",
    "Fear":     "Anxious",
    "Happy":    "Happy",
    "Mixed":    "Neutral",   # no perfect match; neutral is the closest anchor
    "Neutral":  "Neutral",
    "Sad":      "Sad",
    "Surprise": "Surprised",
}
