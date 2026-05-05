"""Constants for YAAD ECG emotion module."""
import os

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
_MODULE_DIR = os.path.dirname(__file__)
YAAD_DIR = os.path.normpath(os.path.join(_MODULE_DIR, "../../Datasets/YAAD/ECG_GSR_Emotions"))

ECG_MULTIMODAL_DIR  = os.path.join(YAAD_DIR, "Raw Data", "Multimodal", "ECG")
ECG_SINGLEMODAL_DIR = os.path.join(YAAD_DIR, "Raw Data", "Single Modal", "ECG")
GSR_MULTIMODAL_DIR  = os.path.join(YAAD_DIR, "Raw Data", "Multimodal", "GSR")

ANNOTATION_MULTIMODAL  = os.path.join(YAAD_DIR, "Self-Annotation Labels",
                                       "Self-annotation Multimodal_Use.xlsx")
ANNOTATION_SINGLEMODAL = os.path.join(YAAD_DIR, "Self-Annotation Labels",
                                       "Self-annotation Single Modal_Use.xlsx")

CACHE_DIR = os.path.join(_MODULE_DIR, "cache")

# ---------------------------------------------------------------------------
# Signal parameters
# ---------------------------------------------------------------------------
ECG_SAMPLES = 5000   # samples per trial (full 39-second recording at ~128 Hz)

# ---------------------------------------------------------------------------
# Label space
# ---------------------------------------------------------------------------
EMOTION_NAMES = ["Anger", "Disgust", "Fear", "Happy", "Mixed", "Neutral", "Sad", "Surprise"]
NUM_CLASSES   = len(EMOTION_NAMES)

# Russell circumplex (valence, arousal) ∈ [-1, 1]
EMOTION_TO_VA = {
    "Anger":    (-0.85,  0.85),
    "Disgust":  (-0.82, -0.20),
    "Fear":     (-0.40,  0.65),
    "Happy":    ( 0.85,  0.35),
    "Mixed":    ( 0.00,  0.00),
    "Neutral":  ( 0.00, -0.30),
    "Sad":      (-0.75, -0.65),
    "Surprise": ( 0.10,  0.80),
}

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
# 6 features selected by correlation analysis in the reference notebook
# (correlation with emotion label > 0.1 after LabelEncoder)
SELECTED_FEATURES = ["sdrr", "rmssd", "sd1", "sd2", "skew", "kurt"]
N_FEATURES = len(SELECTED_FEATURES)

# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
TEST_SIZE     = 0.2
RANDOM_STATE  = 42
N_ESTIMATORS  = 100   # Random Forest trees
