"""Constants for the EEG → emotion module (SEED-VII dataset)."""

import os

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
SEED_DATA_DIR = os.environ.get("SEED_VII_DIR", "/path/to/SEED-VII")
SEED_FEATURES_DIR = os.path.join(SEED_DATA_DIR, "EEG_features")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

# ---------------------------------------------------------------------------
# EEG parameters
# ---------------------------------------------------------------------------
N_EEG_CHANNELS = 62
N_BANDS = 5
DE_DIM = N_BANDS * N_EEG_CHANNELS  # 310

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
N_SUBJECTS = 20
N_VIDEOS = 80
N_SESSIONS = 4
VIDEOS_PER_SESSION = 20
NUM_CLASSES = 7

# Session → video ranges (1-indexed, inclusive)
SESSION_VIDEOS = {
    1: list(range(1, 21)),    # videos 1-20
    2: list(range(21, 41)),   # videos 21-40
    3: list(range(41, 61)),   # videos 41-60
    4: list(range(61, 81)),   # videos 61-80
}

# ---------------------------------------------------------------------------
# Emotion definitions (alphabetical, 0-indexed class labels)
# ---------------------------------------------------------------------------
EMOTION_NAMES = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Speech model VA anchors — identical to face_emotion's _FACE_SPEECH_VA (minus Contempt)
# so all modalities map into the same VA space for sympatheia integration.
EMOTION_TO_VA = {
    "Anger":    (-0.85,  0.85),
    "Disgust":  (-0.82, -0.20),
    "Fear":     (-0.40,  0.65),
    "Happy":    ( 0.85,  0.35),
    "Neutral":  ( 0.00,  0.00),
    "Sad":      (-0.75, -0.65),
    "Surprise": ( 0.10,  0.80),
}

# ---------------------------------------------------------------------------
# Video → emotion mapping (from emotion_label_and_stimuli_order.xlsx)
# Same for all 20 subjects.  Keys are 1-indexed video IDs.
# ---------------------------------------------------------------------------
VIDEO_EMOTIONS = {
    # Session 1 (videos 1-20)
    1: "Happy", 2: "Neutral", 3: "Disgust", 4: "Sad", 5: "Anger",
    6: "Anger", 7: "Sad", 8: "Disgust", 9: "Neutral", 10: "Happy",
    11: "Happy", 12: "Neutral", 13: "Disgust", 14: "Sad", 15: "Anger",
    16: "Anger", 17: "Sad", 18: "Disgust", 19: "Neutral", 20: "Happy",
    # Session 2 (videos 21-40)
    21: "Anger", 22: "Sad", 23: "Fear", 24: "Neutral", 25: "Surprise",
    26: "Surprise", 27: "Neutral", 28: "Fear", 29: "Sad", 30: "Anger",
    31: "Anger", 32: "Sad", 33: "Fear", 34: "Neutral", 35: "Surprise",
    36: "Surprise", 37: "Neutral", 38: "Fear", 39: "Sad", 40: "Anger",
    # Session 3 (videos 41-60)
    41: "Happy", 42: "Surprise", 43: "Disgust", 44: "Fear", 45: "Anger",
    46: "Anger", 47: "Fear", 48: "Disgust", 49: "Surprise", 50: "Happy",
    51: "Happy", 52: "Surprise", 53: "Disgust", 54: "Fear", 55: "Anger",
    56: "Anger", 57: "Fear", 58: "Disgust", 59: "Surprise", 60: "Happy",
    # Session 4 (videos 61-80)
    61: "Disgust", 62: "Sad", 63: "Fear", 64: "Surprise", 65: "Happy",
    66: "Happy", 67: "Surprise", 68: "Fear", 69: "Sad", 70: "Disgust",
    71: "Disgust", 72: "Sad", 73: "Fear", 74: "Surprise", 75: "Happy",
    76: "Happy", 77: "Surprise", 78: "Fear", 79: "Sad", 80: "Disgust",
}

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15
DROPOUT = 0.3
SEED = 42
DEVICE = "cuda"

# Within-subject training
WS_BATCH_SIZE = 64
WS_EPOCHS = 100
WS_LR = 3e-4   # paper tunes in {3e-5, 1e-4, 3e-4}
WS_WEIGHT_DECAY = 1e-4
WS_PATIENCE = 20
WS_VAL_FRAC = 0.125   # fraction of videos held out per class for val (early stopping)
LABEL_SMOOTHING = 0.1

# Model architecture
HIDDEN_1 = 256
HIDDEN_2 = 128
HIDDEN_3 = 64

# DGCNN architecture (via torcheeg, Song et al. 2020)
DGCNN_HID_CHANNELS = 32   # hidden dim per graph conv layer
DGCNN_N_LAYERS     = 2    # number of dynamic graph conv layers

# MAET architecture (Jiang et al., IEEE TAffComp 2025)
# La=2 adaptive transformer blocks + Lm=1 mixture transformer block = depth 3
MAET_EMBED_DIM      = 32
MAET_DEPTH          = 3
MAET_NUM_HEADS      = 4
MAET_EEG_SEQ_LEN    = 5    # number of multi-view embeddings (v)
MAET_DROP_RATE      = 0.0
MAET_ATTN_DROP_RATE = 0.0
MAET_DROP_PATH_RATE = 0.1
MAET_MIX_START      = 2    # mixffn_start_layer_index: blocks 0,1 = adaptive, block 2 = mixture
