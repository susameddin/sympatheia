"""Constants for the Face → VA module (AffectNet dataset)."""

import os

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
DATASET_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../Datasets/AffectNet_Balanced",
)
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

# ---------------------------------------------------------------------------
# Emotion labels (alphabetical — matches ImageFolder class ordering)
# ---------------------------------------------------------------------------
NUM_CLASSES = 8
EMOTION_NAMES = [
    "Anger", "Contempt", "Disgust", "Fear",
    "Happy", "Neutral", "Sad", "Surprise",
]

# Russell's circumplex model: emotion → (valence, arousal) in [-1, 1]
EMOTION_TO_VA = {
    "Anger":    (-0.60, +0.70),
    "Contempt": (-0.40, +0.20),
    "Disgust":  (-0.70, +0.30),
    "Fear":     (-0.50, +0.70),
    "Happy":    (+0.80, +0.50),
    "Neutral":  ( 0.00,  0.00),
    "Sad":      (-0.60, -0.30),
    "Surprise": (+0.40, +0.70),
}

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
INPUT_SIZE = 96  # resize 75x75 → 96x96 for ResNet18
MEAN = [0.485, 0.456, 0.406]  # ImageNet stats
STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 10
DROPOUT = 0.3
SEED = 42
DEVICE = "cuda"
