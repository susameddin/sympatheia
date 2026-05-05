"""Constants for the Face → VA module (AffectNet+ dataset)."""

import os

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
DATASET_DIR = os.environ.get("AFFECTNET_DIR", "/path/to/AffectNet+/human_annotated")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

# ---------------------------------------------------------------------------
# Emotion labels — 8 classes, alphabetical (Contempt kept separate)
# ---------------------------------------------------------------------------
NUM_CLASSES = 8
EMOTION_NAMES = [
    "Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise",
]

# AffectNet+ human-label → model class index
# Standard AffectNet: 0=Neutral,1=Happy,2=Sad,3=Surprise,4=Fear,5=Disgust,6=Anger,7=Contempt
AFFECTNET_LABEL_MAP = {
    0: 5,  # Neutral  → idx 5
    1: 4,  # Happy    → idx 4
    2: 6,  # Sad      → idx 6
    3: 7,  # Surprise → idx 7
    4: 3,  # Fear     → idx 3
    5: 2,  # Disgust  → idx 2
    6: 0,  # Anger    → idx 0
    7: 1,  # Contempt → idx 1
}

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
INPUT_SIZE = 224  # AffectNet+ native resolution
MEAN = [0.485, 0.456, 0.406]  # ImageNet stats
STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 48
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 10
DROPOUT = 0.5
SEED = 42
DEVICE = "cuda"

# ---------------------------------------------------------------------------
# Regularization / augmentation flags
# ---------------------------------------------------------------------------
LABEL_SMOOTHING = 0.1   # 0.0 = disabled
USE_SAMPLER     = False  # WeightedRandomSampler (replaces loss weighting when True)
USE_MIXUP       = False  # MixUp in training loop
MIXUP_ALPHA     = 0.4   # Beta distribution alpha for MixUp
BACKBONE        = "efficientnet_b0"  # "resnet18" | "resnet50" | "efficientnet_b0"

# ---------------------------------------------------------------------------
# HSEmotion inference model (VGGFace2 → AffectNet pretrained)
# ---------------------------------------------------------------------------
# Options: "enet_b0_8_va_mtl"  (8 emotions + VA output, best for this project)
#          "enet_b0_8_best_afew"  (best on acted video, no VA output)
#          "enet_b0_8_best_vgaf"  (best on in-the-wild video, no VA output)
#          "enet_b2_8"  (larger EfficientNet-B2, no VA output)
HSEMOTION_MODEL = "enet_b0_8_va_mtl"
