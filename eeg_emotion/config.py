"""Constants for the EEG → VA module (DEAP dataset)."""

import os
import random

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
DEAP_DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../Datasets/DEAP/data_preprocessed_python",
)
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

# ---------------------------------------------------------------------------
# DEAP recording parameters
# ---------------------------------------------------------------------------
FS = 128            # sampling frequency (Hz) — DEAP preprocessed
N_EEG_CHANNELS = 32 # EEG channels are indices 0-31 in the 40-channel array
N_TRIALS = 40
N_SUBJECTS = 32

# Label columns in subject['labels']  (40 x 4)
# Official order: valence=0, arousal=1, dominance=2, liking=3
LABEL_COL_VALENCE = 0
LABEL_COL_AROUSAL = 1

# Rescale raw labels [1, 9] → [-1, 1]:  (x - 5) / 4
LABEL_CENTER = 5.0
LABEL_SCALE  = 4.0

# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
WINDOW_SEC     = 4      # seconds
HOP_SEC        = 2      # seconds
WINDOW_SAMPLES = WINDOW_SEC * FS   # 512
HOP_SAMPLES    = HOP_SEC   * FS    # 256

# ---------------------------------------------------------------------------
# Subject splits — shuffled with fixed seed to prevent distribution skew
# from sequential subject ordering in the DEAP release.
# ---------------------------------------------------------------------------
_rng = random.Random(42)
_subjects = list(range(1, 33))
_rng.shuffle(_subjects)
TRAIN_SUBJECTS = _subjects[:26]    # 26 subjects
VAL_SUBJECTS   = _subjects[26:29]  # 3 subjects
TEST_SUBJECTS  = _subjects[29:]    # 3 subjects

# ---------------------------------------------------------------------------
# Model hyperparameters  (EEGNet)
# ---------------------------------------------------------------------------
F1            = 16
D             = 2
F2            = 32
DROPOUT       = 0.5

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE   = 256
EPOCHS       = 80
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 15
SEED         = 42
DEVICE       = "cuda"

# Loss: alpha * CCC_loss + (1 - alpha) * MSE_loss
# Pure CCC (alpha=1) collapses to constant predictions early in training
# because its gradient is ~0 when prediction variance is near 0.
# Mixing in MSE keeps predictions spread out and stabilises training.
LOSS_ALPHA = 0.5

# ---------------------------------------------------------------------------
# DE (Differential Entropy) feature extraction
# Five canonical EEG frequency bands used in emotion recognition (SEED/DEAP).
# ---------------------------------------------------------------------------
BANDS = {
    "delta": (1,  4),
    "theta": (4,  8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}
N_BANDS = 5
DE_DIM  = N_BANDS * N_EEG_CHANNELS   # 5 * 32 = 160

# Within-subject trial split
WITHIN_SUBJECT_TRAIN_FRAC = 0.8   # 32 of 40 trials for training, 8 for test
WITHIN_SUBJECT_SEED       = 42

# DE model hyperparameters (MLP regression head)
DE_HIDDEN_1 = 256
DE_HIDDEN_2 = 128
DE_HIDDEN_3 = 64
DE_DROPOUT  = 0.3

# Within-subject training hyperparameters
# Smaller per-subject dataset → smaller batch, more epochs, lower LR
WS_BATCH_SIZE   = 64
WS_EPOCHS       = 150
WS_LR           = 5e-4
WS_WEIGHT_DECAY = 1e-4
WS_PATIENCE     = 20
