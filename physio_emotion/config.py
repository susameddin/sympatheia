"""Constants for the physiological signals → VA module (DEAP dataset)."""

import os
import random

# ---------------------------------------------------------------------------
# Dataset paths  (same DEAP files as eeg_emotion)
# ---------------------------------------------------------------------------
DEAP_DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../Datasets/DEAP/data_preprocessed_python",
)
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

# ---------------------------------------------------------------------------
# DEAP channel indices for peripheral signals
#   Full 40-channel layout (0-indexed):
#     0-31  : EEG
#     32    : hEOG
#     33    : vEOG
#     34    : zEMG (zygomaticus)
#     35    : tEMG (trapezius)
#     36    : GSR  (galvanic skin response / EDA)
#     37    : Respiration
#     38    : BVP  (blood volume pulse / PPG — heart activity proxy)
#     39    : Temperature
# ---------------------------------------------------------------------------
CH_BVP  = 38   # blood volume pulse (ECG proxy)
CH_GSR  = 36   # EDA / skin conductance
CH_RESP = 37   # respiration belt
CH_TEMP = 39   # peripheral temperature
CH_zEMG = 34   # zygomaticus major EMG
CH_tEMG = 35   # trapezius EMG

# All channels used by the multi-channel model (order matters for stream indexing)
PHYSIO_CHANNELS = [CH_BVP, CH_GSR, CH_RESP, CH_TEMP, CH_zEMG, CH_tEMG]
N_PHYSIO_CH     = len(PHYSIO_CHANNELS)   # 6

FS = 128  # Hz

# Label columns in subject['labels']
LABEL_COL_VALENCE = 0
LABEL_COL_AROUSAL = 1
LABEL_CENTER      = 5.0
LABEL_SCALE       = 4.0

# ---------------------------------------------------------------------------
# Windowing  (same as eeg_emotion for consistency)
# ---------------------------------------------------------------------------
WINDOW_SEC     = 4
HOP_SEC        = 2
WINDOW_SAMPLES = WINDOW_SEC * FS   # 512
HOP_SAMPLES    = HOP_SEC   * FS    # 256

# ---------------------------------------------------------------------------
# Subject splits  (same as eeg_emotion)
# ---------------------------------------------------------------------------
_rng = random.Random(42)
_subjects = list(range(1, 33))
_rng.shuffle(_subjects)
TRAIN_SUBJECTS = _subjects[:26]
VAL_SUBJECTS   = _subjects[26:29]
TEST_SUBJECTS  = _subjects[29:]

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
CNN_CHANNELS = [1, 32, 64, 128]   # per-stream 1-D CNN channel progression
CNN_KERNELS  = [9, 5, 3]          # kernel sizes for the 3 conv layers
FUSION_HIDDEN = 64
DROPOUT       = 0.5
ATTN_HIDDEN   = 32   # hidden size in per-stream attention network

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

# Loss mixing weight (see eeg_emotion/config.py for rationale)
LOSS_ALPHA = 0.5

# Within-subject trial split (mirrors eeg_emotion settings)
N_SUBJECTS                = 32
WITHIN_SUBJECT_TRAIN_FRAC = 0.8   # 32 of 40 trials for training, 8 for test
WITHIN_SUBJECT_SEED       = 42

# Within-subject training hyperparameters
WS_BATCH_SIZE   = 64
WS_EPOCHS       = 150
WS_LR           = 5e-4
WS_WEIGHT_DECAY = 1e-4
WS_PATIENCE     = 20
