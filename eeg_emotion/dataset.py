"""DEAP EEG dataset for continuous valence-arousal regression."""

import os
import pickle

import numpy as np
import torch
from scipy.signal import butter, sosfilt
from torch.utils.data import Dataset

from .config import (
    DEAP_DATA_DIR,
    FS,
    N_EEG_CHANNELS,
    LABEL_COL_VALENCE,
    LABEL_COL_AROUSAL,
    LABEL_CENTER,
    LABEL_SCALE,
    WINDOW_SAMPLES,
    HOP_SAMPLES,
    BANDS,
    DE_DIM,
    WITHIN_SUBJECT_TRAIN_FRAC,
    WITHIN_SUBJECT_SEED,
)


def load_subject(data_dir: str, subject_id: int) -> tuple:
    """Load one DEAP subject file.

    Returns:
        eeg    : (40, 32, 8064)  float32 — EEG channels, z-score normalised per channel
        va     : (40, 2)         float32 — [valence, arousal] rescaled to [-1, 1]
    """
    path = os.path.join(data_dir, f"s{subject_id:02d}.dat")
    with open(path, "rb") as f:
        subject = pickle.load(f, encoding="latin1")

    data   = subject["data"]    # (40, 40, 8064) — 40 channels total
    labels = subject["labels"]  # (40, 4)

    # Extract EEG channels 0–31
    eeg = data[:, :N_EEG_CHANNELS, :].astype(np.float32)  # (40, 32, 8064)

    # Z-score per channel across all trials of this subject
    flat = eeg.reshape(N_EEG_CHANNELS, -1)   # (32, 40*8064)
    mean = flat.mean(axis=1)                  # (32,)
    std  = flat.std(axis=1) + 1e-8
    eeg  = (eeg - mean[np.newaxis, :, np.newaxis]) / std[np.newaxis, :, np.newaxis]

    # Rescale labels: [1, 9] → [-1, 1]
    va = labels[:, [LABEL_COL_VALENCE, LABEL_COL_AROUSAL]].astype(np.float32)
    va = (va - LABEL_CENTER) / LABEL_SCALE

    return eeg, va


class DEAPEEGDataset(Dataset):
    """Windowed DEAP EEG dataset for VA regression.

    Each item:
        x : float32 tensor (1, 32, WINDOW_SAMPLES)  — ready for EEGNet  (B, 1, C, T)
        y : float32 tensor (2,)                      — [valence, arousal] in [-1, 1]
    """

    def __init__(
        self,
        subject_ids: list,
        data_dir: str = DEAP_DATA_DIR,
        window_samples: int = WINDOW_SAMPLES,
        hop_samples: int = HOP_SAMPLES,
        augment: bool = False,
    ):
        self.window_samples = window_samples
        self.hop_samples    = hop_samples
        self.augment        = augment

        # Pre-load all windows into memory
        self.x_list: list[np.ndarray] = []   # each (1, 32, W)
        self.y_list: list[np.ndarray] = []   # each (2,)

        for sid in subject_ids:
            eeg, va = load_subject(data_dir, sid)   # (40, 32, T), (40, 2)
            T = eeg.shape[2]
            for trial in range(eeg.shape[0]):
                start = 0
                while start + window_samples <= T:
                    window = eeg[trial, :, start : start + window_samples]  # (32, W)
                    self.x_list.append(window[np.newaxis])                   # (1, 32, W)
                    self.y_list.append(va[trial])
                    start += hop_samples

    def __len__(self) -> int:
        return len(self.x_list)

    def __getitem__(self, idx: int):
        x = self.x_list[idx].copy()   # (1, 32, W)
        y = self.y_list[idx].copy()   # (2,)

        if self.augment:
            # Gaussian noise
            if np.random.rand() < 0.5:
                x = x + 0.01 * np.random.randn(*x.shape).astype(np.float32)
            # Channel dropout (zero out a random channel)
            if np.random.rand() < 0.3:
                ch = np.random.randint(0, x.shape[1])
                x[0, ch, :] = 0.0

        return torch.from_numpy(x), torch.from_numpy(y)


# ---------------------------------------------------------------------------
# Differential Entropy (DE) feature extraction
# ---------------------------------------------------------------------------

def extract_de_features(eeg_window: np.ndarray, fs: int = FS) -> np.ndarray:
    """Compute Differential Entropy features from a single EEG window.

    For each of 5 frequency bands (delta, theta, alpha, beta, gamma), the
    window is bandpass-filtered with a 4th-order Butterworth filter (SOS form
    for numerical stability), then log(variance + 1e-8) is computed per
    channel.  This is DE = log(σ²), the standard feature used in SEED/DEAP
    emotion recognition literature.

    Args:
        eeg_window : (N_EEG_CHANNELS, WINDOW_SAMPLES) float32 — z-score normalised.
        fs         : Sampling frequency in Hz.

    Returns:
        de : (DE_DIM,) = (N_BANDS * N_EEG_CHANNELS,) = (160,) float32.
             Band order: delta, theta, alpha, beta, gamma.
    """
    nyq = fs / 2.0
    de_list = []
    for lo, hi in BANDS.values():
        sos = butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
        filtered = sosfilt(sos, eeg_window, axis=1)          # (C, T)
        band_de  = np.log(np.var(filtered, axis=1) + 1e-8)  # (C,)
        de_list.append(band_de)
    return np.concatenate(de_list).astype(np.float32)        # (160,)


# ---------------------------------------------------------------------------
# Within-subject DE dataset
# ---------------------------------------------------------------------------

class DEAPEEGWithinSubjectDataset(Dataset):
    """Within-subject windowed DE-feature dataset for a single DEAP subject.

    Splits the 40 trials of one subject into train (80%) and test (20%) by
    trial index, preventing any temporal leakage between the two splits.
    DE features are pre-computed at construction time for efficiency.

    Each item:
        x : float32 tensor (DE_DIM,) = (160,)  — DE feature vector
        y : float32 tensor (2,)                 — [valence, arousal] in [-1, 1]

    Args:
        subject_id    : int in [1, 32]
        split         : "train" or "test"
        data_dir      : path to DEAP preprocessed pickle files
        window_samples: EEG samples per window (default 512 = 4 s × 128 Hz)
        hop_samples   : hop size in samples (default 256 = 2 s × 128 Hz)
        train_frac    : fraction of trials used for training (0.8 → 32/40)
        seed          : base RNG seed; actual seed is seed + subject_id so
                        each subject gets a different but reproducible shuffle
    """

    def __init__(
        self,
        subject_id:     int,
        split:          str,
        data_dir:       str   = DEAP_DATA_DIR,
        window_samples: int   = WINDOW_SAMPLES,
        hop_samples:    int   = HOP_SAMPLES,
        train_frac:     float = WITHIN_SUBJECT_TRAIN_FRAC,
        seed:           int   = WITHIN_SUBJECT_SEED,
    ):
        assert split in ("train", "test", "all"), \
            f"split must be 'train', 'test', or 'all', got {split!r}"

        eeg, va  = load_subject(data_dir, subject_id)   # (40, 32, T), (40, 2)
        n_trials = eeg.shape[0]
        T        = eeg.shape[2]

        if split == "all":
            # Use all trials — for training the final deployed model
            selected = np.arange(n_trials)
        else:
            # Reproducible per-subject trial shuffle (seed varies per subject)
            rng       = np.random.RandomState(seed + subject_id)
            trial_idx = np.arange(n_trials)
            rng.shuffle(trial_idx)

            n_train  = int(round(n_trials * train_frac))   # 32
            selected = trial_idx[:n_train] if split == "train" else trial_idx[n_train:]

        self.x_list: list[np.ndarray] = []
        self.y_list: list[np.ndarray] = []

        for trial in selected:
            start = 0
            while start + window_samples <= T:
                window = eeg[trial, :, start : start + window_samples]  # (32, W)
                de     = extract_de_features(window)                     # (160,)
                self.x_list.append(de)
                self.y_list.append(va[trial])
                start += hop_samples

    def __len__(self) -> int:
        return len(self.x_list)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.x_list[idx].copy()),
            torch.from_numpy(self.y_list[idx].copy()),
        )
