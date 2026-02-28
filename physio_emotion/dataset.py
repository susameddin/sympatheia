"""DEAP physiological (BVP + GSR) dataset for continuous valence-arousal regression."""

import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import (
    CH_BVP,
    CH_GSR,
    DEAP_DATA_DIR,
    HOP_SAMPLES,
    LABEL_CENTER,
    LABEL_COL_AROUSAL,
    LABEL_COL_VALENCE,
    LABEL_SCALE,
    PHYSIO_CHANNELS,
    N_PHYSIO_CH,
    WINDOW_SAMPLES,
    WITHIN_SUBJECT_TRAIN_FRAC,
    WITHIN_SUBJECT_SEED,
)


def load_subject_physio(data_dir: str, subject_id: int) -> tuple:
    """Load all PHYSIO_CHANNELS from one DEAP subject file.

    Returns:
        signals : (40, N_PHYSIO_CH, T) float32 — z-score normalised per channel
        va      : (40, 2)              float32 — [valence, arousal] in [-1, 1]
    """
    path = os.path.join(data_dir, f"s{subject_id:02d}.dat")
    with open(path, "rb") as f:
        subject = pickle.load(f, encoding="latin1")

    data   = subject["data"]    # (40, 40, T)
    labels = subject["labels"]  # (40, 4)

    # Extract all peripheral channels: (40, N_PHYSIO_CH, T)
    signals = data[:, PHYSIO_CHANNELS, :].astype(np.float32)

    # Z-score each channel independently across all trials of this subject
    for ch in range(N_PHYSIO_CH):
        flat = signals[:, ch, :].ravel()
        m    = flat.mean()
        s    = flat.std() + 1e-8
        signals[:, ch, :] = (signals[:, ch, :] - m) / s

    va = labels[:, [LABEL_COL_VALENCE, LABEL_COL_AROUSAL]].astype(np.float32)
    va = (va - LABEL_CENTER) / LABEL_SCALE   # [1,9] → [-1,1]

    return signals, va


class DEAPPhysioDataset(Dataset):
    """Windowed DEAP physiological dataset for VA regression.

    Uses all PHYSIO_CHANNELS (BVP, GSR, Respiration, Temperature, zEMG, tEMG).

    Each item:
        signals : float32 tensor (N_PHYSIO_CH, WINDOW_SAMPLES) — all 6 channels
        y       : float32 tensor (2,)                          — [valence, arousal] in [-1, 1]
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

        self.signals_list: list[np.ndarray] = []   # each (N_PHYSIO_CH, W)
        self.y_list:       list[np.ndarray] = []

        for sid in subject_ids:
            signals, va = load_subject_physio(data_dir, sid)   # (40, N_PHYSIO_CH, T), (40, 2)
            T = signals.shape[2]
            for trial in range(signals.shape[0]):
                start = 0
                while start + window_samples <= T:
                    self.signals_list.append(signals[trial, :, start : start + window_samples])
                    self.y_list.append(va[trial])
                    start += hop_samples

    def __len__(self) -> int:
        return len(self.y_list)

    def __getitem__(self, idx: int):
        sig = self.signals_list[idx].copy()   # (N_PHYSIO_CH, W)
        y   = self.y_list[idx].copy()         # (2,)

        if self.augment:
            if np.random.rand() < 0.5:
                sig = sig + 0.01 * np.random.randn(*sig.shape).astype(np.float32)

        return (
            torch.from_numpy(sig),
            torch.from_numpy(y),
        )


# ---------------------------------------------------------------------------
# Within-subject physio dataset
# ---------------------------------------------------------------------------

class DEAPPhysioWithinSubjectDataset(Dataset):
    """Within-subject windowed physiological dataset for a single DEAP subject.

    Mirrors DEAPEEGWithinSubjectDataset but for 6-channel physiological signals.
    Splits the 40 trials into train (80%) / test (20%) by trial index.

    Each item:
        signals : float32 tensor (N_PHYSIO_CH, WINDOW_SAMPLES) — 6 channels
        y       : float32 tensor (2,)                          — [valence, arousal]
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

        signals, va  = load_subject_physio(data_dir, subject_id)   # (40, N_PHYSIO_CH, T), (40, 2)
        n_trials     = signals.shape[0]
        T            = signals.shape[2]

        if split == "all":
            selected = np.arange(n_trials)
        else:
            rng       = np.random.RandomState(seed + subject_id)
            trial_idx = np.arange(n_trials)
            rng.shuffle(trial_idx)

            n_train  = int(round(n_trials * train_frac))   # 32
            selected = trial_idx[:n_train] if split == "train" else trial_idx[n_train:]

        self.signals_list: list[np.ndarray] = []
        self.y_list:       list[np.ndarray] = []

        for trial in selected:
            start = 0
            while start + window_samples <= T:
                self.signals_list.append(signals[trial, :, start : start + window_samples])
                self.y_list.append(va[trial])
                start += hop_samples

    def __len__(self) -> int:
        return len(self.y_list)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.signals_list[idx].copy()),
            torch.from_numpy(self.y_list[idx].copy()),
        )
