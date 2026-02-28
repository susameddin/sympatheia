"""EEGNet and DE-MLP models with VA regression heads + predictor wrappers."""

import os

import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

from .config import (
    CACHE_DIR, LOSS_ALPHA, N_EEG_CHANNELS, WINDOW_SAMPLES, HOP_SAMPLES,
    F1, D, F2, DROPOUT,
    DE_DIM, DE_HIDDEN_1, DE_HIDDEN_2, DE_HIDDEN_3, DE_DROPOUT, BANDS, FS,
)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def ccc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Concordance Correlation Coefficient loss: 1 - mean(CCC) over output dims.

    Args:
        pred   : (B, 2)  — predicted [valence, arousal]
        target : (B, 2)  — ground-truth [valence, arousal]
    Returns:
        scalar loss in [0, 2] (0 = perfect prediction)
    """
    pred_mean   = pred.mean(0)
    target_mean = target.mean(0)
    pred_var    = pred.var(0,  unbiased=False)
    target_var  = target.var(0, unbiased=False)
    cov = ((pred - pred_mean) * (target - target_mean)).mean(0)
    ccc = 2.0 * cov / (pred_var + target_var + (pred_mean - target_mean) ** 2 + 1e-8)
    return (1.0 - ccc).mean()


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = LOSS_ALPHA,
) -> torch.Tensor:
    """alpha * CCC_loss + (1 - alpha) * MSE_loss.

    Pure CCC loss has a degenerate fixed point: when prediction variance → 0
    the gradient also → 0, so the model gets stuck predicting a constant.
    The MSE term keeps the prediction variance alive throughout training.
    """
    return alpha * ccc_loss(pred, target) + (1.0 - alpha) * F.mse_loss(pred, target)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class EEGNetVA(nn.Module):
    """EEGNet backbone (Lawhern et al., 2018) with a continuous VA regression head.

    Input:  (B, 1, C, T)  — C EEG channels, T time samples
    Output: (B, 2)        — [valence, arousal] in (-1, 1) via Tanh

    This is EEGNet v4 with the classification head replaced by a 2-output
    regression head trained with CCC loss.
    """

    def __init__(
        self,
        chans: int = N_EEG_CHANNELS,
        samples: int = WINDOW_SAMPLES,
        F1: int = F1,
        D: int = D,
        F2: int = F2,
        kernel_length: int = 64,
        pool1: int = 4,
        pool2: int = 8,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        # Block 1 — Temporal convolution
        self.conv1 = nn.Conv2d(
            1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise spatial convolution
        self.depthwise = nn.Conv2d(F1, F1 * D, (chans, 1), groups=F1, bias=False)
        self.bn2       = nn.BatchNorm2d(F1 * D)
        self.act2      = nn.ELU(inplace=True)
        self.pool2     = nn.AvgPool2d((1, pool1))
        self.drop2     = nn.Dropout(dropout)

        # Separable convolution (depthwise temporal + pointwise)
        self.sep_depth = nn.Conv2d(
            F1 * D, F1 * D, (1, 16), groups=F1 * D, padding=(0, 8), bias=False
        )
        self.sep_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3       = nn.BatchNorm2d(F2)
        self.act3      = nn.ELU(inplace=True)
        self.pool3     = nn.AvgPool2d((1, pool2))
        self.drop3     = nn.Dropout(dropout)

        # Compute flat feature size dynamically
        with torch.no_grad():
            flat_size = self._forward_features(
                torch.zeros(1, 1, chans, samples)
            ).shape[1]

        # VA regression head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(flat_size, 2),
            nn.Tanh(),
        )

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.conv1(x))
        x = self.drop2(self.pool2(self.act2(self.bn2(self.depthwise(x)))))
        x = self.drop3(
            self.pool3(self.act3(self.bn3(self.sep_point(self.sep_depth(x)))))
        )
        return torch.flatten(x, start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._forward_features(x))


# ---------------------------------------------------------------------------
# DE-feature MLP model (within-subject)
# ---------------------------------------------------------------------------

class EEGDEModel(nn.Module):
    """MLP regressor operating on 160-dim Differential Entropy feature vectors.

    Architecture following SEED-IV / DEAP literature convention:
        FC(160→256) → BN → ELU → Dropout(0.3)
        FC(256→128) → BN → ELU → Dropout(0.3)
        FC(128→64)  → BN → ELU
        FC(64→2)    → Tanh

    Input:  (B, DE_DIM) = (B, 160)  — 5 bands × 32 EEG channels
    Output: (B, 2)                   — [valence, arousal] in (-1, 1)

    ~83 K trainable parameters; trains in <30 s per subject on a GPU.
    """

    def __init__(
        self,
        in_dim:  int   = DE_DIM,       # 160
        hidden1: int   = DE_HIDDEN_1,  # 256
        hidden2: int   = DE_HIDDEN_2,  # 128
        hidden3: int   = DE_HIDDEN_3,  # 64
        dropout: float = DE_DROPOUT,   # 0.3
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,  hidden1), nn.BatchNorm1d(hidden1), nn.ELU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2), nn.BatchNorm1d(hidden2), nn.ELU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden2, hidden3), nn.BatchNorm1d(hidden3), nn.ELU(inplace=True),
            nn.Linear(hidden3, 2),       nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, 160). Returns: (B, 2)."""
        return self.net(x)


# ---------------------------------------------------------------------------
# Predictor (inference wrapper)
# ---------------------------------------------------------------------------

class EEGVAPredictor:
    """Load trained EEG models and predict (valence, arousal) from EEG data.

    Supports two modes:

    1. Cross-subject mode (original behaviour, backward compatible):
       Loads EEGNetVA checkpoint (eeg_va_deap.pt) trained on 26 subjects.
       Call: predictor.predict_va(eeg_array)

    2. Within-subject mode (new, higher accuracy):
       Loads the per-subject EEGDEModel checkpoint (eeg_de_s{sid:02d}.pt)
       trained on that subject's own trials with DE features.
       Call: predictor.predict_va(eeg_array, subject_id=5)

    If a within-subject checkpoint is requested but not found on disk, the
    predictor transparently falls back to the cross-subject model.
    """

    def __init__(
        self,
        weights_path:   str = os.path.join(CACHE_DIR, "eeg_va_deap.pt"),
        de_weights_dir: str = CACHE_DIR,
        device:         str = "cuda",
        window_samples: int = WINDOW_SAMPLES,
        hop_samples:    int = HOP_SAMPLES,
    ):
        self.device         = device
        self.window_samples = window_samples
        self.hop_samples    = hop_samples
        self.de_weights_dir = de_weights_dir

        # Cross-subject EEGNet model (always loaded as fallback)
        self._cross_model = EEGNetVA(chans=N_EEG_CHANNELS, samples=window_samples).to(device)
        if os.path.exists(weights_path):
            self._cross_model.load_state_dict(
                torch.load(weights_path, map_location=device, weights_only=True)
            )
        self._cross_model.eval()

        # Cache of lazily-loaded per-subject DE models: {subject_id: EEGDEModel}
        self._ws_cache: dict = {}

    def _get_ws_model(self, subject_id: int):
        """Load (and cache) the per-subject DE model; returns None if not found."""
        if subject_id not in self._ws_cache:
            ckpt = os.path.join(self.de_weights_dir, f"eeg_de_s{subject_id:02d}.pt")
            if not os.path.exists(ckpt):
                return None
            model = EEGDEModel().to(self.device)
            model.load_state_dict(
                torch.load(ckpt, map_location=self.device, weights_only=True)
            )
            model.eval()
            self._ws_cache[subject_id] = model
        return self._ws_cache[subject_id]

    def _extract_de_windows(self, eeg: np.ndarray) -> np.ndarray:
        """Extract DE feature vectors from overlapping windows of a raw EEG trial.

        Args:
            eeg : (32, T) float32 — z-score normalised EEG.
        Returns:
            (N, 160) float32 — DE feature matrix, one row per window.
        """
        from scipy.signal import butter, sosfilt  # local import; scipy is optional dep
        nyq = FS / 2.0
        T   = eeg.shape[1]
        de_windows = []
        start = 0
        while start + self.window_samples <= T:
            window = eeg[:, start : start + self.window_samples]  # (32, W)
            de_list = []
            for lo, hi in BANDS.values():
                sos      = butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
                filtered = sosfilt(sos, window, axis=1)
                de_list.append(np.log(np.var(filtered, axis=1) + 1e-8))
            de_windows.append(np.concatenate(de_list).astype(np.float32))
            start += self.hop_samples

        if not de_windows:
            pad = np.zeros((eeg.shape[0], self.window_samples), dtype=np.float32)
            pad[:, :T] = eeg
            de_list = []
            for lo, hi in BANDS.values():
                sos      = butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
                filtered = sosfilt(sos, pad, axis=1)
                de_list.append(np.log(np.var(filtered, axis=1) + 1e-8))
            de_windows.append(np.concatenate(de_list).astype(np.float32))

        return np.stack(de_windows)  # (N, 160)

    def predict_va(self, eeg: np.ndarray, subject_id: int = None) -> tuple:
        """Predict (valence, arousal) from a z-score-normalised EEG trial.

        Args:
            eeg        : (32, T) float32 — z-score normalised per channel.
            subject_id : int in [1, 32] or None.
                         Providing a subject_id uses the higher-accuracy
                         within-subject DE model for that subject if available;
                         otherwise falls back to cross-subject EEGNet.
        Returns:
            (valence, arousal) : floats in [-1, 1]
        """
        # Within-subject path
        if subject_id is not None:
            ws_model = self._get_ws_model(subject_id)
            if ws_model is not None:
                de_wins = self._extract_de_windows(eeg)            # (N, 160)
                x = torch.from_numpy(de_wins).to(self.device)
                with torch.no_grad():
                    preds = ws_model(x).cpu().numpy()              # (N, 2)
                va = preds.mean(0)
                return float(va[0]), float(va[1])

        # Cross-subject path (original behaviour)
        T = eeg.shape[1]
        windows = []
        start = 0
        while start + self.window_samples <= T:
            windows.append(eeg[:, start : start + self.window_samples])
            start += self.hop_samples
        if not windows:
            pad = np.zeros((eeg.shape[0], self.window_samples), dtype=np.float32)
            pad[:, :T] = eeg
            windows = [pad]

        x = np.stack([w[np.newaxis] for w in windows]).astype(np.float32)
        with torch.no_grad():
            preds = self._cross_model(
                torch.from_numpy(x).to(self.device)
            ).cpu().numpy()
        va = preds.mean(0)
        return float(va[0]), float(va[1])
