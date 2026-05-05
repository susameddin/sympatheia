"""Models and inference wrappers for YAAD ECG emotion recognition."""
from __future__ import annotations

import os

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

from .config import (
    CACHE_DIR,
    ECG_SAMPLES,
    EMOTION_NAMES,
    EMOTION_TO_VA,
    N_ESTIMATORS,
    NUM_CLASSES,
    RANDOM_STATE,
)
from .features import extract_selected_features


# ---------------------------------------------------------------------------
# PyTorch: 1D CNN
# ---------------------------------------------------------------------------

class ECG1DCNN(nn.Module):
    """4-block 1D CNN on raw ECG signal.

    Input:  (B, 1, T)  — single-channel ECG, T ≥ 64
    Output: (B, NUM_CLASSES) logits
    """

    def __init__(self, n_classes: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: (B,1,T) → (B,32,T/2)
            nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            # Block 2: → (B,64,T/4)
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            # Block 3: → (B,128,T/8)
            nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            # Block 4: → (B,256,T/16)
            nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),   # → (B,256,1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


# ---------------------------------------------------------------------------
# PyTorch: 1D ResNet
# ---------------------------------------------------------------------------

class _ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.skip(x))


class ECGResNet1D(nn.Module):
    """Lightweight 1D ResNet on raw ECG signal.

    Input:  (B, 1, T)
    Output: (B, NUM_CLASSES) logits
    """

    def __init__(self, n_classes: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(_ResBlock1D(64, 64),  _ResBlock1D(64, 64))
        self.layer2 = nn.Sequential(_ResBlock1D(64, 128, stride=2), _ResBlock1D(128, 128))
        self.layer3 = nn.Sequential(_ResBlock1D(128, 256, stride=2), _ResBlock1D(256, 256))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Squeeze-and-Excite block
# ---------------------------------------------------------------------------

class _SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, max(1, channels // reduction)),
            nn.ReLU(),
            nn.Linear(max(1, channels // reduction), channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x).unsqueeze(-1)


class _ResBlock1DSE(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.se = _SEBlock1D(out_ch)
        self.skip = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.se(self.conv(x)) + self.skip(x))


# ---------------------------------------------------------------------------
# ResNet variants
# ---------------------------------------------------------------------------

class ECGResNet1DSE(nn.Module):
    """ECGResNet1D with Squeeze-and-Excite in every residual block."""

    def __init__(self, n_classes: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(_ResBlock1DSE(64, 64),  _ResBlock1DSE(64, 64))
        self.layer2 = nn.Sequential(_ResBlock1DSE(64, 128, stride=2), _ResBlock1DSE(128, 128))
        self.layer3 = nn.Sequential(_ResBlock1DSE(128, 256, stride=2), _ResBlock1DSE(256, 256))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return self.head(x)


class ECGResNet1DDeep(nn.Module):
    """ECGResNet1D with an extra layer4 (256→512 channels).

    Args:
        in_channels: 1 for single-modality (ECG or GSR), 2 for ECG+GSR stacked.
    """

    def __init__(self, n_classes: int = NUM_CLASSES, dropout: float = 0.5,
                 in_channels: int = 1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(_ResBlock1D(64, 64),   _ResBlock1D(64, 64))
        self.layer2 = nn.Sequential(_ResBlock1D(64, 128, stride=2),  _ResBlock1D(128, 128))
        self.layer3 = nn.Sequential(_ResBlock1D(128, 256, stride=2), _ResBlock1D(256, 256))
        self.layer4 = nn.Sequential(_ResBlock1D(256, 512, stride=2), _ResBlock1D(512, 512))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(512, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.head(x)


class ECGResNet1DDeepSE(nn.Module):
    """ECGResNet1D with layer4 + Squeeze-and-Excite."""

    def __init__(self, n_classes: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(_ResBlock1DSE(64, 64),   _ResBlock1DSE(64, 64))
        self.layer2 = nn.Sequential(_ResBlock1DSE(64, 128, stride=2),  _ResBlock1DSE(128, 128))
        self.layer3 = nn.Sequential(_ResBlock1DSE(128, 256, stride=2), _ResBlock1DSE(256, 256))
        self.layer4 = nn.Sequential(_ResBlock1DSE(256, 512, stride=2), _ResBlock1DSE(512, 512))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(512, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.head(x)


class ECGResNet1DHybrid(nn.Module):
    """ResNet1D body fused with handcrafted HRV features before classification.

    Input:  x_signal (B, 1, T),  x_hrv (B, n_hrv)
    Output: (B, n_classes) logits
    """

    def __init__(self, n_hrv: int = 17, n_classes: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(_ResBlock1D(64, 64),   _ResBlock1D(64, 64))
        self.layer2 = nn.Sequential(_ResBlock1D(64, 128, stride=2),  _ResBlock1D(128, 128))
        self.layer3 = nn.Sequential(_ResBlock1D(128, 256, stride=2), _ResBlock1D(256, 256))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 + n_hrv, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(128, n_classes),
        )

    def forward(self, x_signal: torch.Tensor, x_hrv: torch.Tensor) -> torch.Tensor:
        x = self.stem(x_signal)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)          # (B, 256)
        x = torch.cat([x, x_hrv], dim=1)    # (B, 256 + n_hrv)
        return self.classifier(x)


_MODEL_FILENAME  = "ecg_rf_model.pkl"          # legacy RF (kept for backward compat)
_RESNET_FILENAME = "ecg_resnet_deep_model.pt"  # production ResNet1D+Deep

# Precomputed VA matrix (NUM_CLASSES, 2) in EMOTION_NAMES order
_VA_MATRIX = np.array([EMOTION_TO_VA[e] for e in EMOTION_NAMES], dtype=np.float32)


# ---------------------------------------------------------------------------
# Training helpers (RF — kept for train.py backward compat)
# ---------------------------------------------------------------------------

def train_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = N_ESTIMATORS,
    seed: int = RANDOM_STATE,
) -> RandomForestClassifier:
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    clf.fit(X_train, y_train)
    return clf


def save_model(model: RandomForestClassifier, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str) -> RandomForestClassifier:
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Inference wrapper  (ResNet1D+Deep)
# ---------------------------------------------------------------------------

class ECGEmotionPredictor:
    """Inference wrapper for the YAAD ECG ResNet1D+Deep model.

    Args:
        weights_dir: directory containing ``ecg_resnet_deep_model.pt``
        device:      torch device string, e.g. ``"cpu"`` or ``"cuda"``.
                     Defaults to CUDA if available, else CPU.

    Usage::

        predictor = ECGEmotionPredictor()
        result = predictor.predict_emotion(ecg_array)  # 1-D, ≥ECG_SAMPLES points
        v, a   = predictor.predict_va(ecg_array)
    """

    def __init__(self, weights_dir: str = CACHE_DIR, device: str | None = None):
        self.weights_dir = weights_dir
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model: ECGResNet1DDeep | None = None

    def _get_model(self) -> ECGResNet1DDeep:
        if self._model is None:
            ckpt_path = os.path.join(self.weights_dir, _RESNET_FILENAME)
            if not os.path.exists(ckpt_path):
                raise RuntimeError(
                    f"No trained model at {ckpt_path}. "
                    "Run: python -m ecg_emotion.train"
                )
            model = ECGResNet1DDeep(n_classes=NUM_CLASSES)
            model.load_state_dict(
                torch.load(ckpt_path, map_location=self.device, weights_only=True)
            )
            model.eval()
            self._model = model.to(self.device)
        return self._model

    def predict_emotion(self, ecg_signal: np.ndarray) -> dict:
        """Predict emotion from a raw 1-D ECG signal.

        Args:
            ecg_signal: 1-D numpy array, ≥ ECG_SAMPLES points.

        Returns:
            dict with keys:
                top_emotion: str
                probs:       {emotion_name: float}  (sums to 1.0)
        """
        sig = np.asarray(ecg_signal, dtype=np.float32).flatten()[:ECG_SAMPLES]
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        x = torch.from_numpy(sig[None, None, :]).to(self.device)  # (1, 1, ECG_SAMPLES)
        with torch.no_grad():
            probs = torch.softmax(self._get_model()(x), dim=1).cpu().numpy()[0]
        top_idx = int(np.argmax(probs))
        return {
            "top_emotion": EMOTION_NAMES[top_idx],
            "probs": {EMOTION_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)},
        }

    def predict_va(self, ecg_signal: np.ndarray) -> tuple[float, float]:
        """Predict valence and arousal via Russell circumplex soft mapping.

        Returns:
            (valence, arousal) both ∈ [-1, 1]
        """
        result = self.predict_emotion(ecg_signal)
        probs  = np.array([result["probs"][e] for e in EMOTION_NAMES], dtype=np.float32)
        va     = probs @ _VA_MATRIX
        return float(np.clip(va[0], -1, 1)), float(np.clip(va[1], -1, 1))
