"""Predictor classes for YAAD ECG / GSR → Valence-Arousal.

Three predictors, all sharing the same interface:

    YAADECGPredictor   — ECG signal only (ResNet1D+Deep, in_channels=1)
    YAADGSRPredictor   — GSR signal only (ResNet1D+Deep, in_channels=1)
    YAADFusionPredictor — ECG + GSR soft-voting ensemble

Emotion softmax probabilities are weighted-summed against the speech model's
VA anchors (EMOTION_VA_MAPPING from constants.py) to produce (valence, arousal).
"""
from __future__ import annotations

import os
import numpy as np
import torch

from .ecg_emotion.models import ECGResNet1DDeep
from src.constants import EMOTION_VA_MAPPING as SPEECH_ANCHORS

from .config import (
    CACHE_DIR, ECG_SAMPLES, ECG_WEIGHTS, EMOTION_NAMES,
    GSR_WEIGHTS, NUM_CLASSES, YAAD_TO_SPEECH,
)

# (8, 2) VA matrix in speech model coordinates, indexed by EMOTION_NAMES order
_YAAD_SPEECH_VA = np.array(
    [SPEECH_ANCHORS[YAAD_TO_SPEECH[e]] for e in EMOTION_NAMES], dtype=np.float32
)


def _znorm(sig: np.ndarray) -> np.ndarray:
    """Per-sample z-score normalisation."""
    mu, sd = sig.mean(), sig.std()
    return (sig - mu) / (sd + 1e-8)


def _load_resnet(weights_path: str, device: torch.device) -> ECGResNet1DDeep:
    model = ECGResNet1DDeep(n_classes=NUM_CLASSES, in_channels=1)
    if not os.path.exists(weights_path):
        raise RuntimeError(
            f"No checkpoint at {weights_path}. "
            "Run: python -m integration.yaad_module.train"
        )
    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    model.eval()
    return model.to(device)


def _predict_probs(model: ECGResNet1DDeep,
                   signal: np.ndarray,
                   device: torch.device) -> np.ndarray:
    """Preprocess signal and return softmax probability array (NUM_CLASSES,)."""
    sig = np.asarray(signal, dtype=np.float32).flatten()[:ECG_SAMPLES]
    sig = _znorm(sig)
    x = torch.from_numpy(sig[None, None, :]).to(device)
    with torch.no_grad():
        return torch.softmax(model(x), dim=1).cpu().numpy()[0]


# ---------------------------------------------------------------------------
# ECG predictor
# ---------------------------------------------------------------------------

class YAADECGPredictor:
    """ECG signal → (valence, arousal) via ResNet1D+Deep trained on YAAD.

    Usage::

        p = YAADECGPredictor()
        v, a = p.predict_va(ecg_array)           # 1-D, ≥ ECG_SAMPLES points
        top, probs = p.predict_emotion(ecg_array)
    """

    def __init__(self, weights_path: str = ECG_WEIGHTS, device: str = "cuda"):
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self._model = _load_resnet(weights_path, self.device)

    def predict_emotion(self, ecg_signal: np.ndarray) -> tuple[str, dict]:
        probs = _predict_probs(self._model, ecg_signal, self.device)
        top   = EMOTION_NAMES[int(np.argmax(probs))]
        return top, {n: float(p) for n, p in zip(EMOTION_NAMES, probs)}

    def predict_va(self, ecg_signal: np.ndarray) -> tuple[float, float]:
        probs = _predict_probs(self._model, ecg_signal, self.device)
        va    = probs @ _YAAD_SPEECH_VA
        return float(np.clip(va[0], -1, 1)), float(np.clip(va[1], -1, 1))


# ---------------------------------------------------------------------------
# GSR predictor
# ---------------------------------------------------------------------------

class YAADGSRPredictor:
    """GSR signal → (valence, arousal) via ResNet1D+Deep trained on YAAD multimodal.

    Same interface as YAADECGPredictor.  Train the GSR model first::

        python -m integration.yaad_module.train
    """

    def __init__(self, weights_path: str = GSR_WEIGHTS, device: str = "cuda"):
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self._model = _load_resnet(weights_path, self.device)

    def predict_emotion(self, gsr_signal: np.ndarray) -> tuple[str, dict]:
        probs = _predict_probs(self._model, gsr_signal, self.device)
        top   = EMOTION_NAMES[int(np.argmax(probs))]
        return top, {n: float(p) for n, p in zip(EMOTION_NAMES, probs)}

    def predict_va(self, gsr_signal: np.ndarray) -> tuple[float, float]:
        probs = _predict_probs(self._model, gsr_signal, self.device)
        va    = probs @ _YAAD_SPEECH_VA
        return float(np.clip(va[0], -1, 1)), float(np.clip(va[1], -1, 1))


# ---------------------------------------------------------------------------
# Fusion predictor (soft voting)
# ---------------------------------------------------------------------------

class YAADFusionPredictor:
    """ECG + GSR soft-voting ensemble → (valence, arousal).

    Averages per-class softmax probabilities from ECG and GSR predictors
    before computing the argmax emotion and VA weighted sum.

    Usage::

        p = YAADFusionPredictor()
        v, a = p.predict_va(ecg_array, gsr_array)
        top, probs = p.predict_emotion(ecg_array, gsr_array)
    """

    def __init__(self,
                 ecg_weights: str = ECG_WEIGHTS,
                 gsr_weights: str = GSR_WEIGHTS,
                 device: str = "cuda"):
        self._ecg = YAADECGPredictor(weights_path=ecg_weights, device=device)
        self._gsr = YAADGSRPredictor(weights_path=gsr_weights, device=device)
        self.device = self._ecg.device

    def _fused_probs(self, ecg_signal: np.ndarray,
                     gsr_signal: np.ndarray) -> np.ndarray:
        p_ecg = _predict_probs(self._ecg._model, ecg_signal, self.device)
        p_gsr = _predict_probs(self._gsr._model, gsr_signal, self.device)
        return (p_ecg + p_gsr) / 2.0

    def predict_emotion(self, ecg_signal: np.ndarray,
                        gsr_signal: np.ndarray) -> tuple[str, dict]:
        probs = self._fused_probs(ecg_signal, gsr_signal)
        top   = EMOTION_NAMES[int(np.argmax(probs))]
        return top, {n: float(p) for n, p in zip(EMOTION_NAMES, probs)}

    def predict_va(self, ecg_signal: np.ndarray,
                   gsr_signal: np.ndarray) -> tuple[float, float]:
        probs = self._fused_probs(ecg_signal, gsr_signal)
        va    = probs @ _YAAD_SPEECH_VA
        return float(np.clip(va[0], -1, 1)), float(np.clip(va[1], -1, 1))
