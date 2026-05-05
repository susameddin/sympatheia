"""EEG (SEED-VII) → speech-model VA utilities."""
from __future__ import annotations

import numpy as np

from src.constants import SPEECH_ANCHORS

_EEG_TO_SPEECH = {
    "Anger":    "angry",
    "Disgust":  "disgusted",
    "Fear":     "anxious",
    "Happy":    "happy",
    "Neutral":  "neutral",
    "Sad":      "sad",
    "Surprise": "surprised",
}

_EEG_RUSSELL_VA = {
    "Anger":    (-0.60, +0.70),
    "Disgust":  (-0.70, +0.30),
    "Fear":     (-0.50, +0.70),
    "Happy":    (+0.80, +0.50),
    "Neutral":  ( 0.00,  0.00),
    "Sad":      (-0.60, -0.30),
    "Surprise": (+0.40, +0.70),
}

EEG_SPEECH_VA = {
    emo: SPEECH_ANCHORS[_EEG_TO_SPEECH[emo]] for emo in _EEG_RUSSELL_VA
}


class EEGVASpeechMapper:
    """Map SEED-VII EEG softmax probabilities to speech-model VA space.

    Uses speech anchor coordinates as the VA matrix; weighted sum of softmax
    probabilities produces (valence, arousal) in the speech model's VA space.
    """

    _EMOTION_NAMES = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    def __init__(self):
        self._va_matrix = np.array(
            [EEG_SPEECH_VA[e] for e in self._EMOTION_NAMES], dtype=np.float32
        )  # (7, 2)
        self._russell_matrix = np.array(
            [_EEG_RUSSELL_VA[e] for e in self._EMOTION_NAMES], dtype=np.float32
        )  # (7, 2)

    def adapt_from_probs(self, probs) -> tuple:
        """Map softmax probabilities to speech-model VA.

        Args:
            probs: (7,) array or dict {emotion_name: prob}
        Returns:
            (valence, arousal) in speech model VA space
        """
        if isinstance(probs, dict):
            p = np.array([probs[e] for e in self._EMOTION_NAMES], dtype=np.float32)
        else:
            p = np.asarray(probs, dtype=np.float32)
        va = p @ self._va_matrix  # (2,)
        return float(np.clip(va[0], -1, 1)), float(np.clip(va[1], -1, 1))

    def adapt(self, v: float, a: float) -> tuple:
        """Remap a Russell-space VA point to speech space via nearest anchor."""
        dists = np.sum((self._russell_matrix - np.array([v, a])) ** 2, axis=1)
        idx = int(np.argmin(dists))
        sv, sa = self._va_matrix[idx]
        return float(sv), float(sa)
