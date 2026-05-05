"""EEG → Emotion classification (SEED-VII) with VA mapping via Russell's circumplex."""

from .models import DGCNNEEGClassifier, EEGEmotionClassifier, EEGVAPredictor

__all__ = ["DGCNNEEGClassifier", "EEGEmotionClassifier", "EEGVAPredictor"]
