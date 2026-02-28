"""EEG → Valence-Arousal decoding using EEGNet and DE-MLP models on DEAP."""

from .models import EEGNetVA, EEGDEModel, EEGVAPredictor

__all__ = ["EEGNetVA", "EEGDEModel", "EEGVAPredictor"]
