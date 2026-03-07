"""Face image → Valence-Arousal prediction via emotion classification on AffectNet."""

from .models import FaceEmotionModel, FaceVAPredictor

__all__ = ["FaceEmotionModel", "FaceVAPredictor"]
