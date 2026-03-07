"""ResNet18 face emotion classifier with VA prediction wrapper."""

import os

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from .config import (
    CACHE_DIR, DROPOUT, EMOTION_NAMES, EMOTION_TO_VA,
    INPUT_SIZE, MEAN, NUM_CLASSES, STD,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FaceEmotionModel(nn.Module):
    """ResNet18 fine-tuned for 8-class face emotion classification.

    Input:  (B, 3, 96, 96)  — RGB face image
    Output: (B, 8)           — raw logits (use CrossEntropyLoss)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = DROPOUT):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Predictor (inference wrapper)
# ---------------------------------------------------------------------------

class FaceVAPredictor:
    """Load trained face emotion model and predict (valence, arousal).

    The model classifies into 8 emotions, then maps softmax probabilities
    to continuous (valence, arousal) via a probability-weighted sum of
    each emotion's VA coordinates (Russell's circumplex model).

    Usage:
        predictor = FaceVAPredictor()
        v, a = predictor.predict_va(pil_image)
        v, a = predictor.predict_va(numpy_array)     # (H, W, 3) uint8
        v, a = predictor.predict_va("/path/img.png")  # file path
    """

    def __init__(
        self,
        weights_path: str = os.path.join(CACHE_DIR, "face_emotion.pt"),
        device: str = "cuda",
    ):
        self.device = device

        self._model = FaceEmotionModel()
        if os.path.exists(weights_path):
            self._model.load_state_dict(
                torch.load(weights_path, map_location=device, weights_only=True)
            )
        self._model.to(device).eval()

        self._transform = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

        # (8, 2) matrix: each row is [valence, arousal] for the emotion
        self._va_matrix = np.array(
            [EMOTION_TO_VA[e] for e in EMOTION_NAMES], dtype=np.float32
        )

    def _to_pil(self, image):
        """Convert various input types to a PIL RGB image."""
        from PIL import Image

        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        # Assume PIL Image
        return image.convert("RGB")

    def predict_va(self, image) -> tuple:
        """Predict (valence, arousal) from a face image.

        Args:
            image : PIL Image, numpy array (H, W, 3), or file path string.
        Returns:
            (valence, arousal) : floats in [-1, 1]
        """
        pil_img = self._to_pil(image)
        x = self._transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self._model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (8,)

        va = probs @ self._va_matrix  # (2,)
        return float(np.clip(va[0], -1, 1)), float(np.clip(va[1], -1, 1))

    def predict_emotion(self, image) -> tuple:
        """Predict the top emotion and all probabilities.

        Returns:
            (top_emotion_name, {emotion_name: probability})
        """
        pil_img = self._to_pil(image)
        x = self._transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self._model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        prob_dict = {name: float(p) for name, p in zip(EMOTION_NAMES, probs)}
        top_idx = int(np.argmax(probs))
        return EMOTION_NAMES[top_idx], prob_dict
