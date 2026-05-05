"""Face emotion classifier with multi-backbone support and VA prediction wrapper."""

import os

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights, ResNet50_Weights, EfficientNet_B0_Weights,
    efficientnet_b0, resnet18, resnet50,
)
from torchvision import transforms

from .config import (
    BACKBONE, CACHE_DIR, DROPOUT, EMOTION_NAMES, HSEMOTION_MODEL,
    INPUT_SIZE, MEAN, NUM_CLASSES, STD,
)

# Speech model VA anchors: emotion category → (valence, arousal) in speech model space.
# These are NOT AffectNet VA labels — they are the target VA coordinates for the speech
# synthesis model. Emotion probabilities are weighted-summed against these anchors.
_FACE_SPEECH_VA = {
    "Anger":    (-0.85,  0.85),
    "Contempt": (-0.65,  0.15),
    "Disgust":  (-0.82, -0.20),
    "Fear":     (-0.40,  0.65),
    "Happy":    ( 0.85,  0.35),
    "Neutral":  ( 0.00,  0.00),
    "Sad":      (-0.75, -0.65),
    "Surprise": ( 0.10,  0.80),
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FaceEmotionModel(nn.Module):
    """Multi-backbone face emotion classifier.

    Input:  (B, 3, 224, 224) — RGB face image
    Output: (B, NUM_CLASSES) — raw logits (use CrossEntropyLoss)

    Supported backbones:
        "resnet18"       — Dropout(p) → Linear(512, C)
        "resnet50"       — Dropout(p) → Linear(2048, 256) → ReLU → Dropout(p) → Linear(256, C)
        "efficientnet_b0"— Dropout(p) → Linear(1280, 256) → ReLU → Dropout(p) → Linear(256, C)
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
        backbone: str = BACKBONE,
    ):
        super().__init__()
        backbone = backbone.lower()

        if backbone == "resnet18":
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(512, num_classes),
            )
        elif backbone == "resnet50":
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(2048, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )
        elif backbone == "efficientnet_b0":
            self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(1280, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )
        else:
            raise ValueError(
                f"Unknown backbone: {backbone!r}. Choose from resnet18, resnet50, efficientnet_b0"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Predictor (inference wrapper)
# ---------------------------------------------------------------------------

class FaceVAPredictor:
    """Load trained FaceEmotionModel and predict (valence, arousal).

    Emotion softmax probabilities are weighted-summed against the speech
    model's VA anchors (_FACE_SPEECH_VA) to produce (valence, arousal) in
    the speech model's coordinate space.

    Usage:
        predictor = FaceVAPredictor()
        v, a = predictor.predict_va(pil_image)
        v, a = predictor.predict_va(numpy_array)      # (H, W, 3) uint8
        v, a = predictor.predict_va("/path/img.png")   # file path
        top_emo, probs = predictor.predict_emotion(image)

    After training with a non-default backbone:
        predictor = FaceVAPredictor(backbone="efficientnet_b0")
    """

    def __init__(
        self,
        weights_path: str = os.path.join(CACHE_DIR, "face_emotion.pt"),
        device: str = "cuda",
        backbone: str = BACKBONE,
    ):
        self.device = device

        self._model = FaceEmotionModel(backbone=backbone)
        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=device, weights_only=True)
            model_state = self._model.state_dict()
            compatible = {
                k: v for k, v in state.items()
                if k in model_state and v.shape == model_state[k].shape
            }
            skipped = len(state) - len(compatible)
            if skipped:
                print(
                    f"[FaceVAPredictor] WARNING: skipped {skipped} incompatible key(s). "
                    "Run face_emotion.train to update the checkpoint."
                )
            model_state.update(compatible)
            self._model.load_state_dict(model_state)
        self._model.to(device).eval()

        self._transform = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

        # (8, 2): row i = [valence, arousal] in speech model space for emotion i
        self._va_matrix = np.array(
            [_FACE_SPEECH_VA[e] for e in EMOTION_NAMES], dtype=np.float32
        )

    def _to_pil(self, image):
        from PIL import Image
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        return image.convert("RGB")

    def predict_va(self, image) -> tuple:
        """Predict (valence, arousal) in speech model VA space.

        Args:
            image : PIL Image, numpy array (H, W, 3), or file path string.
        Returns:
            (valence, arousal) : floats in [-1, 1]
        """
        pil_img = self._to_pil(image)
        x = self._transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self._model(x), dim=1).cpu().numpy()[0]
        va = probs @ self._va_matrix
        return float(np.clip(va[0], -1, 1)), float(np.clip(va[1], -1, 1))

    def predict_emotion(self, image) -> tuple:
        """Predict the top emotion and all probabilities.

        Returns:
            (top_emotion_name, {emotion_name: probability})
        """
        pil_img = self._to_pil(image)
        x = self._transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self._model(x), dim=1).cpu().numpy()[0]
        prob_dict = {name: float(p) for name, p in zip(EMOTION_NAMES, probs)}
        return EMOTION_NAMES[int(np.argmax(probs))], prob_dict


# ---------------------------------------------------------------------------
# HSEmotion predictor — same API, backed by enet_b0_8_va_mtl
# ---------------------------------------------------------------------------

_HSEMO_IDX_TO_NAME = {
    0: "Anger", 1: "Contempt", 2: "Disgust", 3: "Fear",
    4: "Happy", 5: "Neutral", 6: "Sad", 7: "Surprise",
}

_HSEMO_VA_MATRIX = np.array(
    [_FACE_SPEECH_VA[_HSEMO_IDX_TO_NAME[i]] for i in range(8)], dtype=np.float32
)


_HSEMOTION_ONNX = os.path.expanduser("~/.emotiefflib/enet_b0_8_va_mtl.onnx")
_HSEMOTION_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_HSEMOTION_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class HSEmotionFacePredictor:
    """Drop-in replacement for FaceVAPredictor backed by HSEmotion enet_b0_8_va_mtl.

    Uses the ONNX model via onnxruntime — no timm, no emotiefflib needed.
    Works in any env with onnxruntime + numpy (e.g. glm4voice3).

    The .onnx file is downloaded once by the 's' env. If missing, run:
        conda run -n s python -c "
            from emotiefflib.facial_analysis import EmotiEffLibRecognizer
            EmotiEffLibRecognizer(engine='onnx', model_name='enet_b0_8_va_mtl')"
    """

    def __init__(self, onnx_path: str = _HSEMOTION_ONNX):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        # Limit threads to avoid pthread_setaffinity_np crashes on cluster nodes
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self._input_name = self._sess.get_inputs()[0].name

    def _preprocess(self, image) -> np.ndarray:
        """Convert any input to (1, 3, 224, 224) float32 normalised array."""
        from PIL import Image as PILImage
        if isinstance(image, str):
            img = PILImage.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = PILImage.fromarray(image).convert("RGB")
        else:
            img = image.convert("RGB")
        img = img.resize((224, 224), PILImage.BILINEAR)
        arr = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0  # (3,224,224)
        arr = (arr - _HSEMOTION_MEAN) / _HSEMOTION_STD
        return arr[np.newaxis]  # (1, 3, 224, 224)

    def _scores(self, image) -> np.ndarray:
        """Return raw (10,) scores: [:8] emotion logits, [8:] valence/arousal."""
        x = self._preprocess(image)
        return self._sess.run(None, {self._input_name: x})[0][0]  # (10,)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def predict_va(self, image) -> tuple:
        """(valence, arousal) in speech model VA space."""
        scores = self._scores(image)
        emo_probs = self._softmax(scores[:8])
        va = emo_probs @ _HSEMO_VA_MATRIX
        return float(np.clip(va[0], -1, 1)), float(np.clip(va[1], -1, 1))

    def predict_emotion(self, image) -> tuple:
        """(top_emotion_name, {emotion_name: probability})"""
        scores = self._scores(image)
        emo_probs = self._softmax(scores[:8])
        top_name = _HSEMO_IDX_TO_NAME[int(np.argmax(emo_probs))]
        prob_dict = {_HSEMO_IDX_TO_NAME[i]: float(p) for i, p in enumerate(emo_probs)}
        return top_name, prob_dict
