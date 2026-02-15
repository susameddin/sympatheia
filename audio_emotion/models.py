"""Audio emotion recognition models for valence-arousal prediction."""

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from .config import AUDEERING_MODEL_NAME, SAMPLE_RATE


# ---------------------------------------------------------------------------
# Custom model classes required by the audeering model card
# (standard HF pipeline does not support this model)
# ---------------------------------------------------------------------------

class RegressionHead(nn.Module):
    """Classification head for dimensional emotion prediction."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    """wav2vec2-based speech emotion model (arousal, dominance, valence)."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits


# ---------------------------------------------------------------------------
# Predictor wrapper
# ---------------------------------------------------------------------------

class AudeeringVAPredictor:
    """Predicts valence, arousal, dominance from audio using the audeering model.

    Model outputs are in [0, 1] range and rescaled to [-1, 1] to match
    the project's VA space convention.
    """

    def __init__(self, device="cuda"):
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(AUDEERING_MODEL_NAME)
        self.model = EmotionModel.from_pretrained(AUDEERING_MODEL_NAME).to(device)
        self.model.eval()

    def _load_audio(self, audio_path: str) -> np.ndarray:
        """Load audio file and resample to 16kHz mono."""
        waveform, sr = torchaudio.load(audio_path)
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample if needed
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)
        return waveform.squeeze().numpy()

    def predict(self, audio_path: str) -> dict:
        """Predict VA values from an audio file.

        Returns dict with valence, arousal, dominance in [-1, 1] range,
        plus the raw [0, 1] values from the model.
        """
        signal = self._load_audio(audio_path)

        # Process through wav2vec2 processor
        inputs = self.processor(signal, sampling_rate=SAMPLE_RATE)
        input_values = np.array(inputs["input_values"][0]).reshape(1, -1)
        input_tensor = torch.from_numpy(input_values).to(self.device)

        with torch.no_grad():
            _, logits = self.model(input_tensor)

        # Model output order: arousal, dominance, valence
        raw = logits.squeeze().cpu().numpy()
        raw_arousal, raw_dominance, raw_valence = float(raw[0]), float(raw[1]), float(raw[2])

        # Rescale [0, 1] -> [-1, 1] and clamp
        def rescale(x):
            return max(-1.0, min(1.0, 2.0 * x - 1.0))

        return {
            "valence": rescale(raw_valence),
            "arousal": rescale(raw_arousal),
            "dominance": rescale(raw_dominance),
            "raw_valence": raw_valence,
            "raw_arousal": raw_arousal,
            "raw_dominance": raw_dominance,
        }


class AudioEmotionRecognizer:
    """Facade for audio emotion recognition.

    Provides a clean interface for integration with the speech model.
    """

    def __init__(self, device="cuda"):
        self.predictor = AudeeringVAPredictor(device=device)

    def predict_va(self, audio_path: str) -> tuple:
        """Predict (valence, arousal) from audio. Values in [-1, 1]."""
        result = self.predictor.predict(audio_path)
        return result["valence"], result["arousal"]

    def predict_full(self, audio_path: str) -> dict:
        """Predict full emotion details from audio."""
        return self.predictor.predict(audio_path)
