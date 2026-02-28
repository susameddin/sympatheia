"""Dual-stream 1D CNN for physiological (BVP + GSR) VA regression."""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    CACHE_DIR,
    CNN_CHANNELS,
    CNN_KERNELS,
    DROPOUT,
    FUSION_HIDDEN,
    HOP_SAMPLES,
    LOSS_ALPHA,
    PHYSIO_CHANNELS,
    N_PHYSIO_CH,
    ATTN_HIDDEN,
    WINDOW_SAMPLES,
)


# ---------------------------------------------------------------------------
# Loss  (same CCC loss as eeg_emotion)
# ---------------------------------------------------------------------------

def ccc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 − mean CCC over output dimensions."""
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
    """alpha * CCC_loss + (1 - alpha) * MSE_loss — prevents variance collapse."""
    return alpha * ccc_loss(pred, target) + (1.0 - alpha) * F.mse_loss(pred, target)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _CNN1DStream(nn.Module):
    """Single-modality 1-D CNN stream.

    Input:  (B, 1, T)
    Output: (B, out_channels)  — global-average-pooled feature vector
    """

    def __init__(self, channels: list, kernels: list, dropout: float):
        super().__init__()
        assert len(channels) - 1 == len(kernels)
        layers = []
        for in_ch, out_ch, k in zip(channels[:-1], channels[1:], kernels):
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(dropout),
            ]
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.net(x)).squeeze(-1)   # (B, out_channels)


class PhysioVAModel(nn.Module):
    """Dual-stream 1-D CNN for BVP and GSR → VA regression.

    Two independent CNN streams process BVP and GSR; their feature vectors
    are concatenated and passed through an MLP to predict (valence, arousal).

    Input:
        bvp : (B, 1, T)  — blood volume pulse window
        gsr : (B, 1, T)  — galvanic skin response window
    Output:
        (B, 2)  — [valence, arousal] in (-1, 1) via Tanh
    """

    def __init__(
        self,
        cnn_channels: list = CNN_CHANNELS,   # [1, 32, 64, 128]
        cnn_kernels:  list = CNN_KERNELS,     # [9, 5, 3]
        fusion_hidden: int = FUSION_HIDDEN,
        dropout: float     = DROPOUT,
    ):
        super().__init__()
        feat_dim = cnn_channels[-1]           # output dim per stream (128)

        self.bvp_stream = _CNN1DStream(cnn_channels, cnn_kernels, dropout)
        self.gsr_stream = _CNN1DStream(cnn_channels, cnn_kernels, dropout)

        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * 2, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 2),
            nn.Tanh(),
        )

    def forward(self, bvp: torch.Tensor, gsr: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([self.bvp_stream(bvp), self.gsr_stream(gsr)], dim=1)
        return self.fusion(feat)


# ---------------------------------------------------------------------------
# Multi-channel model with channel attention (6 physiological streams)
# ---------------------------------------------------------------------------

class PhysioMultiChannelModel(nn.Module):
    """N-stream 1-D CNN with channel attention for multi-channel physiological signals.

    Uses all 6 DEAP peripheral channels:
        BVP (38), GSR (36), Respiration (37), Temperature (39), zEMG (34), tEMG (35)

    Architecture:
        N independent _CNN1DStream modules, each (B,1,T) → (B, feat_dim=128)
        Stack: (B, N, 128)
        Channel attention: per-stream linear(128→32→1) → softmax over N
        Weighted sum: (B, 128)
        Fusion MLP: Linear(128→64) → ReLU → Dropout → Linear(64→2) → Tanh

    Input:
        signals : (B, N_PHYSIO_CH, T) — all 6 channels pre-z-scored
    Output:
        (B, 2)  — [valence, arousal] in (-1, 1)
    """

    def __init__(
        self,
        n_channels:    int   = N_PHYSIO_CH,      # 6
        cnn_channels:  list  = CNN_CHANNELS,      # [1, 32, 64, 128]
        cnn_kernels:   list  = CNN_KERNELS,       # [9, 5, 3]
        attn_hidden:   int   = ATTN_HIDDEN,       # 32
        fusion_hidden: int   = FUSION_HIDDEN,     # 64
        dropout:       float = DROPOUT,           # 0.5
    ):
        super().__init__()
        feat_dim = cnn_channels[-1]   # 128

        # One CNN stream per channel
        self.streams = nn.ModuleList([
            _CNN1DStream(cnn_channels, cnn_kernels, dropout) for _ in range(n_channels)
        ])

        # Channel attention: maps each stream's feature vector to a scalar weight
        self.attn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, attn_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(attn_hidden, 1),
            )
            for _ in range(n_channels)
        ])

        # Fusion MLP on the attention-weighted feature
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 2),
            nn.Tanh(),
        )

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """Args: signals (B, N, T). Returns: (B, 2)."""
        n = signals.shape[1]
        feats   = [self.streams[i](signals[:, i : i + 1, :]) for i in range(n)]  # list of (B,128)
        feat_stack = torch.stack(feats, dim=1)                                    # (B, N, 128)

        # Channel attention weights
        logits = torch.cat([self.attn[i](feats[i]).unsqueeze(1) for i in range(n)], dim=1)  # (B,N,1)
        weights = torch.softmax(logits, dim=1)                                               # (B,N,1)

        # Weighted sum over channels
        fused = (feat_stack * weights).sum(dim=1)   # (B, 128)
        return self.fusion(fused)


# ---------------------------------------------------------------------------
# Predictor (inference wrapper)
# ---------------------------------------------------------------------------

class PhysioVAPredictor:
    """Load a trained PhysioMultiChannelModel and predict (valence, arousal).

    Supports two modes (same pattern as EEGVAPredictor):

    1. Cross-subject mode: uses physio_va_deap.pt trained on 26 subjects.
    2. Within-subject mode: if subject_id given, looks for physio_ws_s{sid:02d}.pt
       and uses the subject-specific model (much better accuracy).

    Signal input: either a dict of named arrays (new API) or positional
    (bvp, gsr) arrays for backward compatibility.
    """

    # Channel name → index in PHYSIO_CHANNELS
    _CH_NAMES = ["bvp", "gsr", "resp", "temp", "zemg", "temg"]

    def __init__(
        self,
        weights_path:   str = os.path.join(CACHE_DIR, "physio_va_deap.pt"),
        ws_weights_dir: str = CACHE_DIR,
        device:         str = "cuda",
        window_samples: int = WINDOW_SAMPLES,
        hop_samples:    int = HOP_SAMPLES,
    ):
        self.device         = device
        self.window_samples = window_samples
        self.hop_samples    = hop_samples
        self.ws_weights_dir = ws_weights_dir

        # Cross-subject model (fallback)
        self._cross_model = PhysioMultiChannelModel().to(device)
        if os.path.exists(weights_path):
            self._cross_model.load_state_dict(
                torch.load(weights_path, map_location=device, weights_only=True)
            )
        self._cross_model.eval()

        # Cache for per-subject within-subject models
        self._ws_cache: dict = {}

    def _get_ws_model(self, subject_id: int):
        """Load (and cache) the per-subject model; returns None if not found."""
        if subject_id not in self._ws_cache:
            ckpt = os.path.join(self.ws_weights_dir, f"physio_ws_s{subject_id:02d}.pt")
            if not os.path.exists(ckpt):
                return None
            model = PhysioMultiChannelModel().to(self.device)
            model.load_state_dict(
                torch.load(ckpt, map_location=self.device, weights_only=True)
            )
            model.eval()
            self._ws_cache[subject_id] = model
        return self._ws_cache[subject_id]

    def _epoch(self, signal: np.ndarray) -> np.ndarray:
        """Slice a 1-D signal into overlapping windows → (N, W)."""
        T = signal.shape[0]
        windows = []
        start = 0
        while start + self.window_samples <= T:
            windows.append(signal[start : start + self.window_samples])
            start += self.hop_samples
        if not windows:
            pad = np.zeros(self.window_samples, dtype=np.float32)
            pad[:T] = signal
            windows = [pad]
        return np.stack(windows).astype(np.float32)   # (N, W)

    def _build_batch(self, sig_dict: dict) -> np.ndarray:
        """Build (N, N_PHYSIO_CH, W) batch from a signal dict; zero-pad missing channels."""
        first_sig = next(iter(sig_dict.values()))
        T = first_sig.shape[0]
        N = max(1, (T - self.window_samples) // self.hop_samples + 1)
        batch = np.zeros((N, N_PHYSIO_CH, self.window_samples), dtype=np.float32)
        for ch_idx, name in enumerate(self._CH_NAMES):
            if name in sig_dict:
                wins  = self._epoch(sig_dict[name])
                n_win = min(wins.shape[0], N)
                batch[:n_win, ch_idx, :] = wins[:n_win]
        return batch

    def predict_va(self, signals_or_bvp, gsr: np.ndarray = None, subject_id: int = None) -> tuple:
        """Predict (valence, arousal) from physiological signals.

        Args:
            signals_or_bvp : dict  {"bvp": ..., "gsr": ..., ...}  — new 6-channel API
                             OR np.ndarray (T,) — bvp array (backward-compat)
            gsr            : np.ndarray (T,) — only used in backward-compat call
            subject_id     : int [1,32] or None.  If given and within-subject
                             checkpoint exists, uses the per-subject model.
        Returns:
            (valence, arousal) : floats in [-1, 1]
        """
        if isinstance(signals_or_bvp, dict):
            sig_dict = signals_or_bvp
        else:
            sig_dict = {"bvp": signals_or_bvp, "gsr": gsr}

        batch = self._build_batch(sig_dict)   # (N, N_PHYSIO_CH, W)
        x     = torch.from_numpy(batch).to(self.device)

        # Choose model: within-subject if available, else cross-subject
        model = self._get_ws_model(subject_id) if subject_id is not None else None
        if model is None:
            model = self._cross_model

        with torch.no_grad():
            preds = model(x).cpu().numpy()   # (N, 2)

        va = preds.mean(0)
        return float(va[0]), float(va[1])
