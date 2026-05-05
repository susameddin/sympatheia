"""EEG emotion classifier (SEED-VII) and VA predictor via softmax→Russell mapping.

Models:
  - EEGEmotionClassifier : legacy MLP (kept for reference)
  - DGCNNEEGClassifier   : DGCNN wrapper (kept for reference)
  - MAETEEGClassifier    : MAET wrapper — current default (Jiang et al. 2025)
"""

import os
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    CACHE_DIR,
    DE_DIM,
    DGCNN_HID_CHANNELS,
    DGCNN_N_LAYERS,
    DROPOUT,
    EMOTION_NAMES,
    EMOTION_TO_VA,
    HIDDEN_1,
    HIDDEN_2,
    HIDDEN_3,
    MAET_ATTN_DROP_RATE,
    MAET_DEPTH,
    MAET_DROP_PATH_RATE,
    MAET_DROP_RATE,
    MAET_EMBED_DIM,
    MAET_EEG_SEQ_LEN,
    MAET_MIX_START,
    MAET_NUM_HEADS,
    NUM_CLASSES,
    N_EEG_CHANNELS,
    N_BANDS,
)


# ---------------------------------------------------------------------------
# Legacy: MLP classifier
# ---------------------------------------------------------------------------

class EEGEmotionClassifier(nn.Module):
    """MLP classifier on 310-dim DE features → 7-class emotion logits."""

    def __init__(
        self,
        in_dim: int = DE_DIM,
        hidden1: int = HIDDEN_1,
        hidden2: int = HIDDEN_2,
        hidden3: int = HIDDEN_3,
        n_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.ELU(inplace=True),
            nn.Linear(hidden3, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Legacy: DGCNN classifier (kept for reference, not used in training)
# ---------------------------------------------------------------------------

class DGCNNEEGClassifier(nn.Module):
    """DGCNN wrapper — kept for reference only; use MAETEEGClassifier instead."""

    def __init__(
        self,
        in_dim: int = DE_DIM,
        n_nodes: int = N_EEG_CHANNELS,
        n_bands: int = N_BANDS,
        hid_channels: int = DGCNN_HID_CHANNELS,
        n_layers: int = DGCNN_N_LAYERS,
        n_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        from torcheeg.models import DGCNN as _DGCNN
        self.n_nodes = n_nodes
        self.n_bands = n_bands
        self._dgcnn = _DGCNN(
            in_channels=n_bands,
            num_electrodes=n_nodes,
            hid_channels=hid_channels,
            num_layers=n_layers,
            num_classes=n_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        return self._dgcnn(x.reshape(B, self.n_bands, self.n_nodes).transpose(1, 2))


# ---------------------------------------------------------------------------
# MAET implementation (Jiang et al., IEEE TAffComp 2025)
# Adapted from https://github.com/935963004/MAET
# No external dependencies beyond PyTorch.
# ---------------------------------------------------------------------------

def _drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob).div_(keep_prob)
    return x * random_tensor


class _DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)


class _Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class _Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim)) if qkv_bias else None
        self.v_bias = nn.Parameter(torch.zeros(dim)) if qkv_bias else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat([self.q_bias,
                                   torch.zeros_like(self.v_bias, requires_grad=False),
                                   self.v_bias])
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q.float() * self.scale) @ k.float().transpose(-2, -1)
        attn = attn.softmax(dim=-1).type_as(x)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class _Block(nn.Module):
    """Transformer block with modality-specific FFN experts (adaptive) or shared Mix-FFN (mixture)."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0,
                 attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, with_mixffn=False,
                 layer_scale_init=0.1, max_eeg_len=6):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = _Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                               attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        hidden = int(dim * mlp_ratio)
        self.norm2_eeg = norm_layer(dim)
        self.norm2_eye = norm_layer(dim)
        self.mlp_eeg = _Mlp(dim, hidden, act_layer=act_layer, drop=drop)
        self.mlp_eye = _Mlp(dim, hidden, act_layer=act_layer, drop=drop)
        self.mlp_mix = _Mlp(dim, hidden, act_layer=act_layer, drop=drop) if with_mixffn else None
        self.norm2_mix = norm_layer(dim) if with_mixffn else None
        self.gamma1 = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init is not None else 1.0
        self.gamma2 = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init is not None else 1.0
        self.max_eeg_len = max_eeg_len  # eeg_seq_len + 1 (CLS)

    def forward(self, x, modality_type=None):
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        if modality_type == "eeg":
            x = x + self.drop_path(self.gamma2 * self.mlp_eeg(self.norm2_eeg(x)))
        elif modality_type == "eye":
            x = x + self.drop_path(self.gamma2 * self.mlp_eye(self.norm2_eye(x)))
        else:
            # multimodal: split and apply modality-specific FFN, or mix-FFN
            if self.mlp_mix is None:
                x_eeg = x[:, :self.max_eeg_len]
                x_eye = x[:, self.max_eeg_len:]
                x_eeg = x_eeg + self.drop_path(self.gamma2 * self.mlp_eeg(self.norm2_eeg(x_eeg)))
                x_eye = x_eye + self.drop_path(self.gamma2 * self.mlp_eye(self.norm2_eye(x_eye)))
                x = torch.cat([x_eeg, x_eye], dim=1)
            else:
                x = x + self.drop_path(self.gamma2 * self.mlp_mix(self.norm2_mix(x)))
        return x


class _MultiViewEmbedding(nn.Module):
    """Maps (B, input_dim) → (B, v, embed_dim) via multi-view gated projection."""

    def __init__(self, input_dim, embed_dim, v):
        super().__init__()
        self.embed_dim = embed_dim
        self.v = v
        self.transform1 = nn.Linear(input_dim, embed_dim)          # base embedding
        self.transform2 = nn.Linear(input_dim, embed_dim * v)      # v gating vectors
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        B = x.size(0)
        base = self.transform1(x).unsqueeze(1).expand(-1, self.v, -1)   # (B, v, d)
        gate = self.sigmoid(self.transform2(x)).reshape(B, self.v, self.embed_dim)  # (B, v, d)
        out = base * gate                                                 # (B, v, d)
        out = self.bn(out.permute(0, 2, 1)).permute(0, 2, 1)            # BN over d
        return out


class _Fusion(nn.Module):
    """Attention-weighted fusion of EEG and eye CLS tokens."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, 1))

    def forward(self, eeg_cls, eye_cls):
        o = torch.cat([eeg_cls @ self.weight, eye_cls @ self.weight], dim=-1)
        alpha = o.softmax(dim=-1)
        return eeg_cls * alpha[:, 0:1] + eye_cls * alpha[:, 1:2]


class _MAET(nn.Module):
    """Multimodal Adaptive Emotion Transformer (Jiang et al. 2025).

    Supports EEG-only, eye-only, or joint EEG+eye inputs.
    For EEG-only: call forward(eeg=x, eye=None).
    """

    def __init__(
        self,
        eeg_dim=310,
        eye_dim=33,
        num_classes=7,
        embed_dim=32,
        depth=3,
        eeg_seq_len=5,
        eye_seq_len=5,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        layer_scale_init=0.1,
        mixffn_start_layer_index=2,
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.eeg_seq_len = eeg_seq_len
        self.eye_seq_len = eye_seq_len
        self.mixffn_start_layer_index = mixffn_start_layer_index

        self.eeg_embed = _MultiViewEmbedding(eeg_dim, embed_dim, eeg_seq_len)
        self.eye_embed = _MultiViewEmbedding(eye_dim, embed_dim, eye_seq_len)

        self.eeg_cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.eye_cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.eeg_pos = nn.Parameter(torch.zeros(1, eeg_seq_len + 1, embed_dim))
        self.eye_pos = nn.Parameter(torch.zeros(1, eye_seq_len + 1, embed_dim))
        self.eeg_type = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.eye_type = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            _Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer,
                with_mixffn=(i >= mixffn_start_layer_index),
                layer_scale_init=layer_scale_init,
                max_eeg_len=eeg_seq_len + 1,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.head_eeg = nn.Linear(embed_dim, num_classes)
        self.head_eye = nn.Linear(embed_dim, num_classes)
        self.fusion = _Fusion(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for p in [self.eeg_cls, self.eye_cls, self.eeg_pos, self.eye_pos,
                  self.eeg_type, self.eye_type]:
            torch.nn.init.trunc_normal_(p, std=0.02)

    def forward(self, eeg=None, eye=None):
        if eeg is not None:
            eeg = self.eeg_embed(eeg)
            B = eeg.size(0)
            cls = self.eeg_cls.expand(B, -1, -1)

        if eye is not None:
            eye = self.eye_embed(eye)
            B = eye.size(0)
            cls_eye = self.eye_cls.expand(B, -1, -1)

        if eye is None:
            # EEG-only
            x = torch.cat([cls, eeg], dim=1)
            modality = "eeg"
            x = x + self.eeg_type.expand(B, x.size(1), -1)
            x = x + self.eeg_pos.expand(B, -1, -1)
        elif eeg is None:
            # Eye-only
            x = torch.cat([cls_eye, eye], dim=1)
            modality = "eye"
            x = x + self.eye_type.expand(B, x.size(1), -1)
            x = x + self.eye_pos.expand(B, -1, -1)
        else:
            # Multimodal
            x = torch.cat([cls, eeg, cls_eye, eye], dim=1)
            modality = None
            eeg_type = self.eeg_type.expand(B, self.eeg_seq_len + 1, -1)
            eye_type = self.eye_type.expand(B, self.eye_seq_len + 1, -1)
            x = x + torch.cat([eeg_type, eye_type], dim=1)
            x = x + torch.cat([self.eeg_pos.expand(B, -1, -1),
                                self.eye_pos.expand(B, -1, -1)], dim=1)

        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x, modality_type=modality)
        x = self.norm(x)

        if modality == "eeg":
            # mean-pool over view tokens (exclude CLS at position 0)
            return self.head_eeg(x[:, 1:].mean(dim=1))
        elif modality == "eye":
            return self.head_eye(x[:, 1:].mean(dim=1))
        else:
            eeg_cls_out = x[:, 0]
            eye_cls_out = x[:, self.eeg_seq_len + 1]
            return self.head(self.fusion(eeg_cls_out, eye_cls_out))


class MAETEEGClassifier(nn.Module):
    """MAET wrapper for EEG-only emotion classification.

    Drop-in replacement for DGCNNEEGClassifier:
        forward(x: (B, 310)) → (B, 7) logits
    """

    def __init__(
        self,
        eeg_dim: int = DE_DIM,
        n_classes: int = NUM_CLASSES,
        embed_dim: int = MAET_EMBED_DIM,
        depth: int = MAET_DEPTH,
        num_heads: int = MAET_NUM_HEADS,
        eeg_seq_len: int = MAET_EEG_SEQ_LEN,
        drop_rate: float = MAET_DROP_RATE,
        attn_drop_rate: float = MAET_ATTN_DROP_RATE,
        drop_path_rate: float = MAET_DROP_PATH_RATE,
        mixffn_start_layer_index: int = MAET_MIX_START,
    ):
        super().__init__()
        self._maet = _MAET(
            eeg_dim=eeg_dim,
            num_classes=n_classes,
            embed_dim=embed_dim,
            depth=depth,
            eeg_seq_len=eeg_seq_len,
            num_heads=num_heads,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            mixffn_start_layer_index=mixffn_start_layer_index,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, 310). Returns: (B, 7) logits."""
        return self._maet(eeg=x, eye=None)


# ---------------------------------------------------------------------------
# Predictor (inference wrapper)
# ---------------------------------------------------------------------------

class EEGVAPredictor:
    """Predict (valence, arousal) from EEG via 7-class emotion classification + VA mapping.

    Supports two modes:
    1. Within-subject (primary): per-subject model (eeg_seed_s{id:02d}.pt)
    2. Cross-subject (fallback): single model trained on all subjects (eeg_seed_cross.pt)

    VA anchors match face_emotion's speech model space for cross-modality consistency.
    """

    def __init__(
        self,
        weights_dir: str = CACHE_DIR,
        device: str = "cuda",
    ):
        self.device = device
        self.weights_dir = weights_dir

        self._va_matrix = np.array(
            [EMOTION_TO_VA[e] for e in EMOTION_NAMES], dtype=np.float32
        )
        self._model_cache: dict = {}
        self._subject_norms: dict = {}
        self._load_all_norms()

    def _load_all_norms(self):
        for sid in range(1, 21):
            norm_path = os.path.join(self.weights_dir, f"eeg_seed_norm_s{sid:02d}.npz")
            if not os.path.exists(norm_path):
                continue
            data = np.load(norm_path)
            if "means" in data:
                # New format: per-session stats {session_id: (mean, std)}
                sess_ids = data["session_ids"]
                means = data["means"]
                stds  = data["stds"]
                self._subject_norms[sid] = {
                    int(s): (means[i], stds[i])
                    for i, s in enumerate(sess_ids)
                }
            else:
                # Legacy format (single mean/std over all sessions) — approximate
                self._subject_norms[sid] = {s: (data["mean"], data["std"]) for s in range(1, 5)}

    def _normalize(self, de: np.ndarray, subject_id: int = None,
                   session_id: int = None) -> np.ndarray:
        """Normalize DE features using per-session stats if available."""
        if subject_id is None or subject_id not in self._subject_norms:
            return de
        norm = self._subject_norms[subject_id]
        if isinstance(norm, dict):
            if session_id is not None and session_id in norm:
                mean, std = norm[session_id]
                return (de - mean) / std
            # No session_id provided: use session-1 stats as fallback (inference-time)
            mean, std = next(iter(norm.values()))
            return (de - mean) / std
        # Legacy tuple format
        mean, std = norm
        return (de - mean) / std

    def _get_model(self, subject_id: int = None) -> nn.Module | None:
        if subject_id is not None:
            key = f"ws_{subject_id}"
            ckpt_name = f"eeg_seed_s{subject_id:02d}.pt"
        else:
            key = "cross"
            ckpt_name = "eeg_seed_cross.pt"

        if key not in self._model_cache:
            ckpt = os.path.join(self.weights_dir, ckpt_name)
            if not os.path.exists(ckpt):
                return None
            model = MAETEEGClassifier().to(self.device)
            model.load_state_dict(
                torch.load(ckpt, map_location=self.device, weights_only=True)
            )
            model.eval()
            self._model_cache[key] = model

        return self._model_cache[key]

    def _de_from_raw(self, eeg: np.ndarray) -> np.ndarray:
        """Extract DE features from raw EEG (62, T) at 200Hz.

        Uses 1-second windows with 0.5s hop. Returns (N, 310).
        """
        from scipy.signal import butter, sosfilt

        fs = 200
        nyq = fs / 2.0
        window_samples = fs
        hop_samples = fs // 2

        bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta":  (13, 30),
            "gamma": (30, 45),
        }

        T = eeg.shape[1]
        de_windows = []
        start = 0

        while start + window_samples <= T:
            window = eeg[:, start:start + window_samples]
            de_list = []
            for lo, hi in bands.values():
                sos = butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
                filtered = sosfilt(sos, window, axis=1)
                de_list.append(np.log(np.var(filtered, axis=1) + 1e-8))
            de_windows.append(np.concatenate(de_list).astype(np.float32))
            start += hop_samples

        if not de_windows:
            pad = np.zeros((eeg.shape[0], window_samples), dtype=np.float32)
            pad[:, :T] = eeg
            de_list = []
            for lo, hi in bands.values():
                sos = butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
                filtered = sosfilt(sos, pad, axis=1)
                de_list.append(np.log(np.var(filtered, axis=1) + 1e-8))
            de_windows.append(np.concatenate(de_list).astype(np.float32))

        return np.stack(de_windows)

    def predict_va(self, eeg: np.ndarray, subject_id: int = None) -> tuple:
        """Predict (valence, arousal) from EEG.

        Args:
            eeg: (62, T) raw preprocessed EEG at 200Hz, OR (N, 310) pre-extracted DE features
            subject_id: int in [1, 20] or None.
        Returns:
            (valence, arousal) in [-1, 1]
        """
        model = None
        if subject_id is not None:
            model = self._get_model(subject_id)
        if model is None:
            model = self._get_model(None)
        if model is None:
            raise RuntimeError("No trained model found in " + self.weights_dir)

        if eeg.ndim == 2 and eeg.shape[0] == N_EEG_CHANNELS and eeg.shape[1] > DE_DIM:
            de = self._de_from_raw(eeg)
        elif eeg.ndim == 2 and eeg.shape[1] == DE_DIM:
            de = eeg.astype(np.float32)
        elif eeg.ndim == 1 and eeg.shape[0] == DE_DIM:
            de = eeg.astype(np.float32).reshape(1, -1)
        else:
            raise ValueError(f"Expected (62, T) raw EEG or (N, 310) DE features, got {eeg.shape}")

        de = self._normalize(de, subject_id)
        x = torch.from_numpy(de).to(self.device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        avg_probs = probs.mean(axis=0)
        va = avg_probs @ self._va_matrix
        return float(np.clip(va[0], -1, 1)), float(np.clip(va[1], -1, 1))

    def predict_emotion(self, eeg: np.ndarray, subject_id: int = None) -> tuple:
        """Return top emotion name + probability dict."""
        model = None
        if subject_id is not None:
            model = self._get_model(subject_id)
        if model is None:
            model = self._get_model(None)
        if model is None:
            raise RuntimeError("No trained model found in " + self.weights_dir)

        if eeg.ndim == 2 and eeg.shape[0] == N_EEG_CHANNELS and eeg.shape[1] > DE_DIM:
            de = self._de_from_raw(eeg)
        elif eeg.ndim == 2 and eeg.shape[1] == DE_DIM:
            de = eeg.astype(np.float32)
        elif eeg.ndim == 1 and eeg.shape[0] == DE_DIM:
            de = eeg.astype(np.float32).reshape(1, -1)
        else:
            raise ValueError(f"Expected (62, T) raw EEG or (N, 310) DE features, got {eeg.shape}")

        de = self._normalize(de, subject_id)
        x = torch.from_numpy(de).to(self.device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        avg_probs = probs.mean(axis=0)
        top_idx = int(np.argmax(avg_probs))
        prob_dict = {EMOTION_NAMES[i]: float(avg_probs[i]) for i in range(len(EMOTION_NAMES))}
        return EMOTION_NAMES[top_idx], prob_dict
