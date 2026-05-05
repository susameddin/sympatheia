"""SEED-VII EEG dataset: load pre-extracted DE_LDS features for 7-class emotion classification."""

import random

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

from .config import (
    SEED_FEATURES_DIR,
    EMOTION_NAMES,
    VIDEO_EMOTIONS,
    SESSION_VIDEOS,
    N_VIDEOS,
    NUM_CLASSES,
    SEED,
)


def load_subject_features(
    subject_id: int,
    features_dir: str = SEED_FEATURES_DIR,
    use_lds: bool = True,
) -> tuple:
    """Load all DE_LDS features and labels for one SEED-VII subject.

    Args:
        subject_id: int in [1, 20]
        features_dir: path to EEG_features directory containing {1..20}.mat
        use_lds: if True use de_LDS (LDS-smoothed), else raw de

    Returns:
        features: list of 80 arrays, each (N_windows, 310) float32
        labels:   list of 80 int labels (0-6, matching EMOTION_NAMES index)
        video_ids: list of 80 video IDs (1-80)
    """
    mat = sio.loadmat(f"{features_dir}/{subject_id}.mat")
    prefix = "de_LDS_" if use_lds else "de_"

    features = []
    labels = []
    video_ids = []

    for vid in range(1, N_VIDEOS + 1):
        key = f"{prefix}{vid}"
        feat = mat[key]  # (N_win, 5, 62)
        n_win = feat.shape[0]
        feat_flat = feat.reshape(n_win, -1).astype(np.float32)  # (N_win, 310)

        emotion_name = VIDEO_EMOTIONS[vid]
        label = EMOTION_NAMES.index(emotion_name)

        features.append(feat_flat)
        labels.append(label)
        video_ids.append(vid)

    return features, labels, video_ids


def _zscore_by_session(
    features: list,
    video_ids_list: list,
    session_norm_stats: dict = None,
) -> list:
    """Z-score normalize features per session.

    Sessions are recorded on different days, so normalizing within each session
    removes session-to-session drift while preserving within-session discriminative
    patterns. This is the standard approach in SEED literature.

    Args:
        features: list of arrays, one per video
        video_ids_list: corresponding video IDs (1-indexed)
        session_norm_stats: optional pre-computed {session_id: (mean, std)} dict.
            When provided, uses these fixed statistics instead of computing from
            the current feature list. Allows consistent normalization across
            train/val/test splits (all normalized using training-set statistics).
    """
    # Group features by session
    session_indices = {}  # session_id -> list of indices into features
    for i, vid in enumerate(video_ids_list):
        for sess, sess_vids in SESSION_VIDEOS.items():
            if vid in sess_vids:
                session_indices.setdefault(sess, []).append(i)
                break

    normed = [None] * len(features)
    for sess, indices in session_indices.items():
        if session_norm_stats is not None and sess in session_norm_stats:
            mean, std = session_norm_stats[sess]
        else:
            sess_feats = np.concatenate([features[i] for i in indices], axis=0)
            mean = sess_feats.mean(axis=0)
            std = sess_feats.std(axis=0) + 1e-8
        for i in indices:
            normed[i] = (features[i] - mean) / std

    return normed


def compute_session_norm_stats(
    subject_id: int,
    features_dir: str = SEED_FEATURES_DIR,
    use_lds: bool = True,
) -> dict:
    """Compute per-session z-score statistics from ALL 80 videos of a subject.

    Returns:
        {session_id: (mean, std)} for sessions 1-4, where mean/std are (310,) float32.
    """
    feats, _, vids = load_subject_features(subject_id, features_dir, use_lds)
    session_indices = {}
    for i, vid in enumerate(vids):
        for sess, sess_vids in SESSION_VIDEOS.items():
            if vid in sess_vids:
                session_indices.setdefault(sess, []).append(i)
                break
    stats = {}
    for sess, indices in session_indices.items():
        sess_feats = np.concatenate([feats[i] for i in indices], axis=0)
        stats[sess] = (sess_feats.mean(axis=0), sess_feats.std(axis=0) + 1e-8)
    return stats


class SEEDEEGDataset(Dataset):
    """SEED-VII DE_LDS feature dataset for 7-class emotion classification.

    Each item:
        x: (310,) float32 — session-normalized DE_LDS feature vector
        y: int — emotion label (0-6)
    """

    def __init__(
        self,
        subject_ids: list,
        features_dir: str = SEED_FEATURES_DIR,
        video_ids: list = None,
        use_lds: bool = True,
        normalize: bool = True,
        session_norm_stats: dict = None,
    ):
        """
        Args:
            subject_ids: list of subject IDs to include [1..20]
            features_dir: path to EEG_features dir
            video_ids: if given, only include these video IDs (for session-based splitting)
            use_lds: use LDS-smoothed features (recommended)
            normalize: z-score normalize per session (recommended)
            session_norm_stats: pre-computed {session_id: (mean, std)} from
                compute_session_norm_stats(). When provided, ensures all splits
                (train/val/test) are normalized with the same statistics.
        """
        self.x_list: list[np.ndarray] = []
        self.y_list: list[int] = []

        for sid in subject_ids:
            feats, labs, vids = load_subject_features(sid, features_dir, use_lds)

            # Z-score normalize per session (all 4 sessions independently)
            if normalize:
                feats = _zscore_by_session(feats, vids, session_norm_stats)

            for feat, lab, vid in zip(feats, labs, vids):
                if video_ids is not None and vid not in video_ids:
                    continue
                n_win = feat.shape[0]
                for w in range(n_win):
                    self.x_list.append(feat[w].astype(np.float32))
                    self.y_list.append(lab)

    def __len__(self) -> int:
        return len(self.x_list)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.x_list[idx]),
            torch.tensor(self.y_list[idx], dtype=torch.long),
        )


def get_paper_fold_split(fold_idx: int) -> tuple:
    """4-fold CV matching the paper's Fig. 1 structure.

    Each fold has exactly 5 consecutive videos from each of the 4 sessions,
    so both train and test always contain data from all 4 sessions.
    This matches the evaluation protocol in Jiang et al. (IEEE TAffComp 2025).

    Args:
        fold_idx: 0-3

    Returns:
        (train_vids, test_vids): each a sorted list of 1-indexed video IDs.
        test_vids has 20 videos (5/session); train_vids has 60 videos (15/session).
    """
    test_vids = []
    train_vids = []
    for session_id in range(1, 5):
        vids = SESSION_VIDEOS[session_id]   # 20 videos, ordered
        lo = fold_idx * 5
        hi = lo + 5
        test_vids.extend(vids[lo:hi])
        train_vids.extend(vids[:lo] + vids[hi:])
    return sorted(train_vids), sorted(test_vids)


def get_session_split(test_session: int) -> tuple:
    """Return (train_video_ids, test_video_ids) for leave-one-session-out CV.

    Args:
        test_session: session number to hold out (1-4)

    Returns:
        train_videos: list of video IDs for training
        test_videos: list of video IDs for testing
    """
    test_videos = SESSION_VIDEOS[test_session]
    train_videos = []
    for s in range(1, 5):
        if s != test_session:
            train_videos.extend(SESSION_VIDEOS[s])
    return train_videos, test_videos


def stratified_video_split(
    test_frac: float = 0.2,
    val_frac: float = 0.125,
    rng_seed: int = SEED,
) -> tuple:
    """Stratified random split of the 80 SEED-VII videos into train/val/test.

    Splits per emotion class so every split contains all 7 classes.
    With the default fractions (0.2 test, 0.125 val): each class has
    ~11-12 videos → ~2 test, ~1 val, ~8-9 train.

    Args:
        test_frac: fraction of videos per class to hold out as test
        val_frac: fraction of *remaining* (non-test) videos per class for validation
        rng_seed: random seed for reproducibility

    Returns:
        (train_vids, val_vids, test_vids): lists of 1-indexed video IDs
    """
    rng = random.Random(rng_seed)

    # Group video IDs by emotion class
    class_to_vids: dict = {name: [] for name in EMOTION_NAMES}
    for vid, emo in VIDEO_EMOTIONS.items():
        class_to_vids[emo].append(vid)

    train_vids, val_vids, test_vids = [], [], []

    for emo in EMOTION_NAMES:
        vids = sorted(class_to_vids[emo])
        rng.shuffle(vids)
        n_test = max(1, round(len(vids) * test_frac))
        remaining = vids[n_test:]
        n_val = max(1, round(len(remaining) * val_frac))
        test_vids.extend(vids[:n_test])
        val_vids.extend(remaining[:n_val])
        train_vids.extend(remaining[n_val:])

    return sorted(train_vids), sorted(val_vids), sorted(test_vids)


def get_normalization_stats(dataset: "SEEDEEGDataset") -> tuple:
    """Compute mean and std of all feature vectors in a dataset.

    Returns:
        (mean, std): each shape (DE_DIM,) float32 numpy arrays
    """
    x = np.stack(dataset.x_list, axis=0)  # (N, 310)
    mean = x.mean(axis=0).astype(np.float32)
    std = (x.std(axis=0) + 1e-8).astype(np.float32)
    return mean, std
