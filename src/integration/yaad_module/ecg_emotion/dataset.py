"""Train/test split for YAAD ECG features."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from .config import TEST_SIZE, RANDOM_STATE


def make_splits(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random 80/20 train/test split matching the reference notebook.

    Args:
        X: (N, 6) feature matrix
        y: (N,) integer labels

    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=seed)
