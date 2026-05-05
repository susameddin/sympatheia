"""HRV feature extraction for YAAD ECG signals.

Feature functions are ported verbatim from the reference Kaggle notebook:
  https://www.kaggle.com/code/danielfesalbon/ecg-signals-emotion-recognition
  (original credit: https://github.com/chandanacharya1/ECG-Feature-extraction-using-Python)

Public API
----------
extract_all_features(ecg_signal)      → dict of 17 HRV features
extract_selected_features(ecg_signal) → np.ndarray (6,)  [sdrr, rmssd, sd1, sd2, skew, kurt]
build_feature_matrix(records, ...)    → (X, y)  (N, 6) float32 and (N,) int32
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import label as sci_label
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import LabelEncoder

from .config import (
    CACHE_DIR,
    EMOTION_NAMES,
    SELECTED_FEATURES,
)


# ---------------------------------------------------------------------------
# Peak detection helpers (from notebook)
# ---------------------------------------------------------------------------

def _detect_peaks(ecg_signal: pd.Series, threshold: float = 0.3,
                  qrs_filter: np.ndarray | None = None):
    """Cross-correlation peak detection.  Returns (peak_index_series, similarity)."""
    if qrs_filter is None:
        t = np.linspace(1.5 * np.pi, 3.5 * np.pi, 15)
        qrs_filter = np.sin(t)

    ecg_signal = (ecg_signal - ecg_signal.mean()) / ecg_signal.std()
    similarity = np.correlate(ecg_signal, qrs_filter, mode="same")
    similarity = similarity / np.max(similarity)
    return ecg_signal[similarity > threshold].index, similarity


def _group_peaks(p: np.ndarray, threshold: int = 5) -> np.ndarray:
    """Merge nearby peaks; return one median index per QRS complex."""
    output = np.empty(0)
    peak_groups, num_groups = sci_label(np.diff(p) < threshold)
    for i in np.unique(peak_groups)[1:]:
        peak_group = p[np.where(peak_groups == i)]
        output = np.append(output, np.median(peak_group))
    return output


def _rr_intervals(ecg_series: pd.Series) -> np.ndarray:
    """Detect R-peaks and return RR-interval array."""
    peaks, _ = _detect_peaks(ecg_series, threshold=0.3)
    grouped  = _group_peaks(peaks)
    return np.diff(grouped)


# ---------------------------------------------------------------------------
# Feature functions (from notebook — time-domain, nonlinear, statistical)
# ---------------------------------------------------------------------------

def calc_rmssd(sig: np.ndarray) -> float:
    diff_nni = np.diff(sig)
    return float(np.sqrt(np.mean(diff_nni ** 2)))


def calc_avrr(sig: np.ndarray) -> float:
    return float(np.mean(sig))


def calc_sdrr(sig: np.ndarray) -> float:
    import statistics
    return float(statistics.stdev(sig))


def calc_skew(sig: np.ndarray) -> float:
    return float(skew(sig))


def calc_kurt(sig: np.ndarray) -> float:
    return float(kurtosis(sig))


def calc_NNx(ecg_series: pd.Series) -> float:
    rr = _rr_intervals(ecg_series)
    if len(rr) < 2:
        return 0.0
    return float(np.sum(np.abs(np.diff(rr)) > 50))


def calc_pNNx(ecg_series: pd.Series) -> float:
    rr = _rr_intervals(ecg_series)
    if len(rr) < 2:
        return 0.0
    return float(100 * np.sum((np.abs(np.diff(rr)) > 50)) / len(rr))


def calc_SD1(sig: np.ndarray) -> float:
    diff_nn = np.diff(sig)
    return float(np.sqrt(np.std(diff_nn, ddof=1) ** 2 * 0.5))


def calc_SD2(sig: np.ndarray) -> float:
    diff_nn = np.diff(sig)
    return float(np.sqrt(2 * np.std(sig, ddof=1) ** 2
                         - 0.5 * np.std(diff_nn, ddof=1) ** 2))


def calc_SD1overSD2(sig: np.ndarray) -> float:
    diff_nn = np.diff(sig)
    sd1 = np.sqrt(np.std(diff_nn, ddof=1) ** 2 * 0.5)
    sd2 = np.sqrt(2 * np.std(sig, ddof=1) ** 2 - 0.5 * np.std(diff_nn, ddof=1) ** 2)
    return float(sd2 / sd1) if sd1 != 0 else 0.0


def calc_CSI(sig: np.ndarray) -> float:
    diff_nn = np.diff(sig)
    sd1 = np.sqrt(np.std(diff_nn, ddof=1) ** 2 * 0.5)
    sd2 = np.sqrt(2 * np.std(sig, ddof=1) ** 2 - 0.5 * np.std(diff_nn, ddof=1) ** 2)
    L = 4 * sd1
    T = 4 * sd2
    return float(L / T) if T != 0 else 0.0


def calc_CVI(sig: np.ndarray) -> float:
    diff_nn = np.diff(sig)
    sd1 = np.sqrt(np.std(diff_nn, ddof=1) ** 2 * 0.5)
    sd2 = np.sqrt(2 * np.std(sig, ddof=1) ** 2 - 0.5 * np.std(diff_nn, ddof=1) ** 2)
    L = 4 * sd1
    T = 4 * sd2
    val = L * T
    return float(np.log10(val)) if val > 0 else 0.0


def calc_modifiedCVI(sig: np.ndarray) -> float:
    diff_nn = np.diff(sig)
    sd1 = np.sqrt(np.std(diff_nn, ddof=1) ** 2 * 0.5)
    sd2 = np.sqrt(2 * np.std(sig, ddof=1) ** 2 - 0.5 * np.std(diff_nn, ddof=1) ** 2)
    L = 4 * sd1
    T = 4 * sd2
    return float(L ** 2 / T) if T != 0 else 0.0


def calc_meanrr(ecg_series: pd.Series) -> float:
    rr = _rr_intervals(ecg_series)
    return float(np.mean(rr)) if len(rr) > 0 else 0.0


def calc_medianrr(ecg_series: pd.Series) -> float:
    rr = _rr_intervals(ecg_series)
    return float(np.median(rr)) if len(rr) > 0 else 0.0


def calc_hr(ecg_series: pd.Series) -> float:
    rr = _rr_intervals(ecg_series)
    if len(rr) == 0:
        return 0.0
    hr = 60000 / rr
    return float(np.mean(hr))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_all_features(ecg_signal: np.ndarray) -> dict:
    """Extract all 17 HRV/statistical features from a 1-D ECG array.

    Args:
        ecg_signal: 1-D float array, at least ECG_SAMPLES (1000) points.

    Returns:
        dict with keys: meanrr, medianrr, sdrr, rmssd, sdrr_rmssd, hr,
                        NNx, pNNx, sd1, sd2, avrr, skew, kurt,
                        csi, cvi, modifiedcvi, sd1oversd2
    """
    sig    = ecg_signal.flatten().astype(float)
    series = pd.Series(sig, index=range(len(sig)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rmssd_val = calc_rmssd(sig)
        sdrr_val  = calc_sdrr(sig)
        return {
            "meanrr":      calc_meanrr(series),
            "medianrr":    calc_medianrr(series),
            "sdrr":        sdrr_val,
            "rmssd":       rmssd_val,
            "sdrr_rmssd":  sdrr_val / rmssd_val if rmssd_val != 0 else 0.0,
            "hr":          calc_hr(series),
            "NNx":         calc_NNx(series),
            "pNNx":        calc_pNNx(series),
            "sd1":         calc_SD1(sig),
            "sd2":         calc_SD2(sig),
            "avrr":        calc_avrr(sig),
            "skew":        calc_skew(sig),
            "kurt":        calc_kurt(sig),
            "csi":         calc_CSI(sig),
            "cvi":         calc_CVI(sig),
            "modifiedcvi": calc_modifiedCVI(sig),
            "sd1oversd2":  calc_SD1overSD2(sig),
        }


def extract_selected_features(ecg_signal: np.ndarray) -> np.ndarray:
    """Extract the 6 selected features used for classification.

    Returns:
        np.ndarray of shape (6,) in SELECTED_FEATURES order:
        [sdrr, rmssd, sd1, sd2, skew, kurt]
    """
    all_feats = extract_all_features(ecg_signal)
    return np.array([all_feats[k] for k in SELECTED_FEATURES], dtype=np.float32)


_ALL_FEATURE_KEYS = [
    "meanrr", "medianrr", "sdrr", "rmssd", "sdrr_rmssd", "hr",
    "NNx", "pNNx", "sd1", "sd2", "avrr", "skew", "kurt",
    "csi", "cvi", "modifiedcvi", "sd1oversd2",
]
N_ALL_FEATURES = len(_ALL_FEATURE_KEYS)


def _encode_labels(records: list[dict]) -> np.ndarray:
    le = LabelEncoder()
    le.fit(EMOTION_NAMES)
    return np.array([le.transform([r["emotion"]])[0] for r in records], dtype=np.int32)


def build_feature_matrix(
    records: list[dict],
    cache_dir: str = CACHE_DIR,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) feature matrix from loaded YAAD records.

    Args:
        records:   output of loader.load_all_data()
        cache_dir: directory for .npz cache
        force:     recompute even if cache exists

    Returns:
        X: (N, 6)  float32 — selected HRV features
        y: (N,)    int32   — emotion label index into EMOTION_NAMES
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "yaad_features.npz")

    if not force and os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["X"], data["y"]

    le = LabelEncoder()
    le.fit(EMOTION_NAMES)

    X_list, y_list = [], []
    for i, rec in enumerate(records):
        feats = extract_selected_features(rec["ecg"])
        label = le.transform([rec["emotion"]])[0]
        X_list.append(feats)
        y_list.append(label)
        if (i + 1) % 50 == 0:
            print(f"  features: {i + 1}/{len(records)}")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)

    np.savez_compressed(cache_path, X=X, y=y)
    return X, y


def build_all_feature_matrix(
    records: list[dict],
    cache_dir: str = CACHE_DIR,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) with all 17 HRV features (not just the 6 selected ones).

    Returns:
        X: (N, 17) float32
        y: (N,)    int32
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "yaad_features_all17.npz")

    if not force and os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["X"], data["y"]

    X_list = []
    for i, rec in enumerate(records):
        all_f = extract_all_features(rec["ecg"])
        X_list.append([all_f[k] for k in _ALL_FEATURE_KEYS])
        if (i + 1) % 50 == 0:
            print(f"  all-features: {i + 1}/{len(records)}")

    X = np.array(X_list, dtype=np.float32)
    y = _encode_labels(records)

    np.savez_compressed(cache_path, X=X, y=y)
    return X, y
