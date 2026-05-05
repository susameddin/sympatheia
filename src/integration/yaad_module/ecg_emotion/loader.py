"""Load YAAD ECG .dat files and match them to annotation labels."""
from __future__ import annotations

import os
import re
import pickle

import numpy as np
import pandas as pd

from .config import (
    ECG_MULTIMODAL_DIR,
    ECG_SINGLEMODAL_DIR,
    GSR_MULTIMODAL_DIR,
    ANNOTATION_MULTIMODAL,
    ANNOTATION_SINGLEMODAL,
    ECG_SAMPLES,
    CACHE_DIR,
)


def load_annotations() -> pd.DataFrame:
    """Load and merge multimodal + single-modal annotation files.

    Returns a DataFrame with columns:
        participant_id, session_id, video_id, emotion, modal
    """
    mm = pd.read_excel(ANNOTATION_MULTIMODAL)
    mm = mm.rename(columns={"Participant Id": "participant_id",
                             "Session ID":     "session_id",
                             "Video ID":       "video_id",
                             "Emotion":        "emotion"})
    mm["modal"] = "M"

    sm = pd.read_excel(ANNOTATION_SINGLEMODAL)
    sm = sm.rename(columns={"Participant Id": "participant_id",
                             "Session Id":    "session_id",
                             "Video Id":      "video_id",
                             "Emotion":       "emotion"})
    sm["modal"] = "S"

    cols = ["participant_id", "session_id", "video_id", "emotion", "modal"]
    return pd.concat([mm[cols], sm[cols]], ignore_index=True)


def parse_filename(fname: str) -> tuple[int, int, int] | None:
    """Extract (session, participant, video) from an ECG filename.

    Handles both lower- and upper-case 'S':
        ECGdata_s1p3v7.dat  →  (1, 3, 7)
        ECGdata_S2p9v5.dat  →  (2, 9, 5)

    Returns None if the filename does not match the expected pattern.
    """
    basename = os.path.splitext(os.path.basename(fname))[0]
    # strip prefix, case-insensitive
    m = re.match(r"ECGdata_[sS](\d+)p(\d+)v(\d+)$", basename, re.IGNORECASE)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def load_ecg_file(path: str) -> np.ndarray:
    """Load ECG signal from a .dat file (comma-separated on one line).

    Returns the first ECG_SAMPLES values as a float32 array of shape (ECG_SAMPLES,).
    """
    data = np.loadtxt(path, delimiter=",")
    data = data.flatten()[:ECG_SAMPLES]
    return data.astype(np.float32)


def load_gsr_file(path: str) -> np.ndarray:
    """Load GSR signal from a .dat file (one value per line).

    Returns the first ECG_SAMPLES values as a float32 array of shape (ECG_SAMPLES,).
    """
    data = np.loadtxt(path)
    data = data.flatten()[:ECG_SAMPLES]
    return data.astype(np.float32)


def load_all_data(force: bool = False) -> list[dict]:
    """Load every matched ECG file with its emotion label.

    Each record is a dict::

        {
            "ecg":     np.ndarray (ECG_SAMPLES,) float32
            "emotion": str   e.g. "Happy"
            "session_id":     int
            "participant_id": int
            "video_id":       int
            "modal":          "M" | "S"
        }

    Results are cached to ``cache/yaad_raw.pkl``.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "yaad_raw.pkl")

    if not force and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    annot = load_annotations()
    # Build lookup: (session_id, participant_id, video_id) → list[row]
    lookup: dict[tuple, list] = {}
    for _, row in annot.iterrows():
        key = (int(row["session_id"]), int(row["participant_id"]), int(row["video_id"]))
        lookup.setdefault(key, []).append(row)

    records = []
    for ecg_dir in (ECG_MULTIMODAL_DIR, ECG_SINGLEMODAL_DIR):
        for fname in sorted(os.listdir(ecg_dir)):
            if not fname.lower().endswith(".dat"):
                continue
            parsed = parse_filename(fname)
            if parsed is None:
                continue
            session, participant, video = parsed
            key = (session, participant, video)
            if key not in lookup:
                continue
            ecg = load_ecg_file(os.path.join(ecg_dir, fname))
            for row in lookup[key]:
                records.append({
                    "ecg":            ecg,
                    "emotion":        row["emotion"],
                    "session_id":     session,
                    "participant_id": participant,
                    "video_id":       video,
                    "modal":          row["modal"],
                })

    with open(cache_path, "wb") as f:
        pickle.dump(records, f)

    return records


def load_multimodal_data(force: bool = False) -> list[dict]:
    """Load multimodal records that have both ECG and GSR signals.

    Each record dict contains:
        ecg, gsr:         np.ndarray (ECG_SAMPLES,) float32
        emotion:          str
        session_id, participant_id, video_id: int
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "yaad_multimodal.pkl")

    if not force and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    annot = load_annotations()
    lookup: dict[tuple, list] = {}
    for _, row in annot.iterrows():
        key = (int(row["session_id"]), int(row["participant_id"]), int(row["video_id"]))
        lookup.setdefault(key, []).append(row)

    records = []
    for fname in sorted(os.listdir(ECG_MULTIMODAL_DIR)):
        if not fname.lower().endswith(".dat"):
            continue
        parsed = parse_filename(fname)
        if parsed is None:
            continue
        session, participant, video = parsed
        key = (session, participant, video)
        if key not in lookup:
            continue

        ecg_path = os.path.join(ECG_MULTIMODAL_DIR, fname)
        # GSR filename: replace ECGdata prefix with GSRdata
        gsr_fname = fname.replace("ECGdata_", "GSRdata_").replace("ECGdata_S", "GSRdata_S")
        gsr_path  = os.path.join(GSR_MULTIMODAL_DIR, gsr_fname)
        if not os.path.exists(gsr_path):
            # try case variants
            gsr_fname_alt = "GSRdata_" + fname[len("ECGdata_"):]
            gsr_path = os.path.join(GSR_MULTIMODAL_DIR, gsr_fname_alt)
            if not os.path.exists(gsr_path):
                continue

        ecg = load_ecg_file(ecg_path)
        gsr = load_gsr_file(gsr_path)

        for row in lookup[key]:
            records.append({
                "ecg":            ecg,
                "gsr":            gsr,
                "emotion":        row["emotion"],
                "session_id":     session,
                "participant_id": participant,
                "video_id":       video,
            })

    with open(cache_path, "wb") as f:
        pickle.dump(records, f)

    return records
