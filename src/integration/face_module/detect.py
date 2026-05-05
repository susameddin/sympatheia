"""Face detection and cropping for webcam frames."""

import numpy as np


def detect_and_crop_face(
    frame: np.ndarray,
    min_confidence: float = 0.5,
    margin: float = 0.2,
) -> np.ndarray:
    """Detect the largest face in a frame and return the cropped region.

    Tries (in order): MediaPipe → MTCNN → OpenCV Haar → center-crop fallback.
    """
    crop = _detect_with_mediapipe(frame, min_confidence, margin)
    if crop is not None:
        return crop
    crop = _detect_with_mtcnn(frame, min_confidence, margin)
    if crop is not None:
        return crop
    return _detect_with_opencv(frame, margin)


def _detect_with_mediapipe(frame: np.ndarray, min_confidence: float, margin: float):
    """Face detector using MediaPipe (most accurate, no protobuf conflict on 4.25.x)."""
    try:
        import mediapipe as mp
    except ImportError:
        return None

    h, w = frame.shape[:2]
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=min_confidence
    )
    results = detector.process(frame)
    detector.close()

    if not results.detections:
        return None

    best = max(results.detections, key=lambda d: d.score[0])
    bb = best.location_data.relative_bounding_box
    cx = (bb.xmin + bb.width / 2) * w
    cy = (bb.ymin + bb.height / 2) * h
    side = max(bb.width * w, bb.height * h) * (1 + margin)
    x1 = int(max(0, cx - side / 2))
    y1 = int(max(0, cy - side / 2))
    x2 = int(min(w, cx + side / 2))
    y2 = int(min(h, cy + side / 2))
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def _detect_with_mtcnn(frame: np.ndarray, min_confidence: float, margin: float):
    """Face detector using facenet-pytorch MTCNN (PyTorch, no protobuf dependency)."""
    try:
        from facenet_pytorch import MTCNN
        import torch
    except ImportError:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(keep_all=False, device=device, post_process=False)

    from PIL import Image
    pil = Image.fromarray(frame)
    boxes, probs = mtcnn.detect(pil)

    if boxes is None or len(boxes) == 0:
        return None

    best = int(np.argmax(probs))
    if probs[best] < min_confidence:
        return None

    x1, y1, x2, y2 = boxes[best]
    h, w = frame.shape[:2]
    fw, fh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(fw, fh) * (1 + margin)
    x1 = int(max(0, cx - side / 2))
    y1 = int(max(0, cy - side / 2))
    x2 = int(min(w, cx + side / 2))
    y2 = int(min(h, cy + side / 2))
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def _detect_with_opencv(frame: np.ndarray, margin: float) -> np.ndarray:
    """Face detector using OpenCV Haar cascade."""
    try:
        import cv2
    except ImportError:
        return _center_crop(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    cascade_names = [
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_frontalface_default.xml",
    ]
    faces = ()
    for name in cascade_names:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        if cascade.empty():
            continue
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20),
        )
        if len(faces) > 0:
            break

    if len(faces) == 0:
        return _center_crop(frame)

    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    h, w = frame.shape[:2]
    cx, cy = x + fw / 2, y + fh / 2
    side = max(fw, fh) * (1 + margin)
    x1 = int(max(0, cx - side / 2))
    y1 = int(max(0, cy - side / 2))
    x2 = int(min(w, cx + side / 2))
    y2 = int(min(h, cy + side / 2))
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else _center_crop(frame)


def _center_crop(frame: np.ndarray) -> np.ndarray:
    """Fallback: take a centered square crop from the frame."""
    h, w = frame.shape[:2]
    side = min(h, w)
    y1 = (h - side) // 2
    x1 = (w - side) // 2
    return frame[y1:y1 + side, x1:x1 + side]
