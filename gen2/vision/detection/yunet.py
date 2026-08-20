"""
Face detection via OpenCV YuNet.

YuNet is a lightweight single-stage face detector that outputs:
  - bounding box (x, y, w, h)
  - 5 facial landmarks (left eye, right eye, nose, left mouth, right mouth)
  - confidence score

The 5 landmarks are used directly for ArcFace alignment — no separate
landmark model is needed. This is the standard InsightFace-style pipeline.

Thread-safety: cv2.FaceDetectorYN is NOT guaranteed thread-safe.
All detect() calls are serialized by a threading.Lock.
"""
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from gen2.config import Config

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single face detection."""
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2) in frame coords
    confidence: float
    landmarks: np.ndarray              # shape (5, 2), float32, (x, y) per landmark
    cropped_face: np.ndarray           # the bbox crop from the frame (BGR)


class YuNetDetector:
    """OpenCV YuNet face detector with 5-point landmark output."""

    # Landmark indices in YuNet output (after bbox x,y,w,h)
    LANDMARK_INDICES = [
        (4, 5),    # left eye
        (6, 7),    # right eye
        (8, 9),    # nose
        (10, 11),  # left mouth corner
        (12, 13),  # right mouth corner
    ]

    def __init__(self):
        model_path = Config.model_path("detector")
        if not model_path.exists():
            raise FileNotFoundError(f"YuNet model not found: {model_path}")

        self.score_threshold = Config.get("detector", "score_threshold")
        self.nms_threshold = Config.get("detector", "nms_threshold")
        self.top_k = Config.get("detector", "top_k")
        self.min_face_size = Config.get("detector", "min_face_size")
        input_size = tuple(Config.get("detector", "input_size"))

        self._detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            input_size,
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k,
        )
        self._last_shape: tuple[int, int] = (0, 0)  # (w, h)
        self._lock = threading.Lock()
        logger.info(f"YuNet loaded from {model_path}")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect all faces in a BGR frame. Returns list of Detection."""
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        self._ensure_input_size(w, h)

        try:
            with self._lock:
                retval, faces = self._detector.detect(frame)
        except Exception as e:
            logger.error(f"YuNet detect() error: {e}")
            return []

        if faces is None or len(faces) == 0:
            return []

        detections: list[Detection] = []
        for row in faces:
            score = float(row[14]) if len(row) > 14 else 0.0
            if score < self.score_threshold:
                continue

            x, y, fw, fh = row[0], row[1], row[2], row[3]
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(w, int(x + fw))
            y2 = min(h, int(y + fh))

            fw_px = x2 - x1
            fh_px = y2 - y1
            if fw_px < self.min_face_size or fh_px < self.min_face_size:
                continue

            landmarks = np.zeros((5, 2), dtype=np.float32)
            for i, (lx, ly) in enumerate(self.LANDMARK_INDICES):
                landmarks[i] = [row[lx], row[ly]]

            cropped = frame[y1:y2, x1:x2].copy()
            if cropped.size == 0:
                continue

            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                confidence=score,
                landmarks=landmarks,
                cropped_face=cropped,
            ))

        return detections

    def _ensure_input_size(self, w: int, h: int):
        if (w, h) != self._last_shape:
            if hasattr(self._detector, "setInputSize"):
                self._detector.setInputSize((w, h))
            elif hasattr(self._detector, "setInputShape"):
                self._detector.setInputShape((w, h))
            self._last_shape = (w, h)
