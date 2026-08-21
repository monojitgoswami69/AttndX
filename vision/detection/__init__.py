"""
Face detection types shared across detector implementations.

The Detection dataclass is the common output format for all detectors
(YuNet, SCRFD, etc). Downstream components (alignment, quality, tracking)
depend only on this interface, not on any specific detector.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """A single face detection."""
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2) in frame coords
    confidence: float
    landmarks: np.ndarray              # shape (5, 2), float32, (x, y) per landmark
    cropped_face: np.ndarray           # the bbox crop from the frame (BGR)
