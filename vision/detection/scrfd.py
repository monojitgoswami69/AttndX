"""
Face detection via InsightFace SCRFD (det_10g.onnx).

SCRFD is a high-accuracy FPN-based face detector that outputs:
  - bounding box (x1, y1, x2, y2)
  - 5 facial landmarks (left eye, right eye, nose, left mouth, right mouth)
  - confidence score

This wraps InsightFace's SCRFD class to produce the same Detection dataclass
as the YuNet detector, making it a drop-in replacement.

Advantages over YuNet:
  - Significantly higher detection accuracy (WiderFace Hard: ~95% vs ~82%)
  - Better handling of extreme poses, partial occlusion, small faces
  - Designed and calibrated to pair with InsightFace ArcFace models

Thread-safety: The underlying ONNX Runtime session is thread-safe per spec,
but we serialize detect() calls with a lock to avoid any issues with internal
caching state.
"""
import logging
import threading
from pathlib import Path

import cv2
import numpy as np

from config import Config
from vision.detection import Detection

logger = logging.getLogger(__name__)


class SCRFDDetector:
    """InsightFace SCRFD face detector with 5-point landmark output.

    Uses the insightface.model_zoo.scrfd.SCRFD class internally,
    producing Detection objects compatible with the rest of the pipeline.
    """

    def __init__(self):
        model_path = Config.model_path("detector")
        if not model_path.exists():
            raise FileNotFoundError(f"SCRFD model not found: {model_path}")

        self.score_threshold = Config.get("detector", "score_threshold")
        self.nms_threshold = Config.get("detector", "nms_threshold")
        self.min_face_size = Config.get("detector", "min_face_size")
        input_size = tuple(Config.get("detector", "input_size"))

        from insightface.model_zoo.scrfd import SCRFD

        # Determine ONNX Runtime providers
        import onnxruntime as ort
        providers = Config.get("onnx", "providers")
        available = ort.get_available_providers()
        active_providers = [p for p in providers if p in available]
        if not active_providers:
            active_providers = ["CPUExecutionProvider"]

        # Create ONNX session with our provider preferences
        session = ort.InferenceSession(
            str(model_path), providers=active_providers,
        )

        self._detector = SCRFD(model_file=str(model_path), session=session)
        self._detector.prepare(
            ctx_id=-1,  # provider already set via session
            det_thresh=self.score_threshold,
            nms_thresh=self.nms_threshold,
            input_size=input_size,
        )
        self._lock = threading.Lock()
        logger.info(
            f"SCRFD loaded from {model_path.name} "
            f"(providers: {active_providers}, "
            f"det_thresh: {self.score_threshold}, "
            f"input_size: {input_size})"
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect all faces in a BGR frame. Returns list of Detection."""
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]

        try:
            with self._lock:
                bboxes, kpss = self._detector.detect(frame, max_num=0)
        except Exception as e:
            logger.error(f"SCRFD detect() error: {e}")
            return []

        if bboxes is None or len(bboxes) == 0:
            return []

        detections: list[Detection] = []
        for i in range(len(bboxes)):
            score = float(bboxes[i, 4])
            if score < self.score_threshold:
                continue

            x1 = max(0, int(bboxes[i, 0]))
            y1 = max(0, int(bboxes[i, 1]))
            x2 = min(w, int(bboxes[i, 2]))
            y2 = min(h, int(bboxes[i, 3]))

            fw = x2 - x1
            fh = y2 - y1
            if fw < self.min_face_size or fh < self.min_face_size:
                continue

            # Landmarks: (5, 2) float32
            landmarks = np.zeros((5, 2), dtype=np.float32)
            if kpss is not None and i < len(kpss):
                landmarks = kpss[i].astype(np.float32)

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
