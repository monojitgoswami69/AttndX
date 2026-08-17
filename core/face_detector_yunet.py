"""
Fast face detection module using OpenCV's YuNet detector.

YuNet is a lightweight single-stage face detector that ships as an ONNX model
and runs through OpenCV's DNN module on CPU. It is dramatically faster than
the YOLOv8-person + Haar-cascade two-stage pipeline, does not require the
Ultralytics stack, and avoids MediaPipe's Metal-service crash on macOS.

The detector downloads its ~230KB ONNX model on first use and exposes the same
interface as YOLOFaceDetector so it is a drop-in replacement.
"""

import os
import threading
import urllib.request
import numpy as np
import cv2
from core.config import Config


_MODEL_DIR = Config.BASE_DIR / "models"
_YUNET_FILE = "face_detection_yunet_2023mar.onnx"
_YUNET_URLS = [
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://storage.googleapis.com/opencv-zoo/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
]


def _download_yunet(dest: str) -> bool:
    for url in _YUNET_URLS:
        try:
            print(f"[YuNetDetector] Downloading model: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                with open(dest, "wb") as f:
                    f.write(resp.read())
            print(f"[YuNetDetector] Saved to {dest}")
            return True
        except Exception as e:
            print(f"[YuNetDetector] Download failed: {e}")
            if os.path.exists(dest):
                os.remove(dest)
    return False


class YuNetFaceDetector:
    """Detect faces using OpenCV's YuNet (cv2.FaceDetectorYN).

    Same public interface as YOLOFaceDetector:
        detect_faces(frame) -> list[dict] with bbox/confidence/cropped_face
        detect_single_face(frame) -> dict | None
        draw_detections(frame, detections, names) -> annotated frame
    """

    def __init__(self, confidence: float | None = None):
        self.confidence = confidence if confidence is not None else Config.YOLO_CONFIDENCE
        self.padding = Config.FACE_PADDING

        model_path = _MODEL_DIR / _YUNET_FILE
        if not model_path.exists():
            _MODEL_DIR.mkdir(parents=True, exist_ok=True)
            if not _download_yunet(str(model_path)):
                raise RuntimeError(
                    "Could not download YuNet face detector model. "
                    f"Please manually place {_YUNET_FILE} in {model_path}"
                )

        # Create the detector with a default input size; updated per-frame.
        self._detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),
            score_threshold=self.confidence,
            nms_threshold=0.3,
            top_k=50,
        )
        self._last_input_shape: tuple[int, int] = (0, 0)
        # cv2.FaceDetectorYN is not guaranteed thread-safe; serialize detect()
        # calls so the WebRTC preview thread and the check thread can't race.
        self._detect_lock = threading.Lock()
        print(f"[YuNetDetector] Loaded YuNet model from {model_path}")

    def _ensure_input_shape(self, w: int, h: int) -> None:
        if (w, h) != self._last_input_shape:
            # OpenCV 5.x renamed setInputShape -> setInputSize; support both.
            if hasattr(self._detector, "setInputSize"):
                self._detector.setInputSize((w, h))
            elif hasattr(self._detector, "setInputShape"):
                self._detector.setInputShape((w, h))
            self._last_input_shape = (w, h)

    def detect_faces(self, frame: np.ndarray) -> list[dict]:
        """Detect all faces in a BGR frame.

        Returns a list of dicts with:
            bbox: (x1, y1, x2, y2)
            confidence: float 0-1
            cropped_face: numpy array of the padded face crop
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        self._ensure_input_shape(w, h)

        try:
            with self._detect_lock:
                retval, faces = self._detector.detect(frame)
        except Exception as e:
            print(f"[YuNetDetector] detect() error: {e}")
            return []

        if faces is None or len(faces) == 0:
            return []

        detections: list[dict] = []
        for row in faces:
            x, y, fw, fh = row[0], row[1], row[2], row[3]
            score = float(row[14]) if len(row) > 14 else 0.0
            if score < self.confidence:
                continue

            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(w, int(x + fw))
            y2 = min(h, int(y + fh))

            fw_px = x2 - x1
            fh_px = y2 - y1
            if fw_px < Config.MIN_FACE_SIZE or fh_px < Config.MIN_FACE_SIZE:
                continue

            # Proportional padding for full head coverage (matches YOLO detector)
            pad_x = max(self.padding, int(fw_px * 0.18))
            pad_y = max(self.padding, int(fh_px * 0.22))
            px1 = max(0, x1 - pad_x)
            py1 = max(0, y1 - pad_y)
            px2 = min(w, x2 + pad_x)
            py2 = min(h, y2 + pad_y)

            cropped = frame[py1:py2, px1:px2].copy()
            if cropped.size == 0:
                continue

            detections.append({
                "bbox": (px1, py1, px2, py2),
                "confidence": score,
                "cropped_face": cropped,
            })

        return detections

    def detect_single_face(self, frame: np.ndarray) -> dict | None:
        """Detect exactly one face; return the highest-confidence one if many."""
        detections = self.detect_faces(frame)
        if not detections:
            return None
        if len(detections) == 1:
            return detections[0]
        return max(detections, key=lambda d: d["confidence"])

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: list[dict],
        names: list[str | None] | None = None,
    ) -> np.ndarray:
        """Draw bounding boxes and labels on a copy of the frame."""
        annotated = frame.copy()

        if names is None:
            names = [None] * len(detections)

        for det, name in zip(detections, names):
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]

            is_known = name is not None and name.lower() != "unknown"
            color = (0, 200, 0) if is_known else (0, 0, 220)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = name if is_known else "Unknown"
            label_text = f"{label} ({conf:.2f})"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            label_y = max(y1 - 10, th + 10)
            cv2.rectangle(
                annotated,
                (x1, label_y - th - 6),
                (x1 + tw + 8, label_y + 4),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label_text,
                (x1 + 4, label_y - 2),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        return annotated
