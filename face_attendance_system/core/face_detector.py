"""
YOLOv8-based face detection module.
Handles loading the YOLO model, detecting faces with bounding boxes,
and drawing annotated results on frames.
"""

import numpy as np
import cv2
from ultralytics import YOLO
from core.config import Config


class YOLOFaceDetector:
    """Detect faces in frames using a YOLOv8 model."""

    def __init__(self, model_name: str | None = None, confidence: float | None = None):
        """
        Initialize the YOLOv8 face detector.

        Args:
            model_name: YOLO model file name. Auto-downloads if not present.
            confidence: Minimum detection confidence threshold.
        """
        self.model_name = model_name or Config.YOLO_MODEL_NAME
        self.confidence = confidence if confidence is not None else Config.YOLO_CONFIDENCE
        self.padding = Config.FACE_PADDING

        print(f"[FaceDetector] Loading YOLO model: {self.model_name}")
        self.model = YOLO(self.model_name)
        print("[FaceDetector] Model loaded successfully.")

    def detect_faces(self, frame: np.ndarray) -> list[dict]:
        """
        Detect all faces in a frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            List of detection dicts, each containing:
                - bbox: (x1, y1, x2, y2) integers
                - confidence: float 0-1
                - cropped_face: numpy array of the cropped face region (with padding)
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False,
        )

        detections = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            for box in result.boxes:
                # Filter: only accept "person" class (class 0) detections
                # YOLOv8n is trained on COCO; class 0 = person
                # We use person detection + crop upper body/face region
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # For a general YOLO model, approximate face region
                # from person bounding box (upper portion)
                box_h = y2 - y1
                box_w = x2 - x1

                # If detection is tall (full body), crop to upper ~30%
                aspect = box_h / max(box_w, 1)
                if aspect > 1.5:
                    y2 = y1 + int(box_h * 0.3)

                # Apply padding
                pad = self.padding
                x1_pad = max(0, x1 - pad)
                y1_pad = max(0, y1 - pad)
                x2_pad = min(w, x2 + pad)
                y2_pad = min(h, y2 + pad)

                # Crop the face region
                cropped = frame[y1_pad:y2_pad, x1_pad:x2_pad].copy()

                if cropped.size == 0:
                    continue

                detections.append({
                    "bbox": (x1_pad, y1_pad, x2_pad, y2_pad),
                    "confidence": conf,
                    "cropped_face": cropped,
                })

        return detections

    def detect_single_face(self, frame: np.ndarray) -> dict | None:
        """
        Detect exactly one face in the frame.

        Args:
            frame: BGR image as numpy array.

        Returns:
            Detection dict if exactly 1 face found, else None.
        """
        detections = self.detect_faces(frame)
        if len(detections) == 1:
            return detections[0]
        return None

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: list[dict],
        names: list[str | None] | None = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame.

        Args:
            frame: BGR image to annotate (will be copied).
            detections: List of detection dicts from detect_faces().
            names: Optional parallel list of recognized names.
                   None or "Unknown" → red box; otherwise → green box.

        Returns:
            Annotated copy of the frame.
        """
        annotated = frame.copy()

        if names is None:
            names = [None] * len(detections)

        for det, name in zip(detections, names):
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]

            # Color: green for known, red for unknown
            is_known = name is not None and name.lower() != "unknown"
            color = (0, 200, 0) if is_known else (0, 0, 220)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Build label text
            label = name if is_known else "Unknown"
            label_text = f"{label} ({conf:.2f})"

            # Label background
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
