"""
Face detection module.
Uses YOLOv8 to detect persons (COCO class 0), then refines to actual
face bounding boxes via OpenCV Haar cascade. This two-stage approach
ensures we get proper face crops for the embedding model instead of
coarse person-body crops.

Fallback: if Haar cascade finds no face within a person crop, the
upper-body region is used as an approximate face crop.
"""

import numpy as np
import cv2
from ultralytics import YOLO
from core.config import Config


class YOLOFaceDetector:
    """Detect faces in frames using YOLOv8 person detection + Haar cascade face refinement."""

    # COCO class 0 = "person"
    PERSON_CLASS = 0

    def __init__(self, model_name: str | None = None, confidence: float | None = None):
        """
        Initialize the face detector.

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

        # Haar cascade for face refinement within person crops
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._face_cascade_alt = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        )
        print("[FaceDetector] Haar cascade face refiner loaded.")

    def _detect_faces_haar(self, person_crop: np.ndarray, offset_x: int, offset_y: int):
        """
        Run Haar cascade face detection within a person crop.

        Returns a list of (x1, y1, x2, y2, conf) tuples in *full-frame*
        coordinates (offsets applied).
        """
        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
        # Equalize histogram to improve detection in varied lighting
        gray = cv2.equalizeHist(gray)

        faces = []
        for cascade in (self._face_cascade_alt, self._face_cascade):
            if cascade.empty():
                continue
            detected = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            if len(detected) > 0:
                faces = detected
                break

        results = []
        for (fx, fy, fw, fh) in faces:
            x1 = offset_x + fx
            y1 = offset_y + fy
            x2 = offset_x + fx + fw
            y2 = offset_y + fy + fh
            # Confidence proxy: Haar doesn't give one; use 0.9 as a constant
            results.append((x1, y1, x2, y2, 0.9))
        return results

    def detect_faces(self, frame: np.ndarray) -> list[dict]:
        """
        Detect all faces in a frame.

        Two-stage pipeline:
          1. YOLO detects persons (class 0 only — filters out chairs, laptops, etc.)
          2. Haar cascade finds the actual face within each person crop.

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
            classes=[self.PERSON_CLASS],  # ONLY detect persons
            verbose=False,
        )

        detections = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            for box in result.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0]) if hasattr(box, 'cls') and box.cls is not None else -1

                # Safety: only accept person class
                if cls != self.PERSON_CLASS:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                box_h = y2 - y1
                box_w = x2 - x1

                # --- Stage 2: Face detection within person crop ---
                # Search the entire person bounding box (or upper 85% for tall standing boxes)
                # so the mouth, chin, and beard are never cut off
                aspect = box_h / max(box_w, 1)
                search_h = int(box_h * 0.5) if aspect > 2.0 else box_h
                person_crop = frame[y1:y1 + search_h, x1:x2]
                if person_crop.size == 0:
                    person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue

                haar_faces = self._detect_faces_haar(person_crop, x1, y1)

                if haar_faces:
                    for (fx1, fy1, fx2, fy2, fconf) in haar_faces:
                        fw = fx2 - fx1
                        fh = fy2 - fy1
                        # Proportional padding around the face for full head coverage
                        pad_x = max(self.padding, int(fw * 0.18))
                        pad_y = max(self.padding, int(fh * 0.22))
                        fx1p = max(0, fx1 - pad_x)
                        fy1p = max(0, fy1 - pad_y)
                        fx2p = min(w, fx2 + pad_x)
                        fy2p = min(h, fy2 + pad_y)

                        cropped = frame[fy1p:fy2p, fx1p:fx2p].copy()
                        if cropped.size == 0:
                            continue

                        detections.append({
                            "bbox": (fx1p, fy1p, fx2p, fy2p),
                            "confidence": min(conf, fconf),
                            "cropped_face": cropped,
                        })
                else:
                    # Fallback: adaptive upper-body crop based on aspect ratio
                    if aspect > 1.6:
                        # Full body standing: head is roughly top 35%
                        fb_y2 = y1 + int(box_h * 0.35)
                    elif aspect > 1.1:
                        # Half-body / torso: head is top 65%
                        fb_y2 = y1 + int(box_h * 0.65)
                    else:
                        # Close-up / headshot: head is top 85%
                        fb_y2 = y1 + int(box_h * 0.85)

                    pad = self.padding
                    x1p = max(0, x1 - pad)
                    y1p = max(0, y1 - pad)
                    x2p = min(w, x2 + pad)
                    y2p = min(h, fb_y2 + pad)

                    cropped = frame[y1p:y2p, x1p:x2p].copy()
                    if cropped.size == 0:
                        continue

                    detections.append({
                        "bbox": (x1p, y1p, x2p, y2p),
                        "confidence": conf,
                        "cropped_face": cropped,
                    })

        # If YOLO found no persons, attempt direct face detection on the full frame
        if not detections:
            frame_faces = self._detect_faces_haar(frame, 0, 0)
            for (fx1, fy1, fx2, fy2, fconf) in frame_faces:
                fw = fx2 - fx1
                fh = fy2 - fy1
                pad_x = max(self.padding, int(fw * 0.18))
                pad_y = max(self.padding, int(fh * 0.22))
                fx1p = max(0, fx1 - pad_x)
                fy1p = max(0, fy1 - pad_y)
                fx2p = min(w, fx2 + pad_x)
                fy2p = min(h, fy2 + pad_y)
                cropped = frame[fy1p:fy2p, fx1p:fx2p].copy()
                if cropped.size > 0:
                    detections.append({
                        "bbox": (fx1p, fy1p, fx2p, fy2p),
                        "confidence": fconf,
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
        # If multiple faces, return the one with highest confidence
        if len(detections) > 1:
            return max(detections, key=lambda d: d["confidence"])
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
                   None or "Unknown" -> red box; otherwise -> green box.

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
