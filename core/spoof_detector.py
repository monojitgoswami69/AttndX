"""
Deep-learning-based face anti-spoofing using MiniFASNetV2.

MiniFASNetV2 is a lightweight CNN (434K params, ~1.7MB ONNX) trained on
real spoofing datasets (CASIA-FASD / Replay-Attack). It classifies a face
crop into 3 classes:
    0 = print attack (printed photo)
    1 = real (live)
    2 = screen attack (phone/screen photo)

Preprocessing (matches the Silent-Face-Anti-Spoofing pipeline exactly):
    1. Crop the face from the FULL FRAME at scale=2.0 (2x the face bbox,
       including surrounding context — needed to detect phone/screen edges).
    2. Resize to 80x80 (BGR, no color conversion).
    3. Convert to float32 — NO division by 255 (the repo's to_tensor
       has .div(255) commented out, so the model expects [0, 255] range).
    4. Transpose HWC → CHW → NCHW.
"""

import numpy as np
import cv2
from core.config import Config

_MODEL_PATH = Config.BASE_DIR / "models" / "MiniFASNetV2.onnx"
_INPUT_SIZE = 80  # MiniFASNetV2 expects 80x80 input
_CROP_SCALE = 2.0  # Crop 2x the face bbox (includes context for phone-edge detection)


class SpoofDetector:
    """Model-based face anti-spoofing via MiniFASNetV2 ONNX."""

    def __init__(self):
        self._session = None
        self._input_name = None
        try:
            if not _MODEL_PATH.exists():
                print(f"[SpoofDetector] Model not found at {_MODEL_PATH}")
                return
            import onnxruntime as ort
            providers = ort.get_available_providers()
            self._session = ort.InferenceSession(
                str(_MODEL_PATH), providers=providers,
            )
            self._input_name = self._session.get_inputs()[0].name
            print(
                f"[SpoofDetector] MiniFASNetV2 loaded from {_MODEL_PATH} "
                f"(providers: {providers})"
            )
        except Exception as e:
            print(f"[SpoofDetector] Failed to load MiniFASNetV2: {e}")

    def is_available(self) -> bool:
        return self._session is not None

    def _crop_with_context(self, frame, face_bbox):
        """Crop the face at scale=2.0 (2x the face bbox) from the full frame.

        This includes surrounding context (background, phone edges, screen
        bezels) which the model needs to distinguish real faces from spoofs.
        Mirrors CropImage._get_new_box from the Silent-Face-Anti-Spoofing repo.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = face_bbox
        box_w = x2 - x1
        box_h = y2 - y1

        if box_w <= 0 or box_h <= 0:
            return None

        # Expand by scale (2.0 = 2x the face size)
        scale = min(
            (h - 1) / box_h,
            (w - 1) / box_w,
            _CROP_SCALE,
        )
        new_w = int(box_w * scale)
        new_h = int(box_h * scale)
        cx = x1 + box_w / 2
        cy = y1 + box_h / 2

        left = int(cx - new_w / 2)
        top = int(cy - new_h / 2)
        right = int(cx + new_w / 2)
        bottom = int(cy + new_h / 2)

        # Clamp to frame boundaries
        if left < 0:
            right -= left
            left = 0
        if top < 0:
            bottom -= top
            top = 0
        if right > w - 1:
            left -= right - (w - 1)
            right = w - 1
        if bottom > h - 1:
            top -= bottom - (h - 1)
            bottom = h - 1

        left = max(0, left)
        top = max(0, top)
        right = min(w, right)
        bottom = min(h, bottom)

        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None
        return crop

    def check(self, frame, face_bbox):
        """
        Classify a face as real or spoof.

        Args:
            frame: The FULL BGR frame (not just the face crop).
            face_bbox: (x1, y1, x2, y2) bounding box of the face in the frame.

        Returns:
            {"is_live": bool, "confidence": float, "spoofing_type": str|None,
             "real_prob": float, "print_prob": float, "screen_prob": float}
            or None if the model is unavailable.
        """
        if self._session is None or frame is None or frame.size == 0:
            return None

        # Crop with context (scale=2.0)
        crop = self._crop_with_context(frame, face_bbox)
        if crop is None:
            return None

        # Preprocess — NO /255 (model expects [0, 255] range)
        img = cv2.resize(crop, (_INPUT_SIZE, _INPUT_SIZE))
        img = img.astype(np.float32)  # keep [0, 255]
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        # Inference
        try:
            output = self._session.run(None, {self._input_name: img})[0]
        except Exception as e:
            print(f"[SpoofDetector] Inference error: {e}")
            return None

        # Softmax
        logits = output[0]
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()

        # Classes (verified from Silent-Face-Anti-Spoofing test.py):
        #   0 = print attack (fake)
        #   1 = real (live)
        #   2 = screen attack (fake)
        real_prob = float(probs[1])
        print_prob = float(probs[0])
        screen_prob = float(probs[2])

        is_live = real_prob >= Config.SPOOF_REAL_THRESHOLD

        spoofing_type = None
        if not is_live:
            if screen_prob >= print_prob:
                spoofing_type = "screen_photo"
            else:
                spoofing_type = "printed_photo"

        return {
            "is_live": is_live,
            "confidence": round(real_prob, 3),
            "spoofing_type": spoofing_type,
            "real_prob": round(real_prob, 3),
            "print_prob": round(print_prob, 3),
            "screen_prob": round(screen_prob, 3),
        }
