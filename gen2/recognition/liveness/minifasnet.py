"""
Liveness / anti-spoofing via MiniFASNetV2 ONNX.

Model: MiniFASNetV2 (Silent-Face-Anti-Spoofing)
  Input:  [batch, 3, 80, 80] float32, BGR, range [0, 255] (NO /255 normalization)
  Output: [batch, 3] logits → softmax → [print_prob, real_prob, screen_prob]

Classes:
  0 = print attack (spoof)
  1 = real (live)
  2 = screen attack (spoof)

The model requires the face cropped with 2x context (scale=2.0) from
the FULL FRAME (not just the tight face crop). This context includes
phone/screen edges and background, which the model needs to distinguish
real faces from photos of screens/prints.

This is a SEPARATE decision from recognition. A face can be:
  - recognized but NOT live (phone photo of a known person)
  - live but NOT recognized (unknown person in front of camera)
  - both recognized and live (valid attendance)
  - neither (unknown person + spoof attempt)
"""
import logging
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
import onnxruntime as ort

from gen2.config import Config

logger = logging.getLogger(__name__)


class LivenessState(Enum):
    LIVE = "live"
    SPOOF = "spoof"
    UNCERTAIN = "uncertain"
    NOT_CHECKED = "not_checked"
    ERROR = "error"


@dataclass
class LivenessResult:
    state: LivenessState
    real_prob: float = 0.0
    print_prob: float = 0.0
    screen_prob: float = 0.0
    spoofing_type: str | None = None
    error: str | None = None


class MiniFASNetLiveness:
    """MiniFASNetV2 anti-spoofing detector."""

    def __init__(self):
        model_path = Config.model_path("liveness")
        if not model_path.exists():
            logger.warning(f"Liveness model not found: {model_path}")
            self._session = None
            return

        providers = Config.get("onnx", "providers")
        available = ort.get_available_providers()
        active = [p for p in providers if p in available] or ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(
            str(model_path), providers=active,
        )
        self._input_name = self._session.get_inputs()[0].name
        self._input_size = Config.get("liveness", "input_size", default=80)
        self._crop_scale = Config.get("liveness", "crop_scale", default=2.0)
        self._real_threshold = Config.get("liveness", "real_threshold", default=0.50)
        logger.info(f"MiniFASNetV2 loaded from {model_path.name}")

    def is_available(self) -> bool:
        return self._session is not None

    def check(self, frame: np.ndarray, face_bbox: tuple[int, int, int, int]) -> LivenessResult:
        """Check if a face is live or spoof.
        frame: the FULL BGR frame
        face_bbox: (x1, y1, x2, y2) of the face in the frame
        """
        if not self.is_available():
            return LivenessResult(
                state=LivenessState.NOT_CHECKED,
                error="model_not_loaded",
            )
        if frame is None or frame.size == 0:
            return LivenessResult(
                state=LivenessState.ERROR,
                error="empty_frame",
            )

        crop = self._crop_with_context(frame, face_bbox)
        if crop is None:
            return LivenessResult(
                state=LivenessState.ERROR,
                error="crop_failed",
            )

        # Preprocess — NO /255 (model expects [0, 255] range)
        img = cv2.resize(crop, (self._input_size, self._input_size))
        img = img.astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        try:
            output = self._session.run(None, {self._input_name: img})[0]
        except Exception as e:
            logger.error(f"MiniFASNet inference error: {e}")
            return LivenessResult(
                state=LivenessState.ERROR,
                error="inference_error",
            )

        logits = output[0]
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()

        real_prob = float(probs[1])
        print_prob = float(probs[0])
        screen_prob = float(probs[2])

        if real_prob >= self._real_threshold:
            state = LivenessState.LIVE
            spoof_type = None
        else:
            state = LivenessState.SPOOF
            spoof_type = "screen_photo" if screen_prob >= print_prob else "printed_photo"

        return LivenessResult(
            state=state,
            real_prob=real_prob,
            print_prob=print_prob,
            screen_prob=screen_prob,
            spoofing_type=spoof_type,
        )

    def _crop_with_context(self, frame: np.ndarray,
                           face_bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        """Crop the face at 2x scale (includes surrounding context)."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = face_bbox
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 0 or box_h <= 0:
            return None

        scale = min((h - 1) / box_h, (w - 1) / box_w, self._crop_scale)
        new_w = int(box_w * scale)
        new_h = int(box_h * scale)
        cx = x1 + box_w / 2
        cy = y1 + box_h / 2

        left = int(cx - new_w / 2)
        top = int(cy - new_h / 2)
        right = int(cx + new_w / 2)
        bottom = int(cy + new_h / 2)

        # Clamp to frame
        if left < 0:
            right -= left
            left = 0
        if top < 0:
            bottom -= top
            top = 0
        if right > w:
            left -= right - w
            right = w
        if bottom > h:
            top -= bottom - h
            bottom = h

        left = max(0, left)
        top = max(0, top)
        right = min(w, right)
        bottom = min(h, bottom)

        crop = frame[top:bottom, left:right]
        return crop if crop.size > 0 else None
