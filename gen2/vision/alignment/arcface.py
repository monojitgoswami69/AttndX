"""
Geometric alignment using 5-point similarity transform.

This is the standard ArcFace alignment pipeline:
  1. Use 5 landmarks (eyes, nose, mouth corners) from YuNet
  2. Compute a similarity transform (translation + rotation + uniform scale,
     NO shear) that maps the detected landmarks to the canonical ArcFace
     template at 112x112
  3. Warp the source frame using the affine transform
  4. Output: 112x112 BGR aligned face crop

The canonical template comes from the InsightFace reference implementation.
The similarity transform is computed via cv2.estimateAffinePartial2D which
finds the optimal (rotation, translation, scale) mapping between two point
sets without shear — this is the correct transform for face alignment.

Deterministic: the same landmarks always produce the same aligned crop.
"""
import logging

import cv2
import numpy as np

from gen2.config import Config
from gen2.vision.detection.yunet import Detection

logger = logging.getLogger(__name__)


class ArcFaceAligner:
    """Aligns faces to 112x112 using 5-point similarity transform."""

    def __init__(self):
        self.output_size = tuple(Config.get("alignment", "output_size"))
        template = Config.get("alignment", "template")
        self.template = np.array(template, dtype=np.float32)
        interp = Config.get("alignment", "interpolation", "linear")
        self.interpolation = cv2.INTER_LINEAR if interp == "linear" else cv2.INTER_CUBIC
        border = Config.get("alignment", "border_mode", "constant")
        self.border_flag = cv2.BORDER_CONSTANT if border == "constant" else cv2.BORDER_REPLICATE
        logger.info(f"ArcFaceAligner: output={self.output_size}, template={len(template)} points")

    def align(self, frame: np.ndarray, detection: Detection) -> np.ndarray | None:
        """Produce a 112x112 BGR aligned face crop from the frame + detection.
        Returns None if alignment fails."""
        if frame is None or frame.size == 0:
            return None
        if detection is None or detection.landmarks is None:
            return None

        src = detection.landmarks.astype(np.float32)
        dst = self.template.astype(np.float32)

        # estimateAffinePartial2D finds the best-fit similarity transform
        # (4 DOF: tx, ty, theta, scale) — no shear, no independent xy scaling.
        # This is the correct transform for face alignment.
        transform, inliers = cv2.estimateAffinePartial2D(
            src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2),
            method=cv2.LMEDS,
        )
        if transform is None:
            logger.warning("Alignment failed: could not estimate transform")
            return None

        aligned = cv2.warpAffine(
            frame,
            transform,
            self.output_size,
            flags=self.interpolation,
            borderMode=self.border_flag,
            borderValue=(0, 0, 0),
        )
        return aligned

    def align_batch(self, frame: np.ndarray,
                    detections: list[Detection]) -> list[np.ndarray | None]:
        """Align all detections from a single frame."""
        return [self.align(frame, d) for d in detections]
