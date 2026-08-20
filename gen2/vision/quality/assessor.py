"""
Face quality assessment gate.

Evaluates multiple quality dimensions and returns an explicit decision:
  ACCEPT  — frame is good enough for embedding
  REJECT  — frame is too poor, with a reason code

Quality dimensions:
  - face size (bbox dimensions)
  - sharpness (Laplacian variance)
  - brightness (mean gray value)
  - contrast (gray std dev)
  - pose (estimated from landmark geometry)
  - detector confidence
  - landmark validity (do landmarks look reasonable)

Reason codes:
  FACE_TOO_SMALL
  TOO_BLURRY
  TOO_DARK
  OVEREXPOSED
  LOW_CONTRAST
  EXCESSIVE_POSE
  LOW_DETECTION_CONFIDENCE
  LANDMARK_FAILURE
"""
import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from gen2.config import Config
from gen2.vision.detection.yunet import Detection

logger = logging.getLogger(__name__)


@dataclass
class QualityResult:
    overall_score: float       # 0.0 - 1.0
    accepted: bool
    reason: str                # "ACCEPT" or a rejection code
    details: dict = field(default_factory=dict)


class FaceQualityAssessor:
    """Multi-dimensional face quality gate."""

    def __init__(self):
        self.min_face_size = Config.get("quality", "min_face_size")
        self.blur_threshold = Config.get("quality", "blur_threshold")
        self.brightness_min = Config.get("quality", "brightness_min")
        self.brightness_max = Config.get("quality", "brightness_max")
        self.contrast_min = Config.get("quality", "contrast_min")
        self.max_yaw = Config.get("quality", "max_pose_yaw")
        self.max_pitch = Config.get("quality", "max_pose_pitch")
        self.max_roll = Config.get("quality", "max_pose_roll")
        self.min_detector_conf = Config.get("quality", "min_detector_confidence")
        self.min_overall = Config.get("quality", "min_overall_score")

    def assess(self, detection: Detection, aligned_face: np.ndarray | None = None) -> QualityResult:
        """Assess quality of a detection. If aligned_face is provided, use it;
        otherwise use the raw cropped_face."""
        details = {}
        reasons = []

        # ── Detector confidence ──
        conf = detection.confidence
        details["detector_confidence"] = conf
        if conf < self.min_detector_conf:
            reasons.append("LOW_DETECTION_CONFIDENCE")

        # ── Face size ──
        x1, y1, x2, y2 = detection.bbox
        fw, fh = x2 - x1, y2 - y1
        details["face_width"] = fw
        details["face_height"] = fh
        if fw < self.min_face_size or fh < self.min_face_size:
            reasons.append("FACE_TOO_SMALL")

        # ── Landmark validity ──
        lm = detection.landmarks
        lm_valid = np.all(np.isfinite(lm)) and np.all(lm >= 0)
        details["landmarks_valid"] = bool(lm_valid)
        if not lm_valid:
            reasons.append("LANDMARK_FAILURE")

        # ── Pose estimation from landmarks ──
        pose = self._estimate_pose(lm)
        details["pose"] = pose
        if abs(pose["yaw"]) > self.max_yaw:
            reasons.append("EXCESSIVE_POSE")
        if abs(pose["pitch"]) > self.max_pitch:
            reasons.append("EXCESSIVE_POSE")
        if abs(pose["roll"]) > self.max_roll:
            reasons.append("EXCESSIVE_POSE")

        # ── Image quality (from cropped face) ──
        crop = detection.cropped_face
        if crop is not None and crop.size > 0:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            # Sharpness
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            details["blur_score"] = blur_score
            if blur_score < self.blur_threshold:
                reasons.append("TOO_BLURRY")

            # Brightness
            brightness = float(np.mean(gray))
            details["brightness"] = brightness
            if brightness < self.brightness_min:
                reasons.append("TOO_DARK")
            elif brightness > self.brightness_max:
                reasons.append("OVEREXPOSED")

            # Contrast
            contrast = float(np.std(gray))
            details["contrast"] = contrast
            if contrast < self.contrast_min:
                reasons.append("LOW_CONTRAST")
        else:
            reasons.append("CROP_EMPTY")
            blur_score = 0.0
            brightness = 0.0
            contrast = 0.0

        # ── Overall score ──
        # Normalize each sub-score to [0, 1]
        sub_scores = []
        if conf >= self.min_detector_conf:
            sub_scores.append(min(1.0, conf))
        else:
            sub_scores.append(conf / self.min_detector_conf)

        size_ratio = min((fw * fh), (self.min_face_size * 2) ** 2) / ((self.min_face_size * 2) ** 2)
        sub_scores.append(size_ratio)

        blur_norm = min(1.0, blur_score / (self.blur_threshold * 4))
        sub_scores.append(blur_norm)

        b_mid = (self.brightness_min + self.brightness_max) / 2
        b_half = (self.brightness_max - self.brightness_min) / 2
        if b_half > 0:
            b_score = max(0.0, 1.0 - abs(brightness - b_mid) / b_half * 0.5)
        else:
            b_score = 0.5
        sub_scores.append(b_score)

        contrast_score = min(1.0, contrast / (self.contrast_min * 4))
        sub_scores.append(contrast_score)

        overall = float(np.mean(sub_scores)) if sub_scores else 0.0
        overall = max(0.0, min(1.0, overall))
        details["overall_score"] = overall

        # ── Decision ──
        if reasons:
            return QualityResult(
                overall_score=overall,
                accepted=False,
                reason=reasons[0],
                details=details,
            )

        if overall < self.min_overall:
            return QualityResult(
                overall_score=overall,
                accepted=False,
                reason="LOW_QUALITY",
                details=details,
            )

        return QualityResult(
            overall_score=overall,
            accepted=True,
            reason="ACCEPT",
            details=details,
        )

    def _estimate_pose(self, landmarks: np.ndarray) -> dict:
        """Estimate rough pose angles from 5 landmarks.
        Returns yaw, pitch, roll in degrees (approximate)."""
        if landmarks is None or len(landmarks) != 5:
            return {"yaw": 0, "pitch": 0, "roll": 0}

        le = landmarks[0]  # left eye
        re = landmarks[1]  # right eye
        nose = landmarks[2]
        lm = landmarks[3]  # left mouth
        rm = landmarks[4]  # right mouth

        # Roll: angle of the eye line
        eye_dx = re[0] - le[0]
        eye_dy = re[1] - le[1]
        roll = float(np.degrees(np.arctan2(eye_dy, eye_dx)))

        # Yaw: nose offset relative to eye midpoint
        eye_mid = (le + re) / 2.0
        eye_dist = np.linalg.norm(re - le)
        if eye_dist > 1e-6:
            yaw_offset = (nose[0] - eye_mid[0]) / eye_dist
            # Map [-1, 1] to [-90, 90] degrees approximately
            yaw = float(np.clip(yaw_offset * 45, -90, 90))
        else:
            yaw = 0.0

        # Pitch: nose position relative to eye-mouth midpoint
        mouth_mid = (lm + rm) / 2.0
        eye_to_mouth = mouth_mid[1] - eye_mid[1]
        if eye_to_mouth > 1e-6:
            pitch_offset = (nose[1] - eye_mid[1]) / eye_to_mouth
            # Normal range ~0.55-0.65; deviation indicates pitch
            pitch = float(np.clip((pitch_offset - 0.6) * 90, -90, 90))
        else:
            pitch = 0.0

        return {"yaw": yaw, "pitch": pitch, "roll": roll}
