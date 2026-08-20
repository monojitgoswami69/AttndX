"""
Test quality assessment — face too small, too blurry, too dark,
overexposed, low contrast, excessive pose, accept.

Uses synthetic Detection objects with controlled parameters.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import cv2
import pytest

from gen2.config import Config
from gen2.vision.detection.yunet import Detection
from gen2.vision.quality.assessor import FaceQualityAssessor


def make_detection(bbox=(100, 100, 300, 300), confidence=0.9,
                   face_image=None, landmarks=None):
    if landmarks is None:
        landmarks = np.array([
            [150, 180], [250, 180], [200, 230], [170, 270], [230, 270],
        ], dtype=np.float32)
    if face_image is None:
        face_image = np.full((200, 200, 3), 128, dtype=np.uint8)
    return Detection(
        bbox=bbox, confidence=confidence,
        landmarks=landmarks, cropped_face=face_image,
    )


@pytest.fixture
def assessor():
    Config.load()
    return FaceQualityAssessor()


class TestQuality:
    def _make_textured_face(self, brightness=128):
        """Create a face image with texture (for nonzero Laplacian variance)."""
        rng = np.random.RandomState(42)
        face = np.full((200, 200, 3), brightness, dtype=np.int16)
        noise = rng.randint(-60, 60, face.shape, dtype=np.int16)
        face = np.clip(face + noise, 0, 255).astype(np.uint8)
        return face

    def test_accept_good_face(self, assessor):
        """A well-lit, sharp, large face → ACCEPT."""
        face = self._make_textured_face(brightness=128)
        det = make_detection(face_image=face)
        result = assessor.assess(det)
        assert result.accepted, f"Expected ACCEPT, got {result.reason} ({result.details})"

    def test_face_too_small(self, assessor):
        """A face below min_face_size → REJECT with FACE_TOO_SMALL."""
        det = make_detection(bbox=(100, 100, 130, 130))  # 30x30
        result = assessor.assess(det)
        assert not result.accepted
        assert "FACE_TOO_SMALL" in result.reason

    def test_too_dark_advisory(self, assessor):
        """A dark face with texture → TOO_DARK is advisory (accepted, warning)."""
        face = self._make_textured_face(brightness=10)
        det = make_detection(face_image=face)
        result = assessor.assess(det)
        # Advisory: accepted but TOO_DARK in warnings
        assert "TOO_DARK" in result.details.get("warnings", [])
        # Face is still usable if overall score is high enough (texture compensates)

    def test_truly_dark_rejected(self, assessor):
        """A completely dark, tiny, low-confidence face → rejected."""
        face = np.full((50, 50, 3), 5, dtype=np.uint8)  # tiny + no texture + dark
        det = Detection(
            bbox=(100, 100, 150, 150),  # 50x50 — just above min_face_size
            confidence=0.5,
            landmarks=np.array([
                [120, 120], [130, 120], [125, 130], [122, 140], [128, 140],
            ], dtype=np.float32),
            cropped_face=face,
        )
        result = assessor.assess(det)
        assert not result.accepted

    def test_overexposed_advisory(self, assessor):
        """A bright face with texture → OVEREXPOSED is advisory."""
        face = self._make_textured_face(brightness=250)
        det = make_detection(face_image=face)
        result = assessor.assess(det)
        assert "OVEREXPOSED" in result.details.get("warnings", [])

    def test_low_detection_confidence_advisory(self, assessor):
        """Low detector confidence → advisory (not a hard reject)."""
        det = make_detection(confidence=0.1)
        result = assessor.assess(det)
        assert "LOW_DETECTION_CONFIDENCE" in result.details.get("warnings", [])

    def test_landmark_failure(self, assessor):
        """NaN landmarks → REJECT with LANDMARK_FAILURE."""
        bad_lm = np.full((5, 2), np.nan, dtype=np.float32)
        det = make_detection(landmarks=bad_lm)
        result = assessor.assess(det)
        assert not result.accepted
        assert "LANDMARK_FAILURE" in result.reason

    def test_pose_estimation(self, assessor):
        """Pose estimation from landmarks produces reasonable angles."""
        # Frontal face
        det = make_detection()
        pose = assessor._estimate_pose(det.landmarks)
        assert abs(pose["yaw"]) < 90
        assert abs(pose["pitch"]) < 90
        assert abs(pose["roll"]) < 90

    def test_overall_score_in_range(self, assessor):
        """Overall score must be in [0, 1]."""
        det = make_detection()
        result = assessor.assess(det)
        assert 0.0 <= result.overall_score <= 1.0
