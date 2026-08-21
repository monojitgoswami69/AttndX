"""
Test alignment — deterministic transform, output shape, landmark validity.

Uses synthetic landmarks and synthetic frames to test the alignment math
without requiring real face images.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import cv2
import pytest

from config import Config
from vision.alignment.arcface import ArcFaceAligner
from vision.detection import Detection


@pytest.fixture(scope="module")
def aligner():
    Config.load()
    return ArcFaceAligner()


@pytest.fixture
def synthetic_frame():
    """A 480x640 BGR frame with a drawn face-like pattern."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 150), (300, 250), (128, 128, 128), -1)
    return frame


@pytest.fixture
def synthetic_detection():
    """A Detection with 5 landmarks roughly in face position."""
    landmarks = np.array([
        [230, 180],  # left eye
        [270, 180],  # right eye
        [250, 210],  # nose
        [235, 230],  # left mouth
        [265, 230],  # right mouth
    ], dtype=np.float32)
    return Detection(
        bbox=(200, 150, 300, 250),
        confidence=0.9,
        landmarks=landmarks,
        cropped_face=np.zeros((100, 100, 3), dtype=np.uint8),
    )


class TestAlignment:
    def test_output_shape(self, aligner, synthetic_frame, synthetic_detection):
        """Aligned face must be 112x112."""
        aligned = aligner.align(synthetic_frame, synthetic_detection)
        assert aligned is not None
        assert aligned.shape == (112, 112, 3)

    def test_deterministic(self, aligner, synthetic_frame, synthetic_detection):
        """Same input → same output (deterministic)."""
        a1 = aligner.align(synthetic_frame, synthetic_detection)
        a2 = aligner.align(synthetic_frame, synthetic_detection)
        assert np.array_equal(a1, a2)

    def test_none_frame(self, aligner, synthetic_detection):
        """None frame → None."""
        assert aligner.align(None, synthetic_detection) is None

    def test_none_detection(self, aligner, synthetic_frame):
        """None detection → None."""
        assert aligner.align(synthetic_frame, None) is None

    def test_no_landmarks(self, aligner, synthetic_frame):
        """Detection with None landmarks → None."""
        det = Detection(
            bbox=(0, 0, 10, 10), confidence=0.5,
            landmarks=None, cropped_face=np.zeros((10, 10, 3), dtype=np.uint8),
        )
        assert aligner.align(synthetic_frame, det) is None

    def test_batch_alignment(self, aligner, synthetic_frame, synthetic_detection):
        """Batch alignment produces one result per detection."""
        dets = [synthetic_detection, synthetic_detection, synthetic_detection]
        results = aligner.align_batch(synthetic_frame, dets)
        assert len(results) == 3
        for r in results:
            assert r is not None
            assert r.shape == (112, 112, 3)
