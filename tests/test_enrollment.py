"""
Test enrollment — valid enrollment, blurry enrollment, multiple faces,
inconsistent samples, duplicate registration, identity exists.

Uses synthetic frames and synthetic embeddings to test enrollment logic
without requiring real face images.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from config import Config
from storage.db import BiometricDB
from recognition.matching.engine import IdentityIndex
from enrollment.service import EnrollmentService, EnrollmentStatus
from vision.detection import Detection
from vision.quality.assessor import QualityResult


def make_synthetic_detection(frame_shape=(480, 640, 3)) -> Detection:
    """A Detection with valid landmarks pointing to a face-like region."""
    landmarks = np.array([
        [250, 200], [350, 200], [300, 280], [260, 330], [340, 330],
    ], dtype=np.float32)
    return Detection(
        bbox=(200, 150, 400, 400),
        confidence=0.95,
        landmarks=landmarks,
        cropped_face=np.zeros((250, 200, 3), dtype=np.uint8),
    )


def make_good_quality_result() -> QualityResult:
    return QualityResult(
        overall_score=0.8,
        accepted=True,
        reason="ACCEPT",
        details={"blur_score": 200, "brightness": 128, "contrast": 60},
    )


def make_bad_quality_result() -> QualityResult:
    return QualityResult(
        overall_score=0.2,
        accepted=False,
        reason="TOO_BLURRY",
        details={"blur_score": 10},
    )


def make_unit_vec(seed, dim=512):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def tmp_db(tmp_path):
    return BiometricDB(tmp_path / "test_bio.db")


@pytest.fixture
def mock_components():
    """Mock detector, aligner, quality, embedder with predictable behavior."""
    Config.load()

    detector = MagicMock()
    aligner = MagicMock()
    quality = MagicMock()
    embedder = MagicMock()

    # By default: detect one face, align succeeds, quality good, embed valid
    detector.detect.return_value = [make_synthetic_detection()]
    aligner.align.return_value = np.zeros((112, 112, 3), dtype=np.uint8)
    quality.assess.return_value = make_good_quality_result()
    embedder.embed.return_value = MagicMock(
        valid=True,
        vector=make_unit_vec(42),
        error=None,
        pipeline_version="test_v1",
    )
    embedder.pipeline_version = "test_v1"

    return detector, aligner, quality, embedder


class TestEnrollment:
    def test_valid_enrollment(self, tmp_db, mock_components):
        """Valid frames with good quality → SUCCESS."""
        detector, aligner, quality, embedder = mock_components
        index = IdentityIndex()
        svc = EnrollmentService(detector, aligner, quality, embedder,
                               tmp_db, index)

        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        result = svc.enroll("Alice", frames, "STU001")

        assert result.status == EnrollmentStatus.SUCCESS
        assert result.identity_id == "STU001"
        assert result.samples_stored == 5
        assert tmp_db.identity_exists("STU001")
        assert index.size == 1

    def test_insufficient_samples(self, tmp_db, mock_components):
        """Fewer than min_samples → INSUFFICIENT_SAMPLES."""
        detector, aligner, quality, embedder = mock_components
        index = IdentityIndex()
        svc = EnrollmentService(detector, aligner, quality, embedder,
                               tmp_db, index)

        min_samples = Config.get("enrollment", "min_samples")
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)
                  for _ in range(min_samples - 1)]
        result = svc.enroll("Alice", frames, "STU001")

        assert result.status == EnrollmentStatus.INSUFFICIENT_SAMPLES

    def test_no_face_detected(self, tmp_db, mock_components):
        """Frames with no face → rejected."""
        detector, aligner, quality, embedder = mock_components
        detector.detect.return_value = []  # no faces
        index = IdentityIndex()
        svc = EnrollmentService(detector, aligner, quality, embedder,
                               tmp_db, index)

        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        result = svc.enroll("Alice", frames, "STU001")
        assert result.status == EnrollmentStatus.INSUFFICIENT_SAMPLES

    def test_blurry_frames_rejected(self, tmp_db, mock_components):
        """All frames fail quality → INSUFFICIENT_SAMPLES."""
        detector, aligner, quality, embedder = mock_components
        quality.assess.return_value = make_bad_quality_result()
        index = IdentityIndex()
        svc = EnrollmentService(detector, aligner, quality, embedder,
                               tmp_db, index)

        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        result = svc.enroll("Alice", frames, "STU001")
        assert result.status == EnrollmentStatus.INSUFFICIENT_SAMPLES

    def test_identity_already_exists(self, tmp_db, mock_components):
        """Duplicate ID → IDENTITY_EXISTS."""
        detector, aligner, quality, embedder = mock_components
        index = IdentityIndex()
        # Pre-enroll
        tmp_db.add_identity("STU001", "Alice",
                           Config.pipeline_version_string())
        svc = EnrollmentService(detector, aligner, quality, embedder,
                               tmp_db, index)

        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        result = svc.enroll("Bob", frames, "STU001")
        assert result.status == EnrollmentStatus.IDENTITY_EXISTS

    def test_inconsistent_samples_rejected(self, tmp_db, mock_components):
        """Samples with very different embeddings → INCONSISTENT_SAMPLES."""
        detector, aligner, quality, embedder = mock_components

        # Each frame produces a different random embedding (inconsistent)
        from recognition.embeddings.arcface_onnx import EmbeddingResult
        call_count = [0]
        def mock_embed(face):
            call_count[0] += 1
            return EmbeddingResult(
                vector=make_unit_vec(call_count[0] * 100),
                valid=True, error=None,
                pipeline_version="test_v1",
            )
        embedder.embed.side_effect = mock_embed

        index = IdentityIndex()
        svc = EnrollmentService(detector, aligner, quality, embedder,
                               tmp_db, index)

        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        result = svc.enroll("Alice", frames, "STU001")
        # Should be inconsistent (different random vectors have ~0 similarity)
        assert result.status == EnrollmentStatus.INCONSISTENT_SAMPLES

    def test_duplicate_registration_detected(self, tmp_db, mock_components):
        """Registering the same person twice → POSSIBLE_DUPLICATE or AMBIGUOUS."""
        detector, aligner, quality, embedder = mock_components
        index = IdentityIndex()

        # First enrollment
        svc = EnrollmentService(detector, aligner, quality, embedder,
                               tmp_db, index)
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        result1 = svc.enroll("Alice", frames, "STU001")
        assert result1.status == EnrollmentStatus.SUCCESS

        # Second enrollment with same embeddings → should detect duplicate
        result2 = svc.enroll("Alice2", frames, "STU002")
        assert result2.status in (EnrollmentStatus.POSSIBLE_DUPLICATE,
                                  EnrollmentStatus.AMBIGUOUS_DUPLICATE)

    def test_delete_identity(self, tmp_db, mock_components):
        """Deleting an identity removes it from DB and index."""
        detector, aligner, quality, embedder = mock_components
        index = IdentityIndex()
        svc = EnrollmentService(detector, aligner, quality, embedder,
                               tmp_db, index)

        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        svc.enroll("Alice", frames, "STU001")
        assert index.size == 1

        assert svc.delete_identity("STU001")
        assert index.size == 0
        assert not tmp_db.identity_exists("STU001")

    def test_index_updated_after_enrollment(self, tmp_db, mock_components):
        """The in-memory index is updated immediately after enrollment
        (no restart required)."""
        detector, aligner, quality, embedder = mock_components
        index = IdentityIndex()
        svc = EnrollmentService(detector, aligner, quality, embedder,
                               tmp_db, index)

        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        result = svc.enroll("Alice", frames, "STU001")
        assert result.status == EnrollmentStatus.SUCCESS

        # Index should have the new identity
        assert index.size == 1
        # Searching with the same embedding should find it
        from recognition.matching.engine import RecognitionEngine, RecognitionState
        engine = RecognitionEngine(index)
        rec = engine.recognize(make_unit_vec(42))
        assert rec.state == RecognitionState.KNOWN
        assert rec.identity_id == "STU001"
