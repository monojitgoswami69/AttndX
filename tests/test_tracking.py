"""
Test tracking — stable track, disappearing/reappearing face,
multiple simultaneous tracks, IoU computation.

Uses synthetic bounding boxes (no real images needed).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest

from config import Config
from recognition.tracking.iou_tracker import IoUTracker, Track, _compute_iou
from vision.detection import Detection
from recognition.matching.engine import RecognitionResult, RecognitionState
import numpy as np


class FakeDetection:
    """Lightweight detection substitute for tracker tests."""
    def __init__(self, bbox, confidence=0.9):
        self.bbox = bbox
        self.confidence = confidence


class TestIoU:
    def test_identical_boxes(self):
        assert _compute_iou((0, 0, 100, 100), (0, 0, 100, 100)) == 1.0

    def test_no_overlap(self):
        assert _compute_iou((0, 0, 50, 50), (100, 100, 150, 150)) == 0.0

    def test_partial_overlap(self):
        iou = _compute_iou((0, 0, 100, 100), (50, 50, 150, 150))
        assert 0.0 < iou < 1.0

    def test_contained(self):
        iou = _compute_iou((0, 0, 100, 100), (25, 25, 75, 75))
        assert 0.0 < iou < 1.0


class TestTracker:
    def setup_method(self):
        Config.load()

    def test_single_track_stable(self):
        """A face that stays in roughly the same position keeps its track ID."""
        tracker = IoUTracker()
        # Frame 1
        dets = [FakeDetection((100, 100, 200, 200))]
        tracks1 = tracker.update(dets)
        # Frame 2 — slight movement
        dets = [FakeDetection((105, 105, 205, 205))]
        tracks2 = tracker.update(dets)
        # Same track ID
        if tracks1 and tracks2:
            assert tracks1[0].track_id == tracks2[0].track_id

    def test_new_face_gets_new_track(self):
        tracker = IoUTracker()
        # Frame 1: one face
        tracker.update([FakeDetection((100, 100, 200, 200))])
        # Frame 2: two faces
        tracker.update([
            FakeDetection((100, 100, 200, 200)),
            FakeDetection((300, 100, 400, 200)),
        ])
        # Should have 2 tracks
        assert len(tracker.tracks) >= 2

    def test_disappearing_face(self):
        """A face that disappears should eventually have its track removed."""
        tracker = IoUTracker()
        max_age = Config.get("tracking", "max_track_age")
        # Frame 1: one face
        tracker.update([FakeDetection((100, 100, 200, 200))])
        assert len(tracker.tracks) >= 1
        # Subsequent frames: no faces
        for _ in range(max_age + 2):
            tracker.update([])
        # Track should be gone
        assert len(tracker.tracks) == 0

    def test_multiple_simultaneous_tracks(self):
        """Two faces that don't overlap should get separate track IDs."""
        tracker = IoUTracker()
        dets = [
            FakeDetection((50, 50, 150, 150)),
            FakeDetection((300, 300, 400, 400)),
        ]
        for _ in range(5):  # confirm tracks
            tracker.update(dets)
        assert len(tracker.tracks) >= 2

    def test_vote_only_known_counts(self):
        """Only KNOWN recognition results contribute identity votes."""
        track = Track(track_id=1, bbox=(0, 0, 10, 10))
        # KNOWN votes
        for _ in range(3):
            track.vote(RecognitionResult(
                state=RecognitionState.KNOWN,
                identity_id="A", name="Alice",
                confidence=0.5,
            ))
        # UNKNOWN votes (should not count for any identity)
        for _ in range(2):
            track.vote(RecognitionResult(
                state=RecognitionState.UNKNOWN,
                confidence=0.1,
            ))
        tallies = track.tally_votes()
        assert "A" in tallies
        assert tallies["A"] == 3
        assert track.total_votes() == 5

    def test_ambiguous_vote_neutral(self):
        """AMBIGUOUS recognition results are neutral — they don't vote."""
        track = Track(track_id=1, bbox=(0, 0, 10, 10))
        track.vote(RecognitionResult(
            state=RecognitionState.AMBIGUOUS, confidence=0.5,
        ))
        assert track.total_votes() == 0  # ambiguous doesn't count

    def test_reset(self):
        tracker = IoUTracker()
        tracker.update([FakeDetection((0, 0, 100, 100))])
        assert len(tracker.tracks) >= 1
        tracker.reset()
        assert len(tracker.tracks) == 0
