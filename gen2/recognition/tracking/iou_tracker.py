"""
IoU-based face tracker with temporal identity stabilization.

The tracker:
  1. Associates detections across frames using IoU (Intersection-over-Union)
  2. Maintains a track lifecycle (tentative → confirmed → lost → removed)
  3. Accumulates identity votes per track (from recognition engine)
  4. Confirms an identity only after sufficient consistent evidence

Identity stabilization policy:
  - A track accumulates recognition votes in a ring buffer
  - An identity is "confirmed" when vote_min_count votes AND
    vote_min_fraction of votes agree on the same identity
  - If no identity reaches the threshold, the track remains UNKNOWN
  - A single frame's recognition result does NOT determine attendance
  - An ambiguous recognition (AMBIGUOUS state) does NOT count as a vote
    for any identity — it is neutral

This prevents:
  - Frame-to-frame identity flipping
  - A single misrecognition poisoning the track's identity
  - Unknown-to-known promotion without evidence
"""
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from gen2.config import Config
from gen2.recognition.matching.engine import RecognitionResult, RecognitionState

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """A single tracked face."""
    track_id: int
    bbox: tuple[int, int, int, int]   # last known bbox
    age: int = 0                     # frames since creation
    hits: int = 0                    # total detections matched
    misses: int = 0                  # consecutive frames without detection
    confirmed: bool = False         # confirmed after min_track_hits
    # Identity voting
    votes: deque = field(default_factory=lambda: deque(maxlen=10))
    confirmed_identity_id: Optional[str] = None
    confirmed_name: Optional[str] = None
    confirmed_confidence: float = 0.0
    # Last recognition result (for UI/debug)
    last_recognition: Optional[RecognitionResult] = None

    def vote(self, result: RecognitionResult):
        """Cast a recognition vote for this track."""
        self.last_recognition = result
        if result.state == RecognitionState.KNOWN:
            self.votes.append(("known", result.identity_id, result.confidence))
        elif result.state == RecognitionState.UNKNOWN:
            self.votes.append(("unknown", None, result.confidence))
        # AMBIGUOUS and REJECTED votes are NOT appended — they are neutral
        # and do not affect the identity tally.

    def tally_votes(self) -> dict[str, int]:
        """Tally identity votes. Returns {identity_id: count}.
        Only KNOWN votes count. UNKNOWN votes are tracked separately."""
        tallies: dict[str, int] = {}
        for state, identity_id, _ in self.votes:
            if state == "known" and identity_id is not None:
                tallies[identity_id] = tallies.get(identity_id, 0) + 1
        return tallies

    def total_votes(self) -> int:
        return len(self.votes)


class IoUTracker:
    """IoU-based multi-face tracker with identity voting."""

    def __init__(self):
        self.iou_threshold = Config.get("tracking", "iou_threshold")
        self.max_track_age = Config.get("tracking", "max_track_age")
        self.min_track_hits = Config.get("tracking", "min_track_hits")
        self.vote_buffer_size = Config.get("tracking", "vote_buffer_size")
        self.vote_min_fraction = Config.get("tracking", "vote_min_fraction")
        self.vote_min_count = Config.get("tracking", "vote_min_count")

        self._tracks: dict[int, Track] = {}
        self._next_track_id = 1
        self._frame_count = 0

    @property
    def tracks(self) -> dict[int, Track]:
        return self._tracks

    def update(self, detections: list) -> list[Track]:
        """Update tracks with new detections.
        detections: list of Detection objects (from YuNetDetector).
        Returns list of active confirmed tracks."""
        self._frame_count += 1

        # Extract bboxes from detections
        det_bboxes = [d.bbox for d in detections]

        # ── Association via IoU ──
        # Build cost matrix: IoU between each track and each detection
        active_track_ids = [tid for tid, t in self._tracks.items()
                           if t.misses < self.max_track_age]
        assigned_tracks: set[int] = set()
        assigned_dets: set[int] = set()

        if active_track_ids and det_bboxes:
            # Compute IoU matrix
            iou_matrix = np.zeros((len(active_track_ids), len(det_bboxes)),
                                 dtype=np.float32)
            for ti, tid in enumerate(active_track_ids):
                for di, bbox in enumerate(det_bboxes):
                    iou_matrix[ti, di] = _compute_iou(
                        self._tracks[tid].bbox, bbox)

            # Greedy assignment: pick highest IoU pairs
            while True:
                if iou_matrix.size == 0:
                    break
                max_iou = iou_matrix.max()
                if max_iou < self.iou_threshold:
                    break
                max_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                ti, di = int(max_idx[0]), int(max_idx[1])

                tid = active_track_ids[ti]
                self._tracks[tid].bbox = det_bboxes[di]
                self._tracks[tid].hits += 1
                self._tracks[tid].misses = 0
                self._tracks[tid].age += 1
                if self._tracks[tid].hits >= self.min_track_hits:
                    self._tracks[tid].confirmed = True

                assigned_tracks.add(tid)
                assigned_dets.add(di)
                # Mask out this row and column
                iou_matrix[ti, :] = 0
                iou_matrix[:, di] = 0

        # ── Unassigned tracks: increment misses ──
        for tid in active_track_ids:
            if tid not in assigned_tracks:
                self._tracks[tid].misses += 1
                self._tracks[tid].age += 1

        # ── Unassigned detections: create new tracks ──
        for di, bbox in enumerate(det_bboxes):
            if di not in assigned_dets:
                tid = self._next_track_id
                self._next_track_id += 1
                self._tracks[tid] = Track(
                    track_id=tid,
                    bbox=bbox,
                    age=1,
                    hits=1,
                )

        # ── Remove stale tracks ──
        stale = [tid for tid, t in self._tracks.items()
                if t.misses >= self.max_track_age]
        for tid in stale:
            del self._tracks[tid]

        # ── Update identity confirmations ──
        self._update_confirmations()

        # ── Return active confirmed tracks ──
        return [t for t in self._tracks.values()
                if t.misses == 0 and t.confirmed]

    def _update_confirmations(self):
        """Check each track's vote buffer and confirm identity if threshold met."""
        for track in self._tracks.values():
            if track.total_votes() < self.vote_min_count:
                continue

            tallies = track.tally_votes()
            if not tallies:
                continue

            best_id = max(tallies, key=tallies.get)
            best_count = tallies[best_id]
            fraction = best_count / track.total_votes()

            if best_count >= self.vote_min_count and fraction >= self.vote_min_fraction:
                track.confirmed_identity_id = best_id
                # Find name from the last matching vote
                for state, identity_id, conf in reversed(track.votes):
                    if state == "known" and identity_id == best_id:
                        track.confirmed_name = None  # filled by engine
                        track.confirmed_confidence = conf
                        break

    def get_track_for_bbox(self, bbox: tuple[int, int, int, int]) -> Track | None:
        """Find the track that best matches a given bbox (for associating
        recognition results back to tracks)."""
        best_track = None
        best_iou = self.iou_threshold
        for track in self._tracks.values():
            iou = _compute_iou(track.bbox, bbox)
            if iou > best_iou:
                best_iou = iou
                best_track = track
        return best_track

    def reset(self):
        """Reset all tracks (e.g., on session end)."""
        self._tracks.clear()
        self._next_track_id = 1
        self._frame_count = 0


def _compute_iou(bbox1: tuple, bbox2: tuple) -> float:
    """Compute IoU between two bboxes (x1, y1, x2, y2)."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0
