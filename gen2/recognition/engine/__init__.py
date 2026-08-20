"""
Unified recognition engine — ties detection, alignment, quality,
embedding, matching, and liveness together.

This is the single entry point for "recognize this frame".
It processes each face independently:
  1. Detect all faces
  2. For each face: align → quality gate → embed → match
  3. Associate results to tracks
  4. Optionally run liveness on recognized faces

Key properties:
  - Each face is processed independently
  - A failure on one face (bad quality, embed error) does NOT affect others
  - Results include explicit states (KNOWN/UNKNOWN/AMBIGUOUS/REJECTED)
  - Liveness is a separate decision from recognition
"""
import logging
from dataclasses import dataclass, field

import numpy as np

from gen2.config import Config
from gen2.recognition.embeddings.arcface_onnx import ArcFaceEmbedder
from gen2.recognition.liveness.minifasnet import LivenessResult, LivenessState, MiniFASNetLiveness
from gen2.recognition.matching.engine import RecognitionResult, RecognitionState, RecognitionEngine
from gen2.recognition.tracking.iou_tracker import IoUTracker, Track
from gen2.vision.alignment.arcface import ArcFaceAligner
from gen2.vision.detection.yunet import Detection, YuNetDetector
from gen2.vision.quality.assessor import FaceQualityAssessor, QualityResult

logger = logging.getLogger(__name__)


@dataclass
class FaceResult:
    """Result for a single face in a frame."""
    track_id: int | None = None
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    detection_confidence: float = 0.0
    recognition: RecognitionResult | None = None
    liveness: LivenessResult | None = None
    quality: QualityResult | None = None
    # Final attendance-relevant state
    confirmed_identity_id: str | None = None
    confirmed_name: str | None = None
    confirmed: bool = False


@dataclass
class FrameResult:
    """Result for an entire frame."""
    faces: list[FaceResult] = field(default_factory=list)
    num_detected: int = 0
    num_recognized: int = 0
    num_unknown: int = 0
    num_ambiguous: int = 0
    num_rejected: int = 0
    processing_error: str | None = None


class RecognitionPipeline:
    """End-to-end recognition pipeline for a single frame."""

    def __init__(self, detector: YuNetDetector, aligner: ArcFaceAligner,
                 quality_assessor: FaceQualityAssessor, embedder: ArcFaceEmbedder,
                 engine: RecognitionEngine, tracker: IoUTracker,
                 liveness: MiniFASNetLiveness | None = None):
        self.detector = detector
        self.aligner = aligner
        self.quality = quality_assessor
        self.embedder = embedder
        self.engine = engine
        self.tracker = tracker
        self.liveness = liveness
        self._run_liveness_on_preview = Config.get("liveness", "run_on_preview", default=False)

    def process_frame(self, frame: np.ndarray,
                      run_liveness: bool = False) -> FrameResult:
        """Process one camera frame end-to-end.
        Each face is handled independently — errors are isolated."""
        result = FrameResult()

        if frame is None or frame.size == 0:
            result.processing_error = "empty_frame"
            return result

        # ── 1. Detect all faces ──
        try:
            detections = self.detector.detect(frame)
        except Exception as e:
            logger.error(f"Detection error: {e}")
            result.processing_error = f"detection_error: {e}"
            return result

        result.num_detected = len(detections)
        if not detections:
            # Still update tracker (marks all tracks as missed)
            self.tracker.update([])
            return result

        # ── 2. Update tracker ──
        active_tracks = self.tracker.update(detections)

        # ── 3. For each detection: align → quality → embed → match ──
        for det in detections:
            face_result = FaceResult(
                bbox=det.bbox,
                detection_confidence=det.confidence,
            )

            # Find associated track
            track = self.tracker.get_track_for_bbox(det.bbox)
            if track is not None:
                face_result.track_id = track.track_id

            # Align
            try:
                aligned = self.aligner.align(frame, det)
            except Exception as e:
                logger.error(f"Alignment error for face: {e}")
                face_result.recognition = RecognitionResult(
                    state=RecognitionState.REJECTED,
                    error="alignment_error",
                    reason=f"Alignment failed: {e}",
                )
                result.num_rejected += 1
                result.faces.append(face_result)
                continue

            if aligned is None:
                face_result.recognition = RecognitionResult(
                    state=RecognitionState.REJECTED,
                    error="alignment_failed",
                    reason="Alignment returned None",
                )
                result.num_rejected += 1
                result.faces.append(face_result)
                continue

            # Quality gate
            try:
                q_result = self.quality.assess(det, aligned)
                face_result.quality = q_result
            except Exception as e:
                logger.error(f"Quality error: {e}")
                q_result = QualityResult(
                    overall_score=0.0, accepted=False,
                    reason="QUALITY_ERROR",
                )
                face_result.quality = q_result

            if not q_result.accepted:
                face_result.recognition = RecognitionResult(
                    state=RecognitionState.REJECTED,
                    error=f"quality_rejected:{q_result.reason}",
                    reason=f"Quality: {q_result.reason}",
                )
                result.num_rejected += 1
                # Still vote on the track (REJECTED is neutral)
                if track is not None:
                    track.vote(face_result.recognition)
                result.faces.append(face_result)
                continue

            # Embed
            try:
                emb_result = self.embedder.embed(aligned)
            except Exception as e:
                logger.error(f"Embedding error: {e}")
                face_result.recognition = RecognitionResult(
                    state=RecognitionState.REJECTED,
                    error="embedding_error",
                    reason=f"Embedding failed: {e}",
                )
                result.num_rejected += 1
                result.faces.append(face_result)
                continue

            if not emb_result.valid:
                face_result.recognition = RecognitionResult(
                    state=RecognitionState.REJECTED,
                    error=f"embedding_invalid:{emb_result.error}",
                    reason=f"Embedding invalid: {emb_result.error}",
                )
                result.num_rejected += 1
                result.faces.append(face_result)
                continue

            # Recognize
            try:
                rec_result = self.engine.recognize(emb_result.vector)
            except Exception as e:
                logger.error(f"Recognition error: {e}")
                rec_result = RecognitionResult(
                    state=RecognitionState.REJECTED,
                    error="recognition_error",
                    reason=f"Recognition failed: {e}",
                )
                result.num_rejected += 1
            else:
                if rec_result.state == RecognitionState.KNOWN:
                    result.num_recognized += 1
                elif rec_result.state == RecognitionState.UNKNOWN:
                    result.num_unknown += 1
                elif rec_result.state == RecognitionState.AMBIGUOUS:
                    result.num_ambiguous += 1
                elif rec_result.state == RecognitionState.REJECTED:
                    result.num_rejected += 1

            face_result.recognition = rec_result

            # Vote on track
            if track is not None:
                track.vote(rec_result)
                # Carry confirmed identity
                if track.confirmed_identity_id is not None:
                    face_result.confirmed_identity_id = track.confirmed_identity_id
                    face_result.confirmed = True

            # Liveness (optional, per-call)
            should_run_liveness = run_liveness and self.liveness is not None
            if self._run_liveness_on_preview:
                should_run_liveness = self.liveness is not None

            if should_run_liveness and rec_result.state == RecognitionState.KNOWN:
                try:
                    liveness_result = self.liveness.check(frame, det.bbox)
                    face_result.liveness = liveness_result
                except Exception as e:
                    logger.error(f"Liveness error: {e}")
                    face_result.liveness = LivenessResult(
                        state=LivenessState.ERROR,
                        error="liveness_error",
                    )

            result.faces.append(face_result)

        return result
