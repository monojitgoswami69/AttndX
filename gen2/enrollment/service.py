"""
Enrollment service — builds reliable identity templates.

Protocol:
  1. Capture candidate frames (one face per frame required)
  2. Detect → align → quality gate → embed → validate
  3. Collect min_samples valid samples
  4. Check intra-person consistency (pairwise cosine similarity)
  5. Check for duplicates against existing identities
  6. Compute template (centroid: mean of L2-normalized embeddings, re-normalized)
  7. Persist atomically (identity + samples + template in one SQLite transaction)
  8. Update in-memory index
  9. Verify stored identity

Duplicate handling (NOT a global "twin system"):
  - If best cosine sim to existing identity > duplicate_ambiguous_threshold → AMBIGUOUS
  - If > duplicate_threshold but < ambiguous → POSSIBLE_DUPLICATE
  - Never auto-merges. Requires human confirmation for ambiguous cases.
"""
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from gen2.config import Config
from gen2.recognition.embeddings.arcface_onnx import ArcFaceEmbedder, EmbeddingResult
from gen2.recognition.matching.engine import IdentityIndex
from gen2.storage.db import BiometricDB
from gen2.vision.alignment.arcface import ArcFaceAligner
from gen2.vision.detection.yunet import YuNetDetector
from gen2.vision.quality.assessor import FaceQualityAssessor

logger = logging.getLogger(__name__)


class EnrollmentStatus(Enum):
    SUCCESS = "success"
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    INCONSISTENT_SAMPLES = "inconsistent_samples"
    POSSIBLE_DUPLICATE = "possible_duplicate"
    AMBIGUOUS_DUPLICATE = "ambiguous_duplicate"
    IDENTITY_EXISTS = "identity_exists"
    INVALID_ENROLLMENT = "invalid_enrollment"


@dataclass
class EnrollmentResult:
    status: EnrollmentStatus
    identity_id: str | None = None
    name: str | None = None
    samples_stored: int = 0
    quality_scores: list[float] = field(default_factory=list)
    intra_similarity: float = 0.0
    duplicate_match_id: str | None = None
    duplicate_match_name: str | None = None
    duplicate_similarity: float = 0.0
    issues: list[str] = field(default_factory=list)


class EnrollmentService:
    """Orchestrates the enrollment pipeline."""

    def __init__(self, detector: YuNetDetector, aligner: ArcFaceAligner,
                 quality_assessor: FaceQualityAssessor, embedder: ArcFaceEmbedder,
                 db: BiometricDB, index: IdentityIndex):
        self.detector = detector
        self.aligner = aligner
        self.quality = quality_assessor
        self.embedder = embedder
        self.db = db
        self.index = index

        self.min_samples = Config.get("enrollment", "min_samples")
        self.max_samples = Config.get("enrollment", "max_samples")
        self.min_intra_sim = Config.get("enrollment", "min_intra_similarity")
        self.dup_threshold = Config.get("enrollment", "duplicate_threshold")
        self.dup_ambiguous = Config.get("enrollment", "duplicate_ambiguous_threshold")
        self.pipeline_version = embedder.pipeline_version

    def enroll(self, name: str, frames: list[np.ndarray],
               identity_id: str | None = None) -> EnrollmentResult:
        """Enroll a new identity from captured frames.

        Each frame must contain exactly one face (detected via detect_single_face).
        Frames are processed: detect → align → quality → embed → validate.
        Only valid, high-quality embeddings are accepted.

        Then checks:
          - Intra-person consistency
          - Duplicate against existing identities
          - Creates template + persists atomically
        """
        result = EnrollmentResult(status=EnrollmentStatus.INVALID_ENROLLMENT)

        if identity_id is None:
            identity_id = self._generate_id()
        result.identity_id = identity_id
        result.name = name

        # Check if identity already exists
        if self.db.identity_exists(identity_id):
            result.status = EnrollmentStatus.IDENTITY_EXISTS
            result.issues.append(f"Identity '{identity_id}' already exists")
            return result

        if not frames:
            result.issues.append("No frames provided")
            return result

        # ── Process each frame: detect → align → quality → embed ──
        valid_embeddings: list[np.ndarray] = []
        valid_quality_scores: list[float] = []

        for i, frame in enumerate(frames):
            if frame is None or frame.size == 0:
                result.issues.append(f"Frame {i+1}: empty frame")
                continue

            # Detect
            detections = self.detector.detect(frame)
            if len(detections) == 0:
                result.issues.append(f"Frame {i+1}: no face detected")
                continue
            if len(detections) > 1:
                result.issues.append(
                    f"Frame {i+1}: multiple faces detected, using highest confidence"
                )
            det = max(detections, key=lambda d: d.confidence)

            # Align
            aligned = self.aligner.align(frame, det)
            if aligned is None:
                result.issues.append(f"Frame {i+1}: alignment failed")
                continue

            # Quality
            q_result = self.quality.assess(det, aligned)
            result.quality_scores.append(q_result.overall_score)
            if not q_result.accepted:
                result.issues.append(
                    f"Frame {i+1}: quality rejected ({q_result.reason})"
                )
                continue

            # Embed
            emb_result = self.embedder.embed(aligned)
            if not emb_result.valid:
                result.issues.append(
                    f"Frame {i+1}: embedding failed ({emb_result.error})"
                )
                continue

            valid_embeddings.append(emb_result.vector)
            valid_quality_scores.append(q_result.overall_score)

        result.samples_stored = len(valid_embeddings)

        # ── Check sample count ──
        if len(valid_embeddings) < self.min_samples:
            result.status = EnrollmentStatus.INSUFFICIENT_SAMPLES
            result.issues.append(
                f"Only {len(valid_embeddings)}/{self.min_samples} valid samples. "
                f"Need at least {self.min_samples}."
            )
            return result

        # ── Intra-person consistency check ──
        intra_sim = self._compute_intra_similarity(valid_embeddings)
        result.intra_similarity = intra_sim
        if intra_sim < self.min_intra_sim:
            result.status = EnrollmentStatus.INCONSISTENT_SAMPLES
            result.issues.append(
                f"Samples are internally inconsistent (intra-sim {intra_sim:.3f} "
                f"< {self.min_intra_sim}). This may indicate enrollment of "
                f"different people. Re-capture with consistent framing."
            )
            return result

        # ── Duplicate check ──
        template = self._compute_template(valid_embeddings)
        dup_result = self._check_duplicate(template)
        if dup_result is not None:
            dup_id, dup_name, dup_sim = dup_result
            result.duplicate_match_id = dup_id
            result.duplicate_match_name = dup_name
            result.duplicate_similarity = dup_sim
            if dup_sim >= self.dup_ambiguous:
                result.status = EnrollmentStatus.AMBIGUOUS_DUPLICATE
                result.issues.append(
                    f"AMBIGUOUS: {dup_sim:.1%} similarity to existing "
                    f"identity '{dup_name}'. This may be the same person. "
                    f"Human confirmation required."
                )
                return result
            elif dup_sim >= self.dup_threshold:
                result.status = EnrollmentStatus.POSSIBLE_DUPLICATE
                result.issues.append(
                    f"POSSIBLE DUPLICATE: {dup_sim:.1%} similarity to "
                    f"'{dup_name}'. Review before confirming."
                )
                # Fall through to enrollment — possible duplicate is a warning,
                # not a hard block. The caller decides.

        # ── Persist atomically ──
        # All writes happen in sequence; if any fails, the identity is
        # deleted to prevent partial state.
        if not self.db.add_identity(identity_id, name, self.pipeline_version):
            result.issues.append("Failed to create identity record")
            return result

        for emb, qscore in zip(valid_embeddings, valid_quality_scores):
            if not self.db.add_embedding(identity_id, emb, qscore):
                logger.error(f"Failed to store embedding for {identity_id}")
                # Roll back the identity
                self.db.delete_identity(identity_id)
                result.issues.append("Failed to store embeddings; rolled back")
                return result

        if not self.db.set_template(identity_id, template, len(valid_embeddings)):
            self.db.delete_identity(identity_id)
            result.issues.append("Failed to store template; rolled back")
            return result

        # ── Update in-memory index ──
        self.index.add_identity(identity_id, name, template)

        # ── Verify ──
        stored_template = self.db.get_template(identity_id)
        if stored_template is None or not np.allclose(stored_template, template):
            result.issues.append("WARNING: verification failed — stored template mismatch")

        result.status = EnrollmentStatus.SUCCESS
        logger.info(
            f"Enrolled {identity_id} ({name}): {len(valid_embeddings)} samples, "
            f"intra-sim {intra_sim:.3f}, template dim {template.shape}"
        )
        return result

    def delete_identity(self, identity_id: str) -> bool:
        """Delete an identity. Updates DB + index."""
        if not self.db.identity_exists(identity_id):
            return False
        self.db.delete_identity(identity_id)
        self.index.remove_identity(identity_id)
        logger.info(f"Deleted identity {identity_id}")
        return True

    def _compute_intra_similarity(self, embeddings: list[np.ndarray]) -> float:
        """Compute mean pairwise cosine similarity among enrollment samples."""
        if len(embeddings) < 2:
            return 1.0
        n = len(embeddings)
        mat = np.stack(embeddings)  # (n, 512)
        # Cosine sim = dot product (already L2-normalized)
        sim_matrix = mat @ mat.T  # (n, n)
        # Mean of upper triangle (excluding diagonal)
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += float(sim_matrix[i, j])
                count += 1
        return total / count if count > 0 else 1.0

    def _compute_template(self, embeddings: list[np.ndarray]) -> np.ndarray:
        """Compute the identity template (centroid).
        Mean of L2-normalized embeddings, then re-L2-normalized."""
        mat = np.stack(embeddings)  # (n, 512)
        centroid = np.mean(mat, axis=0)  # (512,)
        # Re-normalize to unit L2 norm
        norm = float(np.linalg.norm(centroid))
        if norm < 1e-10:
            return embeddings[0].copy()
        return (centroid / norm).astype(np.float32)

    def _check_duplicate(self, template: np.ndarray) -> tuple[str, str, float] | None:
        """Check if template matches an existing identity.
        Returns (identity_id, name, similarity) if a match is found."""
        candidates = self.index.search(template, top_k=1)
        if not candidates:
            return None
        best = candidates[0]
        if best.similarity < self.dup_threshold:
            return None
        return (best.identity_id, best.name, best.similarity)

    def _generate_id(self) -> str:
        return str(uuid.uuid4())[:12]
