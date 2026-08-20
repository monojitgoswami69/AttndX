"""
ArcFace embedding generation via ONNX Runtime.

Model: arcfaceresnet100-11-int8.onnx
  Input:  [1, 3, 112, 112] float32, NCHW, RGB, normalized to [-1, 1]
  Output: [1, 512] float32

Preprocessing (matches ONNX Model Zoo ArcFace spec):
  1. Input: 112x112 BGR aligned face (from ArcFaceAligner)
  2. BGR → RGB
  3. (x - 127.5) / 127.5  → [-1, 1]
  4. HWC → CHW → NCHW (1, 3, 112, 112)

Postprocessing:
  1. Flatten to (512,)
  2. L2-normalize
  3. Validate: finite, correct dim, norm ≈ 1.0

Every embedding is tagged with the pipeline version for compatibility tracking.
Loaded once at startup, reused for all inference. Thread-safe per ONNX-RT spec.
"""
import logging
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

from gen2.config import Config

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    vector: np.ndarray | None    # (512,) float32, L2-normalized
    valid: bool
    error: str | None = None
    pipeline_version: str = ""


class ArcFaceEmbedder:
    """ArcFace ONNX embedder with validation and versioning."""

    def __init__(self):
        model_path = Config.model_path("embedder")
        if not model_path.exists():
            raise FileNotFoundError(f"ArcFace model not found: {model_path}")

        providers = Config.get("onnx", "providers")
        # Filter to only available providers
        available = ort.get_available_providers()
        active_providers = [p for p in providers if p in available]
        if not active_providers:
            active_providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(
            str(model_path), providers=active_providers,
        )
        self._input_name = self._session.get_inputs()[0].name
        self._dim = Config.get("embedding", "dimension")
        self._pipeline_version = Config.pipeline_version_string()
        logger.info(
            f"ArcFace loaded from {model_path.name} "
            f"(providers: {active_providers}, dim: {self._dim})"
        )

    @property
    def pipeline_version(self) -> str:
        return self._pipeline_version

    @property
    def dimension(self) -> int:
        return self._dim

    def _preprocess(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Preprocess a 112x112 BGR aligned face for ArcFace.
        Returns NCHW float32 in [-1, 1]."""
        img = aligned_bgr.astype(np.float32)
        # BGR → RGB
        img = img[:, :, ::-1]
        # Normalize to [-1, 1]
        img = (img - 127.5) / 127.5
        # HWC → CHW → NCHW
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return np.ascontiguousarray(img)

    def embed(self, aligned_bgr: np.ndarray) -> EmbeddingResult:
        """Generate a 512-d L2-normalized embedding from a 112x112 BGR aligned face.
        Returns EmbeddingResult with the vector or an error."""
        if aligned_bgr is None or aligned_bgr.size == 0:
            return EmbeddingResult(None, False, "empty_input", self._pipeline_version)

        h, w = aligned_bgr.shape[:2]
        if h != 112 or w != 112:
            return EmbeddingResult(None, False,
                                    f"wrong_size_{h}x{w}_expected_112x112",
                                    self._pipeline_version)

        try:
            tensor = self._preprocess(aligned_bgr)
            outputs = self._session.run(None, {self._input_name: tensor})
            raw = outputs[0].flatten().astype(np.float32)
        except Exception as e:
            logger.error(f"ArcFace inference error: {e}")
            return EmbeddingResult(None, False, f"inference_error", self._pipeline_version)

        # Validate raw output
        if not np.all(np.isfinite(raw)):
            return EmbeddingResult(None, False, "nan_or_inf", self._pipeline_version)
        if raw.shape[0] != self._dim:
            return EmbeddingResult(None, False,
                                   f"dim_mismatch_{raw.shape[0]}_vs_{self._dim}",
                                   self._pipeline_version)

        # L2-normalize
        norm = float(np.linalg.norm(raw))
        if norm < 1e-10:
            return EmbeddingResult(None, False, "zero_norm", self._pipeline_version)
        vector = (raw / norm).astype(np.float32)

        # Validate normalized
        if not np.all(np.isfinite(vector)):
            return EmbeddingResult(None, False, "nan_after_norm", self._pipeline_version)
        final_norm = float(np.linalg.norm(vector))
        if final_norm < 0.99 or final_norm > 1.01:
            return EmbeddingResult(None, False, f"bad_norm_{final_norm:.4f}",
                                   self._pipeline_version)

        return EmbeddingResult(vector, True, None, self._pipeline_version)

    def embed_batch(self, aligned_faces: list[np.ndarray]) -> list[EmbeddingResult]:
        """Generate embeddings for multiple aligned faces.
        Each face is processed independently — a failure on one face
        does not affect others."""
        results = []
        for face in aligned_faces:
            results.append(self.embed(face))
        return results
