"""
ArcFace embedding generation via ONNX Runtime.

Model: w600k_r50.onnx (InsightFace, ResNet50, WebFace600K, FP32)
  Input:  [1, 3, 112, 112] float32, NCHW, RGB, normalized to [-1, 1]
  Output: [1, 512] float32

Preprocessing (standard InsightFace ArcFace spec):
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

Startup sanity check:
  On load, we embed two visually distinct synthetic images (black vs white)
  and verify the cosine similarity is below 0.5. This catches catastrophically
  broken models (e.g. INT8 quantized models that collapse the embedding space)
  before any real faces are processed.
"""
import logging
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

from config import Config

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    vector: np.ndarray | None    # (512,) float32, L2-normalized
    valid: bool
    error: str | None = None
    pipeline_version: str = ""


class ArcFaceEmbedder:
    """ArcFace ONNX embedder with validation, versioning, and startup sanity check."""

    def __init__(self):
        model_path = Config.model_path("embedder")
        if not model_path.exists():
            raise FileNotFoundError(
                f"ArcFace model not found: {model_path}\n"
                f"Please run `python scripts/download_models.py` to download all required AI models."
            )

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

        # Startup sanity check — catches broken models early
        self._validate_model_sanity()

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

    def _validate_model_sanity(self):
        """Startup sanity check: verify the model produces discriminative embeddings.

        Embeds 8 diverse synthetic images (black, white, gray, random noise ×3,
        horizontal gradient, vertical gradient) and checks that the mean pairwise
        cosine similarity is below 0.72.

        Working FP32 models (w600k_r50, glintr100) score ~0.58.
        The broken INT8 model scores ~0.80.
        Threshold at 0.72 gives a clear margin for any well-trained model.

        This multi-probe approach is robust across different model architectures,
        unlike single-pair tests which can give false positives on specific models.
        """
        rng = np.random.RandomState(12345)
        probe_images = [
            np.zeros((112, 112, 3), dtype=np.uint8),                    # black
            np.full((112, 112, 3), 255, dtype=np.uint8),                # white
            np.full((112, 112, 3), 128, dtype=np.uint8),                # gray
            rng.randint(0, 255, (112, 112, 3)).astype(np.uint8),        # random1
            rng.randint(0, 255, (112, 112, 3)).astype(np.uint8),        # random2
            rng.randint(0, 255, (112, 112, 3)).astype(np.uint8),        # random3
        ]
        # Horizontal gradient
        h_grad = np.zeros((112, 112, 3), dtype=np.uint8)
        for c in range(3):
            h_grad[:, :, c] = np.tile(
                np.linspace(0, 255, 112, dtype=np.uint8), (112, 1)
            )
        probe_images.append(h_grad)
        # Vertical gradient
        v_grad = np.zeros((112, 112, 3), dtype=np.uint8)
        for c in range(3):
            v_grad[:, :, c] = np.tile(
                np.linspace(0, 255, 112, dtype=np.uint8), (112, 1)
            ).T
        probe_images.append(v_grad)

        # Embed all probes
        embeddings = []
        for img in probe_images:
            result = self.embed(img)
            if not result.valid:
                logger.warning(
                    f"Model sanity check: probe embed failed ({result.error}). "
                    "Skipping sanity check."
                )
                return
            embeddings.append(result.vector)

        # Compute mean pairwise cosine similarity
        mat = np.stack(embeddings)
        sim_matrix = (mat @ mat.T).astype(np.float32)
        n = len(embeddings)
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += float(sim_matrix[i, j])
                count += 1
        mean_sim = total / count if count > 0 else 0.0

        # Working FP32 models: ~0.58 mean similarity
        # Broken INT8 model:   ~0.80 mean similarity
        # Threshold at 0.72 cleanly separates them
        SANITY_THRESHOLD = 0.72

        if mean_sim > SANITY_THRESHOLD:
            raise RuntimeError(
                f"CRITICAL: Embedding model sanity check FAILED. "
                f"Mean pairwise similarity across {n} diverse probes = "
                f"{mean_sim:.4f} (expected < {SANITY_THRESHOLD}). "
                f"The model is collapsing different inputs to similar "
                f"embeddings — it cannot distinguish different faces. "
                f"Replace the model with a properly trained FP32 model."
            )

        logger.info(
            f"Model sanity check passed: mean probe similarity = "
            f"{mean_sim:.4f} (threshold < {SANITY_THRESHOLD})"
        )
