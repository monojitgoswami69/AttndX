"""
Test embedding generation — shape, dtype, finiteness, normalization.

Uses real ArcFace ONNX model on a synthetic aligned face (112x112 zeros).
The output won't be a meaningful face embedding, but it must be:
  - shape (512,)
  - dtype float32
  - finite (no NaN/Inf)
  - L2-normalized (norm ≈ 1.0)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pytest

from gen2.config import Config
from gen2.recognition.embeddings.arcface_onnx import ArcFaceEmbedder


@pytest.fixture(scope="module")
def embedder():
    Config.load()
    try:
        return ArcFaceEmbedder()
    except FileNotFoundError:
        pytest.skip("ArcFace model not available")


class TestEmbedding:
    def test_valid_embedding_shape(self, embedder):
        """Embedding must be (512,) float32."""
        face = np.zeros((112, 112, 3), dtype=np.uint8)
        result = embedder.embed(face)
        assert result.valid
        assert result.vector is not None
        assert result.vector.shape == (512,)

    def test_valid_embedding_dtype(self, embedder):
        """Embedding must be float32."""
        face = np.zeros((112, 112, 3), dtype=np.uint8)
        result = embedder.embed(face)
        assert result.valid
        assert result.vector.dtype == np.float32

    def test_embedding_finite(self, embedder):
        """Embedding must not contain NaN or Inf."""
        face = np.zeros((112, 112, 3), dtype=np.uint8)
        result = embedder.embed(face)
        assert result.valid
        assert np.all(np.isfinite(result.vector))

    def test_embedding_l2_normalized(self, embedder):
        """Embedding must be L2-normalized (norm ≈ 1.0)."""
        face = np.zeros((112, 112, 3), dtype=np.uint8)
        result = embedder.embed(face)
        assert result.valid
        norm = float(np.linalg.norm(result.vector))
        assert 0.99 <= norm <= 1.01, f"Norm was {norm}"

    def test_deterministic(self, embedder):
        """Same input → same output."""
        face = np.zeros((112, 112, 3), dtype=np.uint8)
        r1 = embedder.embed(face)
        r2 = embedder.embed(face)
        assert r1.valid and r2.valid
        assert np.allclose(r1.vector, r2.vector, atol=1e-5)

    def test_wrong_size_rejected(self, embedder):
        """Wrong-sized input → rejected with error."""
        face = np.zeros((100, 100, 3), dtype=np.uint8)
        result = embedder.embed(face)
        assert not result.valid
        assert result.error is not None
        assert "size" in result.error.lower()

    def test_none_input_rejected(self, embedder):
        """None input → rejected."""
        result = embedder.embed(None)
        assert not result.valid
        assert result.error == "empty_input"

    def test_pipeline_version_set(self, embedder):
        """Embedding must carry a pipeline version string."""
        assert embedder.pipeline_version != ""
        assert "arcface" in embedder.pipeline_version.lower()

    def test_batch_independence(self, embedder):
        """A failure on one face must not affect others in a batch."""
        faces = [
            np.zeros((112, 112, 3), dtype=np.uint8),  # valid
            np.zeros((50, 50, 3), dtype=np.uint8),    # wrong size
            np.zeros((112, 112, 3), dtype=np.uint8),  # valid
        ]
        results = embedder.embed_batch(faces)
        assert len(results) == 3
        assert results[0].valid
        assert not results[1].valid
        assert results[2].valid
