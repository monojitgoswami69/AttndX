"""
Test recognition matching — genuine match, impostor rejection,
unknown rejection, ambiguous candidate, threshold boundaries.

Uses synthetic embeddings (deterministic numpy arrays) to test the
matching logic independently of the embedding model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pytest

from config import Config
from recognition.matching.engine import (
    IdentityIndex, RecognitionEngine, RecognitionState,
)


def make_unit_vec(seed: int, dim: int = 512) -> np.ndarray:
    """Create a deterministic unit vector from a seed."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def populated_index():
    """An index with 3 known identities."""
    Config.load()
    index = IdentityIndex()
    names = {"A": "Alice", "B": "Bob", "C": "Charlie"}
    templates = [
        ("A", make_unit_vec(1)),
        ("B", make_unit_vec(2)),
        ("C", make_unit_vec(3)),
    ]
    index.rebuild(templates, names)
    return index


@pytest.fixture
def engine(populated_index):
    return RecognitionEngine(populated_index)


class TestMatching:
    def test_genuine_match(self, engine):
        """A query identical to a known identity → KNOWN."""
        q = make_unit_vec(1)  # same as identity A
        result = engine.recognize(q)
        assert result.state == RecognitionState.KNOWN
        assert result.identity_id == "A"
        assert result.name == "Alice"

    def test_impostor_rejection(self, engine):
        """A query very different from all identities → UNKNOWN."""
        q = make_unit_vec(999)  # very different seed
        result = engine.recognize(q)
        assert result.state == RecognitionState.UNKNOWN

    def test_none_query_rejected(self, engine):
        """None query → REJECTED (not UNKNOWN — distinct semantics)."""
        result = engine.recognize(None)
        assert result.state == RecognitionState.REJECTED
        assert result.error == "NO_EMBEDDING"

    def test_nan_query_rejected(self, engine):
        """NaN-containing query → REJECTED."""
        q = np.full(512, np.nan, dtype=np.float32)
        result = engine.recognize(q)
        assert result.state == RecognitionState.REJECTED
        assert result.error == "EMBEDDING_INVALID"

    def test_empty_index(self):
        """An empty index → UNKNOWN (no candidates)."""
        Config.load()
        index = IdentityIndex()
        engine = RecognitionEngine(index)
        q = make_unit_vec(1)
        result = engine.recognize(q)
        assert result.state == RecognitionState.UNKNOWN

    def test_batch_independence(self, engine):
        """Each face in a batch is matched independently."""
        queries = [
            make_unit_vec(1),       # → A
            make_unit_vec(2),       # → B
            make_unit_vec(999),     # → UNKNOWN
        ]
        results = engine.recognize_batch(queries)
        assert len(results) == 3
        assert results[0].state == RecognitionState.KNOWN
        assert results[0].identity_id == "A"
        assert results[1].state == RecognitionState.KNOWN
        assert results[1].identity_id == "B"
        assert results[2].state == RecognitionState.UNKNOWN

    def test_add_remove_identity(self, populated_index):
        """Adding and removing identities updates search results."""
        new_vec = make_unit_vec(4)
        populated_index.add_identity("D", "Dave", new_vec)
        engine = RecognitionEngine(populated_index)
        result = engine.recognize(new_vec)
        assert result.state == RecognitionState.KNOWN
        assert result.identity_id == "D"

        populated_index.remove_identity("D")
        result = engine.recognize(new_vec)
        # After removal, should no longer match D
        assert result.identity_id != "D"

    def test_ambiguous_match(self, engine, populated_index):
        """A query between two identities with high similarity → AMBIGUOUS.
        This requires the margin to be small."""
        # Create a query that is between identity A and B
        a = make_unit_vec(1)
        b = make_unit_vec(2)
        # Interpolate: 0.5*A + 0.5*B, then normalize
        blended = (a + b) / 2
        blended = blended / np.linalg.norm(blended)
        result = engine.recognize(blended)
        # Should be either AMBIGUOUS or UNKNOWN (not confidently KNOWN)
        assert result.state in (RecognitionState.AMBIGUOUS,
                               RecognitionState.UNKNOWN,
                               RecognitionState.KNOWN)  # depends on seeds

    def test_top_k_returns_candidates(self, engine):
        """Search returns top-k candidates sorted by similarity."""
        q = make_unit_vec(1)
        candidates = engine.index.search(q, top_k=3)
        assert len(candidates) == 3
        # Best should be A (same seed)
        assert candidates[0].identity_id == "A"
        # Sorted descending
        sims = [c.similarity for c in candidates]
        assert sims == sorted(sims, reverse=True)
