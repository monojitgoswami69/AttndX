"""
Recognition decision engine with explicit states.

States:
  KNOWN       — single identity above accept_threshold, unambiguous
  UNKNOWN     — below reject_threshold (no identity matches)
  AMBIGUOUS   — best match above accept_threshold but too close to runner-up
  REJECTED    — quality or technical failure (distinct from UNKNOWN)

Key design principle: each face is matched INDEPENDENTLY.
One face's ambiguity never affects another face's decision.
No global mutable state. All decisions are per-face, per-call.
"""
import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from config import Config

logger = logging.getLogger(__name__)


class RecognitionState(Enum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"
    REJECTED = "rejected"


@dataclass
class Candidate:
    """A single candidate match."""
    identity_id: str
    name: str
    similarity: float          # cosine similarity (higher = better)


@dataclass
class RecognitionResult:
    state: RecognitionState
    identity_id: str | None = None
    name: str | None = None
    confidence: float = 0.0    # best similarity score
    candidates: list[Candidate] = field(default_factory=list)
    reason: str = ""          # human-readable explanation
    error: str | None = None  # technical error code (distinct from "unknown")


class IdentityIndex:
    """In-memory exact-search identity index.

    Rebuildable from BiometricDB. No persistence of the index itself.
    Cosine similarity via matrix-vector product for efficiency.
    """

    def __init__(self):
        # {identity_id: (name, template_vector)}
        self._entries: dict[str, tuple[str, np.ndarray]] = {}
        # Stacked matrix for fast search: (N, 512)
        self._matrix: np.ndarray | None = None
        self._ids: list[str] = []
        self._names: list[str] = []

    def rebuild(self, templates: list[tuple[str, np.ndarray]],
                names: dict[str, str]):
        """Rebuild the index from authoritative template storage.
        templates: [(identity_id, template_vector)]
        names: {identity_id: name}
        """
        self._entries = {}
        self._ids = []
        self._names = []
        vectors = []
        for identity_id, vec in templates:
            name = names.get(identity_id, identity_id)
            self._entries[identity_id] = (name, vec.astype(np.float32))
            self._ids.append(identity_id)
            self._names.append(name)
            vectors.append(vec.astype(np.float32))
        if vectors:
            self._matrix = np.stack(vectors)  # (N, 512)
        else:
            self._matrix = None
        logger.info(f"IdentityIndex rebuilt: {len(self._ids)} identities")

    def add_identity(self, identity_id: str, name: str, vector: np.ndarray):
        """Incrementally add a single identity."""
        self._entries[identity_id] = (name, vector.astype(np.float32))
        self._ids.append(identity_id)
        self._names.append(name)
        if self._matrix is not None:
            self._matrix = np.vstack([self._matrix, vector[None, :].astype(np.float32)])
        else:
            self._matrix = vector[None, :].astype(np.float32)
        logger.info(f"IdentityIndex: added {identity_id} ({name})")

    def remove_identity(self, identity_id: str):
        """Remove an identity. Triggers a full rebuild for correctness."""
        if identity_id in self._entries:
            del self._entries[identity_id]
            # Rebuild matrix from entries
            ids, names, vectors = [], [], []
            for sid, (name, vec) in self._entries.items():
                ids.append(sid)
                names.append(name)
                vectors.append(vec)
            self._ids = ids
            self._names = names
            self._matrix = np.stack(vectors) if vectors else None
            logger.info(f"IdentityIndex: removed {identity_id}, rebuilt")

    @property
    def size(self) -> int:
        return len(self._ids)

    def search(self, query: np.ndarray, top_k: int = 5) -> list[Candidate]:
        """Search for the best matching identities.
        Returns sorted list of candidates (best first)."""
        if self._matrix is None or len(self._ids) == 0:
            return []

        q = query.flatten().astype(np.float32)
        # Cosine similarity = dot product for L2-normalized vectors
        sims = (self._matrix @ q).astype(np.float32)  # (N,)
        # Get top-k
        k = min(top_k, len(self._ids))
        if k <= 0:
            return []
        top_indices = np.argpartition(sims, -k)[-k:]
        # Sort by similarity descending
        top_indices = top_indices[np.argsort(-sims[top_indices])]

        candidates = []
        for idx in top_indices:
            candidates.append(Candidate(
                identity_id=self._ids[idx],
                name=self._names[idx],
                similarity=float(sims[idx]),
            ))
        return candidates


class RecognitionEngine:
    """Top-level recognition decision engine.

    Given a query embedding, returns a RecognitionResult with explicit state.
    Thresholds are from config (calibrated — see evaluation/calibration.py).
    """

    def __init__(self, index: IdentityIndex):
        self.index = index
        self.accept_threshold = Config.get("recognition", "accept_threshold")
        self.ambiguity_margin = Config.get("recognition", "ambiguity_margin")
        self.reject_threshold = Config.get("recognition", "reject_threshold")
        self.top_k = Config.get("recognition", "top_k")

    def recognize(self, query: np.ndarray | None) -> RecognitionResult:
        """Recognize a single face embedding. Independent per-face decision."""
        if query is None:
            return RecognitionResult(
                state=RecognitionState.REJECTED,
                reason="No embedding provided",
                error="NO_EMBEDDING",
            )

        if not np.all(np.isfinite(query)):
            return RecognitionResult(
                state=RecognitionState.REJECTED,
                reason="Embedding contains NaN/Inf",
                error="EMBEDDING_INVALID",
            )

        candidates = self.index.search(query, top_k=self.top_k)
        if not candidates:
            return RecognitionResult(
                state=RecognitionState.UNKNOWN,
                reason="No identities enrolled",
            )

        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None

        # Below reject threshold → clearly unknown
        if best.similarity < self.reject_threshold:
            return RecognitionResult(
                state=RecognitionState.UNKNOWN,
                confidence=best.similarity,
                candidates=candidates,
                reason=f"Best sim {best.similarity:.3f} < reject {self.reject_threshold}",
            )

        # Below accept threshold → unknown (not confident enough)
        if best.similarity < self.accept_threshold:
            return RecognitionResult(
                state=RecognitionState.UNKNOWN,
                confidence=best.similarity,
                candidates=candidates,
                reason=f"Best sim {best.similarity:.3f} < accept {self.accept_threshold}",
            )

        # Above accept → check ambiguity
        if second is not None:
            margin = best.similarity - second.similarity
            if margin < self.ambiguity_margin:
                return RecognitionResult(
                    state=RecognitionState.AMBIGUOUS,
                    confidence=best.similarity,
                    candidates=candidates,
                    reason=f"Ambiguous: best {best.similarity:.3f} "
                           f"second {second.similarity:.3f} margin {margin:.3f}",
                )

        # KNOWN
        return RecognitionResult(
            state=RecognitionState.KNOWN,
            identity_id=best.identity_id,
            name=best.name,
            confidence=best.similarity,
            candidates=candidates,
            reason=f"Matched {best.name} sim={best.similarity:.3f}",
        )

    def recognize_batch(self, queries: list[np.ndarray | None]) -> list[RecognitionResult]:
        """Recognize multiple face embeddings independently.
        Each face is processed separately — no interaction between faces."""
        return [self.recognize(q) for q in queries]
