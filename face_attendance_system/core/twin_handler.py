"""
Twin detection and handling module.
Detects high-similarity student pairs during registration,
and resolves identity conflicts during attendance checks.
"""

import pickle
import uuid
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine as cosine_distance
from core.config import Config


class TwinHandler:
    """Detects, registers, and resolves twin/lookalike student pairs."""

    def __init__(self, db_path=None):
        self.db_path = Path(db_path) if db_path else Config.TWIN_PAIRS_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.twin_threshold = Config.TWIN_SIMILARITY_THRESHOLD
        self.high_risk_threshold = Config.HIGH_RISK_TWIN_THRESHOLD
        self.min_twin_diff = Config.MIN_TWIN_DIFFERENCE

        # {pair_id: {student_a, student_b, similarity, status}}
        self.twin_pairs: dict[str, dict] = {}
        self.load()

    # ──────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────

    def save(self):
        try:
            with open(self.db_path, "wb") as f:
                pickle.dump(self.twin_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        except Exception as e:
            print(f"[TwinHandler] Save error: {e}")
            return False

    def load(self):
        if self.db_path.exists():
            try:
                with open(self.db_path, "rb") as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    self.twin_pairs = loaded
                    print(f"[TwinHandler] Loaded {len(self.twin_pairs)} twin pairs.")
                    return True
            except Exception as e:
                print(f"[TwinHandler] Load error: {e}")
        self.twin_pairs = {}
        return False

    # ──────────────────────────────────────────────
    # Registration-time twin check
    # ──────────────────────────────────────────────

    def _cosine_similarity(self, a, b):
        return 1.0 - cosine_distance(a.flatten(), b.flatten())

    def _avg_similarity(self, embeddings_a, embeddings_b):
        """Compute average of best pairwise similarities."""
        if not embeddings_a or not embeddings_b:
            return 0.0
        sims = []
        for ea in embeddings_a:
            best = max(self._cosine_similarity(ea, eb) for eb in embeddings_b)
            sims.append(best)
        return float(np.mean(sims))

    def check_for_twins(self, new_student_id, new_embeddings, face_database):
        """
        Compare new student's embeddings against all existing students.

        Returns:
            {has_twin, twin_student_id, twin_student_name, max_similarity,
             recommendation: "SAFE"|"POTENTIAL_TWIN"|"HIGH_RISK_TWIN"}
        """
        result = {
            "has_twin": False,
            "twin_student_id": None,
            "twin_student_name": None,
            "max_similarity": 0.0,
            "recommendation": "SAFE",
        }

        if not new_embeddings:
            return result

        students = face_database.get_all_students()
        best_sim = 0.0
        best_sid = None
        best_name = None

        for sid, data in students.items():
            if sid == new_student_id:
                continue
            existing_embs = data.get("embeddings", [])
            if not existing_embs:
                continue

            sim = self._avg_similarity(new_embeddings, existing_embs)
            if sim > best_sim:
                best_sim = sim
                best_sid = sid
                best_name = data.get("name", sid)

        result["max_similarity"] = round(best_sim, 4)

        if best_sim >= self.high_risk_threshold:
            result["has_twin"] = True
            result["twin_student_id"] = best_sid
            result["twin_student_name"] = best_name
            result["recommendation"] = "HIGH_RISK_TWIN"
        elif best_sim >= self.twin_threshold:
            result["has_twin"] = True
            result["twin_student_id"] = best_sid
            result["twin_student_name"] = best_name
            result["recommendation"] = "POTENTIAL_TWIN"

        return result

    # ──────────────────────────────────────────────
    # Twin pair management
    # ──────────────────────────────────────────────

    def register_twin_pair(self, student_a_id, student_b_id, similarity):
        """Register a confirmed twin/lookalike pair."""
        # Check if pair already exists
        for pid, pair in self.twin_pairs.items():
            ids = {pair["student_a"], pair["student_b"]}
            if {student_a_id, student_b_id} == ids:
                pair["similarity"] = max(pair["similarity"], similarity)
                self.save()
                return pid

        pair_id = str(uuid.uuid4())[:8]
        self.twin_pairs[pair_id] = {
            "student_a": student_a_id,
            "student_b": student_b_id,
            "similarity": round(similarity, 4),
            "status": "confirmed",
        }
        self.save()
        print(f"[TwinHandler] Registered twin pair {pair_id}: {student_a_id} <-> {student_b_id} ({similarity:.2%})")
        return pair_id

    def is_twin_pair(self, student_id_1, student_id_2):
        """Check if two students are a registered twin pair."""
        for pair in self.twin_pairs.values():
            ids = {pair["student_a"], pair["student_b"]}
            if {student_id_1, student_id_2} == ids:
                return True
        return False

    def get_twin_of(self, student_id):
        """Return the twin's student_id, or None."""
        for pair in self.twin_pairs.values():
            if pair["student_a"] == student_id:
                return pair["student_b"]
            if pair["student_b"] == student_id:
                return pair["student_a"]
        return None

    def get_all_twin_pairs(self):
        """Return all registered twin pairs as a list of dicts."""
        return [
            {"pair_id": pid, **data}
            for pid, data in self.twin_pairs.items()
        ]

    # ──────────────────────────────────────────────
    # Runtime twin resolution
    # ──────────────────────────────────────────────

    def resolve_twin_match(self, query_embedding,
                           candidate_a_id, candidate_a_embeddings,
                           candidate_b_id, candidate_b_embeddings):
        """
        When a face matches both members of a twin pair,
        determine which twin it actually is.

        Returns:
            {resolved, assigned_to, confidence_a, confidence_b,
             difference, status: "RESOLVED"|"UNCERTAIN"}
        """
        if query_embedding is None:
            return {
                "resolved": False, "assigned_to": None,
                "confidence_a": 0.0, "confidence_b": 0.0,
                "difference": 0.0, "status": "UNCERTAIN",
            }

        # Best similarity with each candidate
        best_a = max(
            (self._cosine_similarity(query_embedding, e) for e in candidate_a_embeddings),
            default=0.0,
        )
        best_b = max(
            (self._cosine_similarity(query_embedding, e) for e in candidate_b_embeddings),
            default=0.0,
        )

        difference = abs(best_a - best_b)

        if difference >= self.min_twin_diff:
            assigned = candidate_a_id if best_a > best_b else candidate_b_id
            return {
                "resolved": True,
                "assigned_to": assigned,
                "confidence_a": round(best_a, 4),
                "confidence_b": round(best_b, 4),
                "difference": round(difference, 4),
                "status": "RESOLVED",
            }
        else:
            # Too close to call
            assigned = candidate_a_id if best_a >= best_b else candidate_b_id
            return {
                "resolved": False,
                "assigned_to": assigned,
                "confidence_a": round(best_a, 4),
                "confidence_b": round(best_b, 4),
                "difference": round(difference, 4),
                "status": "UNCERTAIN",
            }

    def remove_pair_for_student(self, student_id):
        """Remove all twin pairs involving a student (used on deletion)."""
        to_remove = [
            pid for pid, pair in self.twin_pairs.items()
            if pair["student_a"] == student_id or pair["student_b"] == student_id
        ]
        for pid in to_remove:
            del self.twin_pairs[pid]
        if to_remove:
            self.save()
        return len(to_remove)
