"""
Face matching module using cosine similarity.
Compares query embeddings against a registered face database,
performs greedy assignment, and resolves twin/lookalike conflicts.
"""

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from core.config import Config


class FaceMatcher:
    """Match face embeddings against a registered database with twin resolution."""

    def __init__(self, threshold=None):
        self.threshold = threshold if threshold is not None else Config.SIMILARITY_THRESHOLD
        # {student_id: {"name": str, "embeddings": list[np.ndarray]}}
        self.registered_db = {}
        self.twin_handler = None  # Set externally via set_twin_handler()

    def set_twin_handler(self, twin_handler):
        """Attach a TwinHandler for twin-aware matching."""
        self.twin_handler = twin_handler

    def load_database(self, face_db):
        self.registered_db = face_db
        print(f"[FaceMatcher] Loaded database with {len(self.registered_db)} students.")

    def register_student(self, student_id, name, embeddings):
        if not embeddings:
            return
        self.registered_db[student_id] = {"name": name, "embeddings": embeddings}

    def _cosine_similarity(self, a, b):
        return 1.0 - cosine_distance(a.flatten(), b.flatten())

    def _best_similarity(self, query, stored_embeddings):
        best = -1.0
        for stored in stored_embeddings:
            sim = self._cosine_similarity(query, stored)
            if sim > best:
                best = sim
        return best

    # ──────────────────────────────────────────────
    # Single match (with twin resolution)
    # ──────────────────────────────────────────────

    def find_match(self, query_embedding):
        """
        Find best match for a single query embedding.
        If the match is a twin, resolve which twin it is.

        Returns:
            Dict {student_id, name, confidence, uncertain?, twin_conflict?}
            or None.
        """
        if query_embedding is None:
            return None

        best_match = None
        best_score = -1.0
        second_best = -1.0

        for student_id, data in self.registered_db.items():
            sim = self._best_similarity(query_embedding, data["embeddings"])
            if sim > best_score:
                second_best = best_score
                best_score = sim
                best_match = {
                    "student_id": student_id,
                    "name": data["name"],
                    "confidence": sim,
                }
            elif sim > second_best:
                second_best = sim

        # Require both absolute threshold and a margin to the runner-up
        if best_match is None or best_score < self.threshold:
            return None

        if (best_score - second_best) < Config.MIN_MATCH_MARGIN:
            # Ambiguous match — too close to call
            return None

        # Twin resolution
        if self.twin_handler is not None:
            twin_id = self.twin_handler.get_twin_of(best_match["student_id"])
            if twin_id and twin_id in self.registered_db:
                twin_data = self.registered_db[twin_id]
                resolution = self.twin_handler.resolve_twin_match(
                    query_embedding,
                    best_match["student_id"], self.registered_db[best_match["student_id"]]["embeddings"],
                    twin_id, twin_data["embeddings"],
                )

                if resolution["status"] == "RESOLVED":
                    resolved_id = resolution["assigned_to"]
                    resolved_name = self.registered_db[resolved_id]["name"]
                    best_match = {
                        "student_id": resolved_id,
                        "name": resolved_name,
                        "confidence": max(resolution["confidence_a"], resolution["confidence_b"]),
                        "uncertain": False,
                        "twin_resolved": True,
                    }
                else:
                    # UNCERTAIN — can't distinguish
                    best_id = resolution["assigned_to"]
                    other_id = twin_id if best_id == best_match["student_id"] else best_match["student_id"]
                    best_match["uncertain"] = True
                    best_match["twin_conflict"] = {
                        "student_a": best_id,
                        "name_a": self.registered_db[best_id]["name"],
                        "score_a": resolution["confidence_a"] if best_id == best_match["student_id"] else resolution["confidence_b"],
                        "student_b": other_id,
                        "name_b": self.registered_db[other_id]["name"],
                        "score_b": resolution["confidence_b"] if best_id == best_match["student_id"] else resolution["confidence_a"],
                        "difference": resolution["difference"],
                    }

        return best_match

    # ──────────────────────────────────────────────
    # Batch matching (greedy + twin resolution)
    # ──────────────────────────────────────────────

    def find_all_matches(self, query_embeddings):
        """
        Find matches for multiple query embeddings using greedy assignment.
        Each registered identity can only be matched once.
        Twin pairs are resolved when both twins appear or when a face
        matches a twin pair member.

        Returns:
            List (same length as input) of match dicts or None.
            Match dict: {student_id, name, confidence, uncertain?, twin_conflict?}
        """
        if not query_embeddings or not self.registered_db:
            return [None] * len(query_embeddings)

        num_queries = len(query_embeddings)
        student_ids = list(self.registered_db.keys())
        num_students = len(student_ids)

        # Build similarity matrix
        sim_matrix = np.zeros((num_queries, num_students), dtype=np.float32)
        for qi, q_emb in enumerate(query_embeddings):
            if q_emb is None:
                continue
            for si, sid in enumerate(student_ids):
                sim_matrix[qi, si] = self._best_similarity(
                    q_emb, self.registered_db[sid]["embeddings"]
                )

        # Greedy assignment
        results = [None] * num_queries
        assigned_queries = set()
        assigned_students = set()

        pairs = []
        for qi in range(num_queries):
            for si in range(num_students):
                pairs.append((sim_matrix[qi, si], qi, si))
        pairs.sort(key=lambda x: x[0], reverse=True)

        for sim_score, qi, si in pairs:
            if qi in assigned_queries or si in assigned_students:
                continue
            if sim_score < self.threshold:
                break

            # Check ambiguity among currently unassigned students for this query
            row = sim_matrix[qi]
            unassigned_idxs = [j for j in range(num_students) if j not in assigned_students]
            if not unassigned_idxs:
                continue
            # Best and second-best among unassigned students
            vals = row[unassigned_idxs]
            if vals.size == 0:
                continue
            # indices relative to unassigned_idxs
            rel_sorted = np.argsort(vals)[::-1]
            best_rel = rel_sorted[0]
            best_idx = unassigned_idxs[best_rel]
            best_val = row[best_idx]
            second_val = vals[rel_sorted[1]] if vals.size > 1 else -1.0

            # Only accept if this pair corresponds to the best unassigned student
            if best_idx != si:
                continue

            # Require margin between best and runner-up to avoid ambiguous assignments
            if (best_val - second_val) < Config.MIN_MATCH_MARGIN:
                continue

            sid = student_ids[si]
            match = {
                "student_id": sid,
                "name": self.registered_db[sid]["name"],
                "confidence": float(sim_score),
            }

            # Twin resolution
            if self.twin_handler is not None:
                twin_id = self.twin_handler.get_twin_of(sid)
                if twin_id and twin_id in self.registered_db:
                    twin_si = student_ids.index(twin_id) if twin_id in student_ids else None

                    # Check if twin was already assigned to a different query
                    if twin_si is not None and twin_si in assigned_students:
                        # Twin already matched to another face — this face is safe
                        pass
                    else:
                        q_emb = query_embeddings[qi]
                        if q_emb is not None:
                            resolution = self.twin_handler.resolve_twin_match(
                                q_emb,
                                sid, self.registered_db[sid]["embeddings"],
                                twin_id, self.registered_db[twin_id]["embeddings"],
                            )

                            if resolution["status"] == "RESOLVED":
                                resolved_id = resolution["assigned_to"]
                                resolved_name = self.registered_db[resolved_id]["name"]
                                match = {
                                    "student_id": resolved_id,
                                    "name": resolved_name,
                                    "confidence": max(resolution["confidence_a"], resolution["confidence_b"]),
                                    "uncertain": False,
                                    "twin_resolved": True,
                                }
                                # Update si to resolved student's index
                                if resolved_id != sid:
                                    si = student_ids.index(resolved_id)
                            else:
                                match["uncertain"] = True
                                a_id = sid
                                b_id = twin_id
                                match["twin_conflict"] = {
                                    "student_a": a_id,
                                    "name_a": self.registered_db[a_id]["name"],
                                    "score_a": resolution["confidence_a"],
                                    "student_b": b_id,
                                    "name_b": self.registered_db[b_id]["name"],
                                    "score_b": resolution["confidence_b"],
                                    "difference": resolution["difference"],
                                }

            results[qi] = match
            assigned_queries.add(qi)
            assigned_students.add(si)

            if len(assigned_queries) == num_queries or len(assigned_students) == num_students:
                break

        return results
