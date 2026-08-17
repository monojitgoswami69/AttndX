"""
Student registration service.
Orchestrates face detection, quality checking, embedding generation,
twin detection, and database storage for registering new students.
"""

import numpy as np
from core.face_detector import YOLOFaceDetector
from core.face_embedder import FaceEmbedder
from core.face_matcher import FaceMatcher
from core.image_preprocessor import ImagePreprocessor
from storage.face_database import FaceDatabase
from core.config import Config


class RegistrationService:
    """Handles student face registration workflow with twin detection."""

    def __init__(self, face_detector, face_embedder, face_database, twin_handler=None):
        self.detector = face_detector
        self.embedder = face_embedder
        self.database = face_database
        self.preprocessor = ImagePreprocessor()
        self.twin_handler = twin_handler

    def register_student_from_frames(self, student_id, name, frames):
        """
        Register a student using captured frames.
        After generating embeddings, checks for twin/lookalike matches.

        Returns:
            Dict with success, registered_count, total_frames, quality_scores,
            issues, twin_info.
        """
        result = {
            "success": False,
            "registered_count": 0,
            "total_frames": len(frames),
            "quality_scores": [],
            "issues": [],
            "twin_info": None,
        }

        if not frames:
            result["issues"].append("No frames provided for registration.")
            return result

        if self.database.student_exists(student_id):
            result["issues"].append(
                f"Student ID '{student_id}' already exists. "
                f"Delete the existing record first to re-register."
            )
            return result

        valid_embeddings = []
        valid_face_images = []

        for i, frame in enumerate(frames):
            frame_label = f"Frame {i + 1}/{len(frames)}"

            detection = self.detector.detect_single_face(frame)
            if detection is None:
                result["issues"].append(
                    f"{frame_label}: Could not detect exactly one face."
                )
                result["quality_scores"].append(0.0)
                continue

            face_crop = detection["cropped_face"]

            quality = self.preprocessor.assess_quality(face_crop)
            result["quality_scores"].append(quality["score"])

            if quality["score"] < 0.3:
                issues_str = "; ".join(quality["issues"])
                result["issues"].append(
                    f"{frame_label}: Poor quality ({quality['score']:.2f}) — {issues_str}"
                )
                continue

            if quality["issues"]:
                for issue in quality["issues"]:
                    result["issues"].append(f"{frame_label}: Warning — {issue}")

            embedding = self.embedder.get_embedding(face_crop)
            if embedding is None:
                result["issues"].append(
                    f"{frame_label}: Could not generate face embedding."
                )
                continue

            valid_embeddings.append(embedding)
            valid_face_images.append(face_crop)

        if not valid_embeddings:
            result["issues"].append(
                "No valid face embeddings could be generated. "
                "Please try again with better lighting and positioning."
            )
            return result

        # ── Twin detection (before storing) ──
        twin_info = None
        if self.twin_handler is not None:
            twin_info = self.twin_handler.check_for_twins(
                student_id, valid_embeddings, self.database
            )
            result["twin_info"] = twin_info

        # ── Duplicate registration blocking ──
        # If any single embedding pair matches an existing student very
        # strongly, this is almost certainly the same person enrolled twice
        # (not a genuine twin). max_pair_similarity is used instead of the
        # averaged value because even a few poor captures drag the average
        # down, while the single best-matching pair stays 0.95+ for the same
        # face. Genuine twins typically stay below 0.92 on the best pair.
        if (
            Config.BLOCK_DUPLICATE_REGISTRATION
            and twin_info
            and twin_info["has_twin"]
            and twin_info.get("max_pair_similarity", 0.0) >= Config.DUPLICATE_BLOCK_THRESHOLD
        ):
            twin_name = twin_info["twin_student_name"]
            pair_pct = twin_info["max_pair_similarity"] * 100
            result["issues"].append(
                f"❌ DUPLICATE BLOCKED: {pair_pct:.1f}% best-pair similarity with "
                f"existing student '{twin_name}'. This looks like the same "
                f"person registered twice. Use a different Student ID or "
                f"delete the existing record first."
            )
            print(
                f"[Registration] Blocked duplicate registration for "
                f"'{name}' ({student_id}) — {pair_pct:.1f}% best-pair match "
                f"with '{twin_name}'"
            )
            return result

        # Store in database (genuine twins are still allowed with a warning)
        success = self.database.add_student(
            student_id=student_id,
            name=name,
            embeddings=valid_embeddings,
            face_images=valid_face_images,
        )

        result["success"] = success
        result["registered_count"] = len(valid_embeddings)

        if success:
            print(
                f"[Registration] Successfully registered '{name}' "
                f"with {len(valid_embeddings)}/{len(frames)} embeddings."
            )

            # Register twin pair if detected
            if twin_info and twin_info["has_twin"]:
                self.twin_handler.register_twin_pair(
                    student_id,
                    twin_info["twin_student_id"],
                    twin_info["max_similarity"],
                )
                rec = twin_info["recommendation"]
                sim_pct = twin_info["max_similarity"] * 100
                twin_name = twin_info["twin_student_name"]

                if rec == "HIGH_RISK_TWIN":
                    result["issues"].append(
                        f"⚠️ Very high similarity ({sim_pct:.1f}%) with {twin_name}. "
                        f"Recommend: capture more images from different angles. "
                        f"Some detections may need teacher verification."
                    )
                else:
                    result["issues"].append(
                        f"⚠️ High facial similarity ({sim_pct:.1f}%) detected with {twin_name}. "
                        f"System will use enhanced matching for this pair."
                    )
        else:
            result["issues"].append("Database save failed.")

        return result

    def recapture_for_twin(self, student_id, frames):
        """
        Capture extra images for a student flagged as a twin.
        These additional embeddings improve twin distinction accuracy.

        Returns:
            Dict with success, added_count, issues.
        """
        result = {"success": False, "added_count": 0, "issues": []}

        student = self.database.get_student(student_id)
        if student is None:
            result["issues"].append(f"Student '{student_id}' not found.")
            return result

        new_embeddings = []
        new_images = []

        for i, frame in enumerate(frames):
            detection = self.detector.detect_single_face(frame)
            if detection is None:
                result["issues"].append(f"Frame {i+1}: No face detected.")
                continue

            face_crop = detection["cropped_face"]
            quality = self.preprocessor.assess_quality(face_crop)
            if quality["score"] < 0.3:
                result["issues"].append(f"Frame {i+1}: Poor quality.")
                continue

            embedding = self.embedder.get_embedding(face_crop)
            if embedding is None:
                result["issues"].append(f"Frame {i+1}: Embedding failed.")
                continue

            new_embeddings.append(embedding)
            new_images.append(face_crop)

        if new_embeddings:
            # Append to existing
            existing_embs = student.get("embeddings", [])
            existing_imgs = student.get("images", [])
            all_embs = existing_embs + new_embeddings
            all_imgs = existing_imgs + new_images
            student["embeddings"] = all_embs
            student["images"] = all_imgs
            self.database.save()
            result["success"] = True
            result["added_count"] = len(new_embeddings)
            print(
                f"[Registration] Added {len(new_embeddings)} extra embeddings "
                f"for twin student '{student_id}'. Total: {len(all_embs)}"
            )
        else:
            result["issues"].append("No valid embeddings from extra captures.")

        return result

    def verify_student(self, student_id):
        student = self.database.get_student(student_id)
        if student is None:
            return {
                "exists": False, "name": None,
                "embedding_count": 0, "image_count": 0,
                "registered_at": None,
            }
        return {
            "exists": True,
            "name": student["name"],
            "embedding_count": len(student.get("embeddings", [])),
            "image_count": len(student.get("images", [])),
            "registered_at": student.get("registered_at"),
        }

    def delete_student(self, student_id):
        # Also clean up twin pairs
        if self.twin_handler is not None:
            self.twin_handler.remove_pair_for_student(student_id)
        return self.database.remove_student(student_id)

    def get_all_registered(self):
        students = self.database.get_all_students()
        summaries = []
        for sid, data in students.items():
            summaries.append({
                "student_id": sid,
                "name": data["name"],
                "embedding_count": len(data.get("embeddings", [])),
                "registered_at": data.get("registered_at", "N/A"),
            })
        return summaries
