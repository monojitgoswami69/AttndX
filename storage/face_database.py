"""
Pickle-based face database for storing registered student faces and embeddings.
Provides CRUD operations with automatic persistence.
"""

import pickle
import os
import datetime
import numpy as np
import cv2
from pathlib import Path
from core.config import Config


class FaceDatabase:
    """Persistent face database using pickle serialization."""

    def __init__(self, db_path: str | Path | None = None, faces_dir: str | Path | None = None):
        """
        Initialize the face database.

        Args:
            db_path: Path to the pickle file. Defaults to Config.FACE_DB_PATH.
            faces_dir: Directory for saved face images. Defaults to Config.REGISTERED_FACES_DIR.
        """
        self.db_path = Path(db_path) if db_path else Config.FACE_DB_PATH
        self.faces_dir = Path(faces_dir) if faces_dir else Config.REGISTERED_FACES_DIR

        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.faces_dir.mkdir(parents=True, exist_ok=True)

        # Internal data structure
        self.data: dict = {"students": {}}

        # Auto-load existing database
        self.load()

    def load(self) -> bool:
        """
        Load the database from the pickle file.

        Returns:
            True if loaded successfully, False if no file exists or load failed.
        """
        if self.db_path.exists():
            try:
                with open(self.db_path, "rb") as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, dict) and "students" in loaded:
                    self.data = loaded
                    print(f"[FaceDB] Loaded {len(self.data['students'])} students from {self.db_path}")
                    return True
                else:
                    print("[FaceDB] Invalid database format, starting fresh.")
                    self.data = {"students": {}}
            except Exception as e:
                print(f"[FaceDB] Error loading database: {e}")
                self.data = {"students": {}}
        else:
            print("[FaceDB] No existing database found, starting fresh.")
        return False

    def save(self) -> bool:
        """
        Save the database to the pickle file.

        Returns:
            True if saved successfully.
        """
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.db_path, "wb") as f:
                pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[FaceDB] Saved {len(self.data['students'])} students to {self.db_path}")
            return True
        except Exception as e:
            print(f"[FaceDB] Error saving database: {e}")
            return False

    def add_student(
        self,
        student_id: str,
        name: str,
        embeddings: list[np.ndarray],
        face_images: list[np.ndarray] | None = None,
    ) -> bool:
        """
        Add a student to the database (or update if already exists).

        Args:
            student_id: Unique student identifier.
            name: Student display name.
            embeddings: List of 512-d face embeddings.
            face_images: Optional list of face crop images to save to disk.

        Returns:
            True if added/updated successfully.
        """
        if not embeddings:
            print(f"[FaceDB] Cannot add student '{name}' — no embeddings provided.")
            return False

        # Save face images to disk
        saved_image_paths = []
        if face_images:
            student_dir = self.faces_dir / student_id
            student_dir.mkdir(parents=True, exist_ok=True)

            for i, img in enumerate(face_images):
                if img is not None and img.size > 0:
                    img_path = student_dir / f"face_{i:03d}.jpg"
                    cv2.imwrite(str(img_path), img)
                    saved_image_paths.append(str(img_path))

        # Store in data dict
        self.data["students"][student_id] = {
            "name": name,
            "embeddings": embeddings,
            "images": saved_image_paths,
            "registered_at": datetime.datetime.now().isoformat(),
        }

        # Auto-save
        self.save()
        print(f"[FaceDB] Added student '{name}' (ID: {student_id}) with {len(embeddings)} embeddings.")
        return True

    def remove_student(self, student_id: str) -> bool:
        """
        Remove a student from the database.

        Args:
            student_id: Student ID to remove.

        Returns:
            True if removed, False if not found.
        """
        if student_id not in self.data["students"]:
            print(f"[FaceDB] Student '{student_id}' not found.")
            return False

        student = self.data["students"].pop(student_id)

        # Remove saved face images
        student_dir = self.faces_dir / student_id
        if student_dir.exists():
            import shutil
            shutil.rmtree(student_dir, ignore_errors=True)

        self.save()
        print(f"[FaceDB] Removed student '{student['name']}' (ID: {student_id}).")
        return True

    def get_student(self, student_id: str) -> dict | None:
        """
        Get a single student record.

        Args:
            student_id: Student ID to look up.

        Returns:
            Student dict {name, embeddings, images, registered_at} or None.
        """
        return self.data["students"].get(student_id)

    def get_all_students(self) -> dict[str, dict]:
        """
        Get all student records.

        Returns:
            Dict mapping student_id → student data.
        """
        return self.data["students"]

    def get_all_embeddings(self) -> dict[str, dict]:
        """
        Get all student embeddings in a format suitable for FaceMatcher.

        Returns:
            Dict: {student_id: {"name": str, "embeddings": list[np.ndarray]}}
        """
        result = {}
        for sid, student in self.data["students"].items():
            result[sid] = {
                "name": student["name"],
                "embeddings": student["embeddings"],
            }
        return result

    def get_student_count(self) -> int:
        """Return the number of registered students."""
        return len(self.data["students"])

    def student_exists(self, student_id: str) -> bool:
        """Check if a student ID is already registered."""
        return student_id in self.data["students"]

    def get_student_face_images(self, student_id: str) -> list[np.ndarray]:
        """
        Load saved face images from disk for a student.

        Args:
            student_id: Student ID.

        Returns:
            List of face images as numpy arrays.
        """
        student = self.get_student(student_id)
        if not student:
            return []

        images = []
        for path in student.get("images", []):
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    images.append(img)
        return images
