"""
Face Registration REST API.
FastAPI APIRouter for registering, verifying, and deleting student faces.
Teammates import this router into their FastAPI backend:

    from integration.face_api import create_face_router
    app.include_router(create_face_router(detector, embedder, face_db))
"""

import base64
import numpy as np
import cv2
from fastapi import APIRouter, HTTPException

from integration.schemas import (
    FaceRegisterRequest,
    FaceRegisterResponse,
    StudentInfo,
    StudentSummary,
    DeleteResponse,
)


def create_face_router(face_detector, face_embedder, face_database) -> APIRouter:
    """
    Factory that creates a face registration APIRouter with injected dependencies.

    Args:
        face_detector: YOLOFaceDetector instance.
        face_embedder: FaceEmbedder instance.
        face_database: FaceDatabase instance.

    Returns:
        Configured APIRouter.
    """
    from services.registration_service import RegistrationService

    reg_service = RegistrationService(face_detector, face_embedder, face_database)
    router = APIRouter(prefix="/api/face", tags=["Face Registration"])

    # ── POST /register ──────────────────────────

    @router.post("/register", response_model=FaceRegisterResponse)
    async def register_student(req: FaceRegisterRequest):
        """
        Register a new student.

        Accepts a list of base64-encoded face images, decodes them,
        runs the full detection → quality → embedding pipeline,
        and stores valid embeddings in the face database.
        """
        # Decode base64 images to OpenCV frames
        frames = []
        for i, b64_str in enumerate(req.images):
            try:
                # Handle data URI prefix if present
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]

                img_bytes = base64.b64decode(b64_str)
                np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None or frame.size == 0:
                    raise ValueError("Decoded image is empty")

                frames.append(frame)

            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to decode image {i + 1}: {str(e)}",
                )

        if not frames:
            raise HTTPException(status_code=400, detail="No valid images provided.")

        # Run registration pipeline
        result = reg_service.register_student_from_frames(
            student_id=req.student_id,
            name=req.name,
            frames=frames,
        )

        return FaceRegisterResponse(
            success=result["success"],
            student_id=req.student_id,
            name=req.name,
            registered_count=result["registered_count"],
            total_images=result["total_frames"],
            quality_scores=result["quality_scores"],
            issues=result["issues"],
        )

    # ── GET /verify/{student_id} ─────────────────

    @router.get("/verify/{student_id}", response_model=StudentInfo)
    async def verify_student(student_id: str):
        """Check if a student is registered and return their info."""
        info = reg_service.verify_student(student_id)
        return StudentInfo(
            exists=info["exists"],
            student_id=student_id,
            name=info["name"],
            embedding_count=info["embedding_count"],
            image_count=info["image_count"],
            registered_at=info["registered_at"],
        )

    # ── DELETE /{student_id} ─────────────────────

    @router.delete("/{student_id}", response_model=DeleteResponse)
    async def delete_student(student_id: str):
        """Remove a student from the face database."""
        student = face_database.get_student(student_id)
        if student is None:
            raise HTTPException(
                status_code=404,
                detail=f"Student '{student_id}' not found.",
            )

        success = reg_service.delete_student(student_id)
        return DeleteResponse(
            success=success,
            message=(
                f"Student '{student_id}' deleted successfully."
                if success
                else f"Failed to delete student '{student_id}'."
            ),
        )

    # ── GET /students ────────────────────────────

    @router.get("/students", response_model=list[StudentSummary])
    async def list_students():
        """List all registered students."""
        registered = reg_service.get_all_registered()
        return [
            StudentSummary(
                student_id=s["student_id"],
                name=s["name"],
                embedding_count=s["embedding_count"],
                registered_at=s["registered_at"],
            )
            for s in registered
        ]

    return router
