"""
Pydantic schemas for the Face Attendance REST API.
Used by the FastAPI integration layer for request/response validation.
"""

from pydantic import BaseModel, Field
from enum import Enum


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class AttendanceStatus(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"


class SessionStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    STOPPED = "stopped"


class CheckStatus(str, Enum):
    COMPLETED = "completed"
    NEXT = "next"
    PENDING = "pending"


# ──────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────

class FaceRegisterRequest(BaseModel):
    """Request body for registering a new student."""
    student_id: str = Field(..., min_length=1, description="Unique student identifier")
    name: str = Field(..., min_length=1, description="Student display name")
    images: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="List of base64-encoded JPEG/PNG face images",
    )


class FaceRegisterResponse(BaseModel):
    """Response after a registration attempt."""
    success: bool = Field(..., description="Whether registration succeeded")
    student_id: str = Field(..., description="Student ID that was registered")
    name: str = Field(..., description="Student name")
    registered_count: int = Field(0, description="Number of valid embeddings stored")
    total_images: int = Field(0, description="Total images submitted")
    quality_scores: list[float] = Field(
        default_factory=list,
        description="Quality score (0-1) for each submitted image",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Human-readable issues encountered during registration",
    )


class StudentInfo(BaseModel):
    """Info about a registered student."""
    exists: bool
    student_id: str
    name: str | None = None
    embedding_count: int = 0
    image_count: int = 0
    registered_at: str | None = None


class StudentSummary(BaseModel):
    """Compact student summary for listings."""
    student_id: str
    name: str
    embedding_count: int
    registered_at: str | None = None


class DeleteResponse(BaseModel):
    """Response after a deletion attempt."""
    success: bool
    message: str


# ──────────────────────────────────────────────
# Attendance Session
# ──────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    """Request body for starting an attendance session."""
    class_name: str = Field(..., min_length=1, description="Name of the class or lecture")
    camera_source: int = Field(0, description="Camera device index (0 = default webcam)")


class StartSessionResponse(BaseModel):
    """Response after starting a session."""
    success: bool
    session_id: str | None = None
    message: str
    check_times: list[int] = Field(default_factory=list)
    mode: str = Field("demo", description="'demo' or 'normal'")


class CheckInfo(BaseModel):
    """Status of an individual attendance check."""
    check_number: int
    time_value: int
    unit: str
    status: CheckStatus
    detected_count: int | None = None


class SessionStatusResponse(BaseModel):
    """Current session status snapshot."""
    active: bool
    session_id: str | None = None
    class_name: str = ""
    status: str = "No active session"
    checks_completed: int = 0
    total_checks: int = 0
    elapsed_seconds: float = 0
    next_check_in: float = 0
    check_running: bool = False
    checks: list[CheckInfo] = Field(default_factory=list)


class StopSessionResponse(BaseModel):
    """Response after stopping a session."""
    success: bool
    message: str
    session_id: str | None = None
    results: list["AttendanceResult"] = Field(default_factory=list)


class AttendanceResult(BaseModel):
    """Attendance result for a single student."""
    student_id: str
    name: str
    check1: bool = False
    check2: bool = False
    check3: bool = False
    check1_spoofed: bool = False
    check2_spoofed: bool = False
    check3_spoofed: bool = False
    checks_present: int = 0
    checks_spoofed: int = 0
    total_checks: int = 0
    status: AttendanceStatus


class SessionResultsResponse(BaseModel):
    """Full results for a completed session."""
    session_id: str
    class_name: str
    date: str
    session_status: str
    total_students: int
    present_count: int
    absent_count: int
    spoofed_count: int = 0
    results: list[AttendanceResult]


# Rebuild forward refs for self-referencing models
StopSessionResponse.model_rebuild()
