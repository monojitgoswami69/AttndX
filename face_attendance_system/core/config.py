"""
Configuration settings for the Face Attendance System.
All tunable parameters, paths, and mode settings are centralized here.
"""

import os
from pathlib import Path


class Config:
    """Central configuration for the face attendance system."""

    # ──────────────────────────────────────────────
    # Project root (face_attendance_system/)
    # ──────────────────────────────────────────────
    BASE_DIR = Path(__file__).resolve().parent.parent

    # ──────────────────────────────────────────────
    # YOLOv8 Face Detection
    # ──────────────────────────────────────────────
    YOLO_MODEL_NAME: str = "yolov8n.pt"
    YOLO_CONFIDENCE: float = 0.5
    FACE_PADDING: int = 20  # pixels added around detected face bbox

    # ──────────────────────────────────────────────
    # InsightFace Embedding
    # ──────────────────────────────────────────────
    EMBEDDING_DIM: int = 512
    INSIGHTFACE_MODEL: str = "buffalo_l"

    # ──────────────────────────────────────────────
    # Face Matching
    # ──────────────────────────────────────────────
    SIMILARITY_THRESHOLD: float = 0.6  # cosine similarity cutoff

    # ──────────────────────────────────────────────
    # Registration
    # ──────────────────────────────────────────────
    IMAGES_PER_REGISTRATION: int = 5
    CAPTURE_DELAY: float = 1.5  # seconds between captures

    # ──────────────────────────────────────────────
    # Attendance Session
    # ──────────────────────────────────────────────
    CHECKS_PER_SESSION: int = 3
    MIN_CHECKS_FOR_PRESENT: int = 2  # must be present in at least N checks
    FRAMES_PER_CHECK: int = 5  # frames sampled per check window

    # ──────────────────────────────────────────────
    # Demo vs Normal Mode
    # ──────────────────────────────────────────────
    DEMO_MODE: bool = True

    # Demo mode: times in seconds from session start
    CHECK_TIMES_DEMO: list[int] = [20, 40, 60]
    FRAME_INTERVAL_DEMO: int = 2  # seconds between frame grabs in demo

    # Normal mode: times in minutes from session start
    CHECK_TIMES_NORMAL: list[int] = [15, 30, 45]
    FRAME_INTERVAL_NORMAL: int = 6  # seconds between frame grabs in normal

    # ──────────────────────────────────────────────
    # Storage Paths
    # ──────────────────────────────────────────────
    STORAGE_DIR: Path = BASE_DIR / "storage"
    FACE_DB_PATH: Path = STORAGE_DIR / "face_database.pkl"
    ATTENDANCE_DB_PATH: Path = STORAGE_DIR / "attendance_records.pkl"
    REGISTERED_FACES_DIR: Path = STORAGE_DIR / "registered_faces"
    SESSION_SNAPSHOTS_DIR: Path = STORAGE_DIR / "session_snapshots"

    # ──────────────────────────────────────────────
    # Camera
    # ──────────────────────────────────────────────
    CAMERA_INDEX: int = 0

    # ──────────────────────────────────────────────
    # Image Preprocessing
    # ──────────────────────────────────────────────
    FACE_INPUT_SIZE: tuple[int, int] = (112, 112)
    MIN_FACE_SIZE: int = 40  # minimum face width/height in pixels
    BLUR_THRESHOLD: float = 50.0  # Laplacian variance threshold
    BRIGHTNESS_LOW: float = 40.0
    BRIGHTNESS_HIGH: float = 220.0

    # ──────────────────────────────────────────────
    # Light Monitoring / Darkness Handling
    # ──────────────────────────────────────────────
    BRIGHTNESS_MIN: float = 40.0         # Below = too dark
    BRIGHTNESS_MAX: float = 240.0        # Above = too bright
    BRIGHTNESS_RECOVERABLE: float = 25.0 # Below = can't enhance
    LOW_LIGHT_ENHANCE: bool = True       # Try to enhance low-light frames
    DARK_RETRY_DELAY: int = 30           # Seconds to wait before retrying
    DARK_RETRY_DELAY_DEMO: int = 8       # Shorter retry in demo mode
    MAX_RETRIES_PER_CHECK: int = 3       # Max retry attempts per check
    BRIGHTNESS_MONITOR_INTERVAL: int = 10 # Seconds between brightness polls

    # ──────────────────────────────────────────────
    # Twin Detection & Handling
    # ──────────────────────────────────────────────
    TWIN_SIMILARITY_THRESHOLD: float = 0.80   # Above = potential twin
    HIGH_RISK_TWIN_THRESHOLD: float = 0.90    # Above = high-risk twin
    MIN_TWIN_DIFFERENCE: float = 0.03         # Min diff to distinguish twins
    TWIN_EXTRA_IMAGES: int = 10               # Extra captures for twin students
    TWIN_PAIRS_DB_PATH: Path = STORAGE_DIR / "twin_pairs.pkl"

    # ──────────────────────────────────────────────
    # Anti-Spoofing / Liveness Detection
    # ──────────────────────────────────────────────
    LIVENESS_ENABLED: bool = True
    LIVENESS_TEXTURE_THRESHOLD: float = 15.0     # LBP variance threshold
    LIVENESS_BLINK_EAR_THRESHOLD: float = 0.20   # EAR below = blink
    LIVENESS_MIN_BLINKS: int = 1                 # Min blinks to confirm live
    LIVENESS_FRAMES_TO_CAPTURE: int = 45         # ~1.5s at 30fps for blink
    LIVENESS_FREQ_THRESHOLD: float = 0.45        # FFT frequency score

    @classmethod
    def get_dark_retry_delay(cls) -> int:
        return cls.DARK_RETRY_DELAY_DEMO if cls.DEMO_MODE else cls.DARK_RETRY_DELAY


    @classmethod
    def get_check_times(cls) -> list[int]:
        """Return check times based on current mode."""
        return cls.CHECK_TIMES_DEMO if cls.DEMO_MODE else cls.CHECK_TIMES_NORMAL

    @classmethod
    def get_frame_interval(cls) -> int:
        """Return frame interval based on current mode."""
        return cls.FRAME_INTERVAL_DEMO if cls.DEMO_MODE else cls.FRAME_INTERVAL_NORMAL

    @classmethod
    def ensure_directories(cls) -> None:
        """Create all required storage directories if they don't exist."""
        cls.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        cls.REGISTERED_FACES_DIR.mkdir(parents=True, exist_ok=True)
        cls.SESSION_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)


# Auto-create directories on import
Config.ensure_directories()
