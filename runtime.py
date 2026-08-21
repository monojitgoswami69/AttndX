"""
Application runtime — loads all models and services once at startup.

All models are loaded eagerly and reused for the entire application lifecycle.
No per-frame model loading. No per-request initialization.
"""
import logging
import sys
from pathlib import Path

# Ensure repo root is on sys.path
_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config import Config
from recognition.embeddings.arcface_onnx import ArcFaceEmbedder
from recognition.liveness.minifasnet import MiniFASNetLiveness
from recognition.matching.engine import IdentityIndex, RecognitionEngine
from recognition.engine import RecognitionPipeline
from recognition.tracking.iou_tracker import IoUTracker
from storage.attendance_db import AttendanceDB
from storage.db import BiometricDB
from vision.alignment.arcface import ArcFaceAligner
from vision.camera.source import CameraSource, ExternalFrameBuffer
from vision.quality.assessor import FaceQualityAssessor

logger = logging.getLogger(__name__)


class AppRuntime:
    """Application runtime that owns all shared resources.

    All components are loaded once and reused. In Streamlit, this is cached
    in st.session_state. The class-level _instance is a convenience for
    non-Streamlit use (scripts, tests) and does NOT enforce a singleton —
    Streamlit's session_state handles that.
    """

    _instance: "AppRuntime | None" = None

    def __init__(self):
        AppRuntime._instance = self

        Config.load()
        Config.ensure_directories()
        self._setup_logging()

        logger.info("Initializing gen2 runtime...")

        # ── Auto-ensure models are present ──
        try:
            from scripts.download_models import check_models_present, download_all_models
            models_ok, missing = check_models_present()
            if not models_ok:
                logger.info(f"Missing models detected: {missing}. Automatically downloading...")
                download_all_models(include_optional=False)
        except Exception as e:
            logger.warning(f"Auto-download check encountered an error: {e}")

        # ── Load models (eagerly, once) ──
        self.detector = self._create_detector()
        self.aligner = ArcFaceAligner()
        self.quality_assessor = FaceQualityAssessor()
        self.embedder = ArcFaceEmbedder()
        self.liveness = MiniFASNetLiveness()

        # ── Storage (SQLite, atomic, recoverable) ──
        self.biometric_db = BiometricDB.safe_open()
        self.attendance_db = AttendanceDB()

        # ── Recognition index (rebuilt from DB) ──
        self.identity_index = IdentityIndex()
        self._rebuild_index()

        # ── Recognition engine ──
        self.recognition_engine = RecognitionEngine(self.identity_index)

        # ── Tracker (one per session) ──
        self.tracker = IoUTracker()

        # ── Pipeline ──
        self.pipeline = RecognitionPipeline(
            detector=self.detector,
            aligner=self.aligner,
            quality_assessor=self.quality_assessor,
            embedder=self.embedder,
            engine=self.recognition_engine,
            tracker=self.tracker,
            liveness=self.liveness,
        )

        # ── Camera ──
        self.camera = CameraSource()
        self.external_buffer = ExternalFrameBuffer()

        logger.info("gen2 runtime initialized.")

    @staticmethod
    def _create_detector():
        """Create the face detector based on config backend setting."""
        backend = Config.get("detector", "backend", default="scrfd")
        if backend == "scrfd":
            from vision.detection.scrfd import SCRFDDetector
            return SCRFDDetector()
        else:
            from vision.detection.yunet import YuNetDetector
            return YuNetDetector()

    def _rebuild_index(self):
        """Rebuild the identity index from authoritative DB storage."""
        templates = self.biometric_db.get_all_templates()
        identities = self.biometric_db.get_all_identities()
        names = {i["identity_id"]: i["name"] for i in identities}
        self.identity_index.rebuild(templates, names)
        logger.info(f"Index rebuilt: {len(templates)} identities")

    def _setup_logging(self):
        level_str = Config.get("logging", "level", default="INFO")
        level = getattr(logging, level_str.upper(), logging.INFO)
        fmt = Config.get("logging", "format",
                         default="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        logging.basicConfig(level=level, format=fmt, force=True)

    @classmethod
    def get(cls) -> "AppRuntime":
        if cls._instance is None:
            cls()
        return cls._instance

    def shutdown(self):
        """Release all hardware resources (camera, etc.)."""
        if hasattr(self, 'camera') and self.camera.is_opened():
            self.camera.release()
            logger.info("Runtime shutdown: camera released")

    @classmethod
    def reset(cls):
        """Reset the singleton (for testing)."""
        if cls._instance is not None:
            cls._instance.shutdown()
        cls._instance = None
