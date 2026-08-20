"""
Application runtime — loads all models and services once at startup.

All models are loaded eagerly and reused for the entire application lifecycle.
No per-frame model loading. No per-request initialization.
"""
import logging
import sys
from pathlib import Path

# Add the parent of gen2/ to sys.path so `gen2` is importable
_GEN2_PARENT = str(Path(__file__).resolve().parent.parent.parent)
if _GEN2_PARENT not in sys.path:
    sys.path.insert(0, _GEN2_PARENT)

from gen2.config import Config
from gen2.recognition.embeddings.arcface_onnx import ArcFaceEmbedder
from gen2.recognition.liveness.minifasnet import MiniFASNetLiveness
from gen2.recognition.matching.engine import IdentityIndex, RecognitionEngine
from gen2.recognition.engine import RecognitionPipeline
from gen2.recognition.tracking.iou_tracker import IoUTracker
from gen2.storage.attendance_db import AttendanceDB
from gen2.storage.db import BiometricDB
from gen2.vision.alignment.arcface import ArcFaceAligner
from gen2.vision.camera.source import CameraSource, ExternalFrameBuffer
from gen2.vision.detection.yunet import YuNetDetector
from gen2.vision.quality.assessor import FaceQualityAssessor

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

        # ── Load models (eagerly, once) ──
        self.detector = YuNetDetector()
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

    @classmethod
    def reset(cls):
        """Reset the singleton (for testing)."""
        cls._instance = None
