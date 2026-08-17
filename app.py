"""
AI Smart Attendance System — Main Streamlit Application.
Entry point: streamlit run app.py
"""

import streamlit as st
import sys
import os

# Ensure the project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import Config


# ──────────────────────────────────────────────
# Page Config (must be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Smart Attendance",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark theme refinements */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    .stSidebar > div:first-child {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f1a 100%);
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 12px 16px;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }

    /* Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    .stButton > button[kind="secondary"] {
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.2);
    }

    /* Cards in containers */
    [data-testid="stVerticalBlock"] > div:has(> [data-testid="stImage"]) {
        border-radius: 12px;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 1rem;
        font-weight: 600;
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Lazy-Init Shared Components (cached in session_state)
# ──────────────────────────────────────────────

def _create_face_detector():
    """Pick the face detector backend based on Config, with fallbacks."""
    backend = Config.FACE_DETECTOR_BACKEND.lower()
    # Try preferred backend first
    if backend == "yunet":
        try:
            from core.face_detector_yunet import YuNetFaceDetector
            print("[App] Using YuNet face detector backend.")
            return YuNetFaceDetector()
        except Exception as e:
            print(f"[App] YuNet backend failed ({e}); falling back to YOLO.")
    elif backend == "mediapipe":
        try:
            from core.face_detector_mediapipe import MediaPipeFaceDetector  # noqa: F401
            print("[App] MediaPipe backend not yet implemented; falling back.")
        except Exception as e:
            print(f"[App] MediaPipe backend unavailable ({e}); falling back.")
    # Fallback: YOLO + Haar cascade
    from core.face_detector import YOLOFaceDetector
    print("[App] Using YOLOv8 + Haar cascade face detector backend (fallback).")
    return YOLOFaceDetector()


def _init_components():
    """Initialize all ML models and services once, store in session_state."""

    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        st.session_state.init_error = None

    if st.session_state.initialized:
        return True

    try:
        with st.spinner("🔄 Loading AI models... This may take a moment on first run."):
            # Ensure storage directories
            Config.ensure_directories()

            # Core models
            from core.face_detector import YOLOFaceDetector
            from core.face_embedder import FaceEmbedder

            if "detector" not in st.session_state:
                st.session_state.detector = _create_face_detector()
            if "embedder" not in st.session_state:
                st.session_state.embedder = FaceEmbedder()

            # Storage
            from storage.face_database import FaceDatabase
            from storage.attendance_store import AttendanceStore

            if "face_db" not in st.session_state:
                st.session_state.face_db = FaceDatabase()
            if "attendance_store" not in st.session_state:
                st.session_state.attendance_store = AttendanceStore()

            # Services
            from services.camera_service import CameraService
            from services.registration_service import RegistrationService
            from services.attendance_service import AttendanceMonitor
            from core.twin_handler import TwinHandler

            if "twin_handler" not in st.session_state:
                st.session_state.twin_handler = TwinHandler()

            if "camera_service" not in st.session_state:
                st.session_state.camera_service = CameraService()

            if "reg_service" not in st.session_state:
                st.session_state.reg_service = RegistrationService(
                    face_detector=st.session_state.detector,
                    face_embedder=st.session_state.embedder,
                    face_database=st.session_state.face_db,
                    twin_handler=st.session_state.twin_handler,
                )

            if "monitor" not in st.session_state:
                st.session_state.monitor = AttendanceMonitor(
                    face_detector=st.session_state.detector,
                    face_embedder=st.session_state.embedder,
                    face_database=st.session_state.face_db,
                    attendance_store=st.session_state.attendance_store,
                    twin_handler=st.session_state.twin_handler,
                )

            st.session_state.initialized = True
            return True

    except Exception as e:
        st.session_state.init_error = str(e)
        return False


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<h1 style='text-align:center; font-size:1.6rem;'>"
        "🎓 AI Smart Attendance"
        "</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:#888; font-size:0.85rem; margin-top:-10px;'>"
        "YOLOv8 + InsightFace"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigation",
        options=["📸 Register", "📋 Attendance", "👥 Students", "📊 Reports"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # System status panel
    st.markdown("### ⚙️ System Status")

    if st.session_state.get("initialized"):
        st.success("🟢 Models Loaded")
        student_count = st.session_state.face_db.get_student_count()
        st.metric("👥 Registered Students", student_count)

        # Twin pairs info
        if st.session_state.get("twin_handler"):
            pairs = st.session_state.twin_handler.get_all_twin_pairs()
            if pairs:
                st.caption(f"👯 {len(pairs)} twin pair(s) registered")

        # Active session info
        if st.session_state.get("monitor") and st.session_state.monitor.is_session_active():
            status = st.session_state.monitor.get_session_status()
            st.warning(f"🔴 Session Active: {status['class_name']}")
            st.caption(
                f"Checks: {status['checks_completed']}/{status['total_checks']}"
            )
        else:
            st.info("⬜ No active session")

        # Demo mode indicator
        if Config.DEMO_MODE:
            st.caption("⚡ Demo Mode ON")
        else:
            st.caption("🕐 Normal Mode")
    else:
        st.warning("⏳ Models not loaded yet")

    st.markdown("---")
    st.caption("v1.0 — Smart Attendance System")


# ──────────────────────────────────────────────
# Main Content
# ──────────────────────────────────────────────

# Initialize components
init_ok = _init_components()

if not init_ok:
    st.error(
        f"❌ **Failed to initialize AI models.**\n\n"
        f"Error: {st.session_state.get('init_error', 'Unknown error')}\n\n"
        f"Please check:\n"
        f"1. All dependencies are installed (`pip install -r requirements.txt`)\n"
        f"2. Sufficient disk space for model downloads\n"
        f"3. Internet connection for first-time model download"
    )
    st.stop()

# Route to selected page
if page == "📸 Register":
    from ui.register_page import render_register_page
    render_register_page(
        detector=st.session_state.detector,
        embedder=st.session_state.embedder,
        reg_service=st.session_state.reg_service,
        camera_service=st.session_state.camera_service,
    )

elif page == "📋 Attendance":
    from ui.attendance_page import render_attendance_page
    render_attendance_page(
        monitor=st.session_state.monitor,
        face_db=st.session_state.face_db,
    )

elif page == "👥 Students":
    from ui.gallery_page import render_gallery_page
    render_gallery_page(
        face_db=st.session_state.face_db,
        reg_service=st.session_state.reg_service,
    )

elif page == "📊 Reports":
    from ui.report_page import render_report_page
    render_report_page(
        attendance_store=st.session_state.attendance_store,
        face_db=st.session_state.face_db,
    )
