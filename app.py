"""
AI Smart Attendance System — Streamlit Application.

Run with:
    streamlit run app.py
    # or: streamlit run app/streamlit_app.py

Pages:
  - Register: enroll new identities
  - Attendance: live attendance session
  - Subjects: manage classes/subjects
  - Gallery: view/delete enrolled identities
  - Reports: past session results
"""
import sys
from pathlib import Path

# Ensure repo root is on sys.path
_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st

from config import Config

st.set_page_config(
    page_title="AI Smart Attendance",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme CSS
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border: none; border-radius: 10px; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def get_runtime():
    """Lazy-initialize the AppRuntime singleton, stored in session_state."""
    if "runtime" not in st.session_state:
        from scripts.download_models import check_models_present
        models_ok, _ = check_models_present()
        spinner_msg = "Downloading missing AI models (first-time setup)..." if not models_ok else "Loading AI models..."
        with st.spinner(spinner_msg):
            from runtime import AppRuntime
            st.session_state.runtime = AppRuntime()
    return st.session_state.runtime


# ── Sidebar ──
with st.sidebar:
    st.markdown("### 🎓 AI Smart Attendance")
    st.caption("YuNet + ArcFace + MiniFASNet")
    st.markdown("---")
    page = st.radio("Navigation", ["Register", "Attendance", "Subjects", "Gallery", "Reports"],
                    index=0, label_visibility="collapsed")
    st.markdown("---")

    try:
        rt = get_runtime()
        st.success("🟢 Models loaded")
        count = rt.biometric_db.count_identities()
        st.metric("Enrolled", count)
        st.caption(f"Index: {rt.identity_index.size} identities")

        att_eng = st.session_state.get("att_engine")
        if att_eng is not None and getattr(att_eng, "session_active", False):
            st.warning("🔴 Session active")
    except Exception as e:
        st.error(f"Init failed: {e}")
        st.stop()

# ── Route to pages ──
if page == "Register":
    from ui.enrollment_page import render_enrollment_page
    render_enrollment_page(rt)

elif page == "Attendance":
    from ui.attendance_page import render_attendance_page
    render_attendance_page(rt)

elif page == "Subjects":
    from ui.subjects_page import render_subjects_page
    render_subjects_page(rt)

elif page == "Gallery":
    from ui.gallery_page import render_gallery_page
    render_gallery_page(rt)

elif page == "Reports":
    from ui.reports_page import render_reports_page
    render_reports_page(rt)
