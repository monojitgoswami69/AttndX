"""
gen2 Streamlit application — entry point.

Run with:
    streamlit run gen2/app/streamlit_app.py

Pages:
  - Register: enroll new identities
  - Attendance: live attendance session
  - Gallery: view/delete enrolled identities
  - Reports: past session results
"""
import sys
from pathlib import Path

# Ensure gen2 parent is on sys.path
_GEN2_PARENT = str(Path(__file__).resolve().parent.parent.parent)
if _GEN2_PARENT not in sys.path:
    sys.path.insert(0, _GEN2_PARENT)

import streamlit as st

from gen2.config import Config

st.set_page_config(
    page_title="gen2 — Face Attendance",
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
        with st.spinner("Loading AI models..."):
            from gen2.app.runtime import AppRuntime
            st.session_state.runtime = AppRuntime()
    return st.session_state.runtime


# ── Sidebar ──
with st.sidebar:
    st.markdown("### 🎓 gen2 Attendance")
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

        if rt.attendance_engine_ref().session_active if hasattr(rt, 'attendance_engine_ref') else False:
            st.warning("🔴 Session active")
    except Exception as e:
        st.error(f"Init failed: {e}")
        st.stop()

# ── Route to pages ──
if page == "Register":
    from gen2.ui.enrollment_page import render_enrollment_page
    render_enrollment_page(rt)

elif page == "Attendance":
    from gen2.ui.attendance_page import render_attendance_page
    render_attendance_page(rt)

elif page == "Subjects":
    from gen2.ui.subjects_page import render_subjects_page
    render_subjects_page(rt)

elif page == "Gallery":
    from gen2.ui.gallery_page import render_gallery_page
    render_gallery_page(rt)

elif page == "Reports":
    from gen2.ui.reports_page import render_reports_page
    render_reports_page(rt)
