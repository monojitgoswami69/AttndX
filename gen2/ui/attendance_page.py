"""Attendance page for gen2 — live session with WebRTC."""
import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd

from gen2.config import Config
from gen2.attendance.engine import AttendanceEngine
from gen2.recognition.matching.engine import RecognitionState


def render_attendance_page(rt):
    st.markdown("## 📋 Attendance Session")

    if rt.identity_index.size == 0:
        st.warning("No identities enrolled. Register students first.")
        return

    st.success(f"✅ {rt.identity_index.size} identities enrolled.")

    # Build attendance engine if not exists
    if "att_engine" not in st.session_state:
        st.session_state.att_engine = AttendanceEngine(
            pipeline=rt.pipeline,
            camera=rt.camera,
            external_buffer=rt.external_buffer,
            biometric_db=rt.biometric_db,
            attendance_db=rt.attendance_db,
        )
    engine = st.session_state.att_engine

    if not engine.session_active:
        # Pre-session controls
        classes = Config.get("attendance")  # not used, just show mode
        demo = Config.get("attendance", "demo_mode")
        times = Config.get("attendance", "check_times_demo" if demo else "check_times_normal")
        unit = "sec" if demo else "min"
        st.info(f"{'⚡ DEMO' if demo else '🕐 NORMAL'} mode — "
                f"checks at {times} {unit}")

        if st.button("▶️ Start Session", type="primary", use_container_width=True):
            sid = engine.start_session("Session")
            if sid:
                st.rerun()
            else:
                st.error("Failed to start. Check camera and enrolled identities.")
        return

    # Active session
    st.markdown("### 🔴 Session In Progress")
    if st.button("⏹️ Stop", type="secondary"):
        engine.stop_session()
        st.rerun()

    # Status bar
    status = engine.get_status()
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Elapsed", f"{int(status.elapsed_seconds//60):02d}:{int(status.elapsed_seconds%60):02d}")
    sc2.metric("Checks", f"{status.checks_completed}/{status.total_checks}")
    sc3.metric("Next In", f"{status.next_check_in:.0f}s")

    # Live frame result display
    latest = engine.get_latest_frame_result()
    if latest:
        cols = st.columns(4)
        cols[0].metric("Detected", latest.num_detected)
        cols[1].metric("Recognized", latest.num_recognized)
        cols[2].metric("Unknown", latest.num_unknown)
        cols[3].metric("Ambiguous", latest.num_ambiguous)

        if latest.faces:
            for face in latest.faces:
                if face.recognition:
                    if face.recognition.state == RecognitionState.KNOWN:
                        st.success(f"✅ Track {face.track_id}: {face.recognition.name} "
                                  f"({face.recognition.confidence:.1%})")
                    elif face.recognition.state == RecognitionState.UNKNOWN:
                        st.info(f"❓ Track {face.track_id}: Unknown")
                    elif face.recognition.state == RecognitionState.AMBIGUOUS:
                        st.warning(f"⚠️ Track {face.track_id}: Ambiguous")
                    elif face.recognition.state == RecognitionState.REJECTED:
                        st.error(f"🔴 Track {face.track_id}: Rejected ({face.recognition.error})")

    time.sleep(0.5)

    # Final results
    if not status.active and status.checks_completed > 0:
        st.markdown("---")
        st.markdown("## 📊 Final Results")
        attendance = rt.attendance_db.get_attendance(engine.session_id)
        if attendance:
            rows = []
            for a in attendance:
                rows.append({
                    "Student": a["name"],
                    "Status": a["status"].upper(),
                    "Present": f"{a['checks_present']}",
                    "Spoofed": a["checks_spoofed"],
                    "Note": a["note"],
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("📥 CSV", df.to_csv(index=False),
                              file_name=f"attendance_{engine.session_id}.csv")
