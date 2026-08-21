"""
Attendance page for gen2 — Native high-performance OpenCV camera streaming.

Directly utilizes native OS camera drivers (DirectShow/MSMF on Windows,
AVFoundation on macOS, V4L2 on Linux) via CameraSource for zero-latency,
rock-solid stability without WebRTC/browser networking overhead.
"""
import logging
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from attendance.engine import AttendanceEngine
from config import Config
from recognition.liveness.minifasnet import LivenessState
from recognition.matching.engine import RecognitionState

logger = logging.getLogger(__name__)

# Explicit Color Codes for Face Recognition Windows (BGR format)
_COLOR_REAL_PERSON = (0, 220, 0)       # 🟢 GREEN: Real enrolled person
_COLOR_POSSIBLE_SPOOF = (0, 215, 255)  # 🟡 YELLOW: Possible spoof attack (screen / print)
_COLOR_UNKNOWN = (0, 0, 230)           # 🔴 RED: Unknown / unregistered person
_COLOR_AMBIGUOUS = (0, 140, 255)       # 🟠 ORANGE: Ambiguous match
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)


def _draw_dashed_rect(frame: np.ndarray, bbox: tuple[int, int, int, int], color: tuple[int, int, int],
                      thickness: int = 2, dash: int = 10, gap: int = 6):
    """Draw a dashed rectangle on frame."""
    x1, y1, x2, y2 = bbox
    for x in range(x1, x2, dash + gap):
        cv2.line(frame, (x, y1), (min(x + dash, x2), y1), color, thickness, cv2.LINE_AA)
    for x in range(x1, x2, dash + gap):
        cv2.line(frame, (x, y2), (min(x + dash, x2), y2), color, thickness, cv2.LINE_AA)
    for y in range(y1, y2, dash + gap):
        cv2.line(frame, (x1, y), (x1, min(y + dash, y2)), color, thickness, cv2.LINE_AA)
    for y in range(y1, y2, dash + gap):
        cv2.line(frame, (x2, y), (x2, min(y + dash, y2)), color, thickness, cv2.LINE_AA)


def _draw_label(frame: np.ndarray, bbox_tl: tuple[int, int], label: str,
                color: tuple[int, int, int], text_color: tuple[int, int, int] | None = None):
    """Draw semi-transparent label banner + text above bounding box."""
    if text_color is None:
        text_color = _WHITE
    x1, y1 = bbox_tl
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 1
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

    label_y = max(y1 - 8, th + 8)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, label_y - th - 6), (x1 + tw + 10, label_y + 4), color, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.putText(frame, label, (x1 + 5, label_y - 2), font, font_scale, text_color, thickness, cv2.LINE_AA)


def _draw_frame_overlays(frame: np.ndarray, frame_result) -> np.ndarray:
    """Draw bounding boxes and status labels:
    - 🟢 GREEN: Real Enrolled Person
    - 🟡 YELLOW: Possible Spoof Attack (Screen / Print)
    - 🔴 RED: Unknown / Impostor
    """
    if frame_result is None or not frame_result.faces:
        return frame

    for face in frame_result.faces:
        bbox = face.bbox
        if bbox is None or bbox == (0, 0, 0, 0):
            continue
        x1, y1, x2, y2 = bbox

        rec = face.recognition
        liveness = face.liveness

        # ─── 1. SPOOF ATTACK DETECTED -> YELLOW BOX ───
        if liveness and liveness.state == LivenessState.SPOOF:
            color = _COLOR_POSSIBLE_SPOOF
            spoof_desc = liveness.spoofing_type.upper() if liveness.spoofing_type else "SPOOF"
            label = f"SPOOF ({spoof_desc})"
            _draw_dashed_rect(frame, (x1, y1, x2, y2), color, thickness=3)
            _draw_label(frame, (x1, y1), label, color, text_color=_BLACK)
            continue

        # ─── 2. REAL ENROLLED PERSON -> GREEN BOX ───
        if rec is not None and rec.state == RecognitionState.KNOWN:
            color = _COLOR_REAL_PERSON
            name = rec.name or "Real Person"
            live_tag = " [LIVE]" if (liveness and liveness.state == LivenessState.LIVE) else ""
            label = f"{name} ({rec.confidence:.0%}){live_tag}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            _draw_label(frame, (x1, y1), label, color, text_color=_WHITE)

            # Track ID badge (top right)
            if face.track_id is not None:
                tid_text = f"T{face.track_id}"
                cv2.putText(frame, tid_text, (x2 - 38, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

            # Confirmed badge (bottom right)
            if face.confirmed:
                cv2.circle(frame, (x2 - 12, y2 - 12), 8, _COLOR_REAL_PERSON, -1, cv2.LINE_AA)
                cv2.putText(frame, "C", (x2 - 16, y2 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, _WHITE, 1, cv2.LINE_AA)
            continue

        # ─── 3. AMBIGUOUS MATCH -> ORANGE BOX ───
        if rec is not None and rec.state == RecognitionState.AMBIGUOUS:
            color = _COLOR_AMBIGUOUS
            label = f"Ambiguous ({rec.confidence:.0%})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            _draw_label(frame, (x1, y1), label, color, text_color=_WHITE)
            continue

        # ─── 4. UNKNOWN / REJECTED -> RED BOX ───
        color = _COLOR_UNKNOWN
        label = "Unknown Person"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        _draw_label(frame, (x1, y1), label, color, text_color=_WHITE)

    return frame


def render_attendance_page(rt):
    """Main attendance page view."""
    st.markdown("## 📋 Attendance Session")

    if rt.identity_index.size == 0:
        st.warning("⚠️ No identities enrolled in biometric database. Please enroll students in the **Register** tab first.")
        return

    st.success(f"✅ Biometric Index Ready: **{rt.identity_index.size}** identities enrolled.")

    # Initialize AttendanceEngine if not exists
    if st.session_state.get("att_engine") is None:
        st.session_state.att_engine = AttendanceEngine(
            pipeline=rt.pipeline,
            camera=rt.camera,
            external_buffer=rt.external_buffer,
            biometric_db=rt.biometric_db,
            attendance_db=rt.attendance_db,
        )
    engine: AttendanceEngine = st.session_state.att_engine

    # ─── Case 1: Pre-session / Completed Session (Idle state) ───
    if not engine.session_active:
        # Show results if a session completed
        if engine.session_id is not None:
            st.markdown("---")
            st.markdown("## 📊 Session Report & Results")
            attendance = rt.attendance_db.get_attendance(engine.session_id)
            if attendance:
                rows = []
                for a in attendance:
                    rows.append({
                        "Student": a["name"],
                        "Status": a["status"].upper(),
                        "Present Checks": f"{a['checks_present']}",
                        "Spoofed Checks": a["checks_spoofed"],
                        "Notes": a["note"],
                    })
                df = pd.DataFrame(rows)

                present = sum(1 for a in attendance if a["status"] == "present")
                absent = sum(1 for a in attendance if a["status"] == "absent")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("👥 Total Enrolled", len(attendance))
                mc2.metric("✅ Marked Present", present)
                mc3.metric("❌ Marked Absent", absent)

                st.dataframe(df, use_container_width=True, hide_index=True)
                st.download_button("📥 Download Session CSV", df.to_csv(index=False),
                                   file_name=f"attendance_{engine.session_id}.csv",
                                   mime="text/csv",
                                   use_container_width=True)
            else:
                st.info("No attendance data recorded for this completed session.")

            st.markdown("---")
            if st.button("🔄 Start New Attendance Session", type="primary", use_container_width=True):
                if rt.camera.is_opened():
                    rt.camera.release()
                st.session_state.pop("att_engine", None)
                st.rerun()
            return

        # Start Session Form
        demo = Config.get("attendance", "demo_mode")
        times = Config.get("attendance", "check_times_demo" if demo else "check_times_normal")
        unit = "sec" if demo else "min"
        st.info(f"⚙️ Operating Mode: **{'⚡ DEMO' if demo else '🕐 PRODUCTION'}** — Scheduled checks at **{times} {unit}**")

        subjects = rt.attendance_db.get_all_subjects()
        if not subjects:
            st.warning("⚠️ No subjects configured yet. Add subjects in the **Subjects** tab first.")
            return

        subject_options = {s["subject_id"]: f"{s['name']}" + (f" ({s['code']})" if s.get("code") else "")
                           for s in subjects}
        selected_sid = st.selectbox(
            "📚 Select Class / Subject",
            options=list(subject_options.keys()),
            format_func=lambda x: subject_options[x],
            key="att_subject_select",
        )
        selected_name = subjects[list(subject_options).index(selected_sid)]["name"]

        if st.button("▶️ Start Live Attendance Session", type="primary", use_container_width=True):
            sid = engine.start_session(selected_name)
            if sid:
                st.rerun()
            else:
                st.error("❌ Failed to start session. Check camera device permissions and enrolled biometric templates.")
        return

    # ─── Case 2: Active Session (Live OpenCV Stream) ───
    st.markdown(f"### 🔴 Live Session: **{engine.class_name}**")

    col_stop, _ = st.columns([1, 4])
    with col_stop:
        if st.button("⏹️ Stop Session", type="secondary", use_container_width=True):
            engine.stop_session()
            if rt.camera.is_opened():
                rt.camera.release()
            st.rerun()

    status_placeholder = st.empty()
    feed_col, info_col = st.columns([2, 1])

    with feed_col:
        video_placeholder = st.empty()
        camera_info_placeholder = st.empty()

    with info_col:
        info_placeholder = st.empty()

    # Frame timing for FPS calculation
    last_frame_time = time.time()
    fps_history = []

    # ─── Live Native OpenCV Inference Loop ───
    while engine.session_active:
        # Ensure camera is opened
        if not rt.camera.is_opened():
            if not rt.camera.open():
                st.error("❌ Camera disconnected or inaccessible. Please reconnect camera.")
                time.sleep(1)
                continue

        frame = rt.camera.read_frame()
        if frame is None or frame.size == 0:
            time.sleep(0.03)
            continue

        # Push to external buffer for background scheduled check thread
        rt.external_buffer.push(frame)

        # Run recognition pipeline
        frame_result = rt.pipeline.process_frame(frame, run_liveness=True)

        with engine._lock:
            engine._latest_frame_result = frame_result

        # Draw overlays
        annotated = _draw_frame_overlays(frame.copy(), frame_result)
        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        # Render video frame smoothly
        video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

        # Calculate live FPS
        now = time.time()
        dt = now - last_frame_time
        last_frame_time = now
        if dt > 0:
            current_fps = 1.0 / dt
            fps_history.append(current_fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history) if fps_history else 30.0

        camera_info_placeholder.caption(
            f"📹 Native Capture: {rt.camera.width}x{rt.camera.height} | Live Feed: **{avg_fps:.1f} FPS** | SCRFD-10G + glintr100"
        )

        # Update top metrics bar
        status = engine.get_status()
        with status_placeholder.container():
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("⏱️ Elapsed Time",
                       f"{int(status.elapsed_seconds // 60):02d}:{int(status.elapsed_seconds % 60):02d}")
            sc2.metric("📊 Checks", f"{status.checks_completed}/{status.total_checks}")
            sc3.metric("⏳ Next Check In", f"{status.next_check_in:.0f}s")
            if status.check_running:
                sc4.metric("📡 Status", "📸 RUNNING CHECK")
            elif status.paused:
                sc4.metric("📡 Status", "⏸️ PAUSED")
            else:
                sc4.metric("📡 Status", "🔴 LIVE STREAM")

        # Update right info panel
        with info_placeholder.container():
            st.markdown("#### 📡 Real-time Recognition")
            ic1, ic2 = st.columns(2)
            ic1.metric("Detected Faces", frame_result.num_detected)
            ic2.metric("Recognized", frame_result.num_recognized)

            if frame_result.faces:
                for face in frame_result.faces:
                    rec = face.recognition
                    liveness = face.liveness

                    if liveness and liveness.state == LivenessState.SPOOF:
                        st.warning(f"🟡 **Possible Spoof Detected**: {liveness.spoofing_type or 'Fake Face'}")
                    elif rec and rec.state == RecognitionState.KNOWN:
                        st.success(f"🟢 **Real Person**: {rec.name} ({rec.confidence:.0%})")
                    elif rec and rec.state == RecognitionState.AMBIGUOUS:
                        st.warning(f"🟠 **Ambiguous Match**: ({rec.confidence:.0%})")
                    else:
                        st.error("🔴 **Unknown Person** (Unregistered)")

            st.markdown("---")
            st.markdown("#### 📋 Check Schedule")
            check_times = engine.check_times
            unit = "s" if engine.demo_mode else "m"
            for i, ct in enumerate(check_times):
                cn = i + 1
                if cn <= status.checks_completed:
                    icon = "✅"
                elif cn == status.checks_completed + 1:
                    icon = "⏳"
                else:
                    icon = "⬜"
                st.caption(f"{icon} Check {cn} @ {ct}{unit}")

            if status.spoofing_count > 0:
                st.error(f"🚫 {status.spoofing_count} spoofing attack(s) blocked!")

        # Short pause to prevent maxing CPU thread while maintaining 30 FPS
        time.sleep(0.03)

    # ─── Cleanup after loop exits ───
    if rt.camera.is_opened():
        rt.camera.release()
    st.rerun()
