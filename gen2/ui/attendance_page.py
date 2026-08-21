"""Attendance page for gen2 — live session with WebRTC."""
import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd

from gen2.config import Config
from gen2.attendance.engine import AttendanceEngine
from gen2.recognition.matching.engine import RecognitionState
from gen2.recognition.liveness.minifasnet import LivenessState

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    import av
    _HAS_WEBRTC = True
except Exception:
    _HAS_WEBRTC = False

# ICE servers for WebRTC connection establishment.
# STUN servers are required for ICE candidate gathering — without them,
# the connection silently fails on Windows and many browsers.
_RTC_CONFIG = RTCConfiguration(
    ice_servers=[
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ]
) if _HAS_WEBRTC else None


def _cleanup_webrtc():
    """Stop any existing WebRTC peer-connection to avoid leaking them."""
    ctx = st.session_state.get("_webrtc_ctx")
    if ctx is not None:
        try:
            if hasattr(ctx, 'state') and ctx.state.playing:
                pass  # streamlit-webrtc handles stop on key removal
        except Exception:
            pass
        st.session_state.pop("_webrtc_ctx", None)
    # Reset connection tracking
    st.session_state.pop("_webrtc_connect_time", None)

# Colors (BGR)
_GREEN = (0, 200, 0)
_RED = (0, 0, 220)
_ORANGE = (0, 165, 255)
_YELLOW = (0, 255, 255)
_GRAY = (128, 128, 128)
_WHITE = (255, 255, 255)
_BLUE = (255, 100, 0)


def _draw_frame_overlays(frame, frame_result):
    """Draw bounding boxes, names, confidence, liveness, and spoofing
    overlays on a copy of the frame."""
    if frame_result is None or not frame_result.faces:
        return frame

    for face in frame_result.faces:
        bbox = face.bbox
        if bbox is None or bbox == (0, 0, 0, 0):
            continue
        x1, y1, x2, y2 = bbox

        rec = face.recognition
        liveness = face.liveness

        # Determine box color and label
        if liveness and liveness.state == LivenessState.SPOOF:
            # Yellow dashed box for spoof
            _draw_dashed_rect(frame, (x1, y1, x2, y2), _YELLOW, 2)
            label = f"SPOOF — {liveness.spoofing_type or 'detected'}"
            _draw_label(frame, (x1, y1), label, _YELLOW, text_color=(0, 0, 0))
            continue

        if rec is None:
            continue

        if rec.state == RecognitionState.KNOWN:
            color = _GREEN
            name = rec.name or "Unknown"
            label = f"{name} ({rec.confidence:.0%})"
            if liveness and liveness.state == LivenessState.LIVE:
                label = f"{name} ({rec.confidence:.0%}) LIVE"
        elif rec.state == RecognitionState.UNKNOWN:
            color = _RED
            label = "Unknown"
        elif rec.state == RecognitionState.AMBIGUOUS:
            color = _ORANGE
            label = "Ambiguous?"
        elif rec.state == RecognitionState.REJECTED:
            color = _GRAY
            label = "Rejected"
        else:
            color = _GRAY
            label = "?"

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Draw label
        _draw_label(frame, (x1, y1), label, color)

        # Draw track ID (small, top-right corner)
        if face.track_id is not None:
            tid_text = f"T{face.track_id}"
            cv2.putText(frame, tid_text, (x2 - 40, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # Confirmed badge
        if face.confirmed:
            cv2.circle(frame, (x2 - 12, y2 - 12), 8, _GREEN, -1, cv2.LINE_AA)
            cv2.putText(frame, "C", (x2 - 16, y2 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, _WHITE, 1, cv2.LINE_AA)

    return frame


def _draw_label(frame, bbox_tl, label, color, text_color=None):
    """Draw a semi-transparent label background + text above the box."""
    if text_color is None:
        text_color = _WHITE
    x1, y1 = bbox_tl
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

    label_y = max(y1 - 8, th + 8)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, label_y - th - 6),
                  (x1 + tw + 10, label_y + 4), color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, label, (x1 + 5, label_y - 2),
                font, font_scale, text_color, thickness, cv2.LINE_AA)


def _draw_dashed_rect(frame, bbox, color, thickness=2, dash=10, gap=6):
    """Draw a dashed rectangle."""
    x1, y1, x2, y2 = bbox
    for x in range(x1, x2, dash + gap):
        cv2.line(frame, (x, y1), (min(x + dash, x2), y1),
                 color, thickness, cv2.LINE_AA)
    for x in range(x1, x2, dash + gap):
        cv2.line(frame, (x, y2), (min(x + dash, x2), y2),
                 color, thickness, cv2.LINE_AA)
    for y in range(y1, y2, dash + gap):
        cv2.line(frame, (x1, y), (x1, min(y + dash, y2)),
                 color, thickness, cv2.LINE_AA)
    for y in range(y1, y2, dash + gap):
        cv2.line(frame, (x2, y), (x2, min(y + dash, y2)),
                 color, thickness, cv2.LINE_AA)


class AttendanceVideoProcessor(VideoProcessorBase if _HAS_WEBRTC else object):
    """Pushes browser frames to the engine's external buffer and returns
    annotated frames with bounding boxes, names, and spoofing warnings."""

    def __init__(self, engine):
        if _HAS_WEBRTC:
            super().__init__()
        self.engine = engine

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Push raw frame to the engine's buffer for the check thread
        if self.engine.external_buffer is not None:
            self.engine.external_buffer.push(img)

        # Run the recognition pipeline for live preview
        annotated = img.copy()
        try:
            frame_result = self.engine.pipeline.process_frame(
                img, run_liveness=False
            )
            with self.engine._lock:
                self.engine._latest_frame_result = frame_result
            annotated = _draw_frame_overlays(annotated, frame_result)
        except Exception:
            pass

        new_frame = av.VideoFrame.from_ndarray(annotated, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame


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

    # ─── Pre-session / Post-session: not actively running ───
    if not engine.session_active:
        # Show final results if a session just completed
        if engine.session_id is not None:
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

                # Summary metrics
                present = sum(1 for a in attendance if a["status"] == "present")
                absent = sum(1 for a in attendance if a["status"] == "absent")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("👥 Total", len(attendance))
                mc2.metric("✅ Present", present)
                mc3.metric("❌ Absent", absent)

                st.dataframe(df, use_container_width=True, hide_index=True)
                st.download_button("📥 Download CSV", df.to_csv(index=False),
                                  file_name=f"attendance_{engine.session_id}.csv")
            else:
                st.info("No attendance data recorded for this session.")

            st.markdown("---")
            if st.button("🔄 Start New Session", type="primary", use_container_width=True):
                _cleanup_webrtc()
                if st.session_state.get("att_engine") is not None:
                    st.session_state.att_engine._release_camera()
                st.session_state.att_engine = None
                st.rerun()
            return

        # No completed session — show start controls
        demo = Config.get("attendance", "demo_mode")
        times = Config.get("attendance", "check_times_demo" if demo else "check_times_normal")
        unit = "sec" if demo else "min"
        st.info(f"{'⚡ DEMO' if demo else '🕐 NORMAL'} mode — "
                f"checks at {times} {unit}")

        # Dynamic subject selection from DB
        subjects = rt.attendance_db.get_all_subjects()
        if not subjects:
            st.warning("No subjects configured. Add subjects in the **Subjects** tab first.")
            return

        subject_options = {s["subject_id"]: f"{s['name']}"
                          + (f" ({s['code']})" if s.get("code") else "")
                          for s in subjects}
        selected_sid = st.selectbox(
            "📚 Select Subject",
            options=list(subject_options.keys()),
            format_func=lambda x: subject_options[x],
            key="att_subject_select",
        )
        selected_name = subjects[list(subject_options).index(selected_sid)]["name"]

        if st.button("▶️ Start Session", type="primary", use_container_width=True):
            sid = engine.start_session(selected_name)
            if sid:
                st.rerun()
            else:
                st.error("Failed to start. Check camera and enrolled identities.")
        return

    # ─── Active session ───
    st.markdown("### 🔴 Session In Progress")

    col_stop, _ = st.columns([1, 3])
    with col_stop:
        if st.button("⏹️ Stop Session", type="secondary"):
            _cleanup_webrtc()
            engine.stop_session()
            st.rerun()

    # Status bar placeholder (updated in-place, no st.rerun)
    status_placeholder = st.empty()

    # Two-column layout: camera feed (left, 2/3) + info (right, 1/3)
    feed_col, info_col = st.columns([2, 1])

    # Camera feed goes directly in the left column — NOT in a placeholder.
    # This keeps the WebRTC component stable across updates.
    with feed_col:
        webrtc_ctx = None
        if _HAS_WEBRTC:
            webrtc_ctx = webrtc_streamer(
                key="attendance-live",
                video_processor_factory=lambda e=engine: AttendanceVideoProcessor(e),
                rtc_configuration=_RTC_CONFIG,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 640},
                        "height": {"ideal": 480},
                        "facingMode": "user",
                    },
                    "audio": False,
                },
                desired_playing_state=True,
                async_processing=True,
            )
            # Cache context so we can clean it up on session stop
            st.session_state["_webrtc_ctx"] = webrtc_ctx

            # ── Connection status feedback ──
            if webrtc_ctx.state.playing:
                # Connected successfully — clear any tracking
                st.session_state.pop("_webrtc_connect_time", None)
            else:
                # Not playing — track how long we've been waiting
                if "_webrtc_connect_time" not in st.session_state:
                    st.session_state["_webrtc_connect_time"] = time.time()
                wait_secs = time.time() - st.session_state["_webrtc_connect_time"]

                if wait_secs < 8:
                    st.info("📷 Connecting to camera... Please allow camera access if prompted.")
                elif wait_secs < 20:
                    st.warning(
                        "⏳ **Camera connection is taking longer than expected.**\n\n"
                        "**Try these steps:**\n"
                        "1. Click **Allow** if the browser asked for camera permission\n"
                        "2. Check that no other app is using the camera\n"
                        "3. Try refreshing the page (F5)\n"
                        "4. On Windows: try Chrome or Edge (Firefox may have issues)"
                    )
                else:
                    st.error(
                        "❌ **Camera connection failed.**\n\n"
                        "The WebRTC video feed could not be established. This can happen when:\n"
                        "- Camera permissions were denied in the browser\n"
                        "- Another application is using the camera\n"
                        "- Your browser or network blocks WebRTC connections\n"
                        "- A firewall is blocking STUN/TURN traffic\n\n"
                        "**Stop the session and try again after fixing the issue.**"
                    )
        else:
            st.warning(
                "⚠️ `streamlit-webrtc` not installed. Camera feed unavailable.\n\n"
                "Install with: `pip install streamlit-webrtc`"
            )
        # Placeholder below the WebRTC component for cv2 fallback
        feed_placeholder = st.empty()

    # Info panel placeholder (right column, updated in-place)
    info_placeholder = info_col.empty()

    # ─── Live update loop (no st.rerun — uses in-place placeholder updates) ───
    while engine.session_active:
        status = engine.get_status()

        # Update status bar in-place
        with status_placeholder.container():
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("⏱️ Elapsed",
                       f"{int(status.elapsed_seconds//60):02d}:{int(status.elapsed_seconds%60):02d}")
            sc2.metric("📊 Checks", f"{status.checks_completed}/{status.total_checks}")
            sc3.metric("⏳ Next In", f"{status.next_check_in:.0f}s")
            if status.check_running:
                sc4.metric("📡 Status", "📸 CHECK")
            elif status.paused:
                sc4.metric("📡 Status", "⏸️ PAUSED")
            else:
                sc4.metric("📡 Status", "🔴 LIVE")

        # Update info panel in-place
        latest = engine.get_latest_frame_result()
        with info_placeholder.container():
            if latest:
                st.markdown("#### 📡 Live Recognition")
                mc1, mc2 = st.columns(2)
                mc1.metric("Detected", latest.num_detected)
                mc2.metric("Recognized", latest.num_recognized)

                if latest.faces:
                    for face in latest.faces:
                        if face.recognition:
                            if face.recognition.state == RecognitionState.KNOWN:
                                st.success(f"✅ {face.recognition.name} "
                                          f"({face.recognition.confidence:.0%})")
                            elif face.recognition.state == RecognitionState.UNKNOWN:
                                st.info("❓ Unknown")
                            elif face.recognition.state == RecognitionState.AMBIGUOUS:
                                st.warning("⚠️ Ambiguous")
                            elif face.recognition.state == RecognitionState.REJECTED:
                                st.error("🔴 Rejected")

                st.markdown("---")
                st.markdown("#### 📋 Schedule")
                check_times = engine.check_times
                unit = "sec" if engine.demo_mode else "min"
                for i, ct in enumerate(check_times):
                    cn = i + 1
                    if cn <= status.checks_completed:
                        icon = "✅"
                    elif cn == status.checks_completed + 1:
                        icon = "⏳"
                    else:
                        icon = "⬜"
                    st.caption(f"{icon} Check {cn} @ {ct}{unit}")
            else:
                st.info("📷 Waiting for frames...")

            if status.spoofing_count > 0:
                st.warning(f"🚫 {status.spoofing_count} spoofing attempt(s) blocked")

        # cv2 fallback: ONLY when streamlit-webrtc is not installed at all.
        # When WebRTC IS available, the browser is the sole camera owner.
        # Do NOT open cv2 camera during WebRTC ICE negotiation — it grabs
        # the device on the server side and never releases it.
        if not _HAS_WEBRTC:
            if not rt.camera.is_opened():
                rt.camera.open()
            if rt.camera.is_opened():
                frame = rt.camera.read_frame()
                if frame is not None:
                    rt.external_buffer.push(frame)
                    try:
                        frame_result = rt.pipeline.process_frame(
                            frame, run_liveness=True)
                        with engine._lock:
                            engine._latest_frame_result = frame_result
                        annotated = _draw_frame_overlays(frame.copy(), frame_result)
                        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        feed_placeholder.image(rgb, channels="RGB")
                    except Exception:
                        pass

        # Throttle refresh
        time.sleep(1)

    # ─── Session just ended (loop exited): clean up and rerun ───
    # Release camera, tear down WebRTC, and rerun so the page re-renders
    # WITHOUT the frozen camera feed — showing only the final results.
    if not _HAS_WEBRTC and rt.camera.is_opened():
        rt.camera.release()
    _cleanup_webrtc()
    st.rerun()
