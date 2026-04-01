"""
Attendance Session Page.
Runs a live-monitored attendance session with scheduled face checks,
real-time camera feed, brightness monitoring, darkness warnings,
and final attendance results with anomaly notes.
"""

import streamlit as st
import cv2
import time
import numpy as np
import pandas as pd
from PIL import Image
from core.config import Config
from utils.drawing import (
    draw_brightness_meter,
    draw_darkness_warning,
    draw_enhancement_indicator,
    draw_retry_countdown,
)


def render_attendance_page(monitor, face_db):
    """Render the attendance session page."""

    st.markdown("## 📋 Attendance Session")

    student_count = face_db.get_student_count()
    if student_count == 0:
        st.warning(
            "⚠️ **No students registered!** "
            "Please register students first in the **Register** tab."
        )
        return

    st.success(f"✅ **{student_count}** students registered and ready.")

    # Show registered students summary
    with st.expander("👥 Registered Students", expanded=False):
        students = face_db.get_all_students()
        cols = st.columns(min(len(students), 4))
        for i, (sid, data) in enumerate(students.items()):
            with cols[i % 4]:
                images = face_db.get_student_face_images(sid)
                if images:
                    rgb = cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB)
                    st.image(rgb, caption=data["name"], use_container_width=True)
                else:
                    st.markdown(f"**{data['name']}**")
                st.caption(f"ID: {sid}")

    st.markdown("---")

    # Mode indicator
    if Config.DEMO_MODE:
        check_times = Config.CHECK_TIMES_DEMO
        st.info(
            f"⚡ **DEMO MODE** — Session compresses to ~90 seconds\n\n"
            f"Checks at: **{check_times[0]}s**, **{check_times[1]}s**, **{check_times[2]}s**\n\n"
            f"💡 *Cover the camera with your hand to simulate darkness!*"
        )
    else:
        check_times = Config.CHECK_TIMES_NORMAL
        st.info(
            f"🕐 **NORMAL MODE** — Full session\n\n"
            f"Checks at: {check_times[0]}min, {check_times[1]}min, {check_times[2]}min"
        )

    # ── Session State Management ──
    if "att_session_active" not in st.session_state:
        st.session_state.att_session_active = False
    if "att_session_id" not in st.session_state:
        st.session_state.att_session_id = None
    if "att_final_shown" not in st.session_state:
        st.session_state.att_final_shown = False

    # ── Pre-Session: Start Controls ──
    if not st.session_state.att_session_active:
        st.session_state.att_final_shown = False
        class_name = st.text_input(
            "📚 Class Name",
            placeholder="e.g. Computer Science 101",
            key="att_class_name",
        )

        if st.button(
            "▶️ START ATTENDANCE SESSION",
            type="primary",
            use_container_width=True,
            disabled=not class_name.strip(),
        ):
            with st.spinner("🔄 Initializing session..."):
                session_id = monitor.start_session(
                    class_name=class_name.strip(),
                    camera_index=Config.CAMERA_INDEX,
                )

            if session_id:
                st.session_state.att_session_active = True
                st.session_state.att_session_id = session_id
                st.session_state.att_final_shown = False
                st.rerun()
            else:
                st.error(
                    "❌ Could not start session. Check that:\n"
                    "- Students are registered\n"
                    "- Webcam is available and not in use"
                )
        return

    # ── Active Session UI ──
    st.markdown("### 🔴 Session In Progress")

    # Stop button
    col_stop, col_info = st.columns([1, 3])
    with col_stop:
        if st.button("⏹️ Stop Session", type="secondary", use_container_width=True):
            with st.spinner("Stopping session..."):
                final = monitor.stop_session()
            st.session_state.att_session_active = False
            if final:
                st.session_state.att_final_results = final
                st.session_state.att_final_shown = True
            st.rerun()

    # Create placeholder containers
    darkness_banner = st.empty()
    status_container = st.empty()
    feed_col, info_col = st.columns([2, 1])
    frame_placeholder = feed_col.empty()
    check_placeholder = info_col.empty()
    results_placeholder = st.empty()

    # ── Live Update Loop ──
    while st.session_state.att_session_active:
        status = monitor.get_session_status()
        brightness = monitor.get_brightness_status()
        retry_info = monitor.get_retry_info()

        if not status["active"]:
            st.session_state.att_session_active = False
            session_data = monitor.attendance_store.get_session(
                st.session_state.att_session_id
            )
            if session_data and session_data.get("final_results"):
                st.session_state.att_final_results = session_data["final_results"]
                st.session_state.att_final_shown = True
            st.rerun()
            break

        # ── Darkness Warning Banner ──
        with darkness_banner.container():
            if status.get("paused"):
                st.markdown(
                    """
                    <div style="
                        background: linear-gradient(135deg, #b71c1c 0%, #880e0e 100%);
                        border: 2px solid #ff5252;
                        border-radius: 12px;
                        padding: 20px;
                        text-align: center;
                        margin-bottom: 10px;
                        animation: pulse 2s ease-in-out infinite;
                    ">
                        <h3 style="margin:0; color:#ff8a80;">⚠️ LOW LIGHT DETECTED — Session Paused</h3>
                        <p style="margin:5px 0 0 0; color:#ffcdd2;">
                            Waiting for lights to be restored...
                        </p>
                    </div>
                    <style>
                        @keyframes pulse {
                            0%, 100% { opacity: 1; }
                            50% { opacity: 0.7; }
                        }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
            elif retry_info.get("active"):
                rem = retry_info.get("seconds_remaining", 0)
                att = retry_info.get("attempt", 1)
                mx = retry_info.get("max_attempts", 3)
                cn = retry_info.get("check_number", 0)
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #e65100 0%, #bf360c 100%);
                        border: 2px solid #ff9800;
                        border-radius: 12px;
                        padding: 16px;
                        text-align: center;
                        margin-bottom: 10px;
                    ">
                        <h4 style="margin:0; color:#ffe0b2;">
                            🔄 Retrying Check {cn} in {rem:.0f}s...
                        </h4>
                        <p style="margin:5px 0 0 0; color:#ffcc80;">
                            Attempt {att}/{mx} — Waiting for sufficient lighting
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif brightness.get("quality_label") == "LOW_LIGHT":
                st.markdown(
                    """
                    <div style="
                        background: linear-gradient(135deg, #f57f17 0%, #e65100 100%);
                        border: 1px solid #ffc107;
                        border-radius: 10px;
                        padding: 12px;
                        text-align: center;
                        margin-bottom: 10px;
                    ">
                        <p style="margin:0; color:#fff8e1;">
                            ⚡ <strong>Low light detected</strong> — Auto-enhancing frames for better recognition
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── Status Bar ──
        elapsed = status["elapsed_seconds"]
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)

        with status_container.container():
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            sc1.metric("⏱️ Elapsed", f"{mins:02d}:{secs:02d}")
            sc2.metric("📊 Checks", f"{status['checks_completed']}/{status['total_checks']}")
            sc3.metric("⏳ Next In", f"{status['next_check_in']:.0f}s")

            # Brightness indicator
            br_val = brightness.get("brightness", 128)
            br_label = brightness.get("quality_label", "GOOD")
            if br_label == "GOOD":
                sc4.metric("💡 Light", f"{br_val:.0f}", delta="Good", delta_color="normal")
            elif br_label == "LOW_LIGHT":
                sc4.metric("💡 Light", f"{br_val:.0f}", delta="Low", delta_color="off")
            elif br_label == "DARK":
                sc4.metric("💡 Light", f"{br_val:.0f}", delta="Dark!", delta_color="inverse")
            else:
                sc4.metric("💡 Light", f"{br_val:.0f}", delta=br_label, delta_color="off")

            # Session status + spoofing
            spoof_count = monitor.get_spoofing_count() if hasattr(monitor, 'get_spoofing_count') else 0
            if status.get("paused"):
                sc5.metric("📡 Status", "⏸️ PAUSED")
            elif status["check_running"]:
                sc5.metric("📡 Status", "📸 CHECK")
            else:
                sc5.metric("📡 Status", "🔴 LIVE")

            # Spoofing alert
            if spoof_count > 0:
                st.markdown(
                    f'<div style="background:rgba(255,0,0,0.15);border:1px solid #f44336;'
                    f'border-radius:8px;padding:8px 16px;text-align:center;'
                    f'margin-bottom:10px;">'
                    f'🚨 <strong style="color:#ff5252;">{spoof_count} spoofing attempt(s) blocked!</strong>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── Live Camera Feed with brightness overlay ──
        frame, detections = monitor.get_live_frame()
        if frame is not None:
            display_frame = frame.copy()

            # Draw brightness meter on frame
            br_val = brightness.get("brightness", 128)
            display_frame = draw_brightness_meter(display_frame, br_val)

            # Draw enhancement indicator if frame was enhanced
            if monitor.is_frame_enhanced:
                display_frame = draw_enhancement_indicator(display_frame)

            # Draw darkness warning overlay if paused
            if status.get("paused"):
                display_frame = draw_darkness_warning(display_frame)

            # Draw retry countdown if active
            if retry_info.get("active"):
                display_frame = draw_retry_countdown(
                    display_frame,
                    retry_info.get("check_number", 0),
                    retry_info.get("seconds_remaining", 0),
                    retry_info.get("attempt", 1),
                    retry_info.get("max_attempts", 3),
                )

            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            with frame_placeholder.container():
                st.image(rgb, caption="Live Feed — Face Detection", use_container_width=True)
                if status["check_running"]:
                    st.warning(f"📸 **Running Check #{status['checks_completed'] + 1}...**")
        else:
            with frame_placeholder.container():
                st.info("📷 Waiting for camera feed...")

        # ── Check Schedule Panel ──
        schedule = monitor.get_check_schedule_info()
        with check_placeholder.container():
            st.markdown("#### 📋 Check Schedule")
            for chk in schedule:
                num = chk["check_number"]
                t = chk["time_value"]
                u = chk["unit"]
                s = chk["status"]

                if s == "completed":
                    icon, label = "✅", "Done"
                elif s == "partial":
                    icon, label = "🟡", "Partial"
                elif s == "retried":
                    icon, label = "🔄", "Retried"
                elif s == "failed_dark":
                    icon, label = "🌑", "Failed (Dark)"
                elif s == "skipped_dark":
                    icon, label = "⛔", "Skipped (Dark)"
                elif s == "next":
                    icon, label = "⏳", "Next"
                else:
                    icon, label = "⬜", "Pending"

                st.markdown(f"{icon} Check {num} @ {t}{u} — **{label}**")

            st.markdown("---")
            st.caption(f"Session: {status['session_id']}")
            st.caption(f"Class: {status['class_name']}")
            st.caption(status["status"])

            # Brightness info
            st.markdown("---")
            st.markdown("#### 💡 Light Status")
            br_label = brightness.get("quality_label", "GOOD")
            br_msg = brightness.get("message", "")
            if br_label == "GOOD":
                st.success(f"🟢 {br_msg}")
            elif br_label == "LOW_LIGHT":
                st.warning(f"🟡 {br_msg}")
            elif br_label == "DARK":
                st.error(f"🔴 {br_msg}")
            elif br_label == "TOO_BRIGHT":
                st.warning(f"🔵 {br_msg}")

        # Rate limit the loop
        time.sleep(0.5)

    # ── Final Results Display ──
    if st.session_state.get("att_final_shown") and st.session_state.get("att_final_results"):
        # ── Twin Review Queue ──
        review_queue = monitor.get_review_queue()
        if review_queue:
            _render_review_queue(review_queue, monitor)

        _render_final_results(
            st.session_state.att_final_results,
            st.session_state.att_session_id,
            monitor.attendance_store,
            face_db,
            monitor,
        )

        if st.button("🔄 Start New Session", type="primary", use_container_width=True):
            st.session_state.att_session_active = False
            st.session_state.att_session_id = None
            st.session_state.att_final_shown = False
            st.session_state.att_final_results = None
            st.rerun()


def _render_review_queue(review_queue, monitor):
    """Render the twin/uncertain detection review queue."""
    unresolved = [r for r in review_queue if not r.get("resolved")]
    if not unresolved:
        return

    st.markdown("---")
    st.markdown(
        f"## ⚠️ {len(unresolved)} Detection(s) Need Teacher Review"
    )
    st.markdown(
        "These detections involve twin/lookalike students. "
        "Please confirm the correct identity."
    )

    for idx, item in enumerate(review_queue):
        if item.get("resolved"):
            continue

        check_num = item.get("check_number", "?")
        name_a = item.get("name_a", "Student A")
        name_b = item.get("name_b", "Student B")
        score_a = item.get("score_a", 0)
        score_b = item.get("score_b", 0)
        sid_a = item.get("student_a", "")
        sid_b = item.get("student_b", "")
        crop_path = item.get("crop_path")

        with st.container():
            st.markdown(
                f"""
                <div style="
                    background: rgba(255,165,0,0.1);
                    border: 1px solid #ff9800;
                    border-radius: 10px;
                    padding: 15px;
                    margin: 10px 0;
                ">
                    <strong>Check #{check_num}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

            rc1, rc2, rc3 = st.columns([1, 2, 1])

            with rc1:
                if crop_path:
                    try:
                        crop_img = cv2.imread(crop_path)
                        if crop_img is not None:
                            rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                            st.image(rgb, caption="Detected Face", use_container_width=True)
                    except Exception:
                        st.info("📷 Face image unavailable")
                else:
                    st.info("📷 No face crop saved")

            with rc2:
                st.markdown(
                    f"**Is this {name_a} ({score_a:.0%}) or {name_b} ({score_b:.0%})?**"
                )
                diff = item.get("difference", 0)
                st.caption(f"Similarity difference: {diff:.4f} (below threshold {Config.MIN_TWIN_DIFFERENCE})")

            with rc3:
                if st.button(
                    f"✅ {name_a}",
                    key=f"resolve_{idx}_a",
                    use_container_width=True,
                ):
                    monitor.resolve_review(idx, sid_a)
                    st.rerun()

                if st.button(
                    f"✅ {name_b}",
                    key=f"resolve_{idx}_b",
                    use_container_width=True,
                ):
                    monitor.resolve_review(idx, sid_b)
                    st.rerun()

                if st.button(
                    "⏭️ Skip",
                    key=f"resolve_{idx}_skip",
                    use_container_width=True,
                ):
                    monitor.resolve_review(idx, item.get("assigned_to", sid_a))
                    st.rerun()


def _render_final_results(final_results, session_id, attendance_store, face_db, monitor=None):
    """Render the final attendance results table with darkness anomaly notes."""

    st.markdown("---")
    st.markdown("## 📊 Final Attendance Results")

    # Anti-spoofing summary
    if monitor and hasattr(monitor, 'get_spoofing_count'):
        spoof_count = monitor.get_spoofing_count()
        if spoof_count > 0:
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#b71c1c,#880e0e);'
                f'border-radius:10px;padding:15px;margin:10px 0;">'
                f'<span style="font-size:1.2em;">🛡️</span> '
                f'<strong style="color:#ffcdd2;">{spoof_count} spoofing attempt(s) blocked</strong>'
                f'<span style="color:#ef9a9a;"> during this session</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.success("🛡️ No spoofing attempts detected — session clean")

    if not final_results:
        st.warning("No results available.")
        return

    session = attendance_store.get_session(session_id) if session_id else None
    checks = session.get("checks", {}) if session else {}
    session_status = session.get("status", "") if session else ""
    session_notes = session.get("notes", []) if session else []
    students = face_db.get_all_students()

    # ── Session cancelled due to darkness ──
    if session_status == "cancelled_dark":
        st.error(
            "❌ **Session cancelled — insufficient lighting for all checks**\n\n"
            "No attendance was recorded. Teacher can mark attendance manually."
        )
        if session_notes:
            with st.expander("📝 Session Notes"):
                for note in session_notes:
                    st.caption(f"• {note}")
        return

    # ── Session notes / anomalies ──
    if session_notes:
        with st.expander("⚠️ Session Notes & Anomalies", expanded=True):
            for note in session_notes:
                st.caption(f"• {note}")

    # Check for darkness-affected checks
    dark_checks = [
        cn for cn, cd in checks.items()
        if cd.get("status") in ("partial", "failed_dark", "skipped_dark", "retried")
    ]
    if dark_checks:
        st.warning(
            f"⚠️ **{len(dark_checks)}** check(s) were affected by lighting conditions. "
            "See the Notes column for details."
        )

    # Build results table
    rows = []
    present_count = late_count = absent_count = review_count = 0

    for sid, result in final_results.items():
        name = result.get("name", students.get(sid, {}).get("name", sid))
        total_present = result["checks_present"]
        status = result["status"]
        note = result.get("note", "")

        # Per-check marks with status info
        check_marks = {}
        for cn_str, check_data in checks.items():
            cn = int(cn_str)
            check_status = check_data.get("status", "completed")
            detected = check_data.get("detected", [])

            if check_status in ("failed_dark", "skipped_dark"):
                check_marks[f"Check {cn}"] = "🌑"  # Darkness
            elif sid in detected:
                check_marks[f"Check {cn}"] = "✅"
            else:
                check_marks[f"Check {cn}"] = "❌"

        # Count valid checks
        valid_checks = sum(
            1 for cd in checks.values()
            if cd.get("status") in ("completed", "partial", "retried")
        )

        row = {"Student": name, "ID": sid}
        # Add review marker for twin-affected students
        if result.get("needs_review"):
            row["Student"] = f"🔍 {name}"
        row.update(check_marks)
        row["Total"] = f"{total_present}/{valid_checks}"
        if result.get("needs_review"):
            row["Status"] = "NEEDS_REVIEW"
        else:
            row["Status"] = status.upper()
        if note:
            row["Notes"] = note
        else:
            row["Notes"] = ""
        rows.append(row)

        if status == "present":
            present_count += 1
        elif status == "late":
            late_count += 1
        elif status == "insufficient_data":
            review_count += 1
        else:
            absent_count += 1

    # Summary metrics
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("👥 Total", len(final_results))
    mc2.metric("✅ Present", present_count)
    mc3.metric("🟡 Late", late_count)
    if review_count > 0:
        mc4.metric("🔍 Needs Review", review_count)
    else:
        mc4.metric("❌ Absent", absent_count)

    # Results table with color coding
    if rows:
        df = pd.DataFrame(rows)

        def color_status(val):
            if val == "PRESENT":
                return "background-color: #1b5e20; color: #a5d6a7"
            elif val == "LATE":
                return "background-color: #e65100; color: #ffcc80"
            elif val == "ABSENT":
                return "background-color: #b71c1c; color: #ef9a9a"
            elif val == "INSUFFICIENT_DATA":
                return "background-color: #4a148c; color: #ce93d8"
            elif val == "NEEDS_REVIEW":
                return "background-color: #e65100; color: #ffab40"
            return ""

        styled = df.style.applymap(color_status, subset=["Status"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("No student data to display.")

    # Check details expander
    if checks:
        with st.expander("📋 Check Details"):
            for cn_str in sorted(checks.keys(), key=lambda x: int(x)):
                cd = checks[cn_str]
                cn = int(cn_str)
                cs = cd.get("status", "completed")
                ct = cd.get("time", "N/A")
                if len(ct) > 18:
                    ct = ct[11:19]
                cn_note = cd.get("note", "")
                det_count = cd.get("count", 0)
                uf = cd.get("usable_frames")
                tf = cd.get("total_frames")

                # Status icon
                if cs == "completed":
                    s_icon = "✅"
                elif cs == "partial":
                    s_icon = "🟡"
                elif cs == "retried":
                    s_icon = "🔄"
                elif cs == "failed_dark":
                    s_icon = "🌑"
                elif cs == "skipped_dark":
                    s_icon = "⛔"
                else:
                    s_icon = "⬜"

                line = f"{s_icon} **Check {cn}** at {ct} — {det_count} detected [{cs}]"
                if uf is not None and tf is not None:
                    line += f" ({uf}/{tf} frames)"
                st.markdown(line)
                if cn_note:
                    st.caption(f"   ↳ {cn_note}")
