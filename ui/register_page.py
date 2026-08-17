"""
Student Registration Page.
Captures face images via webcam, checks quality, generates embeddings,
detects twin/lookalike matches, and stores in the face database.
"""

import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image


def render_register_page(detector, embedder, reg_service, camera_service):
    """Render the student face registration page."""

    st.markdown("## 📸 Student Face Registration")
    st.markdown("Register a new student by capturing **5 face images** from the webcam.")
    st.markdown("---")

    # Full-width camera input CSS
    st.markdown("""
    <style>
        div[data-testid="stCameraInput"] > div {
            max-width: 100% !important;
            width: 100% !important;
        }
        div[data-testid="stCameraInput"] video,
        div[data-testid="stCameraInput"] canvas {
            width: 100% !important;
            max-width: 100% !important;
            border-radius: 12px;
        }
        div[data-testid="stCameraInput"] button {
            width: 100% !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── Input Fields ──
    col1, col2 = st.columns(2)
    with col1:
        student_id = st.text_input(
            "🆔 Student ID", placeholder="e.g. STU001", key="reg_student_id",
        )
    with col2:
        student_name = st.text_input(
            "👤 Student Name", placeholder="e.g. John Doe", key="reg_student_name",
        )

    st.markdown("")

    # ── Registration via st.camera_input ──
    st.markdown("### 📷 Capture Face Images")
    st.info(
        "💡 **Tips for best results:**\n"
        "- Ensure good, even lighting on your face\n"
        "- Look directly at the camera\n"
        "- Keep a neutral expression or slight smile\n"
        "- Make sure only your face is visible"
    )

    if "reg_captures" not in st.session_state:
        st.session_state.reg_captures = []
    if "reg_qualities" not in st.session_state:
        st.session_state.reg_qualities = []
    if "reg_result" not in st.session_state:
        st.session_state.reg_result = None
    if "twin_recapture_mode" not in st.session_state:
        st.session_state.twin_recapture_mode = False
    if "twin_extra_captures" not in st.session_state:
        st.session_state.twin_extra_captures = []

    images_needed = 5
    captures = st.session_state.reg_captures

    st.markdown(f"**Captured: {len(captures)}/{images_needed}**")
    progress = len(captures) / images_needed
    st.progress(progress)

    if len(captures) < images_needed:
        camera_img = st.camera_input(
            f"📸 Capture image {len(captures) + 1} of {images_needed}",
            key=f"cam_capture_{len(captures)}",
        )

        if camera_img is not None:
            file_bytes = np.asarray(bytearray(camera_img.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if frame is not None:
                from core.image_preprocessor import ImagePreprocessor
                preprocessor = ImagePreprocessor()
                detection = detector.detect_single_face(frame)
                if detection is not None:
                    quality = preprocessor.assess_quality(detection["cropped_face"])
                    st.session_state.reg_captures.append(frame)
                    st.session_state.reg_qualities.append(quality)
                    st.success(f"✅ Image {len(captures) + 1} captured! Quality: {quality['score']:.0%}")
                    st.rerun()
                else:
                    st.error("❌ Could not detect exactly one face. Please try again.")
    else:
        st.success("✅ All 5 images captured!")

    # Show captured thumbnails
    if captures:
        st.markdown("### 🖼️ Captured Images")
        thumb_cols = st.columns(min(len(captures), 5))
        for i, (cap, qual) in enumerate(zip(captures, st.session_state.reg_qualities)):
            with thumb_cols[i % 5]:
                rgb = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
                st.image(rgb, caption=f"#{i+1} Q:{qual['score']:.0%}", width="stretch")
                score = qual["score"]
                if score >= 0.7:
                    st.markdown("🟢 Good")
                elif score >= 0.4:
                    st.markdown("🟡 Fair")
                else:
                    st.markdown("🔴 Poor")

    # ── Register Button ──
    st.markdown("---")
    col_reg, col_reset = st.columns([3, 1])

    with col_reg:
        can_register = (
            len(captures) >= 1
            and student_id.strip()
            and student_name.strip()
        )
        if st.button(
            "✅ Register Student",
            type="primary",
            disabled=not can_register,
            width="stretch",
        ):
            if not student_id.strip():
                st.error("Please enter a Student ID.")
            elif not student_name.strip():
                st.error("Please enter a Student Name.")
            else:
                with st.spinner("🔄 Processing face embeddings..."):
                    result = reg_service.register_student_from_frames(
                        student_id=student_id.strip(),
                        name=student_name.strip(),
                        frames=captures,
                    )
                    st.session_state.reg_result = result

                if result["success"]:
                    st.balloons()
                    st.success(
                        f"✅ **{student_name}** registered successfully!\n\n"
                        f"- **Student ID:** {student_id}\n"
                        f"- **Embeddings stored:** {result['registered_count']}/{result['total_frames']}\n"
                    )

                    # ── Twin Detection Warning ──
                    twin_info = result.get("twin_info")
                    if twin_info and twin_info.get("has_twin"):
                        _render_twin_warning(
                            twin_info, student_id, student_name, reg_service
                        )

                    # Clear captures after success
                    st.session_state.reg_captures = []
                    st.session_state.reg_qualities = []
                else:
                    st.error("❌ Registration failed!")
                    for issue in result.get("issues", []):
                        st.warning(f"⚠️ {issue}")

    with col_reset:
        if st.button("🔄 Reset", width="stretch"):
            st.session_state.reg_captures = []
            st.session_state.reg_qualities = []
            st.session_state.reg_result = None
            st.session_state.twin_recapture_mode = False
            st.session_state.twin_extra_captures = []
            st.rerun()

    # Show result details
    if st.session_state.reg_result and not st.session_state.reg_result["success"]:
        result = st.session_state.reg_result
        with st.expander("📋 Detailed Issues"):
            for issue in result.get("issues", []):
                st.markdown(f"- {issue}")

    # ── Twin Recapture Mode ──
    if st.session_state.twin_recapture_mode:
        _render_twin_recapture(detector, reg_service)

    # ── Verify Registration Section ──
    st.markdown("---")
    st.markdown("### 🔍 Verify Registration")

    verify_id = st.text_input(
        "Enter Student ID to verify", placeholder="e.g. STU001", key="verify_id",
    )

    if st.button("🔎 Check", key="verify_btn"):
        if verify_id.strip():
            info = reg_service.verify_student(verify_id.strip())
            if info["exists"]:
                st.success(
                    f"✅ **{info['name']}** is registered!\n\n"
                    f"- Embeddings: {info['embedding_count']}\n"
                    f"- Images: {info['image_count']}\n"
                    f"- Registered: {info['registered_at']}"
                )
            else:
                st.warning(f"⚠️ Student ID '{verify_id}' is not registered.")
        else:
            st.warning("Please enter a Student ID.")


def _render_twin_warning(twin_info, student_id, student_name, reg_service):
    """Show twin/lookalike detection warning with comparison."""
    rec = twin_info["recommendation"]
    sim_pct = twin_info["max_similarity"] * 100
    twin_name = twin_info["twin_student_name"]
    twin_id = twin_info["twin_student_id"]

    if rec == "HIGH_RISK_TWIN":
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #b71c1c 0%, #880e0e 100%);
                border: 2px solid #ff5252;
                border-radius: 12px;
                padding: 20px;
                margin: 15px 0;
            ">
                <h3 style="margin:0; color:#ff8a80;">⚠️ High-Risk Twin/Lookalike Detected!</h3>
                <p style="color:#ffcdd2; margin:8px 0;">
                    <strong>{student_name}</strong>'s face is <strong>{sim_pct:.1f}% similar</strong> 
                    to <strong>{twin_name}</strong> (ID: {twin_id}).
                </p>
                <p style="color:#ef9a9a; font-size:0.9em;">
                    🔴 Very high similarity — some attendance detections may need teacher verification.<br>
                    📷 <strong>Recommended:</strong> Capture additional images from different angles to 
                    improve distinction accuracy.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #e65100 0%, #bf360c 100%);
                border: 2px solid #ff9800;
                border-radius: 12px;
                padding: 20px;
                margin: 15px 0;
            ">
                <h3 style="margin:0; color:#ffe0b2;">⚠️ Twin/Lookalike Detected</h3>
                <p style="color:#ffcc80; margin:8px 0;">
                    <strong>{student_name}</strong>'s face is <strong>{sim_pct:.1f}% similar</strong> 
                    to <strong>{twin_name}</strong> (ID: {twin_id}).
                </p>
                <p style="color:#ffab40; font-size:0.9em;">
                    The system will use enhanced matching for this pair.
                    Capturing additional angles can improve accuracy.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info(
        "💡 **For better accuracy**, capture additional images from different angles.\n\n"
        "Look **left**, **right**, **up**, and **slightly down** during capture."
    )

    if st.button(
        "📷 Capture More Angles",
        key="twin_recapture_btn",
        type="primary",
    ):
        st.session_state.twin_recapture_mode = True
        st.session_state.twin_recapture_student_id = student_id
        st.session_state.twin_extra_captures = []
        st.rerun()


def _render_twin_recapture(detector, reg_service):
    """UI for capturing extra angles for a twin-flagged student."""
    st.markdown("---")
    st.markdown("### 📷 Additional Angle Captures")

    sid = st.session_state.get("twin_recapture_student_id", "")
    from core.config import Config
    extra_needed = Config.TWIN_EXTRA_IMAGES
    extras = st.session_state.twin_extra_captures

    directions = [
        "Look LEFT", "Look RIGHT", "Look slightly UP",
        "Look slightly DOWN", "Tilt head LEFT",
        "Tilt head RIGHT", "Neutral (closer)",
        "Neutral (further)", "Slight smile", "Serious expression",
    ]

    current = len(extras)
    if current < extra_needed:
        direction = directions[current] if current < len(directions) else f"Angle {current+1}"
        st.info(f"**{direction}** — Capture {current+1}/{extra_needed}")

        cam_img = st.camera_input(
            f"📸 Extra capture {current+1}: {direction}",
            key=f"twin_extra_{current}",
        )

        if cam_img is not None:
            file_bytes = np.asarray(bytearray(cam_img.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if frame is not None:
                detection = detector.detect_single_face(frame)
                if detection is not None:
                    st.session_state.twin_extra_captures.append(frame)
                    st.success(f"✅ {direction} captured!")
                    st.rerun()
                else:
                    st.error("❌ No face detected. Try again.")
    else:
        st.success(f"✅ All {extra_needed} extra images captured!")

        if st.button("✅ Submit Extra Images", type="primary", width="stretch"):
            with st.spinner("🔄 Processing additional embeddings..."):
                result = reg_service.recapture_for_twin(sid, extras)

            if result["success"]:
                st.success(
                    f"✅ Added **{result['added_count']}** extra embeddings!\n\n"
                    f"Twin distinction accuracy has been improved."
                )
                st.session_state.twin_recapture_mode = False
                st.session_state.twin_extra_captures = []
            else:
                st.error("❌ Extra capture failed.")
                for issue in result.get("issues", []):
                    st.warning(f"⚠️ {issue}")

    if st.button("⏭️ Skip Extra Captures", key="skip_twin_extra"):
        st.session_state.twin_recapture_mode = False
        st.session_state.twin_extra_captures = []
        st.rerun()
