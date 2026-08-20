"""Enrollment page for gen2."""
import streamlit as st
import cv2
import numpy as np
import time

from gen2.config import Config
from gen2.enrollment.service import EnrollmentService, EnrollmentStatus


def render_enrollment_page(rt):
    st.markdown("## 📸 Student Enrollment")
    st.markdown("Capture high-quality face samples to create a reliable biometric template.")

    # Create enrollment service
    enroll_svc = EnrollmentService(
        detector=rt.detector,
        aligner=rt.aligner,
        quality_assessor=rt.quality_assessor,
        embedder=rt.embedder,
        db=rt.biometric_db,
        index=rt.identity_index,
    )

    # Input fields
    col1, col2 = st.columns(2)
    with col1:
        student_id = st.text_input("Student ID", placeholder="e.g. STU001",
                                    key="enr_id")
    with col2:
        name = st.text_input("Name", placeholder="e.g. Alice", key="enr_name")

    # Session state for captures
    if "enr_captures" not in st.session_state:
        st.session_state.enr_captures = []
    if "enr_qualities" not in st.session_state:
        st.session_state.enr_qualities = []

    max_samples = Config.get("enrollment", "max_samples")
    min_samples = Config.get("enrollment", "min_samples")
    captures = st.session_state.enr_captures

    st.markdown(f"**Captured: {len(captures)}/{max_samples}** (need {min_samples} minimum)")
    st.progress(min(len(captures) / max_samples, 1.0))

    # Camera capture
    if len(captures) < max_samples:
        img = st.camera_input(f"Capture {len(captures)+1}/{max_samples}",
                              key=f"enr_cam_{len(captures)}")
        if img is not None:
            file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if frame is not None:
                # Quick quality preview
                detections = rt.detector.detect(frame)
                if len(detections) == 0:
                    st.error("No face detected. Try again.")
                else:
                    det = max(detections, key=lambda d: d.confidence)
                    aligned = rt.aligner.align(frame, det)
                    q = rt.quality_assessor.assess(det, aligned)
                    st.session_state.enr_captures.append(frame)
                    st.session_state.enr_qualities.append(q)
                    color = "🟢" if q.overall_score >= 0.6 else "🟡" if q.overall_score >= 0.45 else "🔴"
                    st.success(f"{color} Captured (quality: {q.overall_score:.0%})")
                    if not q.accepted:
                        st.warning(f"Quality issue: {q.reason}")
                    st.rerun()

    # Show thumbnails
    if captures:
        st.markdown("### Captured Samples")
        cols = st.columns(min(len(captures), 5))
        for i, (cap, q) in enumerate(zip(captures, st.session_state.enr_qualities)):
            with cols[i % 5]:
                rgb = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
                st.image(rgb, caption=f"#{i+1} Q:{q.overall_score:.0%}",
                         use_container_width=True)

    # Register button
    st.markdown("---")
    can_register = (len(captures) >= min_samples
                    and student_id.strip()
                    and name.strip())
    c1, c2 = st.columns([3, 1])
    with c1:
        if st.button("✅ Enroll Student", type="primary",
                      disabled=not can_register, use_container_width=True):
            with st.spinner("Processing embeddings..."):
                result = enroll_svc.enroll(
                    name=name.strip(),
                    frames=captures,
                    identity_id=student_id.strip(),
                )
            if result.status == EnrollmentStatus.SUCCESS:
                st.balloons()
                st.success(
                    f"✅ **{name}** enrolled! "
                    f"ID: {result.identity_id}, "
                    f"Samples: {result.samples_stored}, "
                    f"Intra-sim: {result.intra_similarity:.1%}"
                )
                st.session_state.enr_captures = []
                st.session_state.enr_qualities = []
            else:
                st.error(f"Enrollment failed: {result.status.value}")
                for issue in result.issues:
                    st.warning(f"⚠️ {issue}")
    with c2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.enr_captures = []
            st.session_state.enr_qualities = []
            st.rerun()
