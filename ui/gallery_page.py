"""
Registered Students Gallery Page.
Displays all registered students with face images, metadata, and delete controls.
"""

import streamlit as st
import cv2
import os


def render_gallery_page(face_db, reg_service):
    """Render the student gallery page."""

    st.markdown("## 👥 Registered Students")

    students = face_db.get_all_students()
    count = face_db.get_student_count()

    if count == 0:
        st.markdown("---")
        st.markdown(
            "<div style='text-align:center; padding:60px 20px;'>"
            "<h3>😔 No students registered yet</h3>"
            "<p style='color:#888;'>Go to the <b>Register</b> tab to add students.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    st.info(f"📊 **{count}** student{'s' if count != 1 else ''} registered")
    st.markdown("---")

    # Confirmation state
    if "delete_confirm" not in st.session_state:
        st.session_state.delete_confirm = None

    # Display in a grid (3 columns)
    cols_per_row = 3
    student_list = list(students.items())

    for row_start in range(0, len(student_list), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            idx = row_start + col_idx
            if idx >= len(student_list):
                break

            sid, data = student_list[idx]
            with cols[col_idx]:
                # Card container
                with st.container(border=True):
                    # Face image
                    images = face_db.get_student_face_images(sid)
                    if images:
                        rgb = cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB)
                        st.image(rgb, width="stretch")
                    else:
                        st.markdown(
                            "<div style='height:150px;background:#2a2a2a;"
                            "display:flex;align-items:center;justify-content:center;"
                            "border-radius:8px;'>"
                            "<span style='font-size:48px;'>👤</span></div>",
                            unsafe_allow_html=True,
                        )

                    # Student info
                    st.markdown(f"### {data['name']}")
                    st.caption(f"🆔 {sid}")
                    emb_count = len(data.get("embeddings", []))
                    st.caption(f"🧠 {emb_count} embeddings")
                    reg_at = data.get("registered_at", "N/A")
                    if reg_at != "N/A" and len(reg_at) > 10:
                        reg_at = reg_at[:10]  # Show date only
                    st.caption(f"📅 {reg_at}")

                    # Delete button with confirmation
                    if st.session_state.delete_confirm == sid:
                        st.warning(f"Delete **{data['name']}**?")
                        dc1, dc2 = st.columns(2)
                        with dc1:
                            if st.button("✅ Yes", key=f"del_yes_{sid}", width="stretch"):
                                reg_service.delete_student(sid)
                                st.session_state.delete_confirm = None
                                st.rerun()
                        with dc2:
                            if st.button("❌ No", key=f"del_no_{sid}", width="stretch"):
                                st.session_state.delete_confirm = None
                                st.rerun()
                    else:
                        if st.button(
                            "🗑️ Delete",
                            key=f"del_{sid}",
                            width="stretch",
                        ):
                            st.session_state.delete_confirm = sid
                            st.rerun()

    # Additional images viewer
    st.markdown("---")
    st.markdown("### 🔍 View All Faces for a Student")
    selected_id = st.selectbox(
        "Select student",
        options=[""] + [f"{sid} — {d['name']}" for sid, d in students.items()],
        key="gallery_select",
    )

    if selected_id and " — " in selected_id:
        sel_sid = selected_id.split(" — ")[0]
        images = face_db.get_student_face_images(sel_sid)
        if images:
            img_cols = st.columns(min(len(images), 5))
            for i, img in enumerate(images):
                with img_cols[i % 5]:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    st.image(rgb, caption=f"Face #{i+1}", width="stretch")
        else:
            st.info("No saved face images found.")
