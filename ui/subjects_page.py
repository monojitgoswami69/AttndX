"""Subjects management page — add, edit, delete subjects."""
import streamlit as st


def render_subjects_page(rt):
    st.markdown("## 📚 Subjects")

    subjects = rt.attendance_db.get_all_subjects()

    # ─── Add subject ───
    with st.expander("➕ Add New Subject", expanded=False):
        col1, col2 = st.columns([2, 1])
        with col1:
            new_name = st.text_input("Subject Name", placeholder="e.g. Data Structures",
                                     key="subj_new_name")
        with col2:
            new_code = st.text_input("Subject Code (optional)",
                                     placeholder="e.g. CS201",
                                     key="subj_new_code")

        if st.button("✅ Add Subject", type="primary", width="stretch"):
            if not new_name.strip():
                st.error("Subject name is required.")
            elif rt.attendance_db.subject_exists(new_name, new_code):
                st.error(f"A subject with this name/code already exists.")
            else:
                result = rt.attendance_db.add_subject(new_name, new_code)
                if result:
                    st.success(f"✅ Added: {result['name']}")
                    st.rerun()
                else:
                    st.error("Failed to add subject.")

    st.markdown("---")

    # ─── List subjects ───
    if not subjects:
        st.info("No subjects yet. Add one above to get started.")
        return

    st.metric("Total Subjects", len(subjects))

    if "editing_subject" not in st.session_state:
        st.session_state.editing_subject = None
    if "delete_confirm" not in st.session_state:
        st.session_state.delete_confirm = None

    for subj in subjects:
        sid = subj["subject_id"]
        with st.container(border=True):
            col1, col2, col3 = st.columns([3, 2, 2])

            with col1:
                if st.session_state.editing_subject == sid:
                    edit_name = st.text_input("Name", value=subj["name"],
                                             key=f"edit_name_{sid}")
                    edit_code = st.text_input("Code", value=subj.get("code", ""),
                                             key=f"edit_code_{sid}")
                else:
                    st.markdown(f"### {subj['name']}")
                    if subj.get("code"):
                        st.caption(f"Code: {subj['code']}")
                    st.caption(f"ID: {sid}")

            with col2:
                if st.session_state.editing_subject == sid:
                    if st.button("💾 Save", key=f"save_{sid}", width="stretch"):
                        if not edit_name.strip():
                            st.error("Name cannot be empty.")
                        else:
                            rt.attendance_db.update_subject(
                                sid, edit_name, edit_code
                            )
                            st.session_state.editing_subject = None
                            st.rerun()
                    if st.button("Cancel", key=f"cancel_{sid}", width="stretch"):
                        st.session_state.editing_subject = None
                        st.rerun()
                elif st.session_state.delete_confirm == sid:
                    st.warning("Delete?")
                    dc1, dc2 = st.columns(2)
                    with dc1:
                        if st.button("Yes", key=f"del_y_{sid}", width="stretch"):
                            rt.attendance_db.delete_subject(sid)
                            st.session_state.delete_confirm = None
                            st.rerun()
                    with dc2:
                        if st.button("No", key=f"del_n_{sid}", width="stretch"):
                            st.session_state.delete_confirm = None
                            st.rerun()

            with col3:
                if st.session_state.editing_subject != sid and \
                   st.session_state.delete_confirm != sid:
                    if st.button("✏️ Edit", key=f"edit_{sid}", width="stretch"):
                        st.session_state.editing_subject = sid
                        st.rerun()
                    if st.button("🗑️ Delete", key=f"del_{sid}", width="stretch"):
                        st.session_state.delete_confirm = sid
                        st.rerun()
