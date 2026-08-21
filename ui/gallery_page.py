"""Gallery page — view and manage enrolled identities."""
import streamlit as st
import cv2
import numpy as np


def render_gallery_page(rt):
    st.markdown("## 👥 Enrolled Identities")

    identities = rt.biometric_db.get_all_identities()
    if not identities:
        st.info("No identities enrolled yet.")
        return

    st.metric("Total", len(identities))

    if "del_confirm" not in st.session_state:
        st.session_state.del_confirm = None

    cols_per_row = 3
    for i in range(0, len(identities), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            if i + j >= len(identities):
                break
            ident = identities[i + j]
            with cols[j]:
                with st.container(border=True):
                    # Display thumbnail from captures if available
                    from config import Config
                    cap_dir = Config.captures_dir() / ident["identity_id"]
                    thumb_path = cap_dir / "aligned_01.png"
                    if not thumb_path.exists():
                        thumb_path = cap_dir / "raw_01.jpg"

                    if thumb_path.exists():
                        try:
                            img = cv2.imread(str(thumb_path))
                            if img is not None:
                                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                st.image(rgb, width=120)
                        except Exception:
                            pass

                    st.markdown(f"### {ident['name']}")
                    st.caption(f"ID: {ident['identity_id']}")
                    st.caption(f"Pipeline: {ident['pipeline_version'][:30]}...")
                    st.caption(f"Enrolled: {ident['enrolled_at'][:10]}")

                    if st.session_state.del_confirm == ident["identity_id"]:
                        st.warning(f"Delete {ident['name']}?")
                        dc1, dc2 = st.columns(2)
                        with dc1:
                            if st.button("Yes", key=f"del_y_{ident['identity_id']}"):
                                from enrollment.service import EnrollmentService
                                es = EnrollmentService(
                                    rt.detector, rt.aligner, rt.quality_assessor,
                                    rt.embedder, rt.biometric_db, rt.identity_index,
                                )
                                es.delete_identity(ident["identity_id"])
                                st.session_state.del_confirm = None
                                st.rerun()
                        with dc2:
                            if st.button("No", key=f"del_n_{ident['identity_id']}"):
                                st.session_state.del_confirm = None
                                st.rerun()
                    else:
                        if st.button("🗑️ Delete",
                                    key=f"del_{ident['identity_id']}"):
                            st.session_state.del_confirm = ident["identity_id"]
                            st.rerun()
