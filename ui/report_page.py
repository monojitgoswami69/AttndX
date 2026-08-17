"""
Attendance Reports Page.
Displays past attendance sessions with summaries and expandable detail tables.
"""

import streamlit as st
import pandas as pd


def render_report_page(attendance_store, face_db):
    """Render the attendance reports page."""

    st.markdown("## 📊 Attendance Reports")

    sessions = attendance_store.get_all_sessions()

    if not sessions:
        st.markdown("---")
        st.markdown(
            "<div style='text-align:center; padding:60px 20px;'>"
            "<h3>📭 No attendance sessions yet</h3>"
            "<p style='color:#888;'>Start an attendance session in the "
            "<b>Attendance</b> tab to generate reports.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # Sort sessions by date (most recent first)
    sorted_sessions = sorted(
        sessions.items(),
        key=lambda x: x[1].get("start_time", ""),
        reverse=True,
    )

    st.info(f"📁 **{len(sorted_sessions)}** session{'s' if len(sorted_sessions) != 1 else ''} recorded")
    st.markdown("---")

    students = face_db.get_all_students()

    for session_id, session_data in sorted_sessions:
        class_name = session_data.get("class_name", "Unknown Class")
        date = session_data.get("date", "N/A")
        status = session_data.get("status", "unknown")
        start_time = session_data.get("start_time", "")
        end_time = session_data.get("end_time", "")
        checks = session_data.get("checks", {})
        final = session_data.get("final_results", {})

        # Count statuses
        present = sum(1 for r in final.values() if r.get("status") == "present")
        absent = sum(1 for r in final.values() if r.get("status") == "absent")
        spoofed = sum(1 for r in final.values() if r.get("checks_spoofed", 0) > 0)
        total = len(final)

        # Status badge
        if status == "completed":
            badge = "✅ Completed"
        elif status == "stopped":
            badge = "⏹️ Stopped Early"
        elif status == "in_progress":
            badge = "🔴 In Progress"
        else:
            badge = f"❓ {status}"

        # Session card
        with st.expander(
            f"📚 **{class_name}** — {date} | "
            f"✅{present} ❌{absent} | {badge}",
            expanded=False,
        ):
            # Metadata
            mc1, mc2, mc3 = st.columns(3)
            mc1.markdown(f"**Session ID:** `{session_id}`")
            mc2.markdown(f"**Checks:** {len(checks)}")
            if start_time:
                time_str = start_time[11:19] if len(start_time) > 18 else start_time
                mc3.markdown(f"**Started:** {time_str}")

            # Summary metrics
            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("👥 Total", total)
            sm2.metric("✅ Present", present)
            sm3.metric("❌ Absent", absent)
            sm4.metric("🚫 Spoofed", spoofed)

            # Detailed results table
            if final:
                st.markdown("#### 📋 Detailed Results")

                rows = []
                for sid, result in final.items():
                    name = result.get("name", students.get(sid, {}).get("name", sid))
                    checks_present = result.get("checks_present", 0)
                    checks_spoofed = result.get("checks_spoofed", 0)
                    att_status = result.get("status", "unknown").upper()

                    # Per-check breakdown
                    check_marks = {}
                    for cn_str, cd in checks.items():
                        cn = int(cn_str)
                        detected = cd.get("detected", [])
                        spoofed = cd.get("spoofed", [])
                        if sid in spoofed:
                            check_marks[f"Check {cn}"] = "🚫"
                        elif sid in detected:
                            check_marks[f"Check {cn}"] = "✅"
                        else:
                            check_marks[f"Check {cn}"] = "❌"

                    row = {"Student": name, "ID": sid}
                    row.update(check_marks)
                    row["Checks Present"] = f"{checks_present}/{len(checks)}"
                    if checks_spoofed > 0:
                        row["Spoofed"] = f"{checks_spoofed}/{len(checks)}"
                    else:
                        row["Spoofed"] = "0"
                    row["Status"] = att_status
                    rows.append(row)

                df = pd.DataFrame(rows)

                def color_status(val):
                    if val == "PRESENT":
                        return "background-color: #1b5e20; color: #a5d6a7"
                    elif val == "ABSENT":
                        return "background-color: #b71c1c; color: #ef9a9a"
                    elif val == "INSUFFICIENT_DATA":
                        return "background-color: #4a148c; color: #ce93d8"
                    return ""

                if hasattr(df.style, "map"):
                    styled = df.style.map(color_status, subset=["Status"])
                elif hasattr(df.style, "applymap"):
                    styled = df.style.applymap(color_status, subset=["Status"])
                else:
                    styled = df
                st.dataframe(styled, width="stretch", hide_index=True)

                # Download CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    data=csv,
                    file_name=f"attendance_{session_id}_{date}.csv",
                    mime="text/csv",
                    key=f"dl_{session_id}",
                )
            else:
                st.info("No detailed results available for this session.")

            # Check timing details
            if checks:
                st.markdown("#### ⏱️ Check Timings")
                for cn_str, cd in sorted(checks.items(), key=lambda x: int(x[0])):
                    cn = int(cn_str)
                    check_time = cd.get("time", "N/A")
                    if len(check_time) > 18:
                        check_time = check_time[11:19]
                    detected_count = cd.get("count", 0)
                    st.caption(
                        f"Check {cn} at {check_time} — "
                        f"Detected {detected_count} student{'s' if detected_count != 1 else ''}"
                    )

            # Delete session
            st.markdown("---")
            if st.button(
                f"🗑️ Delete Session",
                key=f"del_session_{session_id}",
            ):
                attendance_store.delete_session(session_id)
                st.rerun()
