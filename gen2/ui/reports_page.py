"""Reports page — past attendance sessions."""
import streamlit as st
import pandas as pd


def render_reports_page(rt):
    st.markdown("## 📊 Attendance Reports")

    sessions = rt.attendance_db.get_all_sessions()
    if not sessions:
        st.info("No attendance sessions recorded.")
        return

    st.metric("Total Sessions", len(sessions))

    for session in sessions:
        sid = session["session_id"]
        attendance = rt.attendance_db.get_attendance(sid)
        checks = rt.attendance_db.get_checks(sid)
        notes = rt.attendance_db.get_notes(sid)

        badge = {"completed": "✅", "stopped": "⏹️",
                 "in_progress": "🔴", "cancelled_no_data": "❌"}.get(
                     session["status"], "❓")

        with st.expander(f"{badge} {session['class_name']} — {session.get('date','?')} "
                        f"({len(attendance)} students)"):
            if notes:
                with st.expander("📝 Notes"):
                    for n in notes:
                        st.caption(f"• {n}")

            if attendance:
                rows = []
                for a in attendance:
                    rows.append({
                        "Student": a["name"],
                        "Status": a["status"].upper(),
                        "Present": a["checks_present"],
                        "Spoofed": a["checks_spoofed"],
                        "Note": a["note"],
                    })
                df = pd.DataFrame(rows)
                st.dataframe(df, width="stretch", hide_index=True)
                st.download_button("📥 CSV", df.to_csv(index=False),
                                  file_name=f"attendance_{sid}.csv",
                                  key=f"dl_{sid}")

            if checks:
                st.caption("**Check details:**")
                for cn in sorted(checks.keys()):
                    cd = checks[cn]
                    st.caption(f"  Check {cn}: {len(cd['detected'])} detected, "
                              f"{len(cd.get('spoofed',[]))} spoofed [{cd['status']}]")
