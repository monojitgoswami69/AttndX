"""
Pickle-based attendance record store.
Manages attendance sessions, per-check detection records, and final
present/late/absent determinations. Supports darkness-related statuses.
"""

import pickle
import datetime
import uuid
from pathlib import Path
from core.config import Config


class AttendanceStore:
    """Persistent attendance session storage using pickle."""

    def __init__(self, db_path=None):
        self.db_path = Path(db_path) if db_path else Config.ATTENDANCE_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.sessions = {}
        self.load()

    def load(self):
        if self.db_path.exists():
            try:
                with open(self.db_path, "rb") as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    self.sessions = loaded
                    print(f"[AttendanceStore] Loaded {len(self.sessions)} sessions.")
                    return True
            except Exception as e:
                print(f"[AttendanceStore] Error loading: {e}")
        else:
            print("[AttendanceStore] No existing records, starting fresh.")
        self.sessions = {}
        return False

    def save(self):
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.db_path, "wb") as f:
                pickle.dump(self.sessions, f, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        except Exception as e:
            print(f"[AttendanceStore] Error saving: {e}")
            return False

    def create_session(self, class_name):
        session_id = str(uuid.uuid4())[:8]
        now = datetime.datetime.now()
        self.sessions[session_id] = {
            "class_name": class_name,
            "date": now.strftime("%Y-%m-%d"),
            "start_time": now.isoformat(),
            "status": "in_progress",
            "checks": {},
            "final_results": {},
            "notes": [],
            "paused_duration": 0.0,
        }
        self.save()
        print(f"[AttendanceStore] Created session '{session_id}' for '{class_name}'.")
        return session_id

    def record_check(self, session_id, check_number, detected_student_ids,
                     snapshot_path=None, status="completed", note=None,
                     usable_frames=None, total_frames=None):
        """
        Record a check result.

        status: "completed" | "partial" | "failed_dark" | "skipped_dark" | "retried"
        """
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]
        now = datetime.datetime.now()

        check_data = {
            "time": now.isoformat(),
            "detected": detected_student_ids,
            "count": len(detected_student_ids),
            "status": status,
            "snapshot": snapshot_path,
        }

        if note:
            check_data["note"] = note
        if usable_frames is not None:
            check_data["usable_frames"] = usable_frames
        if total_frames is not None:
            check_data["total_frames"] = total_frames

        session["checks"][check_number] = check_data
        self.save()
        print(
            f"[AttendanceStore] Check {check_number} [{status}]: "
            f"detected {len(detected_student_ids)} students."
        )
        return True

    def update_check_status(self, session_id, check_number, status, note=None):
        """Update an existing check's status."""
        if session_id not in self.sessions:
            return False
        session = self.sessions[session_id]
        if check_number in session["checks"]:
            session["checks"][check_number]["status"] = status
            if note:
                session["checks"][check_number]["note"] = note
            self.save()
            return True
        return False

    def add_session_note(self, session_id, note):
        """Add a note to the session."""
        if session_id in self.sessions:
            if "notes" not in self.sessions[session_id]:
                self.sessions[session_id]["notes"] = []
            self.sessions[session_id]["notes"].append(note)
            self.save()

    def update_session_status(self, session_id, status):
        """Update session status (in_progress|paused|completed|stopped|cancelled_dark)."""
        if session_id in self.sessions:
            self.sessions[session_id]["status"] = status
            self.save()

    def add_paused_duration(self, session_id, duration):
        """Add paused time to the session's total paused duration."""
        if session_id in self.sessions:
            current = self.sessions[session_id].get("paused_duration", 0.0)
            self.sessions[session_id]["paused_duration"] = current + duration
            self.save()

    def compute_final(self, session_id, all_student_ids):
        """
        Compute final attendance. Handles edge cases:
        - Only valid checks (completed/partial/retried) count.
        - Dynamic threshold based on completed check count.
        - Adds notes for anomalies.
        """
        if session_id not in self.sessions:
            return {}

        session = self.sessions[session_id]
        checks = session["checks"]

        # Only count checks that actually produced results
        valid_checks = {
            cn: cd for cn, cd in checks.items()
            if cd.get("status") in ("completed", "partial", "retried")
        }
        num_valid = len(valid_checks)

        final_results = {}
        session_notes = list(session.get("notes", []))

        # Determine attendance rules based on valid checks
        if num_valid == 0:
            # No valid checks — cancel session
            session["status"] = "cancelled_dark"
            session_notes.append("Session cancelled: no valid checks due to darkness")
            for sid in all_student_ids:
                final_results[sid] = {
                    "checks_present": 0,
                    "status": "insufficient_data",
                    "note": "No valid checks — manual review needed",
                }
            session["final_results"] = final_results
            session["end_time"] = datetime.datetime.now().isoformat()
            session["notes"] = session_notes
            self.save()
            return final_results

        if num_valid == 1:
            # Only one check — insufficient for reliable attendance
            session_notes.append(
                f"Only {num_valid} valid check — results marked for manual review"
            )

        # Count skipped/failed checks for notes
        skipped = sum(1 for cd in checks.values() if cd.get("status") in ("failed_dark", "skipped_dark"))
        if skipped > 0:
            session_notes.append(f"{skipped} check(s) affected by darkness")

        # Dynamic threshold
        min_for_present = min(Config.MIN_CHECKS_FOR_PRESENT, num_valid)

        for student_id in all_student_ids:
            checks_present = 0
            for cn, cd in valid_checks.items():
                if student_id in cd.get("detected", []):
                    checks_present += 1

            note = ""
            if num_valid == 1:
                status = "insufficient_data"
                note = "Only 1 valid check — manual review needed"
            elif checks_present >= min_for_present:
                status = "present"
            elif checks_present == 1:
                status = "late"
            else:
                status = "absent"

            final_results[student_id] = {
                "checks_present": checks_present,
                "status": status,
                "note": note,
            }

        session["final_results"] = final_results
        if session["status"] not in ("cancelled_dark",):
            session["status"] = "completed"
        session["end_time"] = datetime.datetime.now().isoformat()
        session["notes"] = session_notes
        self.save()

        present_n = sum(1 for r in final_results.values() if r["status"] == "present")
        late_n = sum(1 for r in final_results.values() if r["status"] == "late")
        absent_n = sum(1 for r in final_results.values() if r["status"] == "absent")
        insuf_n = sum(1 for r in final_results.values() if r["status"] == "insufficient_data")
        print(
            f"[AttendanceStore] Final ({num_valid} valid checks): "
            f"{present_n} present, {late_n} late, {absent_n} absent, {insuf_n} review"
        )
        return final_results

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def get_all_sessions(self):
        return self.sessions

    def get_recent_sessions(self, limit=10):
        s = sorted(self.sessions.items(), key=lambda x: x[1].get("start_time", ""), reverse=True)
        return s[:limit]

    def delete_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.save()
            return True
        return False
