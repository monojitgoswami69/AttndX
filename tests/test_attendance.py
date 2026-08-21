"""
Test attendance — idempotency, duplicate attendance prevention,
multiple students, restart behavior.

Uses the AttendanceDB directly with synthetic data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest

from config import Config
from storage.attendance_db import AttendanceDB


@pytest.fixture
def db(tmp_path):
    return AttendanceDB(tmp_path / "test_att.db")


class TestAttendanceIdempotency:
    def test_one_record_per_student_per_session(self, db):
        """DB UNIQUE(session_id, identity_id) prevents duplicates."""
        sid = db.create_session("CS101")
        db.upsert_attendance(sid, "A", "Alice", "present", 3)
        # Insert again — must overwrite, not duplicate
        db.upsert_attendance(sid, "A", "Alice", "present", 3)
        records = db.get_attendance(sid)
        assert len(records) == 1

    def test_multiple_students_in_session(self, db):
        sid = db.create_session("CS101")
        for i in range(10):
            db.upsert_attendance(sid, f"STU{i:03d}", f"Student{i}",
                                "present" if i % 2 == 0 else "absent", i % 3)
        records = db.get_attendance(sid)
        assert len(records) == 10

    def test_check_idempotent(self, db):
        """Recording the same check_number twice overwrites."""
        sid = db.create_session("CS101")
        db.record_check(sid, 1, ["A", "B"], status="completed")
        db.record_check(sid, 1, ["A", "C"], status="completed")
        checks = db.get_checks(sid)
        assert len(checks) == 1
        assert set(checks[1]["detected"]) == {"A", "C"}

    def test_different_sessions_independent(self, db):
        """Two sessions for the same student are independent."""
        sid1 = db.create_session("CS101")
        sid2 = db.create_session("CS101")
        db.upsert_attendance(sid1, "A", "Alice", "present", 3)
        db.upsert_attendance(sid2, "A", "Alice", "absent", 0)
        r1 = db.get_attendance(sid1)
        r2 = db.get_attendance(sid2)
        assert len(r1) == 1 and len(r2) == 1
        assert r1[0]["status"] == "present"
        assert r2[0]["status"] == "absent"

    def test_session_persists_across_reopen(self, tmp_path):
        """Attendance records survive a DB reopen."""
        path = tmp_path / "test_att.db"
        db1 = AttendanceDB(path)
        sid = db1.create_session("CS101")
        db1.upsert_attendance(sid, "A", "Alice", "present", 3)
        db1.record_check(sid, 1, ["A"])

        # Reopen
        db2 = AttendanceDB(path)
        assert db2.get_session(sid) is not None
        assert len(db2.get_attendance(sid)) == 1
        assert len(db2.get_checks(sid)) == 1

    def test_spoofed_not_counted_as_present(self, db):
        """A student in the spoofed list should be distinguishable
        from a student in the detected list."""
        sid = db.create_session("CS101")
        db.record_check(sid, 1, ["A"], spoofed_ids=["B"], status="completed")
        checks = db.get_checks(sid)
        assert "A" in checks[1]["detected"]
        assert "B" in checks[1]["spoofed"]
        assert "B" not in checks[1]["detected"]

    def test_delete_session_removes_all(self, db):
        sid = db.create_session("CS101")
        db.record_check(sid, 1, ["A"])
        db.upsert_attendance(sid, "A", "Alice", "present")
        db.add_note(sid, "test note")
        assert db.delete_session(sid)
        assert db.get_session(sid) is None
        assert len(db.get_checks(sid)) == 0
        assert len(db.get_attendance(sid)) == 0
        assert len(db.get_notes(sid)) == 0
