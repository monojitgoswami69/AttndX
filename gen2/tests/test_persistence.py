"""
Test persistence — SQLite save/load, corruption recovery, index rebuild.

Uses temporary databases and synthetic embeddings.
"""
import sys
import os
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pytest

from gen2.config import Config
from gen2.storage.db import BiometricDB
from gen2.storage.attendance_db import AttendanceDB
from gen2.recognition.matching.engine import IdentityIndex


def make_unit_vec(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def tmp_db_path(tmp_path):
    return tmp_path / "test_biometric.db"


@pytest.fixture
def tmp_attendance_db(tmp_path):
    return tmp_path / "test_attendance.db"


class TestBiometricDB:
    def test_add_and_get_identity(self, tmp_db_path):
        db = BiometricDB(tmp_db_path)
        Config.load()
        pv = Config.pipeline_version_string()
        db.add_identity("STU001", "Alice", pv)
        ident = db.get_identity("STU001")
        assert ident is not None
        assert ident["name"] == "Alice"

    def test_identity_exists(self, tmp_db_path):
        db = BiometricDB(tmp_db_path)
        Config.load()
        pv = Config.pipeline_version_string()
        assert not db.identity_exists("STU001")
        db.add_identity("STU001", "Alice", pv)
        assert db.identity_exists("STU001")

    def test_add_and_get_template(self, tmp_db_path):
        db = BiometricDB(tmp_db_path)
        Config.load()
        pv = Config.pipeline_version_string()
        db.add_identity("STU001", "Alice", pv)
        vec = make_unit_vec(42)
        db.set_template("STU001", vec, 5)
        stored = db.get_template("STU001")
        assert stored is not None
        assert np.allclose(stored, vec, atol=1e-5)

    def test_add_embedding_validates(self, tmp_db_path):
        """Malformed embeddings must be rejected."""
        db = BiometricDB(tmp_db_path)
        Config.load()
        pv = Config.pipeline_version_string()
        db.add_identity("STU001", "Alice", pv)
        # Wrong shape
        bad = np.zeros(256, dtype=np.float32)
        assert not db.add_embedding("STU001", bad)
        # NaN
        bad2 = np.full(512, np.nan, dtype=np.float32)
        assert not db.add_embedding("STU001", bad2)

    def test_delete_identity_cascade(self, tmp_db_path):
        db = BiometricDB(tmp_db_path)
        Config.load()
        pv = Config.pipeline_version_string()
        db.add_identity("STU001", "Alice", pv)
        vec = make_unit_vec(1)
        db.set_template("STU001", vec, 5)
        db.add_embedding("STU001", vec)
        assert db.delete_identity("STU001")
        assert not db.identity_exists("STU001")
        assert db.get_template("STU001") is None

    def test_get_all_templates(self, tmp_db_path):
        db = BiometricDB(tmp_db_path)
        Config.load()
        pv = Config.pipeline_version_string()
        db.add_identity("A", "Alice", pv)
        db.add_identity("B", "Bob", pv)
        va, vb = make_unit_vec(1), make_unit_vec(2)
        db.set_template("A", va, 5)
        db.set_template("B", vb, 5)
        templates = db.get_all_templates()
        assert len(templates) == 2
        ids = [t[0] for t in templates]
        assert "A" in ids and "B" in ids

    def test_corruption_recovery(self, tmp_db_path):
        """If DB is corrupted, safe_open backs it up and starts fresh."""
        Config.load()
        pv = Config.pipeline_version_string()
        # Write valid data first
        db = BiometricDB(tmp_db_path)
        db.add_identity("STU001", "Alice", pv)
        db.close() if hasattr(db, 'close') else None

        # Corrupt the file
        with open(tmp_db_path, "w") as f:
            f.write("CORRUPTED DATABASE FILE CONTENT")

        # safe_open should recover
        db2 = BiometricDB.safe_open(tmp_db_path)
        assert db2.count_identities() == 0  # fresh DB
        # Corrupt backup should exist
        backups = list(tmp_db_path.parent.glob("*.corrupt.*.db"))
        assert len(backups) >= 1

    def test_index_rebuild_from_db(self, tmp_db_path):
        """The index must be fully rebuildable from the DB."""
        Config.load()
        pv = Config.pipeline_version_string()
        db = BiometricDB(tmp_db_path)
        db.add_identity("A", "Alice", pv)
        db.add_identity("B", "Bob", pv)
        va, vb = make_unit_vec(1), make_unit_vec(2)
        db.set_template("A", va, 5)
        db.set_template("B", vb, 5)

        # Build index from DB
        templates = db.get_all_templates()
        names = {i["identity_id"]: i["name"]
                 for i in db.get_all_identities()}
        index = IdentityIndex()
        index.rebuild(templates, names)
        assert index.size == 2

    def test_atomic_write_survives_restart(self, tmp_db_path):
        """Data persists across DB reopen."""
        Config.load()
        pv = Config.pipeline_version_string()
        db1 = BiometricDB(tmp_db_path)
        db1.add_identity("X", "Xavier", pv)
        db1.set_template("X", make_unit_vec(99), 5)

        # Reopen
        db2 = BiometricDB(tmp_db_path)
        assert db2.identity_exists("X")
        t = db2.get_template("X")
        assert t is not None
        assert np.allclose(t, make_unit_vec(99), atol=1e-5)


class TestAttendanceDB:
    def test_create_session(self, tmp_attendance_db):
        db = AttendanceDB(tmp_attendance_db)
        sid = db.create_session("CS101")
        assert sid is not None
        session = db.get_session(sid)
        assert session["class_name"] == "CS101"
        assert session["status"] == "in_progress"

    def test_record_check_idempotent(self, tmp_attendance_db):
        """Recording the same check twice overwrites (not duplicates)."""
        db = AttendanceDB(tmp_attendance_db)
        sid = db.create_session("CS101")
        db.record_check(sid, 1, ["A", "B"], status="completed")
        db.record_check(sid, 1, ["A", "C"], status="completed")
        checks = db.get_checks(sid)
        assert len(checks) == 1  # not 2 — overwrite
        assert set(checks[1]["detected"]) == {"A", "C"}

    def test_attendance_unique_constraint(self, tmp_attendance_db):
        """One attendance record per student per session (DB-level idempotency)."""
        db = AttendanceDB(tmp_attendance_db)
        sid = db.create_session("CS101")
        # Insert twice — must overwrite, not duplicate
        db.upsert_attendance(sid, "A", "Alice", "present", 3)
        db.upsert_attendance(sid, "A", "Alice", "absent", 0)
        records = db.get_attendance(sid)
        assert len(records) == 1
        assert records[0]["status"] == "absent"  # latest write

    def test_multiple_students(self, tmp_attendance_db):
        db = AttendanceDB(tmp_attendance_db)
        sid = db.create_session("CS101")
        db.upsert_attendance(sid, "A", "Alice", "present", 3)
        db.upsert_attendance(sid, "B", "Bob", "absent", 0)
        records = db.get_attendance(sid)
        assert len(records) == 2

    def test_delete_session_cascade(self, tmp_attendance_db):
        db = AttendanceDB(tmp_attendance_db)
        sid = db.create_session("CS101")
        db.record_check(sid, 1, ["A"])
        db.upsert_attendance(sid, "A", "Alice", "present")
        db.delete_session(sid)
        assert db.get_session(sid) is None
        assert len(db.get_checks(sid)) == 0
        assert len(db.get_attendance(sid)) == 0
