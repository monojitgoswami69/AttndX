"""
Attendance database — SQLite with WAL mode.

Stores:
  - sessions (session_id, class_name, start_time, end_time, status, config)
  - checks (check_number, session_id, detected_ids JSON, status, snapshot, ...)
  - attendance (session_id, identity_id, status, checks_present, checks_spoofed, ...)
  - notes (session_id, note, created_at)

UNIQUE constraint on (session_id, identity_id) in attendance table
guarantees one attendance record per student per session = idempotency
at the DB level.
"""
import json
import logging
import sqlite3
import uuid
import datetime
from pathlib import Path
from typing import Optional

from gen2.config import Config

logger = logging.getLogger(__name__)


class AttendanceDB:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        class_name TEXT NOT NULL,
        start_time TEXT NOT NULL,
        end_time TEXT,
        status TEXT NOT NULL DEFAULT 'in_progress',
        mode TEXT NOT NULL DEFAULT 'demo',
        paused_duration REAL DEFAULT 0.0,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS checks (
        session_id TEXT NOT NULL,
        check_number INTEGER NOT NULL,
        time TEXT NOT NULL,
        detected_ids TEXT NOT NULL DEFAULT '[]',
        spoofed_ids TEXT NOT NULL DEFAULT '[]',
        status TEXT NOT NULL DEFAULT 'completed',
        snapshot_path TEXT,
        note TEXT,
        usable_frames INTEGER,
        total_frames INTEGER,
        PRIMARY KEY (session_id, check_number),
        FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS attendance (
        session_id TEXT NOT NULL,
        identity_id TEXT NOT NULL,
        name TEXT NOT NULL,
        status TEXT NOT NULL,
        checks_present INTEGER NOT NULL DEFAULT 0,
        checks_spoofed INTEGER NOT NULL DEFAULT 0,
        needs_review INTEGER NOT NULL DEFAULT 0,
        note TEXT DEFAULT '',
        PRIMARY KEY (session_id, identity_id),
        FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS session_notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        note TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS subjects (
        subject_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        code TEXT DEFAULT '',
        created_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_checks_session ON checks(session_id);
    CREATE INDEX IF NOT EXISTS idx_attendance_session ON attendance(session_id);
    CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
    """

    # ─── Subject Migrations ───
    # Add subjects table if upgrading from an older schema
    SUBJECT_MIGRATION = """
    CREATE TABLE IF NOT EXISTS subjects (
        subject_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        code TEXT DEFAULT '',
        created_at TEXT NOT NULL
    );
    """

    def __init__(self, db_path: Path | str | None = None):
        if db_path is None:
            db_path = Config.attendance_db_path()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10.0,
            isolation_level="DEFERRED",
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._connect()
        conn.executescript(self.SCHEMA)
        conn.commit()
        conn.close()

    # ─── Sessions ───

    def create_session(self, class_name: str, mode: str = "demo") -> str:
        session_id = str(uuid.uuid4())[:12]
        now = _now_iso()
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO sessions (session_id, class_name, start_time, status, mode, created_at) "
                "VALUES (?, ?, ?, 'in_progress', ?, ?)",
                (session_id, class_name, now, mode, now)
            )
            conn.commit()
            logger.info(f"Created session {session_id} for '{class_name}'")
            return session_id
        finally:
            conn.close()

    def get_session(self, session_id: str) -> Optional[dict]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_all_sessions(self) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY start_time DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def update_session_status(self, session_id: str, status: str,
                              end_time: Optional[str] = None):
        conn = self._connect()
        try:
            if end_time:
                conn.execute(
                    "UPDATE sessions SET status=?, end_time=? WHERE session_id=?",
                    (status, end_time, session_id)
                )
            else:
                conn.execute(
                    "UPDATE sessions SET status=? WHERE session_id=?",
                    (status, session_id)
                )
            conn.commit()
        finally:
            conn.close()

    def add_paused_duration(self, session_id: str, seconds: float):
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE sessions SET paused_duration = paused_duration + ? WHERE session_id=?",
                (seconds, session_id)
            )
            conn.commit()
        finally:
            conn.close()

    def delete_session(self, session_id: str) -> bool:
        conn = self._connect()
        try:
            cur = conn.execute(
                "DELETE FROM sessions WHERE session_id=?", (session_id,)
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    # ─── Checks ───

    def record_check(self, session_id: str, check_number: int,
                     detected_ids: list[str], spoofed_ids: list[str] | None = None,
                     status: str = "completed", snapshot_path: str | None = None,
                     note: str | None = None,
                     usable_frames: int | None = None,
                     total_frames: int | None = None) -> bool:
        """Record a check result. Overwrites if check_number already exists
        for this session (idempotent at the check level)."""
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO checks "
                "(session_id, check_number, time, detected_ids, spoofed_ids, status, "
                "snapshot_path, note, usable_frames, total_frames) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, check_number, _now_iso(),
                 json.dumps(detected_ids),
                 json.dumps(spoofed_ids or []),
                 status, snapshot_path, note, usable_frames, total_frames)
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def get_checks(self, session_id: str) -> dict[int, dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM checks WHERE session_id=? ORDER BY check_number",
                (session_id,)
            ).fetchall()
            result = {}
            for r in rows:
                d = dict(r)
                d["detected"] = json.loads(d["detected_ids"])
                d["spoofed"] = json.loads(d["spoofed_ids"])
                result[d["check_number"]] = d
            return result
        finally:
            conn.close()

    def update_check_status(self, session_id: str, check_number: int,
                             status: str, note: str | None = None):
        conn = self._connect()
        try:
            if note:
                conn.execute(
                    "UPDATE checks SET status=?, note=? WHERE session_id=? AND check_number=?",
                    (status, note, session_id, check_number)
                )
            else:
                conn.execute(
                    "UPDATE checks SET status=? WHERE session_id=? AND check_number=?",
                    (status, session_id, check_number)
                )
            conn.commit()
        finally:
            conn.close()

    # ─── Attendance ───

    def upsert_attendance(self, session_id: str, identity_id: str, name: str,
                          status: str, checks_present: int = 0,
                          checks_spoofed: int = 0, needs_review: bool = False,
                          note: str = ""):
        """Upsert one attendance record. The PRIMARY KEY (session_id, identity_id)
        guarantees one record per student per session."""
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO attendance "
                "(session_id, identity_id, name, status, checks_present, "
                "checks_spoofed, needs_review, note) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, identity_id, name, status,
                 checks_present, checks_spoofed,
                 1 if needs_review else 0, note)
            )
            conn.commit()
        finally:
            conn.close()

    def get_attendance(self, session_id: str) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM attendance WHERE session_id=? ORDER BY name",
                (session_id,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ─── Notes ───

    def add_note(self, session_id: str, note: str):
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO session_notes (session_id, note, created_at) VALUES (?, ?, ?)",
                (session_id, note, _now_iso())
            )
            conn.commit()
        finally:
            conn.close()

    def get_notes(self, session_id: str) -> list[str]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT note FROM session_notes WHERE session_id=? ORDER BY id",
                (session_id,)
            ).fetchall()
            return [r["note"] for r in rows]
        finally:
            conn.close()

    # ─── Subjects CRUD ───

    def add_subject(self, name: str, code: str = "") -> dict | None:
        """Add a new subject. Returns the subject dict or None on conflict."""
        import uuid as _uuid
        subject_id = str(_uuid.uuid4())[:12]
        now = _now_iso()
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO subjects (subject_id, name, code, created_at) VALUES (?, ?, ?, ?)",
                (subject_id, name.strip(), code.strip(), now)
            )
            conn.commit()
            logger.info(f"Added subject: {name} ({code})")
            return {"subject_id": subject_id, "name": name.strip(),
                    "code": code.strip(), "created_at": now}
        except sqlite3.IntegrityError as e:
            logger.warning(f"Subject add failed: {e}")
            return None
        finally:
            conn.close()

    def get_all_subjects(self) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM subjects ORDER BY name"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_subject(self, subject_id: str) -> dict | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM subjects WHERE subject_id = ?", (subject_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def update_subject(self, subject_id: str, name: str | None = None,
                       code: str | None = None) -> bool:
        conn = self._connect()
        try:
            if name is not None and code is not None:
                conn.execute(
                    "UPDATE subjects SET name=?, code=? WHERE subject_id=?",
                    (name.strip(), code.strip(), subject_id)
                )
            elif name is not None:
                conn.execute(
                    "UPDATE subjects SET name=? WHERE subject_id=?",
                    (name.strip(), subject_id)
                )
            elif code is not None:
                conn.execute(
                    "UPDATE subjects SET code=? WHERE subject_id=?",
                    (code.strip(), subject_id)
                )
            conn.commit()
            return conn.total_changes > 0
        finally:
            conn.close()

    def delete_subject(self, subject_id: str) -> bool:
        conn = self._connect()
        try:
            cur = conn.execute(
                "DELETE FROM subjects WHERE subject_id=?", (subject_id,)
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def subject_exists(self, name: str, code: str = "") -> bool:
        """Check if a subject with the same name or code already exists."""
        conn = self._connect()
        try:
            if code:
                row = conn.execute(
                    "SELECT 1 FROM subjects WHERE LOWER(name)=LOWER(?) OR LOWER(code)=LOWER(?)",
                    (name, code)
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT 1 FROM subjects WHERE LOWER(name)=LOWER(?)",
                    (name,)
                ).fetchone()
            return row is not None
        finally:
            conn.close()


def _now_iso() -> str:
    return datetime.datetime.now().isoformat()
