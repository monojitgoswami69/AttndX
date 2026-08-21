"""
Biometric database — SQLite with WAL mode.

Stores:
  - identities (student_id, name, enrolled_at, pipeline_version, status)
  - embeddings (embedding_id, identity_id, vector BLOB, source, quality_score, created_at)
  - templates (identity_id, vector BLOB, num_samples, created_at)

The DB is authoritative. The in-memory index is rebuildable from here.
Atomic writes via SQLite WAL + deferred transactions.
Corruption recovery: if the DB file is missing/corrupt, we start fresh
and log a CRITICAL error — but we never silently reset on a transient error.
"""
import json
import logging
import sqlite3
import struct
import time
from pathlib import Path
from typing import Optional

import numpy as np

from config import Config

logger = logging.getLogger(__name__)


def _serialize_embedding(emb: np.ndarray) -> bytes:
    """Serialize a float32 numpy array to compact bytes."""
    assert emb.dtype == np.float32, f"Expected float32, got {emb.dtype}"
    return emb.tobytes()


def _deserialize_embedding(data: bytes, dim: int = 512) -> np.ndarray:
    """Deserialize bytes back to float32 numpy array."""
    arr = np.frombuffer(data, dtype=np.float32)
    if arr.shape[0] != dim:
        logger.warning(f"Embedding dim mismatch: expected {dim}, got {arr.shape[0]}")
    return arr.copy()


class BiometricDB:
    """SQLite-backed biometric identity + embedding store."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS identities (
        identity_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        enrolled_at TEXT NOT NULL,
        pipeline_version TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'active',
        notes TEXT DEFAULT ''
    );

    CREATE TABLE IF NOT EXISTS embeddings (
        embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
        identity_id TEXT NOT NULL,
        vector BLOB NOT NULL,
        dim INTEGER NOT NULL,
        quality_score REAL DEFAULT 0.0,
        source TEXT DEFAULT 'enrollment',
        created_at TEXT NOT NULL,
        FOREIGN KEY (identity_id) REFERENCES identities(identity_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS templates (
        identity_id TEXT PRIMARY KEY,
        vector BLOB NOT NULL,
        dim INTEGER NOT NULL,
        num_samples INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (identity_id) REFERENCES identities(identity_id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_embeddings_identity ON embeddings(identity_id);
    CREATE INDEX IF NOT EXISTS idx_identities_status ON identities(status);
    """

    def __init__(self, db_path: Path | str | None = None):
        if db_path is None:
            db_path = Config.biometric_db_path()
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
        try:
            conn = self._connect()
            conn.executescript(self.SCHEMA)
            conn.commit()
            conn.close()
        except sqlite3.DatabaseError as e:
            logger.critical(f"BiometricDB initialization failed: {e}")
            raise

    # ─── Identity CRUD ───

    def add_identity(self, identity_id: str, name: str,
                     pipeline_version: str) -> bool:
        """Insert a new identity. Returns True on success."""
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO identities (identity_id, name, enrolled_at, pipeline_version, status) "
                "VALUES (?, ?, ?, ?, 'active')",
                (identity_id, name, _now_iso(), pipeline_version)
            )
            conn.commit()
            logger.info(f"Added identity: {identity_id} ({name})")
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"Identity already exists: {identity_id}")
            return False
        finally:
            conn.close()

    def get_identity(self, identity_id: str) -> Optional[dict]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM identities WHERE identity_id = ?", (identity_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_all_identities(self) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM identities WHERE status='active' ORDER BY name"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def identity_exists(self, identity_id: str) -> bool:
        return self.get_identity(identity_id) is not None

    def count_identities(self) -> int:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM identities WHERE status='active'"
            ).fetchone()
            return row[0]
        finally:
            conn.close()

    def delete_identity(self, identity_id: str) -> bool:
        conn = self._connect()
        try:
            cur = conn.execute(
                "DELETE FROM identities WHERE identity_id = ?", (identity_id,)
            )
            conn.commit()
            deleted = cur.rowcount > 0
            if deleted:
                logger.info(f"Deleted identity: {identity_id}")
            return deleted
        finally:
            conn.close()

    # ─── Embedding storage ───

    def add_embedding(self, identity_id: str, vector: np.ndarray,
                      quality_score: float = 0.0, source: str = "enrollment") -> bool:
        """Store a single sample embedding. Validates before storing."""
        if not _validate_embedding(vector):
            logger.error(f"Rejecting malformed embedding for {identity_id}")
            return False
        dim = Config.get("embedding", "dimension", default=512)
        blob = _serialize_embedding(vector.astype(np.float32))
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO embeddings (identity_id, vector, dim, quality_score, source, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (identity_id, blob, dim, quality_score, source, _now_iso())
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError as e:
            logger.error(f"Failed to store embedding: {e}")
            return False
        finally:
            conn.close()

    def set_template(self, identity_id: str, template: np.ndarray,
                     num_samples: int) -> bool:
        """Store the computed template (centroid) for an identity."""
        if not _validate_embedding(template):
            logger.error(f"Rejecting malformed template for {identity_id}")
            return False
        dim = Config.get("embedding", "dimension", default=512)
        blob = _serialize_embedding(template.astype(np.float32))
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO templates (identity_id, vector, dim, num_samples, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (identity_id, blob, dim, num_samples, _now_iso())
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def get_template(self, identity_id: str) -> Optional[np.ndarray]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT vector FROM templates WHERE identity_id = ?", (identity_id,)
            ).fetchone()
            if row is None:
                return None
            return _deserialize_embedding(row["vector"])
        finally:
            conn.close()

    def get_all_templates(self) -> list[tuple[str, np.ndarray]]:
        """Return [(identity_id, template_vector)] for all active identities.
        This is the authoritative source for index rebuild."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT t.identity_id, t.vector FROM templates t "
                "INNER JOIN identities i ON t.identity_id = i.identity_id "
                "WHERE i.status = 'active'"
            ).fetchall()
            return [(r["identity_id"], _deserialize_embedding(r["vector"])) for r in rows]
        finally:
            conn.close()

    def get_embeddings_for_identity(self, identity_id: str) -> list[np.ndarray]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT vector FROM embeddings WHERE identity_id = ? ORDER BY embedding_id",
                (identity_id,)
            ).fetchall()
            return [_deserialize_embedding(r["vector"]) for r in rows]
        finally:
            conn.close()

    # ─── Recovery ───

    @classmethod
    def safe_open(cls, db_path: Path | str | None = None) -> "BiometricDB":
        """Open the DB, recovering from corruption if needed.
        If the DB file is corrupt, back it up and start fresh.
        Never silently reset — log CRITICAL and preserve the old file."""
        if db_path is None:
            db_path = Config.biometric_db_path()
        db_path = Path(db_path)
        try:
            db = cls(db_path)
            # Verify we can actually query
            db.count_identities()
            return db
        except sqlite3.DatabaseError as e:
            logger.critical(f"BiometricDB corruption detected: {e}")
            backup = db_path.with_suffix(f".corrupt.{int(time.time())}.db")
            db_path.rename(backup)
            logger.critical(f"Corrupt DB backed up to {backup}. Starting fresh.")
            return cls(db_path)


# ─── Helpers ───

def _now_iso() -> str:
    import datetime
    return datetime.datetime.now().isoformat()


def _validate_embedding(vec: np.ndarray) -> bool:
    """Validate an embedding before storage.
    Checks: dimension, dtype, finiteness, L2 norm range."""
    dim = Config.get("embedding", "dimension", default=512)
    if vec is None:
        return False
    if vec.shape != (dim,):
        logger.error(f"Embedding shape mismatch: {vec.shape} vs ({dim},)")
        return False
    if vec.dtype != np.float32:
        logger.error(f"Embedding dtype mismatch: {vec.dtype} vs float32")
        return False
    if not np.all(np.isfinite(vec)):
        logger.error("Embedding contains NaN/Inf")
        return False
    norm = float(np.linalg.norm(vec))
    if norm < 0.5 or norm > 1.5:
        logger.error(f"Embedding L2 norm out of range: {norm}")
        return False
    return True
