#!/usr/bin/env python3
"""
Reset the biometric database.

Backs up the existing data/biometric.db, clears all identities,
embeddings, templates, and clears the data/captures/ directory.
Use this when switching embedding models (embeddings are incompatible).
"""
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import Config
Config.load()


def reset_biometric_db():
    db_path = Config.biometric_db_path()
    captures_dir = Config.captures_dir()

    # Backup existing DB
    if db_path.exists():
        timestamp = int(time.time())
        backup_path = db_path.with_suffix(f".bak.{timestamp}.db")
        shutil.copy2(db_path, backup_path)
        print(f"Backed up {db_path.name} → {backup_path.name}")

        # Delete the DB file
        db_path.unlink()
        print(f"Deleted {db_path}")
    else:
        print(f"No existing DB at {db_path}")

    # Clear captures directory
    if captures_dir.exists():
        # Count what's there
        dirs = [d for d in captures_dir.iterdir() if d.is_dir()]
        print(f"Clearing {len(dirs)} enrollment capture directories...")
        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)
            print(f"  Removed {d.name}")
    else:
        print(f"No captures directory at {captures_dir}")

    # Recreate the DB (empty)
    from storage.db import BiometricDB
    db = BiometricDB(db_path)
    count = db.count_identities()
    print(f"\nNew database created: {count} identities (should be 0)")
    print("✅ Biometric database reset complete.")
    print("   All identities must be re-enrolled with the new model.")


if __name__ == "__main__":
    reset_biometric_db()
