#!/usr/bin/env python3
"""Back up biometric and attendance databases."""
import sys
import shutil
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from gen2.config import Config

def main():
    Config.load()
    data_dir = Config.data_dir()
    backup_dir = data_dir / f"backup_{int(time.time())}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    for db_name in ["biometric.db", "attendance.db",
                    "biometric.db-wal", "biometric.db-shm",
                    "attendance.db-wal", "attendance.db-shm"]:
        src = data_dir / db_name
        if src.exists():
            shutil.copy2(src, backup_dir / db_name)
            print(f"  Backed up: {db_name}")

    print(f"\nBackup complete: {backup_dir}")

if __name__ == "__main__":
    main()
