#!/usr/bin/env python3
"""Rebuild the recognition index from the authoritative biometric DB."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from storage.db import BiometricDB
from recognition.matching.engine import IdentityIndex

def main():
    Config.load()
    db = BiometricDB.safe_open()
    templates = db.get_all_templates()
    identities = db.get_all_identities()
    names = {i["identity_id"]: i["name"] for i in identities}

    index = IdentityIndex()
    index.rebuild(templates, names)

    print(f"Index rebuilt: {index.size} identities")
    print(f"  Templates loaded: {len(templates)}")
    print(f"  DB identities: {len(identities)}")

if __name__ == "__main__":
    main()
