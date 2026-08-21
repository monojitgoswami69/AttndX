#!/usr/bin/env python3
"""
Backwards-compatible wrapper for scripts/download_models.py.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.download_models import download_all_models, main

if __name__ == "__main__":
    main()
