"""Pytest configuration and environment setup."""

import sys
from pathlib import Path

# Ensure backend/ directory is on sys.path for test imports
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
