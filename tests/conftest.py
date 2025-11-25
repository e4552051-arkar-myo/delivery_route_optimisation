# tests/conftest.py
import sys
from pathlib import Path

# Project root = parent of the tests/ directory
ROOT = Path(__file__).resolve().parents[1]

# Ensure project root is on PYTHONPATH
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))