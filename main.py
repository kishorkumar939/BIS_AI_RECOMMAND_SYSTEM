import sys
from pathlib import Path

# Ensure backend folder is in python sys.path
backend_path = Path(__file__).resolve().parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from backend.main import app  # noqa: F401
