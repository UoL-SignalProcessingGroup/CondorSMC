from __future__ import annotations
from pathlib import Path
from .paths import get_package_dirs, session_dirs

PACKAGE_ROOT_DIR = Path(__file__).resolve().parent

DIRS = get_package_dirs()
CONDORSMC_OUTPUT_DIR = DIRS.output_dir

def SESSION_OUTPUT_DIR(session_id: str) -> Path:
    return session_dirs(session_id, ensure=True).output
