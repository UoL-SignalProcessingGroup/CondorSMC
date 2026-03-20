from __future__ import annotations
import os
from pathlib import Path
from .paths import get_package_dirs, session_dirs

PACKAGE_ROOT_DIR = Path(__file__).resolve().parent

DIRS = get_package_dirs()
CONDORSMC_OUTPUT_DIR = DIRS.output_dir
PACKAGE_CONFIG_DIR = DIRS.config_dir

# Path to the Python environment archive shipped to HTCondor workers.
# Override with the CONDORSMC_PYTHON_ENV environment variable.
PYTHON_ENV = Path(
    os.environ.get("CONDORSMC_PYTHON_ENV", "/condor_data/sgmcart3/test_env.tar.gz")
)

def SESSION_OUTPUT_DIR(session_id: str) -> Path:
    return session_dirs(session_id, ensure=True).output
