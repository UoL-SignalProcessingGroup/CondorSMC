from __future__ import annotations
import os
from pathlib import Path
from .paths import get_package_dirs, session_dirs
from .config import load_config as _load_config

# ── Package directories ───────────────────────────────────────────────────────

PACKAGE_ROOT_DIR = Path(__file__).resolve().parent

DIRS = get_package_dirs()
CONDORSMC_OUTPUT_DIR = DIRS.output_dir
PACKAGE_CONFIG_DIR   = DIRS.config_dir
PACKAGE_CACHE_DIR    = DIRS.cache_dir
PACKAGE_LOG_DIR      = DIRS.log_dir
PACKAGE_DATA_DIR     = DIRS.data_dir

# ── Session directory helpers ─────────────────────────────────────────────────

def SESSION_OUTPUT_DIR(session_id: str) -> Path:
    return session_dirs(session_id, ensure=True).output

def SESSION_CACHE_DIR(session_id: str) -> Path:
    return session_dirs(session_id).cache

def SESSION_LOG_DIR(session_id: str) -> Path:
    return session_dirs(session_id).log

def SESSION_CONFIG_DIR(session_id: str) -> Path:
    return session_dirs(session_id).config

def SESSION_DATA_DIR(session_id: str) -> Path:
    return session_dirs(session_id).data

# ── Config loader (cached) ────────────────────────────────────────────────────
# Load once at import time; callers can override via condorsmc.yaml / env vars.

_cfg, _cfg_path = _load_config()


def _get(section: str, key: str, env_var: str, default, cast=str):
    """Return a setting via config file > environment variable > default."""
    section_data = _cfg.get(section) if section else _cfg
    val = (section_data or {}).get(key)
    if val is not None:
        return cast(val) if cast is not str else val
    env = os.environ.get(env_var)
    if env is not None:
        return cast(env) if cast is not str else env
    return default


# ── MySQL connection ──────────────────────────────────────────────────────────
# Set via condorsmc.yaml [mysql] section or CONDORSMC_MYSQL_* env vars.

MYSQL_HOST     = _get("mysql", "host",       "CONDORSMC_MYSQL_HOST",     "localhost")
MYSQL_USER     = _get("mysql", "user",       "CONDORSMC_MYSQL_USER",     "")
MYSQL_PASSWORD = _get("mysql", "password",   "CONDORSMC_MYSQL_PASSWORD", "")
MYSQL_DATABASE = _get("mysql", "database",   "CONDORSMC_MYSQL_DATABASE", "condorsmc")
MYSQL_POLL_DELAY = _get("mysql", "poll_delay", "CONDORSMC_MYSQL_POLL_DELAY", 5, cast=float)

# ── Session timing ────────────────────────────────────────────────────────────
# Extra seconds added on top of coordinator_runtime for the session deadline.

SESSION_DEADLINE_BUFFER = _get(
    "timing", "session_deadline_buffer", "CONDORSMC_SESSION_DEADLINE_BUFFER", 300, cast=float
)

# ── Distributed loop timing ───────────────────────────────────────────────────
# Seconds between coordinator / manager / follower polling iterations.
CONDORSMC_TICK_RATE = _get("timing", "tick_rate", "CONDORSMC_TICK_RATE", 5, cast=float)

# Seconds of silence before a daemon is considered stale.
CONDORSMC_FOLLOWER_TIMEOUT = _get(
    "timing", "follower_timeout", "CONDORSMC_FOLLOWER_TIMEOUT", 120, cast=float
)
CONDORSMC_MANAGER_TIMEOUT = _get(
    "timing", "manager_timeout", "CONDORSMC_MANAGER_TIMEOUT", 120, cast=float
)

# How many seconds before a job's deadline the coordinator begins collecting results.
CONDORSMC_FOLLOWER_DEADLINE_BUFFER = _get(
    "timing", "follower_deadline_buffer", "CONDORSMC_FOLLOWER_DEADLINE_BUFFER", 30, cast=float
)
CONDORSMC_MANAGER_DEADLINE_BUFFER = _get(
    "timing", "manager_deadline_buffer", "CONDORSMC_MANAGER_DEADLINE_BUFFER", 30, cast=float
)

# ── HTCondor worker environment ───────────────────────────────────────────────
# Path to the Python environment tarball shipped to worker nodes.
# Must be set via condorsmc.yaml [htcondor] python_env or CONDORSMC_PYTHON_ENV.

_python_env_str = _get("htcondor", "python_env", "CONDORSMC_PYTHON_ENV", "")
PYTHON_ENV: Path | None = Path(_python_env_str) if _python_env_str else None
