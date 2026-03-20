import configparser
import logging
import os
from pathlib import Path

import appdirs  # type: ignore

import condorcmf

# PACKAGE DIRECTORIES
PACKAGE_ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_CONFIG_DIR = Path(
    appdirs.user_config_dir(appname="condorcmf", version=condorcmf.__version__)
)
PACKAGE_CACHE_DIR = Path(
    appdirs.user_cache_dir(appname="condorcmf", version=condorcmf.__version__)
)
PACKAGE_LOG_DIR = Path(
    appdirs.user_log_dir(appname="condorcmf", version=condorcmf.__version__)
)
PACKAGE_DATA_DIR = Path(
    appdirs.user_data_dir(appname="condorcmf", version=condorcmf.__version__)
)

# SESSION DIRECTORIES
SESSION_CONFIG_DIR = lambda session_id: Path(
    appdirs.user_config_dir(appname="condorcmf", version=condorcmf.__version__),
    session_id,
)
SESSION_CACHE_DIR = lambda session_id: Path(
    appdirs.user_cache_dir(appname="condorcmf", version=condorcmf.__version__),
    session_id,
)
SESSION_LOG_DIR = lambda session_id: Path(
    appdirs.user_log_dir(appname="condorcmf", version=condorcmf.__version__),
    session_id,
)
SESSION_DATA_DIR = lambda session_id: Path(
    appdirs.user_data_dir(appname="condorcmf", version=condorcmf.__version__),
    session_id,
)
