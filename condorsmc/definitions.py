import configparser
import logging
import os
from pathlib import Path

import condorsmc

# try:
if os.name != "nt":
    import appdirs  # type: ignore

    # fallback if appdirs fails at runtime
    def _safe_appdir(getter, *args, **kwargs):
        try:
            return Path(getter(*args, **kwargs))
        except Exception:
            fallback_base = Path("/tmp/condorsmc") if os.name != "nt" else Path("C:/tmp/condorsmc")
            return fallback_base / kwargs.get("version", "unknown")
    try:
        PACKAGE_CONFIG_DIR = _safe_appdir(appdirs.user_config_dir, appname="condorsmc", version=condorsmc.__version__)
        PACKAGE_CACHE_DIR = _safe_appdir(appdirs.user_cache_dir, appname="condorsmc", version=condorsmc.__version__)
        PACKAGE_LOG_DIR = _safe_appdir(appdirs.user_log_dir, appname="condorsmc", version=condorsmc.__version__)
        PACKAGE_DATA_DIR = _safe_appdir(appdirs.user_data_dir, appname="condorsmc", version=condorsmc.__version__)
    except:
        pass

    SESSION_CONFIG_DIR = lambda session_id: PACKAGE_CONFIG_DIR / session_id
    SESSION_CACHE_DIR = lambda session_id: PACKAGE_CACHE_DIR / session_id
    SESSION_LOG_DIR = lambda session_id: PACKAGE_LOG_DIR / session_id
    SESSION_DATA_DIR = lambda session_id: PACKAGE_DATA_DIR / session_id
else:
# except ImportError:
    # fallback if appdirs is completely unavailable

    print("We here")
    fallback_base = Path("/tmp/condorsmc") if os.name != "nt" else Path("C:/tmp/condorsmc")

    PACKAGE_CONFIG_DIR = fallback_base / "config"
    PACKAGE_CACHE_DIR = fallback_base / "cache"
    PACKAGE_LOG_DIR = fallback_base / "log"
    PACKAGE_DATA_DIR = fallback_base / "data"

    SESSION_CONFIG_DIR = lambda session_id: PACKAGE_CONFIG_DIR / session_id
    SESSION_CACHE_DIR = lambda session_id: PACKAGE_CACHE_DIR / session_id
    SESSION_LOG_DIR = lambda session_id: PACKAGE_LOG_DIR / session_id
    SESSION_DATA_DIR = lambda session_id: PACKAGE_DATA_DIR / session_id

PACKAGE_ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# CONFIG FILE
LOCAL_CONFIG_DIR = Path(Path.cwd(), "config")

if Path(LOCAL_CONFIG_DIR, "config.cfg").is_file():
    logging.info(f"Found config file in current working directory")
    CONFIG_PATH = Path(LOCAL_CONFIG_DIR, "config.cfg")
elif Path(PACKAGE_CONFIG_DIR, "config.cfg").is_file():
    logging.info("Found config file in package config directory")
    CONFIG_PATH = Path(PACKAGE_CONFIG_DIR, "config.cfg")
else:
    logging.info("Could not find config file in local or package directory")
    raise FileNotFoundError(f"""\
    Config file not found. Please create a config file at one of the following locations:
        {LOCAL_CONFIG_DIR}
        {PACKAGE_CONFIG_DIR}

    A config file should be created for the coordinator, managers and followers and should have the following names:
        coordinator: config.cfg
        manager: manager_config.cfg
        follower: follower_config.cfg

    The config file should have the following format:
        [mysql]
        host = 
        user = 
        password =
        database = condorsmc
        poll_delay = 5

        The config file can also contain the following optional parameters:
    
        [condorsmc]
        tick_rate=5
        job_deadline_buffer=60
        follower_timeout=60
        manager_timeout=60        
    """)

CONFIG = configparser.ConfigParser()
CONFIG.read(CONFIG_PATH)

if not all(
    CONFIG.has_option("mysql", key) for key in ("host", "user", "password", "database")
):
    raise ValueError("Missing required database configuration information")

SESSION_OUTPUT_DIR = lambda session_id: Path(CONDORSMC_OUTPUT_DIR, session_id)

MYSQL_HOST = CONFIG.get("mysql", "host")
MYSQL_USER = CONFIG.get("mysql", "user")
MYSQL_PASSWORD = CONFIG.get("mysql", "password")
MYSQL_DATABASE = CONFIG.get("mysql", "database")
MYSQL_POLL_DELAY = (
    CONFIG.getint("mysql", "poll_delay")
    if CONFIG.has_option("mysql", "poll_delay")
    else 5
)
MYSQL_MAX_POLL_ATTEMPTS = (
    CONFIG.getint("mysql", "max_poll_attempts")
    if CONFIG.has_option("mysql", "max_poll_attempts")
    else 10
)

CONDORSMC_OUTPUT_DIR = (
    Path(CONFIG.get("condorsmc", "output_dir"))
    if CONFIG.has_option("condorsmc", "output_dir")
    else Path(Path.cwd(), "output")
)

SESSION_DEADLINE_BUFFER = (
    CONFIG.getint("condorsmc", "session_deadline_buffer")
    if CONFIG.has_option("condorsmc", "session_deadline_buffer")
    else 180
)

CONDORSMC_TICK_RATE = (
    CONFIG.getint("condorsmc", "tick_rate")
    if CONFIG.has_option("condorsmc", "tick_rate")
    else 5
)

CONDORSMC_FOLLOWER_DEADLINE_BUFFER = (
    CONFIG.getint("condorsmc", "follower_deadline_buffer")
    if CONFIG.has_option("condorsmc", "follower_deadline_buffer")
    else 60
)

CONDORSMC_MANAGER_DEADLINE_BUFFER = (
    CONFIG.getint("condorsmc", "manager_deadline_buffer")
    if CONFIG.has_option("condorsmc", "manager_deadline_buffer")
    else 60
)

CONDORSMC_FOLLOWER_TIMEOUT = (
    CONFIG.getint("condorsmc", "follower_timeout")
    if CONFIG.has_option("condorsmc", "follower_timeout")
    else 60
)

CONDORSMC_MANAGER_TIMEOUT = (
    CONFIG.getint("condorsmc", "manager_timeout")
    if CONFIG.has_option("condorsmc", "manager_timeout")
    else 60
)
