from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from platformdirs import user_config_dir, user_cache_dir, user_log_dir, user_data_dir  # type: ignore
except Exception:
    user_config_dir = user_cache_dir = user_log_dir = user_data_dir = None  # type: ignore


def _safe_dir(getter, *, appname: str, version: str) -> Path:
    if getter is None:
        return Path(tempfile.gettempdir()) / appname / version
    try:
        return Path(getter(appname=appname, version=version))
    except Exception:
        return Path(tempfile.gettempdir()) / appname / version


@dataclass(frozen=True)
class PackageDirs:
    config_dir: Path
    cache_dir: Path
    log_dir: Path
    data_dir: Path
    output_dir: Path


def get_package_dirs(
    appname: str = "condorsmc",
    *,
    version: Optional[str] = None,
    output_dir: Optional[os.PathLike[str] | str] = None,
) -> PackageDirs:
    try:
        from . import __version__ as pkg_version
    except Exception:
        pkg_version = "0"
    version = version or pkg_version

    base_output = Path(
        os.environ.get("CONDORSMC_OUTPUT_DIR", output_dir or (Path.cwd() / "output"))
    )

    return PackageDirs(
        config_dir=_safe_dir(user_config_dir, appname=appname, version=version),
        cache_dir=_safe_dir(user_cache_dir, appname=appname, version=version),
        log_dir=_safe_dir(user_log_dir,   appname=appname, version=version),
        data_dir=_safe_dir(user_data_dir, appname=appname, version=version),
        output_dir=base_output,
    )


@dataclass(frozen=True)
class SessionDirs:
    config: Path
    cache: Path
    log: Path
    data: Path
    output: Path


def session_dirs(
    session_id: str,
    *,
    ensure: bool = False,
    output_dir: Optional[os.PathLike[str] | str] = None,
) -> SessionDirs:
    pkg = get_package_dirs(output_dir=output_dir)
    sess = SessionDirs(
        config=pkg.config_dir / session_id,
        cache=pkg.cache_dir / session_id,
        log=pkg.log_dir / session_id,
        data=pkg.data_dir / session_id,
        output=pkg.output_dir / session_id,
    )
    if ensure:
        for p in (sess.config, sess.cache, sess.log, sess.data, sess.output):
            p.mkdir(parents=True, exist_ok=True)
    return sess
