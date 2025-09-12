from __future__ import annotations

import configparser
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .paths import get_package_dirs

_CANDIDATES_CWD = ("condorsmc.yaml","condorsmc.yml","condorsmc.toml","condorsmc.ini",
                   "config.yaml","config.yml","config.toml","config.ini")
_CANDIDATES_CFG = ("config.yaml","config.yml","config.toml","config.ini")


def _detect_candidates(explicit: Optional[str]) -> list[Path]:
    cands: list[Path] = []
    if explicit:
        cands.append(Path(explicit))
    env = os.getenv("CONDORSMC_CONFIG")
    if env:
        cands.append(Path(env))
    cwd = Path.cwd()
    cands.extend(cwd / name for name in _CANDIDATES_CWD)
    pkg = get_package_dirs()
    cands.extend((pkg.config_dir / name) for name in _CANDIDATES_CFG)
    return [p for p in cands if p.exists()]


def load_config(explicit: Optional[str] = None) -> tuple[dict[str, Any], Optional[Path]]:
    for path in _detect_candidates(explicit):
        try:
            if path.suffix in {".yaml", ".yml"}:
                import yaml  # pyyaml is already in your deps
                return (yaml.safe_load(path.read_text()) or {}), path
            if path.suffix == ".toml":
                try:
                    import tomllib  # py>=3.11
                except ModuleNotFoundError:
                    try:
                        import tomli as tomllib  # optional backport
                    except ModuleNotFoundError:
                        continue
                return tomllib.loads(path.read_text()), path
            if path.suffix in {".ini", ".cfg"}:
                parser = configparser.ConfigParser()
                parser.read(path)
                data = {s: dict(parser.items(s)) for s in parser.sections()}
                return data, path
        except Exception:
            # Skip unreadable or invalid files; try next candidate
            continue
    return {}, None
