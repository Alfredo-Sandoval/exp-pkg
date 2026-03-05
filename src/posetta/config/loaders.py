"""Centralized JSON/YAML load and save helpers.

Single authoritative module for loading and saving JSON/YAML configuration
files. All non-GUI config IO should delegate here.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

from posetta.core.json_utils import load_json_dict, write_json
from posetta.core.path_registry import ensure_dir, resolve_path

_BASE = Path(__file__).parent


def load_json_file(path: str | Path) -> dict[str, Any]:
    """Load a JSON file into a dictionary from any path.

    Supports path expansion (user home, environment variables).

    Args:
        path: Path to the JSON file.

    Returns:
        The loaded configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path_obj = resolve_path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path_obj}")
    return load_json_dict(path_obj)


@cache
def load_json_config(name: str, *, data_dir: Path | None = None) -> dict[str, Any]:
    """Load a JSON config from a data directory (cached).

    Args:
        name: The name of the config file.
        data_dir: Optional override directory for config data.

    Returns:
        The loaded configuration dictionary.
    """
    resolved_dir = _BASE / "data" if data_dir is None else data_dir
    return load_from_data_dir(resolved_dir, name)


def load_from_data_dir(data_dir: Path, name: str) -> dict[str, Any]:
    """Load a JSON config from a specific data directory.

    Args:
        data_dir: The directory containing the config file.
        name: The name of the config file (including extension if needed, usually just name).

    Returns:
        The loaded configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist in the directory.
    """
    path = data_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"Config JSON not found: {path}")
    return load_json_dict(path)


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary from any path.

    Supports path expansion (user home, environment variables).

    Args:
        path: Path to the YAML file.

    Returns:
        The loaded configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        TypeError: If the YAML does not contain a mapping at the top level.
    """
    import yaml

    path_obj = resolve_path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"YAML file not found: {path_obj}")

    data = yaml.safe_load(path_obj.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML file must contain a mapping at the top level: {path_obj}")
    out: dict[str, Any] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise TypeError(f"YAML mapping keys must be strings: {path_obj}")
        out[key] = value
    return out


def save_json(cfg: dict[str, Any], path: str | Path) -> Path:
    """Write a configuration dictionary to a JSON file."""
    path_obj = resolve_path(path)
    ensure_dir(path_obj.parent)
    write_json(path_obj, cfg, indent=2, sort_keys=False)
    return path_obj


def save_yaml(cfg: dict[str, Any], path: str | Path) -> Path:
    """Write a configuration dictionary to a YAML file."""
    import yaml

    path_obj = resolve_path(path)
    ensure_dir(path_obj.parent)
    serialized = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
    if serialized and not serialized.endswith("\n"):
        serialized += "\n"
    path_obj.write_text(serialized, encoding="utf-8")
    return path_obj


__all__ = [
    "load_from_data_dir",
    "load_json_config",
    "load_json_file",
    "load_yaml_file",
    "save_json",
    "save_yaml",
]
