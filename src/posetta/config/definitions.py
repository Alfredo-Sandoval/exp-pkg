"""Static configuration definitions and defaults.

This module is the single source for packaged JSON-backed constants used by
core, GUI, and project services:
- Skeleton metadata and naming aliases
- Preferences defaults
- App/UI defaults
- Project structure defaults
"""

from __future__ import annotations

from typing import Any

from posetta.config.loaders import load_json_config

_CONSTANTS = load_json_config("constants/constants.json")
_SKELETONS_CFG = load_json_config("skeletons.json")
_APP_DEFAULTS = load_json_config("defaults/app_defaults.json")

BUILTIN_COLORMAPS: list[str] = list(_CONSTANTS["visual"]["colormaps"])
METRIC_TYPES: dict[str, str] = _CONSTANTS["metrics"]["metric_types"]
PREFERENCES_DEFAULTS: dict[str, Any] = load_json_config("defaults/preferences_defaults.json")
DISPLAY_ADJUST_DEFAULTS: dict[str, Any] = dict(_APP_DEFAULTS["display_adjust"])
SKELETONS: dict[str, Any] = {k: v for k, v in _SKELETONS_CFG.items() if k != "naming"}

_NAMING = _SKELETONS_CFG["naming"]
SIDE_TOKENS: dict[str, str] = _NAMING["side_tokens"]
NAME_ALIASES: dict[str, str] = _NAMING["name_aliases"]
ROLE_ENUM: set[str] = set(_NAMING["role_enum"])
_PROJECT_DEFAULTS = _APP_DEFAULTS["project"]


def get_skeleton_def(name: str) -> dict[str, Any]:
    """Get a built-in skeleton definition by name.

    Args:
        name: Name of the skeleton.

    Returns:
        The skeleton definition dictionary.

    Raises:
        ValueError: If the skeleton name is not found.
    """
    if name not in SKELETONS:
        raise ValueError(f"Skeleton '{name}' not found in built-in definitions.")
    return SKELETONS[name]


def get_preferences_defaults() -> dict[str, Any]:
    """Get a copy of the default preferences.

    Returns:
        A dictionary of default preferences.
    """
    return PREFERENCES_DEFAULTS.copy()


def resolve_preferences(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve preferences by applying overrides to defaults.

    Args:
        overrides: Optional dictionary of preference overrides.

    Returns:
        A dictionary of resolved preferences.
    """
    prefs = get_preferences_defaults()
    if overrides:
        prefs.update(overrides)
    return prefs


def get_edge_defaults() -> dict[str, Any]:
    """Return default edge/skeleton visualization settings."""
    return dict(_APP_DEFAULTS["edges"])


def get_video_io_defaults() -> dict[str, Any]:
    """Return default video IO settings."""
    return dict(_APP_DEFAULTS["video_io"])


def get_suggestions_defaults() -> dict[str, Any]:
    """Return default suggestion settings."""
    return dict(_APP_DEFAULTS["suggestions"])


def get_project_subdirectories() -> list[str]:
    """Return default project subdirectory names."""
    return [str(x) for x in _PROJECT_DEFAULTS["subdirectories"]]


def get_project_structure() -> dict[str, dict[str, Any]]:
    """Return canonical project structure mapping (rel_path -> metadata)."""
    from posetta.io.manifest import AssetType

    raw = _PROJECT_DEFAULTS["structure"]
    out: dict[str, dict[str, Any]] = {}
    for rel_path, info in raw.items():
        asset_type = AssetType[str(info["asset_type"])]
        role = str(info["role"])
        out[str(rel_path)] = {"asset_type": asset_type, "role": role}
    return out


__all__ = [
    "BUILTIN_COLORMAPS",
    "DISPLAY_ADJUST_DEFAULTS",
    "METRIC_TYPES",
    "NAME_ALIASES",
    "PREFERENCES_DEFAULTS",
    "ROLE_ENUM",
    "SIDE_TOKENS",
    "SKELETONS",
    "get_edge_defaults",
    "get_preferences_defaults",
    "get_project_structure",
    "get_project_subdirectories",
    "get_skeleton_def",
    "get_suggestions_defaults",
    "get_video_io_defaults",
    "resolve_preferences",
]
