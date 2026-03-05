"""Static configuration definitions used by Posetta core IO code."""

from __future__ import annotations

from typing import Any

from posetta.config.loaders import load_json_config

_SKELETONS_CFG = load_json_config("skeletons.json")

SKELETONS: dict[str, Any] = {k: v for k, v in _SKELETONS_CFG.items() if k != "naming"}

_NAMING = _SKELETONS_CFG["naming"]
SIDE_TOKENS: dict[str, str] = _NAMING["side_tokens"]
NAME_ALIASES: dict[str, str] = _NAMING["name_aliases"]
ROLE_ENUM: set[str] = set(_NAMING["role_enum"])


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


__all__ = [
    "NAME_ALIASES",
    "ROLE_ENUM",
    "SIDE_TOKENS",
    "SKELETONS",
    "get_skeleton_def",
]
