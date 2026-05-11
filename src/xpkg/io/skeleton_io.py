"""Filesystem-facing skeleton loading and saving utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .._core.json_utils import load_json_dict, write_json

if TYPE_CHECKING:
    from xpkg.pose.skeleton import Skeleton


def dump_skeleton(
    skeleton: Skeleton,
    path: Path,
    *,
    fmt: str | None = None,
    keep_names: bool = True,
) -> None:
    """Persist a Skeleton to disk."""
    target_format = fmt if fmt is not None else path.suffix.lower().lstrip(".")
    payload = skeleton.to_dict(keep_names=keep_names)
    if target_format == "json":
        write_json(path, payload, indent=2, sort_keys=False)
        return
    raise ValueError(f"Unknown format: {target_format} (only 'json' is supported)")


def load_skeleton(src: Path, **kwargs: Any) -> Skeleton:
    """Load a Skeleton from a supported on-disk representation."""
    if src.suffix.lower() in {".yaml", ".yml"}:
        raise ValueError("YAML skeletons are no longer supported; use JSON.")
    from xpkg.pose.skeleton import Skeleton

    return Skeleton.from_dict(load_json_dict(src), **kwargs)


def load_any_skeleton(
    path: str | Path,
    *,
    format: str | None = None,
    **kwargs: Any,
) -> Skeleton:
    """Load a Skeleton from any supported external format."""
    from xpkg.io.skeleton_loaders import (
        load_skeleton,
        load_skeleton_dlc,
        load_skeleton_sleap,
        load_skeleton_ultralytics,
    )

    del kwargs
    path_obj = Path(path)
    if format:
        fmt = format.lower().strip()
        if fmt == "dlc":
            return load_skeleton_dlc(path_obj)
        if fmt == "sleap":
            return load_skeleton_sleap(path_obj)
        if fmt in {"ultralytics", "yolo"}:
            return load_skeleton_ultralytics(path_obj)
        raise ValueError(f"Unknown format: {format}")
    return load_skeleton(path_obj)
