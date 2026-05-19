"""Filename matching for externally produced DLC-style pose CSVs."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

_BUNDLE_COPY_SUFFIX_PATTERN = re.compile(r"-\d{3}$")
_LABELED_VIDEO_SUFFIXES = ("_lp_default_labeled", "_labeled")

DEFAULT_RAT_SKELETON_EDGES: tuple[tuple[str, str], ...] = (
    ("nose", "head"),
    ("head", "spine1"),
    ("spine1", "spine2"),
    ("spine2", "spine3"),
    ("spine3", "tailbase"),
    ("tailbase", "tail1"),
    ("tail1", "tail2"),
    ("tail2", "tail_tip"),
    ("spine1", "shoulder"),
    ("shoulder", "frontpaw"),
    ("spine3", "hip"),
    ("hip", "knee"),
    ("knee", "backpaw"),
)


def pose_csv_session_name(pose_csv_path: Path) -> str:
    """Return the session token encoded in a supported pose CSV filename."""

    stem = pose_csv_path.stem
    if stem.endswith("__predictions"):
        return stem.removesuffix("__predictions")
    return stem


def normalize_pose_session_name(session_name: str) -> str:
    """Normalize session names emitted by bundle copies and labeled renders."""

    normalized = session_name
    for suffix in _LABELED_VIDEO_SUFFIXES:
        if normalized.endswith(suffix):
            normalized = normalized.removesuffix(suffix)
            break
    return _BUNDLE_COPY_SUFFIX_PATTERN.sub("", normalized)


def normalized_media_session_names(media_path: Path) -> tuple[str, ...]:
    """Return accepted normalized session names for one media file."""

    names: list[str] = []
    for raw_name in (media_path.stem, media_path.name):
        normalized = normalize_pose_session_name(raw_name)
        if normalized not in names:
            names.append(normalized)
    return tuple(names)


def pose_csv_matches_media(pose_csv_path: Path, media_path: Path) -> bool:
    """Return whether a pose CSV filename names the supplied media session."""

    csv_name = normalize_pose_session_name(pose_csv_session_name(pose_csv_path))
    return csv_name in normalized_media_session_names(media_path)


def expected_pose_csv_name(media_path: Path) -> str:
    """Return the legacy source-side pose CSV name for a media file."""

    return f"{media_path.stem}__predictions.csv"


def pose_csv_hint_for_media(media_path: Path) -> str:
    """Return a compact user-facing hint for accepted pose CSV names."""

    normalized_name = normalized_media_session_names(media_path)[0]
    return f"{expected_pose_csv_name(media_path)} or csv/{normalized_name}.csv"


def find_matching_pose_csvs(search_root: Path, media_path: Path) -> tuple[Path, ...]:
    """Find pose CSV files under ``search_root`` that match one media path."""

    matches: list[Path] = []
    for candidate in sorted(search_root.rglob("*.csv")):
        if candidate.is_file() and pose_csv_matches_media(candidate, media_path):
            matches.append(candidate.resolve())
    return tuple(matches)


def resolve_skeleton_edges(
    bodypart_names: Sequence[str],
    *,
    requested_edges: Sequence[tuple[str, str]] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Filter the requested edge list down to bodyparts present in one CSV."""

    name_by_key = {_bodypart_lookup_key(name): name for name in bodypart_names}
    edges = requested_edges or DEFAULT_RAT_SKELETON_EDGES
    resolved: list[tuple[str, str]] = []
    for start_name, end_name in edges:
        actual_start = name_by_key.get(_bodypart_lookup_key(start_name))
        actual_end = name_by_key.get(_bodypart_lookup_key(end_name))
        if actual_start is not None and actual_end is not None:
            resolved.append((actual_start, actual_end))
    if not resolved:
        raise ValueError("No requested skeleton edges were present in the bodyparts.")
    return tuple(resolved)


def _bodypart_lookup_key(bodypart_name: str) -> str:
    return "".join(character for character in bodypart_name.lower() if character.isalnum())


__all__ = [
    "DEFAULT_RAT_SKELETON_EDGES",
    "expected_pose_csv_name",
    "find_matching_pose_csvs",
    "normalize_pose_session_name",
    "normalized_media_session_names",
    "pose_csv_hint_for_media",
    "pose_csv_matches_media",
    "pose_csv_session_name",
    "resolve_skeleton_edges",
]
