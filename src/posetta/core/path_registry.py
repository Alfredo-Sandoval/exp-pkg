"""Centralized registry of path helpers for Posetta."""

from __future__ import annotations

import hashlib
import os
import re
import sys
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname


def resolve_path(path: str | Path | os.PathLike[str]) -> Path:
    """Resolve a path by expanding the user and resolving the absolute path.

    Args:
        path: The input path to resolve.

    Returns:
        The resolved absolute Path.
    """
    return Path(path).expanduser().resolve()


def get_repo_root() -> Path:
    """Get the root directory of the repository.

    Returns:
        The absolute Path to the repository root.
    """
    return resolve_path(Path(__file__)).parents[2]


def get_repo_devtools_dir() -> Path:
    """Get the devtools directory of the repository.

    Returns:
        The absolute Path to the devtools directory.
    """
    return get_repo_root() / "devtools"


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Ensure a directory exists and return its resolved path."""
    directory = Path(path or ".")
    directory.mkdir(parents=True, exist_ok=True)
    return directory.resolve()


def usable_cpu_count() -> int:
    """Gets number of CPUs usable by the current process.

    Returns:
        The number of usable CPUs.

    Raises:
        RuntimeError: If the CPU count cannot be determined.
    """
    if sys.platform.startswith("linux"):
        return len(os.sched_getaffinity(0))
    result = os.cpu_count()
    if result is None:
        raise RuntimeError("os.cpu_count() returned None; cannot determine CPU count")
    return int(result)


def uniquify(seq: Iterable[Hashable]) -> list:
    """Returns unique elements from list, preserving order.

    Args:
        seq: An iterable of hashable elements.

    Returns:
        A list of unique elements in their original order.
    """
    return list(dict.fromkeys(seq))


def parse_uri_path(uri: str) -> str:
    """Parse a URI starting with 'file:///' to a posix path.

    Args:
        uri: The URI string to parse.

    Returns:
        The parsed POSIX path string.
    """
    return Path(url2pathname(urlparse(unquote(uri)).path)).as_posix()


def return_absolute_path(
    possibly_relative_path: str | Path, n_dirs_back: int = 3, *, create_if_missing: bool = False
) -> str:
    """Return an absolute path for Hydra-influenced training flows.

    Args:
        possibly_relative_path: A path (absolute or relative to CWD - n_dirs_back).
        n_dirs_back: How many directories to back up from CWD for relative paths.
        create_if_missing: If True, create the directory if it doesn't exist.

    Returns:
        The absolute path as a string.

    Raises:
        OSError: If the path is invalid and create_if_missing is False.
    """
    path_obj = Path(possibly_relative_path)
    if path_obj.is_absolute():
        abs_path = path_obj
    else:
        cwd_parts = Path.cwd().parts
        desired = Path(*cwd_parts[: max(0, len(cwd_parts) - n_dirs_back)])
        if desired.name == "multirun":
            desired = desired.parent
        abs_path = (desired / path_obj).resolve()
    if not abs_path.exists():
        if create_if_missing:
            abs_path.mkdir(parents=True, exist_ok=True)
        else:
            msg = f"{abs_path} is not a valid path"
            raise OSError(msg)
    return str(abs_path)


def return_absolute_data_paths(
    data_cfg: Any, n_dirs_back: int = 3, *, create_if_missing: bool = True
) -> tuple[str, str]:
    """Return absolute (data_dir, video_dir) tuple given a config-like object.

    Args:
        data_cfg: Configuration object with data_dir and video_dir attributes/keys.
        n_dirs_back: How many directories to back up from CWD for relative paths.
        create_if_missing: If True, create data_dir if it doesn't exist.

    Returns:
        A tuple of (absolute_data_dir, absolute_video_dir) strings.

    Raises:
        TypeError: If data_cfg is not a mapping.
        ValueError: If data_dir or video_dir are invalid.
    """
    if not isinstance(data_cfg, Mapping):
        raise TypeError("data_cfg must be a mapping")

    data_dir = return_absolute_path(
        data_cfg["data_dir"],
        n_dirs_back=n_dirs_back,
        create_if_missing=create_if_missing,
    )
    raw_video_dir = data_cfg["video_dir"]
    video_path = Path(raw_video_dir)
    if not video_path.is_absolute():
        video_path = Path(data_dir) / video_path
    if not Path(data_dir).is_dir():
        msg = "data_dir must be a directory"
        raise ValueError(msg)
    if not (video_path.is_dir() or video_path.is_file()):
        msg = "video_dir must be a directory or file path"
        raise ValueError(msg)
    return data_dir, str(video_path)


@dataclass(slots=True)
class PathId:
    """Stable identifier + label for a concrete filesystem path.

    Attributes:
        id: A stable unique identifier for the path.
        label: A human-friendly slug derived from the path.
        path: The normalized absolute path string.
    """

    id: str
    label: str
    path: str


_SLUG_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def slugify_path_component(path: str | Path) -> str:
    """Return a deterministic, human-friendly slug derived from a path.

    Args:
        path: The input path or string to slugify.

    Returns:
        A slugified string.
    """
    if isinstance(path, Path):
        name = path.name or str(path)
    else:
        name = str(path)
    name = name.strip()
    if not name:
        return "item"
    stem = Path(name).stem or name
    slug = stem.strip().lower()
    slug = _SLUG_NON_ALNUM_RE.sub("-", slug)
    slug = slug.strip("-")
    return slug or "item"


def normalize_separators(path: str) -> str:
    """Return `path` with POSIX-style separators for cross-platform matching.

    Args:
        path: The input path string.

    Returns:
        The path string with normalized separators.
    """
    return path.replace("\\", "/")


def resolve_engine_meta(engine_path: str | Path) -> Path | None:
    """Return meta.json located beside the engine file (no ancestor search).

    Args:
        engine_path: The path to the model artifact file.

    Returns:
        The Path to the meta.json file, or None if not found.
    """

    p = resolve_path(engine_path)
    meta = p.parent / "meta.json"
    return meta if meta.exists() else None


def make_path_id(path: str | Path, *, prefix: str) -> PathId:
    """Return a stable PathId for the given path.

    Args:
        path: The input path.
        prefix: A prefix for the identifier.

    Returns:
        A PathId object containing the stable identifier, label, and normalized path.
    """
    expanded = resolve_path(str(path))
    normalized = normalize_separators(str(expanded))
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:8]
    identifier = f"{prefix}_{digest}"
    label = slugify_path_component(expanded)
    return PathId(id=identifier, label=label, path=normalized)


def resolve_unified_bundle_or_error(
    path: str | Path,
) -> tuple[Path | None, ValueError | FileNotFoundError | None]:
    """Resolve user input to a concrete native archive without raising.

    Returns:
        (resolved_path, None) on success, or (None, exception) on failure.
    """
    candidate = resolve_path(path)

    if candidate.is_dir():
        canonical = candidate / f"{candidate.name}.sta"
        if canonical.exists():
            return canonical.resolve(), None

        archives = sorted(candidate.glob("*.sta")) or sorted(candidate.glob("*.siesta"))
        if not archives:
            return (
                None,
                ValueError(
                    "Expected a native bundle path, got a directory with no bundles: "
                    f"{candidate}"
                ),
            )
        if len(archives) > 1:
            names = ", ".join(archive.name for archive in archives)
            return (
                None,
                ValueError(
                    "Expected a native bundle path, got a directory with multiple bundles "
                    f"({names}): {candidate}"
                ),
            )
        return archives[0].resolve(), None

    if candidate.suffix.lower() not in (".sta", ".siesta"):
        return None, ValueError(f"Expected a native bundle path, got: {candidate}")

    if not candidate.exists():
        return None, FileNotFoundError(f"Native bundle not found: {candidate}")

    return candidate, None


def resolve_unified_bundle(path: str | Path) -> Path:
    """Resolve user input to a concrete native archive (no discovery heuristics).

    Args:
        path: The input path to a native archive or a directory containing one.

    Returns:
        The resolved absolute Path to the native archive.

    Raises:
        ValueError: If the path is not a native archive or directory with a single archive.
        FileNotFoundError: If the archive is not found.
    """
    resolved, err = resolve_unified_bundle_or_error(path)
    if err is not None:
        raise err
    if resolved is None:
        raise RuntimeError("resolve_unified_bundle_or_error returned (None, None)")
    return resolved


def find_project_bundles(project_root: str | Path) -> list[Path]:
    """Return the project archive path if it exists.

    Args:
        project_root: The root directory of the project.

    Returns:
        A list containing the Path to the project archive if it exists, else empty.
    """
    root = Path(project_root)
    for suffix in (".sta", ".siesta"):
        archive = root / f"{root.name}{suffix}"
        if archive.exists():
            return [archive]
    return []


def iter_image_files(directory: str | Path, sort: bool = True) -> Iterable[Path]:
    """Yield image file paths from a directory (non-recursive).

    Args:
        directory: The directory to search for image files.
        sort: Whether to sort the resulting file paths.

    Yields:
        Paths to image files found in the directory.
    """
    path = Path(directory)
    if not path.is_dir():
        return

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    files = path.iterdir()
    if sort:
        files = sorted(files)

    for child in files:
        if child.is_file() and child.suffix.lower() in exts:
            yield child


def resolve_project_roots(
    project_root: str | Path, configured_video_dir: str | Path | None = None
) -> tuple[Path, Path]:
    """Resolve project root and video directory paths deterministically.

    Args:
        project_root: The root directory of the project.
        configured_video_dir: An optional explicit video directory path.

    Returns:
        A tuple of (resolved_project_root, resolved_video_dir) Paths.
    """
    root = resolve_path(project_root)

    video_dir: Path
    if configured_video_dir:
        raw_dir = str(configured_video_dir)
        base_candidate = Path(raw_dir)
        if raw_dir.startswith("~"):
            video_dir = resolve_path(raw_dir)
        elif base_candidate.is_absolute():
            video_dir = resolve_path(base_candidate)
        else:
            video_dir = base_candidate
    else:
        video_dir = Path("videos")

    if not video_dir.is_absolute():
        video_dir = root / video_dir

    return root, video_dir


def locate_annotation_bundle(project_root: Path, annotation_files: list[str]) -> Path | None:
    """Locate the first existing .sta archive from a list of relative/absolute paths.

    Args:
        project_root: The root directory of the project.
        annotation_files: A list of potential annotation file paths.

    Returns:
        The Path to the first existing .sta archive, or None if none found.
    """
    for entry in annotation_files:
        candidate = Path(entry)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        if candidate.suffix.lower() in (".sta", ".siesta") and candidate.exists():
            return candidate
    return None


def get_package_file(filename: str) -> str:
    """Returns full path to specified file within package.

    Args:
        filename: The name of the file within the package.

    Returns:
        The absolute path to the file as a string.
    """
    data_path = files("posetta").joinpath(filename)
    return str(data_path)


__all__ = [
    "PathId",
    "ensure_dir",
    "find_project_bundles",
    "get_package_file",
    "get_repo_devtools_dir",
    "get_repo_root",
    "iter_image_files",
    "locate_annotation_bundle",
    "make_path_id",
    "normalize_separators",
    "parse_uri_path",
    "resolve_engine_meta",
    "resolve_path",
    "resolve_project_roots",
    "resolve_unified_bundle",
    "return_absolute_data_paths",
    "return_absolute_path",
    "slugify_path_component",
    "uniquify",
    "usable_cpu_count",
]
