"""Shared helpers for converter modules and adapter entry points."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from xpkg._core.logging_utils import get_logger
from xpkg._core.path_registry import ensure_dir
from xpkg._core.video_contract import video_total_frames
from xpkg.io.video import Video, available_video_exts, write_video

if TYPE_CHECKING:
    from xpkg.model import Labels as _Labels
    from xpkg.pose.annotations import Point
    from xpkg.pose.skeleton import Keypoint

CliRunner = Callable[[argparse.Namespace, argparse.ArgumentParser], int]
ProgressCallback = Callable[[str], None]

_LOGGER = get_logger(__name__)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


class LabelsVideoRemapProtocol(Protocol):
    """Minimal mutable labels surface required for video remapping."""

    videos: list[Any]
    labeled_frames: list[Any]

    def merge_matching_frames(self) -> None: ...

    def update_cache(self) -> None: ...


@dataclass(slots=True)
class ConversionResult:
    """Outcome of converting an external data format into workspace state."""

    source_dir: Path
    project_root: Path
    videos: list[Path]
    labels: Any
    metadata: dict[str, Any]


def _emit(callback: ProgressCallback | None, message: str) -> None:
    """Emit a progress message via callback, logger, or stdout."""
    if callback is not None:
        callback(message)
        return
    if _LOGGER.hasHandlers():
        _LOGGER.info(message)
        return
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def points_from_coords_scores(
    node_names: Sequence[str | Keypoint],
    coords: np.ndarray,
    scores: np.ndarray,
    *,
    likelihood_threshold: float,
) -> dict[str | Keypoint, Point]:
    """Build visible point objects from parallel keypoint coordinate and score arrays."""

    from xpkg.pose.annotations import Point

    coords_array = np.asarray(coords, dtype=np.float64)
    scores_array = np.asarray(scores, dtype=np.float64)
    node_count = len(node_names)

    if coords_array.shape != (node_count, 2):
        raise ValueError(
            "coords must have shape "
            f"({node_count}, 2), got {coords_array.shape}."
        )
    if scores_array.shape != (node_count,):
        raise ValueError(
            "scores must have shape "
            f"({node_count},), got {scores_array.shape}."
        )

    points: dict[str | Keypoint, Point] = {}
    for node_idx, node_name in enumerate(node_names):
        score = float(scores_array[node_idx])
        if not np.isfinite(score) or score < likelihood_threshold:
            continue

        x_val = float(coords_array[node_idx, 0])
        y_val = float(coords_array[node_idx, 1])
        if np.isnan(x_val) or np.isnan(y_val):
            continue
        points[node_name] = Point(x_val, y_val, visible=True, complete=True)
    return points


def _sorted_frame_list(img_dir: Path) -> list[str]:
    if not img_dir.exists() or not img_dir.is_dir():
        return []

    def key(path: Path) -> tuple[int, str]:
        stem = path.stem
        num = 0
        for candidate in (stem, "".join(ch for ch in stem if ch.isdigit())):
            if candidate and candidate.isdigit():
                num = int(candidate)
                break
        return (num, path.name)

    files = [path for path in img_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    files.sort(key=key)
    return [file_path.as_posix() for file_path in files]


def encode_videos(
    source_dir: Path,
    proj_root: Path,
    *,
    fps: int,
    progress: ProgressCallback | None,
) -> list[Path]:
    """Encode one MP4 per ``labeled-data/<base>`` directory and return paths."""
    labeled = source_dir / "labeled-data"
    if not labeled.exists():
        return []

    proj_videos = proj_root / "videos"
    ensure_dir(proj_videos)
    out_videos: list[Path] = []
    min_frames = 2
    for subdir in sorted(path for path in labeled.iterdir() if path.is_dir()):
        frames = _sorted_frame_list(subdir)
        if not frames:
            continue
        video = Video.from_image_filenames(frames)
        dst = proj_videos / f"{subdir.name}.mp4"
        _emit(progress, "XPKG_IMPORT STEP: build_video")

        frame_indices = list(range(video_total_frames(video)))
        if len(frame_indices) < min_frames:
            if not frame_indices:
                continue
            pad_count = min_frames - len(frame_indices)
            frame_indices.extend([frame_indices[-1]] * pad_count)

        write_video(dst.as_posix(), video, frames=frame_indices, fps=int(fps))
        out_videos.append(dst)
    return out_videos


def _strip_terminal_video_extension(name: str) -> str:
    lower_name = name.lower()
    extensions: list[str] = [str(ext) for ext in available_video_exts()]
    extensions.sort(key=lambda item: len(item), reverse=True)
    for extension in extensions:
        if lower_name.endswith(extension):
            return name[: -len(extension)]
    return name


def _video_match_key(path_like: str | Path) -> str:
    raw_name = Path(str(path_like)).name.strip().lower()
    if not raw_name:
        raise ValueError("Video remap requires a non-empty path component")
    normalized = _strip_terminal_video_extension(raw_name)
    slug = _NON_ALNUM_RE.sub("-", normalized).strip("-")
    if not slug:
        raise ValueError(f"Could not derive a stable video match key from {path_like!r}")
    return slug


def _image_sequence_dir_key(image_filenames: Sequence[str]) -> str | None:
    dir_names = {
        Path(str(image_path)).parent.name
        for image_path in image_filenames
        if str(image_path).strip()
    }
    if len(dir_names) != 1:
        return None
    return _video_match_key(next(iter(dir_names)))


def remap_labels_to_videos(
    labels: LabelsVideoRemapProtocol,
    videos: Sequence[Path],
    project_root: Path,
) -> None:
    """Point label video references at encoded videos using stable basename matching."""

    if not videos:
        return

    mp4_by_key = {
        _video_match_key(video_path): video_path for video_path in videos if video_path.exists()
    }
    existing_by_abs: dict[str, Video] = {}
    mapping: dict[int, Video] = {}
    new_videos: list[Any] = []

    for video in labels.videos:
        if video.filename:
            new_videos.append(video)
            continue

        image_filenames = getattr(video, "image_filenames", None) or []
        target_key = _image_sequence_dir_key(image_filenames)
        if target_key is None:
            new_videos.append(video)
            continue

        target_path = mp4_by_key.get(target_key)
        if target_path is None:
            new_videos.append(video)
            continue

        abs_path = (
            target_path if target_path.is_absolute() else (project_root / target_path)
        ).resolve()
        cache_key = abs_path.as_posix()
        mapped_video = existing_by_abs.get(cache_key)
        if mapped_video is None:
            mapped_video = Video.from_filename(cache_key)
            existing_by_abs[cache_key] = mapped_video
            new_videos.append(mapped_video)
        mapping[id(video)] = mapped_video

    changed = False
    for labeled_frame in labels.labeled_frames:
        mapped_video = mapping.get(id(labeled_frame.video))
        if mapped_video is None:
            continue
        labeled_frame.video = mapped_video
        changed = True

    if new_videos:
        seen: set[int] = set()
        deduped_videos: list[Any] = []
        for video in new_videos:
            video_id = id(video)
            if video_id in seen:
                continue
            seen.add(video_id)
            deduped_videos.append(video)
        labels.videos = deduped_videos

    if changed:
        labels.merge_matching_frames()
        labels.update_cache()


def rebase_image_sequences(
    labels: _Labels,
    src_root: Path,
    dst_root: Path,
) -> None:
    """Move/copy image sequence references from src_root -> dst_root in labels."""

    src_root = src_root.resolve()
    dst_root = dst_root.resolve()

    for video in list(labels.videos or []):
        if video.filename:
            continue
        frames = list(video.image_filenames or [])
        if not frames:
            continue
        updated: list[str] = []
        changed = False
        for frame in frames:
            frame_path = Path(str(frame)).resolve()
            relative_path = frame_path.relative_to(src_root)
            changed = True
            updated.append((dst_root / relative_path).as_posix())
        if changed:
            video._image_filenames = updated


def build_cli_parser(description: str) -> argparse.ArgumentParser:
    """Create a converter CLI parser with the provided description."""

    return argparse.ArgumentParser(description=description)


def add_output_path_argument(
    parser: argparse.ArgumentParser,
    *,
    help_text: str,
) -> None:
    """Register the required ``--out`` argument with converter-specific help text."""

    parser.add_argument("--out", required=True, help=help_text)


def add_bool_toggle_arguments(
    parser: argparse.ArgumentParser,
    *,
    dest: str,
    true_flag: str,
    false_flag: str,
    true_help: str,
    false_help: str,
    default: bool,
) -> None:
    """Register paired boolean flags in a mutually exclusive group."""

    group = parser.add_mutually_exclusive_group()
    group.add_argument(true_flag, dest=dest, action="store_true", help=true_help)
    group.add_argument(false_flag, dest=dest, action="store_false", help=false_help)
    parser.set_defaults(**{dest: default})


def parse_and_run_cli(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None,
    runner: CliRunner,
) -> int:
    """Parse arguments and invoke the CLI runner."""

    args = parser.parse_args(argv)
    return runner(args, parser)


__all__ = [
    "CliRunner",
    "ConversionResult",
    "ProgressCallback",
    "_emit",
    "add_bool_toggle_arguments",
    "add_output_path_argument",
    "build_cli_parser",
    "encode_videos",
    "parse_and_run_cli",
    "points_from_coords_scores",
    "rebase_image_sequences",
    "remap_labels_to_videos",
]
