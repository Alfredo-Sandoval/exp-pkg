"""Shared helpers for data format converters (SLEAP, DLC-style, etc.)."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from posetta.core.logging_utils import get_logger
from posetta.core.path_registry import ensure_dir
from posetta.core.video_contract import video_total_frames
from posetta.io.video import Video, write_video

if TYPE_CHECKING:
    from posetta.io.labels import Labels as _Labels

ProgressCallback = Callable[[str], None]

_LOGGER = get_logger(__name__)


@dataclass(slots=True)
class ConversionResult:
    """Outcome of converting an external data format into a .siesta bundle."""

    source_dir: Path
    project_root: Path
    videos: list[Path]
    siesta_path: Path


def _emit(callback: ProgressCallback | None, message: str) -> None:
    """Emit a progress message via callback, logger, or stdout."""
    if callback is not None:
        callback(message)
    elif _LOGGER.hasHandlers():
        _LOGGER.info(message)
    else:
        sys.stdout.write(message + "\n")
        sys.stdout.flush()


def _sorted_frame_list(img_dir: Path) -> list[str]:
    if not img_dir.exists() or not img_dir.is_dir():
        return []

    def key(p: Path) -> tuple[int, str]:
        stem = p.stem
        num = 0
        for candidate in (stem, "".join(ch for ch in stem if ch.isdigit())):
            if candidate and candidate.isdigit():
                num = int(candidate)
                break
        return (num, p.name)

    files = [p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    files.sort(key=key)
    return [f.as_posix() for f in files]


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
    for sub in sorted([p for p in labeled.iterdir() if p.is_dir()]):
        frames = _sorted_frame_list(sub)
        if not frames:
            continue
        video = Video.from_image_filenames(frames)
        dst = proj_videos / f"{sub.name}.mp4"
        _emit(progress, "SIESTA_IMPORT STEP: build_video")

        idxs = list(range(video_total_frames(video)))
        if len(idxs) < min_frames:
            if not idxs:
                continue
            pad_count = min_frames - len(idxs)
            idxs.extend([idxs[-1]] * pad_count)

        write_video(dst.as_posix(), video, frames=idxs, fps=int(fps))
        out_videos.append(dst)
    return out_videos


def remap_labels_to_videos(
    labels: _Labels,
    videos: Sequence[Path],
    project_root: Path,
) -> None:
    """Point Labels.video references to encoded MP4s when possible."""

    from posetta.core.path_registry import make_path_id
    from posetta.io.video import Video as _Video

    if not videos:
        return

    mp4_by_label = {
        make_path_id(str(p), prefix="video").label: p for p in videos if p and p.exists()
    }
    existing_by_abs: dict[str, Any] = {}
    mapping: dict[Any, Any] = {}
    new_videos: list[Any] = []

    for v in labels.videos:
        fn = v.filename
        imgs = v._image_filenames
        if fn:
            new_videos.append(v)
            continue

        dir_names = {Path(str(img)).parent.name.lower() for img in (imgs or []) if str(img).strip()}
        target_path = None
        if len(dir_names) == 1:
            dir_label = make_path_id(next(iter(dir_names)), prefix="video").label
            target_path = mp4_by_label.get(dir_label)
        if target_path is not None:
            abs_path = (
                target_path if target_path.is_absolute() else (project_root / target_path)
            ).resolve()
            key = abs_path.as_posix()
            mp4_vid = existing_by_abs.get(key)
            if mp4_vid is None:
                mp4_vid = _Video.from_filename(key)
                existing_by_abs[key] = mp4_vid
                new_videos.append(mp4_vid)
            mapping[v] = mp4_vid
        else:
            new_videos.append(v)

    changed = False
    for lf in labels.labeled_frames:
        ov = lf.video
        if ov in mapping:
            lf.video = mapping[ov]
            changed = True

    if new_videos:
        seen: set[int] = set()
        deduped: list[Any] = []
        for v in new_videos:
            vid = id(v)
            if vid in seen:
                continue
            seen.add(vid)
            deduped.append(v)
        labels.videos = deduped

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
            fpath = Path(str(frame)).resolve()
            rel = fpath.relative_to(src_root)
            changed = True
            updated.append((dst_root / rel).as_posix())
        if changed:
            video._image_filenames = updated


__all__ = [
    "ConversionResult",
    "ProgressCallback",
    "encode_videos",
    "rebase_image_sequences",
    "remap_labels_to_videos",
]
