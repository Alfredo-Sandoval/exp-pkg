"""Video encoding, image-sequence rebasing, and label video remapping.

Used by converters that ingest external pose datasets where label records
reference image sequences. We encode each sequence to MP4 and rewrite the
label references to the new video objects.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from xpkg._core.path_registry import ensure_dir
from xpkg.io.converters.progress import ProgressCallback, emit_progress
from xpkg.media import video_total_frames
from xpkg.media.video import Video, available_video_exts, write_video

if TYPE_CHECKING:
    from xpkg.model import Labels as _Labels

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


class LabelsVideoRemapProtocol(Protocol):
    """Mutable label-container surface required when rebasing video references."""

    videos: list[Any]
    labeled_frames: list[Any]

    def merge_matching_frames(self) -> None: ...

    def update_cache(self) -> None: ...


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
        emit_progress(progress, "XPKG_IMPORT STEP: build_video")

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


__all__ = [
    "LabelsVideoRemapProtocol",
    "encode_videos",
    "rebase_image_sequences",
    "remap_labels_to_videos",
]
