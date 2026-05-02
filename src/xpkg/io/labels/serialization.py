"""Serialization helpers for `Labels` JSON state documents and projects."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from xpkg._core.logging_utils import get_logger
from xpkg.io.labels.json_format import read_labels_json_payload, write_labels_json
from xpkg.io.labels.video_types import VideoProtocol
from xpkg.io.video import Video, gui_playback_backend_for_path
from xpkg.pose.annotations import (
    ROI,
    Instance,
    LabeledFrame,
    PointArray,
    SegmentationMask,
    Track,
)
from xpkg.pose.skeleton import SCHEMA_VERSION as SKELETON_SCHEMA_VERSION
from xpkg.pose.skeleton import Keypoint, Skeleton

if TYPE_CHECKING:
    from xpkg.io.labels.model import Labels, SuggestionFrame

logger = get_logger(__name__)


class VideoBuilder(Protocol):
    """Construct a hydrated video object from a serialized video entry."""

    def __call__(
        self,
        filename: str | None,
        *,
        image_filenames: Sequence[str] | None = None,
    ) -> VideoProtocol: ...


HydratedVideoFinalizer = Callable[[VideoProtocol], None]
LABEL_TRACK_ID_DATASET = "track_id"
LABEL_VISIBILITY_DATASET = "visibility"


def _materialize(value: Any) -> Any:
    from xpkg.io.labels.model import Materializable

    if isinstance(value, Materializable):
        return value.materialize()
    return value


def _as_array(
    value: Any,
    *,
    dtype: Any,
    fallback_shape: tuple[int, ...] | None = None,
    name: str = "array",
) -> np.ndarray:
    obj = _materialize(value)
    if obj is None:
        raise ValueError(f"Required field '{name}' missing in labels payload")
    arr = np.asarray(obj, dtype=dtype)
    if arr.ndim == 0:
        arr = arr.reshape(1)

    if fallback_shape is not None and arr.shape != fallback_shape:
        if arr.size == 0 and int(np.prod(fallback_shape, dtype=np.int64)) == 0:
            return np.zeros(fallback_shape, dtype=dtype)
        raise ValueError(
            f"Field '{name}' shape {arr.shape} does not match expected {fallback_shape}"
        )
    return arr


def _to_str(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes | bytearray):
        return value.decode("utf-8")
    return str(value) if value is not None else ""


def build_video_object(
    filename: str | None,
    *,
    image_filenames: Sequence[str] | None = None,
) -> VideoProtocol:
    """Construct a Video from a non-empty filename."""
    if image_filenames is not None:
        frames = [str(path).strip() for path in image_filenames]
        if not frames:
            raise ValueError("Image sequence is empty")
        sequence_root = str(filename).strip() if filename is not None else ""
        video = Video.from_image_filenames(frames)
        if sequence_root:
            video.filename = sequence_root
        return video

    resolved = str(filename or "").strip()
    if not resolved:
        raise ValueError("Video filename is empty")
    return Video.from_filename(
        resolved,
        backend=gui_playback_backend_for_path(resolved),
    )


def finalize_hydrated_video(video_obj: VideoProtocol) -> None:
    """Release hydrated video decode state until the caller needs it."""
    video_obj.close()


def _image_sequence_from_payload(entry: Any, *, video_index: int) -> list[str] | None:
    if entry is None:
        return None
    if not isinstance(entry, Sequence) or isinstance(entry, str | bytes | bytearray):
        raise TypeError(f"videos.image_filenames[{video_index}] must be a sequence of paths")

    out: list[str] = []
    for frame_index, raw_path in enumerate(entry):
        frame_path = str(raw_path).strip()
        if not frame_path:
            raise ValueError(
                f"videos.image_filenames[{video_index}][{frame_index}] cannot be empty"
            )
        out.append(frame_path)
    return out or None


def load_suggestions(
    suggestions: dict[str, Any] | None,
    video_list: list[VideoProtocol],
) -> list[SuggestionFrame]:
    """Hydrate suggestions rows into `SuggestionFrame` objects."""
    from xpkg.io.labels.model import SuggestionFrame

    if not suggestions:
        return []

    raw_video_idx = suggestions.get("video_indices")
    raw_frame_idx = suggestions.get("frame_indices")

    video_idx_arr = (
        np.asarray(_materialize(raw_video_idx), dtype=np.int64).ravel()
        if raw_video_idx is not None
        else np.zeros((0,), dtype=np.int64)
    )
    frame_idx_arr = (
        np.asarray(_materialize(raw_frame_idx), dtype=np.int64).ravel()
        if raw_frame_idx is not None
        else np.zeros((0,), dtype=np.int64)
    )

    if video_idx_arr.shape != frame_idx_arr.shape:
        raise ValueError("suggestions.video_indices/frame_indices length mismatch")

    raw_scores = suggestions.get("scores")
    score_arr = (
        np.asarray(_materialize(raw_scores), dtype=np.float32).ravel()
        if raw_scores is not None
        else None
    )
    if score_arr is not None and score_arr.shape[0] != video_idx_arr.shape[0]:
        raise ValueError("suggestions.scores length mismatch video/frame indices")

    out: list[SuggestionFrame] = []
    for idx, (vi, fi) in enumerate(
        zip(video_idx_arr.tolist(), frame_idx_arr.tolist(), strict=True)
    ):
        video_idx_int = int(vi)
        if video_idx_int < 0 or video_idx_int >= len(video_list):
            raise IndexError(
                f"Suggestion references video index {video_idx_int} "
                f"but only {len(video_list)} videos are loaded"
            )
        score_val = float(score_arr[idx]) if score_arr is not None else None
        out.append(
            SuggestionFrame(
                video=video_list[video_idx_int],
                frame_idx=int(fi),
                score=score_val,
            )
        )
    return out


def _load_frame_arrays(frames_info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    video_index = _as_array(
        frames_info.get("video_index"), dtype=np.int32, name="frames.video_index"
    )
    frame_index = _as_array(
        frames_info.get("frame_index"), dtype=np.int32, name="frames.frame_index"
    )
    num_instances = _as_array(
        frames_info.get("num_instances"), dtype=np.int32, name="frames.num_instances"
    )
    if not (len(video_index) == len(frame_index) == len(num_instances)):
        raise ValueError("frames.video_index/frame_index/num_instances length mismatch")
    return video_index, frame_index, num_instances


def _track_lookup_from_payload(payload: dict[str, Any]) -> dict[int, Track]:
    track_payload = payload.get("tracks")
    track_lookup: dict[int, Track] = {}
    if not isinstance(track_payload, Mapping):
        return track_lookup

    for raw_key, raw_track in track_payload.items():
        if isinstance(raw_key, str):
            stripped = raw_key.strip()
            if not stripped or not stripped.lstrip("+-").isdigit():
                continue
            track_id = int(stripped)
        elif isinstance(raw_key, int | np.integer):
            track_id = int(raw_key)
        else:
            continue
        if track_id < 0:
            continue
        if isinstance(raw_track, Track):
            track_lookup[track_id] = raw_track
            continue
        if isinstance(raw_track, Mapping):
            spawned_on = raw_track.get("spawned_on", track_id)
            track_lookup[track_id] = Track(
                spawned_on=int(spawned_on),
                name=str(raw_track.get("name") or f"track-{track_id}"),
            )
            continue
        track_lookup[track_id] = Track(
            spawned_on=track_id,
            name=str(raw_track or f"track-{track_id}"),
        )
    return track_lookup


def _normalize_keypoints_array(
    raw_keypoints: Any,
    *,
    frame_count: int,
    max_instances: int,
    keypoint_count: int,
) -> np.ndarray:
    keypoints_arr = _as_array(raw_keypoints, dtype=np.float32)
    if keypoints_arr.size == 0:
        return np.full(
            (int(frame_count), int(max_instances), int(keypoint_count), 3),
            np.nan,
            dtype=np.float32,
        )
    if keypoints_arr.ndim == 3:
        keypoints_arr = keypoints_arr[:, np.newaxis, :, :]
    elif keypoints_arr.ndim == 2:
        keypoints_arr = keypoints_arr[np.newaxis, np.newaxis, :, :]
    if keypoints_arr.ndim != 4:
        raise ValueError("keypoints array must be 4D (frames, instances, keypoints, 3)")
    return keypoints_arr


def _parse_keypoint_names(skeleton_info: dict[str, Any], keypoints_arr: np.ndarray) -> list[str]:
    raw_names = list(skeleton_info.get("names") or [])
    keypoint_names: list[str] = []
    for idx, raw_name in enumerate(raw_names):
        name = str(raw_name).strip() if raw_name is not None else ""
        if not name:
            raise ValueError(f"skeleton.names[{idx}] is empty")
        keypoint_names.append(name)

    kp_count = keypoints_arr.shape[2] if keypoints_arr.ndim >= 3 else len(keypoint_names)
    has_data = keypoints_arr.size > 0 and keypoints_arr.shape[0] > 0 and keypoints_arr.shape[1] > 0
    if not keypoint_names and has_data:
        raise ValueError(
            "Skeleton names missing from labels payload but keypoint data is present."
        )
    if keypoint_names and kp_count != len(keypoint_names):
        raise ValueError(
            "keypoints array count "
            f"({kp_count}) does not match skeleton names ({len(keypoint_names)})"
        )
    return keypoint_names


def _build_keypoints(skeleton_info: dict[str, Any], keypoint_names: list[str]) -> list[Keypoint]:
    raw_roles = list(skeleton_info.get("roles") or [])
    keypoints: list[Keypoint] = []
    for idx, name in enumerate(keypoint_names):
        role = None
        if idx < len(raw_roles):
            role = str(raw_roles[idx]).strip() or None
        keypoints.append(Keypoint(id=idx, name=name, role=role))

    symmetry_map = skeleton_info.get("symmetry")
    if isinstance(symmetry_map, dict):
        kp_lookup = {kp.name: kp for kp in keypoints}
        for src_name, dst_name in symmetry_map.items():
            src = str(src_name).strip()
            dst = str(dst_name).strip() if dst_name is not None else ""
            if src and src in kp_lookup:
                kp_lookup[src].mirror_partner = dst or None
    return keypoints


def _parse_links_ids(links_raw: Any, keypoint_count: int) -> list[tuple[int, int]]:
    links_ids: list[tuple[int, int]] = []
    if links_raw is None:
        return links_ids
    links_arr = np.asarray(links_raw)
    if links_arr.ndim == 2 and links_arr.shape[1] >= 2:
        iterable = links_arr
    elif isinstance(links_raw, list | tuple):
        iterable = links_raw
    else:
        iterable = []
    for entry in iterable:
        if not isinstance(entry, list | tuple | np.ndarray) or len(entry) < 2:
            continue
        a_idx = int(entry[0])
        b_idx = int(entry[1])
        if 0 <= a_idx < keypoint_count and 0 <= b_idx < keypoint_count:
            links_ids.append((a_idx, b_idx))
    return links_ids


def _build_skeleton(
    skeleton_info: dict[str, Any], keypoints_arr: np.ndarray
) -> tuple[Skeleton, list[Keypoint]]:
    keypoint_names = _parse_keypoint_names(skeleton_info, keypoints_arr)
    keypoints = _build_keypoints(skeleton_info, keypoint_names)
    links_ids = _parse_links_ids(skeleton_info.get("links"), len(keypoints))
    schema_version_val = skeleton_info.get("schema_version")
    if isinstance(schema_version_val, bytes | bytearray):
        schema_version_val = schema_version_val.decode("utf-8")
    schema_version = str(schema_version_val) if schema_version_val else SKELETON_SCHEMA_VERSION
    skeleton = Skeleton(
        name=str(skeleton_info.get("name") or "Skeleton"),
        keypoints=keypoints,
        links_ids=links_ids,
        schema_version=schema_version,
    )
    return skeleton, keypoints


def _load_label_data_arrays(
    data_info: dict[str, Any],
    keypoints_arr: np.ndarray,
    keypoint_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    frames_dim = keypoints_arr.shape[0] if keypoints_arr.ndim >= 1 else 0
    inst_dim = keypoints_arr.shape[1] if keypoints_arr.ndim >= 2 else 0
    point_dim = keypoints_arr.shape[2] if keypoints_arr.ndim >= 3 else keypoint_count

    flags_raw = data_info.get("flags")
    if flags_raw is None:
        flags_arr = np.zeros((frames_dim, inst_dim, point_dim), dtype=np.uint8)
    else:
        flags_arr = _as_array(
            flags_raw,
            dtype=np.uint8,
            fallback_shape=(frames_dim, inst_dim, point_dim),
            name="data.flags",
        )

    track_ids_raw = data_info.get(LABEL_TRACK_ID_DATASET)
    if track_ids_raw is None:
        track_ids_raw = data_info.get("track_ids")
    if track_ids_raw is None:
        track_ids_arr = np.full((frames_dim, inst_dim), -1, dtype=np.int32)
    else:
        track_ids_arr = _as_array(
            track_ids_raw,
            dtype=np.int32,
            fallback_shape=(frames_dim, inst_dim),
            name=f"data.{LABEL_TRACK_ID_DATASET}",
        )

    visibility_arr: np.ndarray | None = None
    visibility_raw = data_info.get(LABEL_VISIBILITY_DATASET)
    if visibility_raw is not None:
        visibility_arr = _as_array(
            visibility_raw,
            dtype=np.uint8,
            fallback_shape=(frames_dim, inst_dim, point_dim),
            name=f"data.{LABEL_VISIBILITY_DATASET}",
        )
    return flags_arr, track_ids_arr, visibility_arr


def _parse_shapes(videos_info: dict[str, Any]) -> np.ndarray:
    shapes_raw = _materialize(videos_info.get("shapes"))
    if (
        isinstance(shapes_raw, Sequence)
        and not isinstance(shapes_raw, bytes | bytearray | str)
        and len(shapes_raw) == 0
    ):
        return np.zeros((0, 4), dtype=np.int32)
    shapes_arr = (
        np.asarray(shapes_raw, dtype=np.int32)
        if shapes_raw is not None
        else np.zeros((0, 4), dtype=np.int32)
    )
    if shapes_arr.size == 0:
        return np.zeros((0, 4), dtype=np.int32)
    if shapes_raw is not None and (shapes_arr.ndim != 2 or shapes_arr.shape[1] < 4):
        raise ValueError(
            "videos.shapes must be 2D with at least 4 columns (frames,height,width,channels)"
        )
    return shapes_arr


def _total_videos_count(
    videos_info: dict[str, Any], video_index: np.ndarray, frames_dim: int
) -> int:
    resolved_paths = list(videos_info.get("resolved_paths") or [])
    filenames_raw = list(videos_info.get("filenames") or [])
    image_sequences_raw = list(videos_info.get("image_filenames") or [])
    backends_raw = list(videos_info.get("backends") or [])
    sha_raw = list(videos_info.get("sha256") or [])
    shapes_arr = _parse_shapes(videos_info)
    max_video_idx = int(video_index.max()) + 1 if video_index.size else 0
    total_videos = max(
        len(resolved_paths),
        len(filenames_raw),
        len(image_sequences_raw),
        len(backends_raw),
        len(sha_raw),
        shapes_arr.shape[0],
        max_video_idx,
    )
    if total_videos == 0 and frames_dim:
        raise ValueError("Frames are present but no videos listed in labels payload")
    return total_videos


def _hydrate_videos(
    videos_info: dict[str, Any],
    video_index: np.ndarray,
    frames_dim: int,
    *,
    video_builder: VideoBuilder,
    video_finalizer: HydratedVideoFinalizer,
) -> list[VideoProtocol]:
    total_videos = _total_videos_count(videos_info, video_index, frames_dim)
    resolved_paths = list(videos_info.get("resolved_paths") or [])
    video_ids = list(videos_info.get("video_ids") or [])
    video_labels = list(videos_info.get("video_labels") or [])
    filenames_raw = list(videos_info.get("filenames") or [])
    image_sequences_raw = list(videos_info.get("image_filenames") or [])
    backends_raw = list(videos_info.get("backends") or [])
    sha_raw = list(videos_info.get("sha256") or [])

    videos: list[VideoProtocol] = []
    for idx in range(total_videos):
        filepath = ""
        if idx < len(resolved_paths) and resolved_paths[idx]:
            filepath = _to_str(resolved_paths[idx])
        elif idx < len(filenames_raw):
            filepath = _to_str(filenames_raw[idx])

        image_filenames = None
        media_label = filepath
        if idx < len(image_sequences_raw):
            image_filenames = _image_sequence_from_payload(
                image_sequences_raw[idx],
                video_index=idx,
            )
            if image_filenames is not None:
                media_label = image_filenames[0]

        if not filepath and image_filenames is None:
            raise ValueError(
                f"Video entry {idx} missing filename/resolved_paths or image_filenames"
            )

        video_obj = video_builder(filepath or None, image_filenames=image_filenames)
        if idx < len(video_ids):
            video_obj.id = str(video_ids[idx])
        if idx < len(video_labels):
            video_obj.label = str(video_labels[idx])
        if idx < len(backends_raw):
            backend_name = str(backends_raw[idx]).strip()
            if backend_name and backend_name != video_obj.backend:
                logger.debug(
                    "Ignoring serialized video backend '%s' for %s; loaded backend is '%s'.",
                    backend_name,
                    media_label,
                    video_obj.backend,
                )
        if idx < len(sha_raw):
            sha = str(sha_raw[idx]).strip()
            if sha:
                video_obj.sha256 = sha

        video_finalizer(video_obj)
        videos.append(video_obj)

    max_video_idx = int(video_index.max()) + 1 if video_index.size else 0
    if len(videos) < max_video_idx:
        raise FileNotFoundError(
            "Labels payload references missing videos; all video slots must be present."
        )
    return videos


def _total_frames_count(
    metadata: dict[str, Any],
    keypoints_arr: np.ndarray,
    video_index: np.ndarray,
    frame_index: np.ndarray,
    num_instances: np.ndarray,
) -> int:
    frame_candidates = [
        keypoints_arr.shape[0] if keypoints_arr.ndim >= 1 else 0,
        video_index.shape[0],
        frame_index.shape[0],
        num_instances.shape[0],
    ]
    meta_frames = metadata.get("num_frames")
    if isinstance(meta_frames, int | float | np.integer | np.floating):
        frame_candidates.append(int(meta_frames))
    return max(frame_candidates) if frame_candidates else 0

def _instance_has_payload(
    row_idx: int,
    inst_idx: int,
    keypoints_arr: np.ndarray,
    flags_arr: np.ndarray,
    track_ids_arr: np.ndarray,
) -> bool:
    if (
        keypoints_arr.ndim >= 4
        and row_idx < keypoints_arr.shape[0]
        and inst_idx < keypoints_arr.shape[1]
    ):
        block = keypoints_arr[row_idx, inst_idx]
        if block.size and not np.isnan(block[..., :2]).all():
            return True
        if block.size and block.shape[-1] >= 3 and not np.isnan(block[..., 2]).all():
            return True
    if (
        flags_arr.size
        and row_idx < flags_arr.shape[0]
        and inst_idx < flags_arr.shape[1]
        and np.any(flags_arr[row_idx, inst_idx])
    ):
        return True
    if (
        track_ids_arr.size
        and row_idx < track_ids_arr.shape[0]
        and inst_idx < track_ids_arr.shape[1]
        and int(track_ids_arr[row_idx, inst_idx]) >= 0
    ):
        return True
    return False


def _resolve_track(
    row: int,
    inst_idx: int,
    track_ids_arr: np.ndarray,
    track_lookup: dict[int, Track],
) -> Track | None:
    if not track_ids_arr.size:
        return None
    if row >= track_ids_arr.shape[0] or inst_idx >= track_ids_arr.shape[1]:
        return None
    track_id_val = int(track_ids_arr[row, inst_idx])
    if track_id_val < 0:
        return None
    track_obj = track_lookup.get(track_id_val)
    if track_obj is None:
        track_obj = Track(spawned_on=track_id_val, name=f"track-{track_id_val}")
        track_lookup[track_id_val] = track_obj
    return track_obj


def _build_instance_points(
    inst_coords: np.ndarray,
    inst_flags: np.ndarray,
    keypoint_count: int,
    *,
    visibility: np.ndarray | None = None,
) -> PointArray:
    points = PointArray.make_default(keypoint_count)
    points["x"] = inst_coords[..., 0] if inst_coords.shape[-1] >= 1 else np.nan
    points["y"] = inst_coords[..., 1] if inst_coords.shape[-1] >= 2 else np.nan
    has_coords = np.isfinite(points["x"]) & np.isfinite(points["y"])
    if visibility is not None:
        visible = np.asarray(visibility, dtype=bool) & has_coords
    elif inst_coords.shape[-1] >= 3:
        conf = inst_coords[..., 2]
        visible = np.isfinite(conf) & (conf >= 0.5) & has_coords
    else:
        visible = has_coords
    points["visible"] = visible
    points["complete"] = has_coords
    points["flags"] = inst_flags & 0xFF
    return points


def _hydrate_row_instances(
    row: int,
    lf: LabeledFrame,
    *,
    declared_instances: int,
    keypoints_arr: np.ndarray,
    flags_arr: np.ndarray,
    track_ids_arr: np.ndarray,
    visibility_arr: np.ndarray | None,
    track_lookup: dict[int, Track],
    keypoint_count: int,
    skeleton: Skeleton,
) -> None:
    available_instances = keypoints_arr.shape[1] if keypoints_arr.ndim >= 2 else 0
    iter_count = max(declared_instances, available_instances)
    for inst_idx in range(iter_count):
        if inst_idx >= declared_instances and not _instance_has_payload(
            row, inst_idx, keypoints_arr, flags_arr, track_ids_arr
        ):
            continue
        if inst_idx >= keypoints_arr.shape[1]:
            raise ValueError(
                f"Instance index {inst_idx} exceeds keypoints array width {keypoints_arr.shape[1]}"
            )
        track_obj = _resolve_track(row, inst_idx, track_ids_arr, track_lookup)
        inst_coords = keypoints_arr[row, inst_idx]
        inst_flags = flags_arr[row, inst_idx]
        inst_visibility = None
        if (
            visibility_arr is not None
            and row < visibility_arr.shape[0]
            and inst_idx < visibility_arr.shape[1]
        ):
            inst_visibility = visibility_arr[row, inst_idx]
        points = _build_instance_points(
            inst_coords,
            inst_flags,
            keypoint_count,
            visibility=inst_visibility,
        )
        lf.instances.append(
            Instance(skeleton=skeleton, frame=lf, track=track_obj, init_points=points)
        )


def _hydrate_labeled_frames(
    *,
    total_frames: int,
    video_index: np.ndarray,
    frame_index: np.ndarray,
    num_instances: np.ndarray,
    keypoints_arr: np.ndarray,
    flags_arr: np.ndarray,
    track_ids_arr: np.ndarray,
    visibility_arr: np.ndarray | None,
    videos: list[VideoProtocol],
    skeleton: Skeleton,
    keypoint_count: int,
    track_lookup: dict[int, Track] | None = None,
) -> list[LabeledFrame]:
    resolved_track_lookup = {} if track_lookup is None else dict(track_lookup)
    labeled_frames: list[LabeledFrame] = []
    for row in range(total_frames):
        if row % 1000 == 0:
            logger.debug("Processing frame row %d/%d", row, total_frames)
        if row >= video_index.shape[0]:
            raise ValueError(f"frames.video_index missing entry for row {row}")
        vi = int(video_index[row])
        if vi < 0:
            raise ValueError(f"frames.video_index[{row}] is negative")
        if vi >= len(videos):
            raise IndexError(
                f"Labels reference video index {vi} but only {len(videos)} videos are present"
            )
        if row >= frame_index.shape[0]:
            raise ValueError(f"frames.frame_index missing entry for row {row}")
        if row >= num_instances.shape[0]:
            raise ValueError(f"frames.num_instances missing entry for row {row}")
        if row >= keypoints_arr.shape[0]:
            raise ValueError(f"keypoints array missing row {row}")
        declared_instances = int(num_instances[row])
        if declared_instances < 0:
            raise ValueError(f"frames.num_instances[{row}] is negative")

        lf = LabeledFrame(video=cast(Any, videos[vi]), frame_idx=int(frame_index[row]))
        _hydrate_row_instances(
            row,
            lf,
            declared_instances=declared_instances,
            keypoints_arr=keypoints_arr,
            flags_arr=flags_arr,
            track_ids_arr=track_ids_arr,
            visibility_arr=visibility_arr,
            track_lookup=resolved_track_lookup,
            keypoint_count=keypoint_count,
            skeleton=skeleton,
        )
        labeled_frames.append(lf)
    return labeled_frames


def _resolve_segmentation_track(
    raw_track_id: Any,
    track_lookup: dict[int, Track],
) -> Track | None:
    if raw_track_id is None:
        return None
    track_id = int(raw_track_id)
    if track_id < 0:
        return None
    track = track_lookup.get(track_id)
    if track is None:
        track = Track(spawned_on=track_id, name=f"track-{track_id}")
        track_lookup[track_id] = track
    return track


def _frame_lookup_by_video_and_index(
    labeled_frames: list[LabeledFrame],
) -> dict[tuple[VideoProtocol, int], LabeledFrame]:
    return {(frame.video, int(frame.frame_idx)): frame for frame in labeled_frames}


def _attach_segmentation_entries(
    entries: Any,
    *,
    item_type: type[SegmentationMask] | type[ROI],
    attr_name: str,
    videos: list[VideoProtocol],
    labeled_frames: list[LabeledFrame],
    track_lookup: dict[int, Track],
) -> None:
    if not isinstance(entries, Sequence) or isinstance(entries, str | bytes | bytearray):
        return

    frames_by_key = _frame_lookup_by_video_and_index(labeled_frames)
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping):
            raise TypeError(f"{attr_name} entries must be mappings")
        video_idx = int(raw_entry.get("video_index", -1))
        if video_idx < 0 or video_idx >= len(videos):
            raise IndexError(f"{attr_name} entry references invalid video index {video_idx}")
        frame_idx = int(raw_entry.get("frame_index", -1))
        frame_key = (videos[video_idx], frame_idx)
        labeled_frame = frames_by_key.get(frame_key)
        if labeled_frame is None:
            labeled_frame = LabeledFrame(video=cast(Any, videos[video_idx]), frame_idx=frame_idx)
            labeled_frames.append(labeled_frame)
            frames_by_key[frame_key] = labeled_frame

        track = _resolve_segmentation_track(raw_entry.get("track_id"), track_lookup)
        item_payload = dict(raw_entry)
        item_payload.pop("video_index", None)
        item_payload.pop("frame_index", None)
        item = item_type.from_dict(item_payload, track=track)
        getattr(labeled_frame, attr_name).append(item)


def _attach_segmentation_payload(
    segmentation_payload: Any,
    *,
    videos: list[VideoProtocol],
    labeled_frames: list[LabeledFrame],
    track_lookup: dict[int, Track],
) -> None:
    if not isinstance(segmentation_payload, Mapping):
        return
    _attach_segmentation_entries(
        segmentation_payload.get("masks"),
        item_type=SegmentationMask,
        attr_name="masks",
        videos=videos,
        labeled_frames=labeled_frames,
        track_lookup=track_lookup,
    )
    _attach_segmentation_entries(
        segmentation_payload.get("rois"),
        item_type=ROI,
        attr_name="rois",
        videos=videos,
        labeled_frames=labeled_frames,
        track_lookup=track_lookup,
    )


def labels_from_payload(
    cls: type[Labels],
    payload: dict[str, Any] | None,
    *,
    suggestions_payload: dict[str, Any] | None = None,
    video_builder: VideoBuilder | None = None,
    video_finalizer: HydratedVideoFinalizer | None = None,
) -> Labels:
    """Construct `Labels` from a labels payload dictionary."""
    if not payload:
        raise ValueError("Empty labels payload; cannot hydrate Labels")

    frames_info = payload.get("frames") or {}
    data_info = payload.get("data") or {}
    videos_info = payload.get("videos") or {}
    metadata = payload.get("metadata") or {}
    skeleton_info = payload.get("skeleton") or {}
    if suggestions_payload is None and isinstance(payload, dict):
        suggestions_payload = payload.get("suggestions")
    resolved_video_builder = build_video_object if video_builder is None else video_builder
    resolved_video_finalizer = (
        finalize_hydrated_video if video_finalizer is None else video_finalizer
    )

    video_index, frame_index, num_instances = _load_frame_arrays(frames_info)
    raw_keypoints = _materialize(data_info.get("keypoints"))
    if raw_keypoints is None:
        raise ValueError("data.keypoints missing from labels payload")
    if (
        isinstance(raw_keypoints, Sequence)
        and not isinstance(raw_keypoints, bytes | bytearray | str)
        and len(raw_keypoints) == 0
    ):
        raw_keypoints = np.empty(
            (0, 1, len(list(skeleton_info.get("names") or [])), 3),
            dtype=np.float32,
        )

    frame_count = int(frame_index.shape[0])
    max_instances = int(num_instances.max()) if num_instances.size else 1
    max_instances = max(max_instances, 1)
    keypoint_count = len(list(skeleton_info.get("names") or []))
    keypoints_arr = _normalize_keypoints_array(
        raw_keypoints,
        frame_count=frame_count,
        max_instances=max_instances,
        keypoint_count=keypoint_count,
    )
    skeleton, keypoints = _build_skeleton(skeleton_info, keypoints_arr)
    flags_arr, track_ids_arr, visibility_arr = _load_label_data_arrays(
        data_info,
        keypoints_arr,
        len(keypoints),
    )
    track_lookup = _track_lookup_from_payload(payload)
    frames_dim = keypoints_arr.shape[0] if keypoints_arr.ndim >= 1 else 0
    videos = _hydrate_videos(
        videos_info,
        video_index,
        frames_dim,
        video_builder=resolved_video_builder,
        video_finalizer=resolved_video_finalizer,
    )
    total_frames = _total_frames_count(
        metadata, keypoints_arr, video_index, frame_index, num_instances
    )

    logger.debug(
        "Labels.from_payload: %d videos, %d frames, keypoints_shape=%s",
        len(videos),
        total_frames,
        keypoints_arr.shape,
    )
    labeled_frames = _hydrate_labeled_frames(
        total_frames=total_frames,
        video_index=video_index,
        frame_index=frame_index,
        num_instances=num_instances,
        keypoints_arr=keypoints_arr,
        flags_arr=flags_arr,
        track_ids_arr=track_ids_arr,
        visibility_arr=visibility_arr,
        videos=videos,
        skeleton=skeleton,
        keypoint_count=len(keypoints),
        track_lookup=track_lookup,
    )
    _attach_segmentation_payload(
        payload.get("segmentation"),
        videos=videos,
        labeled_frames=labeled_frames,
        track_lookup=track_lookup,
    )

    provenance = payload.get("provenance") or {}
    session = payload.get("session") or {}

    suggestions = load_suggestions(suggestions_payload, videos)
    has_explicit_skeleton = bool(skeleton_info)

    labels_obj = cls(
        labeled_frames=labeled_frames,
        videos=videos,
        skeletons=[skeleton] if keypoints or has_explicit_skeleton else [],
        keypoints=list(keypoints),
        suggestions=suggestions,
        provenance=provenance,
        session=session,
        preferences=metadata.get("preferences", {}),
    )
    for track in track_lookup.values():
        if track not in labels_obj.tracks:
            labels_obj.tracks.append(track)
    labels_obj.validate()
    return labels_obj


def labels_load_file(
    cls: type[Labels],
    filename: str,
    *args: Any,
    video_builder: VideoBuilder | None = None,
    video_finalizer: HydratedVideoFinalizer | None = None,
    **kwargs: Any,
) -> Labels:
    """Load labels from disk."""
    del args, kwargs
    path = Path(filename)
    if path.suffix.lower() == ".expkg":
        raise ValueError("Packed .expkg artifacts must be unpacked before loading labels")

    from xpkg.project.layout import project_current_state_path, resolve_project_root
    from xpkg.project.state_io import read_project_state, state_commit_id
    from xpkg.project.store import (
        _project_state_cache_matches_committed_head,
        current_project_commit_id,
        rebase_project_payload_videos,
        rebuild_project_state_cache,
    )

    project_root = resolve_project_root(path)
    if project_root is not None:
        state_path = project_current_state_path(project_root)
        if state_path.exists():
            state_payload = read_project_state(state_path)
            state_head = state_commit_id(state_payload)
            current_head = current_project_commit_id(project_root)
            if current_head is not None and state_head == current_head:
                if not _project_state_cache_matches_committed_head(
                    project_root,
                    state_path,
                ):
                    rebuilt_state_path = rebuild_project_state_cache(project_root)
                    state_payload = read_project_state(rebuilt_state_path)
                rebase_project_payload_videos(state_payload, project_root)
                obj = labels_from_payload(
                    cls,
                    state_payload,
                    suggestions_payload=state_payload.get("suggestions"),
                    video_builder=video_builder,
                    video_finalizer=video_finalizer,
                )
                obj.validate()
                obj.path = project_root
                return obj

        try:
            rebuilt_state_path = rebuild_project_state_cache(project_root)
        except FileNotFoundError:
            obj = cls()
            obj.path = project_root
            return obj

        rebuilt_payload = read_project_state(rebuilt_state_path)
        rebase_project_payload_videos(rebuilt_payload, project_root)
        obj = labels_from_payload(
            cls,
            rebuilt_payload,
            suggestions_payload=rebuilt_payload.get("suggestions"),
            video_builder=video_builder,
            video_finalizer=video_finalizer,
        )
        obj.validate()
        obj.path = project_root
        return obj

    ext = path.suffix.lower()
    if ext == ".json":
        payload = read_labels_json_payload(path)
        obj = labels_from_payload(
            cls,
            payload,
            suggestions_payload=payload.get("suggestions") if isinstance(payload, dict) else None,
            video_builder=video_builder,
            video_finalizer=video_finalizer,
        )
        obj.validate()
        obj.path = path
        return obj
    raise ValueError(f"No serializer for extension: {ext or '<none>'}")


def labels_save_file(
    labels: Labels,
    filename: str,
    *,
    default_suffix: str = "",
    metadata: dict[str, Any] | None = None,
    **_: Any,
) -> str:
    """Save labels to disk."""
    path = Path(filename)
    from xpkg.project.layout import resolve_project_root
    from xpkg.project.store import save_project_labels

    project_root = resolve_project_root(path)
    if project_root is not None:
        save_project_labels(
            project_root,
            labels,
            metadata=metadata,
        )
        labels.path = project_root
        return project_root.as_posix()

    ext = path.suffix.lower() or default_suffix or ".json"
    if ext == ".json":
        if not path.suffix:
            path = path.with_suffix(".json")
        return write_labels_json(path, labels, metadata=metadata)
    raise ValueError(f"No serializer for extension: {ext}")


__all__ = [
    "build_video_object",
    "finalize_hydrated_video",
    "labels_from_payload",
    "labels_load_file",
    "labels_save_file",
    "load_suggestions",
]
