"""Serialization helpers for `Labels` bundles."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from posetta.core.annotations import Instance, LabeledFrame, PointArray, Track
from posetta.core.logging_utils import get_logger
from posetta.core.path_registry import ensure_dir
from posetta.core.skeleton import SCHEMA_VERSION as SKELETON_SCHEMA_VERSION
from posetta.core.skeleton import Keypoint, Skeleton
from posetta.io.labels.video_types import VideoProtocol
from posetta.io.video import Video, gui_playback_backend_for_path

from .json_format import read_labels_json_payload, write_labels_json

if TYPE_CHECKING:
    from posetta.io.labels.model import Labels, SuggestionFrame

logger = get_logger(__name__)


def _materialize(value: Any) -> Any:
    from posetta.io.labels.model import Materializable

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
        raise ValueError(f"Required field '{name}' missing in .siesta payload")
    arr = np.asarray(obj, dtype=dtype)
    if arr.ndim == 0:
        arr = arr.reshape(1)

    if fallback_shape is not None and arr.shape != fallback_shape:
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
    filename: str,
) -> VideoProtocol:
    """Construct a Video from a non-empty filename."""
    resolved = str(filename or "").strip()
    if not resolved:
        raise ValueError("Video filename is empty")
    return Video.from_filename(
        resolved,
        backend=gui_playback_backend_for_path(resolved),
    )


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
    from posetta.io.labels.model import SuggestionFrame

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


def labels_from_siesta_payload(
    cls: type[Labels],
    payload: dict[str, Any] | None,
    *,
    suggestions_payload: dict[str, Any] | None = None,
) -> Labels:
    """Construct `Labels` from a `.siesta` payload dictionary."""
    if not payload:
        raise ValueError("Empty .siesta payload; cannot hydrate Labels")

    frames_info = payload.get("frames") or {}
    data_info = payload.get("data") or {}
    videos_info = payload.get("videos") or {}
    metadata = payload.get("metadata") or {}
    skeleton_info = payload.get("skeleton") or {}
    if suggestions_payload is None and isinstance(payload, dict):
        suggestions_payload = payload.get("suggestions")

    video_index = _as_array(
        frames_info.get("video_index"), dtype=np.int32, name="frames.video_index"
    )
    frame_index = _as_array(
        frames_info.get("frame_index"), dtype=np.int32, name="frames.frame_index"
    )
    num_instances = _as_array(
        frames_info.get("num_instances"),
        dtype=np.int32,
        name="frames.num_instances",
    )

    if not (len(video_index) == len(frame_index) == len(num_instances)):
        raise ValueError("frames.video_index/frame_index/num_instances length mismatch")

    raw_keypoints = _materialize(data_info.get("keypoints"))
    if raw_keypoints is None:
        raise ValueError("data.keypoints missing from .siesta payload")

    raw_names = list(skeleton_info.get("names") or [])
    keypoint_names: list[str] = []
    for idx, raw_name in enumerate(raw_names):
        name = str(raw_name).strip() if raw_name is not None else ""
        if not name:
            raise ValueError(f"skeleton.names[{idx}] is empty")
        keypoint_names.append(name)
    keypoints_arr = _as_array(raw_keypoints, dtype=np.float32)
    if keypoints_arr.ndim == 3:
        keypoints_arr = keypoints_arr[:, np.newaxis, :, :]
    elif keypoints_arr.ndim == 2:
        keypoints_arr = keypoints_arr[np.newaxis, np.newaxis, :, :]
    if keypoints_arr.ndim != 4:
        raise ValueError("keypoints array must be 4D (frames, instances, keypoints, 3)")

    kp_count = keypoints_arr.shape[2] if keypoints_arr.ndim >= 3 else len(keypoint_names)

    has_data = keypoints_arr.size > 0 and keypoints_arr.shape[0] > 0 and keypoints_arr.shape[1] > 0

    if not keypoint_names and has_data:
        raise ValueError(
            "Skeleton names missing from .siesta payload but keypoint data is present."
        )

    if keypoint_names and kp_count != len(keypoint_names):
        raise ValueError(
            "keypoints array count "
            f"({kp_count}) does not match skeleton names ({len(keypoint_names)})"
        )

    raw_roles = list(skeleton_info.get("roles") or [])
    keypoints: list[Keypoint] = []
    for idx, name in enumerate(keypoint_names):
        role = None
        if idx < len(raw_roles):
            role = str(raw_roles[idx]).strip() or None
        keypoints.append(Keypoint(id=idx, name=name, role=role))

    kp_lookup = {kp.name: kp for kp in keypoints}

    symmetry_map = skeleton_info.get("symmetry")
    if isinstance(symmetry_map, dict):
        for src_name, dst_name in symmetry_map.items():
            src = str(src_name).strip()
            dst = str(dst_name).strip() if dst_name is not None else ""
            if src and src in kp_lookup:
                kp_lookup[src].mirror_partner = dst or None

    links_ids: list[tuple[int, int]] = []
    links_raw = skeleton_info.get("links")
    if links_raw is not None:
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
            if 0 <= a_idx < len(keypoints) and 0 <= b_idx < len(keypoints):
                links_ids.append((a_idx, b_idx))

    schema_version_val = skeleton_info.get("schema_version")
    if isinstance(schema_version_val, bytes | bytearray):
        schema_version_val = schema_version_val.decode("utf-8")
    schema_version = str(schema_version_val) if schema_version_val else SKELETON_SCHEMA_VERSION
    skeleton_name = str(skeleton_info.get("name") or "Skeleton")
    skeleton = Skeleton(
        name=skeleton_name,
        keypoints=keypoints,
        links_ids=links_ids,
        schema_version=schema_version,
    )

    frames_dim = keypoints_arr.shape[0] if keypoints_arr.ndim >= 1 else 0
    inst_dim = keypoints_arr.shape[1] if keypoints_arr.ndim >= 2 else 0
    point_dim = keypoints_arr.shape[2] if keypoints_arr.ndim >= 3 else len(keypoints)

    flags_arr = _as_array(
        data_info.get("flags"),
        dtype=np.uint8,
        fallback_shape=(frames_dim, inst_dim, point_dim),
        name="data.flags",
    )

    track_ids_arr = _as_array(
        data_info.get("track_ids"),
        dtype=np.int32,
        fallback_shape=(frames_dim, inst_dim),
        name="data.track_ids",
    )

    resolved_paths = list(videos_info.get("resolved_paths") or [])
    video_ids = list(videos_info.get("video_ids") or [])
    video_labels = list(videos_info.get("video_labels") or [])
    filenames_raw = list(videos_info.get("filenames") or [])
    image_sequences_raw = list(videos_info.get("image_filenames") or [])
    backends_raw = list(videos_info.get("backends") or [])
    sha_raw = list(videos_info.get("sha256") or [])
    shapes_raw = _materialize(videos_info.get("shapes"))
    shapes_arr = (
        np.asarray(shapes_raw, dtype=np.int32)
        if shapes_raw is not None
        else np.zeros((0, 4), dtype=np.int32)
    )
    if shapes_raw is not None and (shapes_arr.ndim != 2 or shapes_arr.shape[1] < 4):
        raise ValueError(
            "videos.shapes must be 2D with at least 4 columns (frames,height,width,channels)"
        )

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
        raise ValueError("Frames are present but no videos listed in .siesta payload")

    videos: list[VideoProtocol] = []
    for idx in range(total_videos):
        image_filenames = None
        media_label = ""
        if idx < len(image_sequences_raw):
            image_filenames = _image_sequence_from_payload(
                image_sequences_raw[idx],
                video_index=idx,
            )
        if image_filenames is not None:
            video_obj = Video.from_image_filenames(image_filenames)
            media_label = image_filenames[0]
        else:
            filepath = ""
            if idx < len(resolved_paths) and resolved_paths[idx]:
                filepath = _to_str(resolved_paths[idx])
            elif idx < len(filenames_raw):
                filepath = _to_str(filenames_raw[idx])
            if not filepath:
                raise ValueError(
                    f"Video entry {idx} missing filename/resolved_paths or image_filenames"
                )
            video_obj = build_video_object(filepath)
            media_label = filepath

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

        video_obj.close()

        videos.append(video_obj)

    if len(videos) < max_video_idx:
        raise FileNotFoundError(
            "Labels bundle references missing videos; all video slots must be present."
        )

    frame_candidates = [
        keypoints_arr.shape[0] if keypoints_arr.ndim >= 1 else 0,
        video_index.shape[0],
        frame_index.shape[0],
        num_instances.shape[0],
    ]
    meta_frames = metadata.get("num_frames")
    if isinstance(meta_frames, int | float | np.integer | np.floating):
        frame_candidates.append(int(meta_frames))
    total_frames = max(frame_candidates) if frame_candidates else 0

    track_lookup: dict[int, Track] = {}
    labeled_frames: list[LabeledFrame] = []

    def _instance_has_payload(row_idx: int, inst_idx: int) -> bool:
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

    logger.debug(
        "Labels.from_siesta_payload: %d videos, %d frames, keypoints_shape=%s",
        len(videos),
        total_frames,
        keypoints_arr.shape,
    )

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

        video_obj = videos[vi]

        if row >= frame_index.shape[0]:
            raise ValueError(f"frames.frame_index missing entry for row {row}")
        frame_idx_val = int(frame_index[row])

        lf = LabeledFrame(video=cast(Any, video_obj), frame_idx=frame_idx_val)

        if row >= num_instances.shape[0]:
            raise ValueError(f"frames.num_instances missing entry for row {row}")
        declared_instances = int(num_instances[row])
        if declared_instances < 0:
            raise ValueError(f"frames.num_instances[{row}] is negative")
        if row >= keypoints_arr.shape[0]:
            raise ValueError(f"keypoints array missing row {row}")
        available_instances = keypoints_arr.shape[1] if keypoints_arr.ndim >= 2 else 0
        iter_count = max(declared_instances, available_instances)

        for inst_idx in range(iter_count):
            if inst_idx >= declared_instances and not _instance_has_payload(row, inst_idx):
                continue

            if inst_idx >= keypoints_arr.shape[1]:
                raise ValueError(
                    "Instance index "
                    f"{inst_idx} exceeds keypoints array width {keypoints_arr.shape[1]}"
                )

            track_obj: Track | None = None
            track_id_val: int | None = None
            if (
                track_ids_arr.size
                and row < track_ids_arr.shape[0]
                and inst_idx < track_ids_arr.shape[1]
            ):
                candidate = int(track_ids_arr[row, inst_idx])
                if candidate >= 0:
                    track_id_val = candidate
            if track_id_val is not None:
                track_obj = track_lookup.get(track_id_val)
                if track_obj is None:
                    track_obj = Track(spawned_on=track_id_val, name=str(track_id_val))
                    track_lookup[track_id_val] = track_obj

            inst_coords = keypoints_arr[row, inst_idx]
            inst_flags = flags_arr[row, inst_idx]

            points = PointArray.make_default(len(keypoints))

            if inst_coords.shape[-1] >= 1:
                points["x"] = inst_coords[..., 0]
            else:
                points["x"] = np.nan

            if inst_coords.shape[-1] >= 2:
                points["y"] = inst_coords[..., 1]
            else:
                points["y"] = np.nan

            if inst_coords.shape[-1] >= 3:
                conf = inst_coords[..., 2]
                visible = (
                    np.isfinite(conf)
                    & (conf >= 0.5)
                    & np.isfinite(points["x"])
                    & np.isfinite(points["y"])
                )
            else:
                visible = np.zeros(len(keypoints), dtype=bool)

            points["visible"] = visible
            points["complete"] = visible
            points["flags"] = inst_flags & 0xFF

            inst = Instance(skeleton=skeleton, frame=lf, track=track_obj, init_points=points)

            lf.instances.append(inst)

        labeled_frames.append(lf)

    provenance = payload.get("provenance") or {}
    session = payload.get("session") or {}

    suggestions = load_suggestions(suggestions_payload, videos)

    labels_obj = cls(
        labeled_frames=labeled_frames,
        videos=videos,
        skeletons=[skeleton] if keypoints else [],
        keypoints=list(keypoints),
        suggestions=suggestions,
        provenance=provenance,
        session=session,
        preferences=metadata.get("preferences", {}),
    )
    labels_obj.validate()
    return labels_obj


def labels_load_file(cls: type[Labels], filename: str, *args: Any, **kwargs: Any) -> Labels:
    """Load labels from disk."""
    from posetta.io.siesta_format import read_siesta

    path = Path(filename)
    ext = path.suffix.lower()
    if ext == ".json":
        payload = read_labels_json_payload(path)
        obj = labels_from_siesta_payload(
            cls,
            payload,
            suggestions_payload=payload.get("suggestions") if isinstance(payload, dict) else None,
        )
        obj.validate()
        obj.path = path
        return obj
    if ext and ext != ".siesta":
        raise ValueError(f"No serializer for extension: {ext}")

    payload = read_siesta(path, lazy=False)

    labels_payload = payload.get("labels")
    if isinstance(labels_payload, dict):
        labels_payload["metadata"] = payload.get("metadata", {})
        labels_payload["provenance"] = payload.get("provenance", {})
        obj = labels_from_siesta_payload(
            cls,
            labels_payload,
            suggestions_payload=payload.get("suggestions"),
        )
    else:
        obj = labels_from_siesta_payload(
            cls,
            payload,
            suggestions_payload=payload.get("suggestions") if isinstance(payload, dict) else None,
        )

    obj.validate()
    obj.path = path
    return obj


def labels_save_file(
    labels: Labels,
    filename: str,
    *,
    default_suffix: str = "",
    metadata: dict[str, Any] | None = None,
    **_: Any,
) -> str:
    """Save labels to disk."""
    from posetta.io.siesta_format import write_siesta

    path = Path(filename)
    ext = path.suffix.lower() or default_suffix
    if ext == ".json":
        if not path.suffix:
            path = path.with_suffix(".json")
        return write_labels_json(path, labels, metadata=metadata)
    if ext and ext != ".siesta":
        raise ValueError(f"No serializer for extension: {ext}")
    if not path.suffix:
        path = path.with_suffix(".siesta")
    ensure_dir(path.parent)
    write_siesta(path, labels, metadata=metadata)
    labels.path = path
    return str(path)


__all__ = [
    "build_video_object",
    "labels_from_siesta_payload",
    "labels_load_file",
    "labels_save_file",
    "load_suggestions",
]
