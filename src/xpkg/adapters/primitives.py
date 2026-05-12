"""Adapters from canonical xpkg labels/projects to ``primitives`` sessions."""

from __future__ import annotations

from collections.abc import Iterable
from os import PathLike, fspath
from pathlib import Path
from typing import Any

import numpy as np
from primitives import PrimitivesSession
from primitives.models.session import VideoStream
from primitives.skeletons import SkeletonDefinition

from xpkg.io.labels.model import Labels
from xpkg.pose.annotations import Instance, PredictedInstance, is_predicted_instance

VideoSelector = int | str | None
TrackSelector = int | str | None


def _video_candidates(video: Any) -> tuple[str, ...]:
    candidates: list[str] = []
    for raw in (
        getattr(video, "label", None),
        getattr(video, "id", None),
        getattr(video, "filename", None),
    ):
        if raw is None:
            continue
        value = str(raw).strip()
        if not value:
            continue
        candidates.append(value)
        candidates.append(Path(value).name)
    return tuple(dict.fromkeys(candidates))


def _select_video(labels: Labels, video: VideoSelector) -> tuple[int, Any]:
    videos = list(labels.videos)
    if not videos:
        raise ValueError("xpkg labels do not contain any videos.")

    if video is None:
        if len(videos) != 1:
            raise ValueError("xpkg labels contain multiple videos; pass `video=` to select one.")
        return 0, videos[0]

    if isinstance(video, bool):
        raise TypeError("video must be an index, identifier, or label.")

    if isinstance(video, int | np.integer):
        index = int(video)
        if index < 0 or index >= len(videos):
            raise IndexError(f"Video index {index} is out of range for xpkg labels.")
        return index, videos[index]

    wanted = str(video).strip()
    for index, candidate in enumerate(videos):
        if wanted in _video_candidates(candidate):
            return index, candidate
    raise KeyError(f"Could not resolve xpkg video {video!r}.")


def _frame_instances(frame: Any) -> tuple[Instance | PredictedInstance, ...]:
    instances = getattr(frame, "instances", None)
    if instances is not None:
        return tuple(instances)

    combined: list[Instance | PredictedInstance] = []
    for attr in ("user_instances", "predicted_instances"):
        values = getattr(frame, attr, None)
        if values is not None:
            combined.extend(values)
    return tuple(combined)


def _instance_track_key(instance: Instance | PredictedInstance) -> tuple[int, str] | None:
    track = instance.track
    if track is None and instance.from_predicted is not None:
        track = instance.from_predicted.track
    if track is None:
        return None
    return (int(track.id), str(track.name or ""))


def _available_track_keys(
    frames: Iterable[Any],
) -> tuple[tuple[int, str] | None, ...]:
    keys = {
        _instance_track_key(instance)
        for frame in frames
        for instance in _frame_instances(frame)
    }
    return tuple(sorted(keys, key=lambda item: (-1, "") if item is None else item))


def _select_track_key(frames: list[Any], track: TrackSelector) -> tuple[int, str] | None:
    keys = _available_track_keys(frames)
    if not keys:
        raise ValueError("Selected xpkg video does not contain any instances.")

    if track is None:
        if len(keys) != 1:
            raise ValueError(
                "Selected xpkg video contains multiple pose streams; pass `track=` "
                "to choose one."
            )
        return keys[0]

    if isinstance(track, bool):
        raise TypeError("track must be an id, name, or `untracked`.")

    if isinstance(track, int | np.integer):
        track_id = int(track)
        for candidate in keys:
            if candidate is not None and candidate[0] == track_id:
                return candidate
        raise KeyError(f"Could not resolve xpkg track id {track_id!r}.")

    wanted = str(track).strip()
    if wanted.lower() in {"none", "untracked"}:
        if None in keys:
            return None
        raise KeyError("No untracked xpkg pose stream is available.")

    for candidate in keys:
        if candidate is not None and wanted == candidate[1]:
            return candidate
    raise KeyError(f"Could not resolve xpkg track {track!r}.")


def _select_frame_instance(
    frame: Any,
    *,
    selected_track: tuple[int, str] | None,
    use_predicted: bool,
) -> Instance | PredictedInstance | None:
    matching = [
        instance
        for instance in _frame_instances(frame)
        if _instance_track_key(instance) == selected_track
    ]
    if not matching:
        return None

    predicted = [instance for instance in matching if is_predicted_instance(instance)]
    user = [instance for instance in matching if not is_predicted_instance(instance)]

    if use_predicted:
        if len(predicted) == 1:
            return predicted[0]
        if len(predicted) > 1:
            raise ValueError(
                f"Frame {frame.frame_idx} has multiple predicted instances for the "
                "selected xpkg stream."
            )
        if len(user) == 1:
            return user[0]
    else:
        if len(user) == 1:
            return user[0]
        if len(user) > 1:
            raise ValueError(
                f"Frame {frame.frame_idx} has multiple user instances for the "
                "selected xpkg stream."
            )
        if len(predicted) == 1:
            return predicted[0]

    if len(matching) > 1:
        raise ValueError(
            f"Frame {frame.frame_idx} has ambiguous xpkg instances for the selected stream."
        )
    return matching[0]


def _frame_count_for_video(video: Any, frames: list[Any]) -> int:
    video_frames = int(getattr(video, "frames", 0) or 0)
    if video_frames > 0:
        return video_frames
    if not frames:
        return 0
    return max(int(frame.frame_idx) for frame in frames) + 1


def _labels_root(labels: Labels, root: str | Path | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    raw = labels.path
    if raw is None:
        return Path.cwd()
    path = Path(raw)
    return path if path.is_dir() else path.parent


def _video_streams(video: Any) -> list[VideoStream]:
    filename = getattr(video, "filename", None)
    if filename:
        path = Path(str(filename))
    else:
        image_filenames = list(getattr(video, "image_filenames", []))
        if not image_filenames:
            return []
        path = Path(str(image_filenames[0]))

    return [
        VideoStream(
            kind="xpkg",
            path=path,
            description=str(getattr(video, "label", None) or path.name),
            width=int(getattr(video, "width", 0) or 0) or None,
            height=int(getattr(video, "height", 0) or 0) or None,
        )
    ]


def _video_label(video: Any, index: int) -> str:
    for attr in ("label", "id"):
        raw = getattr(video, attr, None)
        if raw is not None and str(raw).strip():
            return str(raw)

    filename = getattr(video, "filename", None)
    if filename is not None and str(filename).strip():
        return Path(str(filename)).name

    return f"video_{index}"


def _skeleton_bodyparts(skeleton: Any) -> tuple[str, ...]:
    names = getattr(skeleton, "keypoint_names", None)
    if names is not None:
        return tuple(str(name) for name in names)
    return tuple(str(getattr(keypoint, "name", keypoint)) for keypoint in skeleton.keypoints)


def _skeleton_edges(
    skeleton: Any,
    *,
    bodyparts: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    links_by_names = getattr(skeleton, "links_by_names", None)
    if callable(links_by_names):
        return tuple((str(src), str(dst)) for src, dst in links_by_names())

    edges: list[tuple[str, str]] = []
    for src_idx, dst_idx in getattr(skeleton, "links_ids", ()):
        src = int(src_idx)
        dst = int(dst_idx)
        if 0 <= src < len(bodyparts) and 0 <= dst < len(bodyparts):
            edges.append((bodyparts[src], bodyparts[dst]))
    return tuple(edges)


def _copy_triads(raw: Any) -> dict[str, tuple[str, str, str]] | None:
    if not raw:
        return None
    triads: dict[str, tuple[str, str, str]] = {}
    for name, value in dict(raw).items():
        if isinstance(value, list | tuple) and len(value) == 3:
            triads[str(name)] = (str(value[0]), str(value[1]), str(value[2]))
    return triads or None


def _copy_str_map(raw: Any) -> dict[str, str] | None:
    if not raw:
        return None
    copied = {str(key): str(value) for key, value in dict(raw).items()}
    return copied or None


def _copy_node_properties(raw: Any) -> dict[str, dict[str, object]] | None:
    if not raw:
        return None
    copied: dict[str, dict[str, object]] = {}
    for key, value in dict(raw).items():
        if isinstance(value, dict):
            copied[str(key)] = dict(value)
    return copied or None


def _skeleton_definition(skeleton: Any) -> SkeletonDefinition:
    bodyparts = _skeleton_bodyparts(skeleton)
    name = str(getattr(skeleton, "name", "xpkg"))
    extras = getattr(skeleton, "extras", {})
    source_path = extras.get("source_path") if isinstance(extras, dict) else None
    path = Path(source_path) if isinstance(source_path, str) else Path("xpkg") / f"{name}.json"
    return SkeletonDefinition(
        name=name,
        bodyparts=bodyparts,
        edges=_skeleton_edges(skeleton, bodyparts=bodyparts),
        path=path,
        triads=_copy_triads(getattr(skeleton, "triads", None)),
        aliases=_copy_str_map(getattr(skeleton, "aliases", None)),
        node_properties=_copy_node_properties(getattr(skeleton, "node_properties", None)),
    )


def _instance_xy_score_array(instance: Instance | PredictedInstance) -> np.ndarray:
    xy_score = instance.xy_score_array(invisible_as_nan=True, missing_score=np.nan)
    array = np.asarray(xy_score, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(
            f"Expected xpkg instance XY-score array with shape (keypoints, 3), got {array.shape}"
        )

    point_count = len(instance.point_records(copy=False))
    if array.shape[0] != point_count:
        raise ValueError(
            "Instance xy_score_array row count does not match point_records: "
            f"{array.shape[0]} != {point_count}"
        )
    return array


def _build_session(
    labels: Labels,
    *,
    selected_video_index: int,
    selected_video: Any,
    selected_frames: list[Any],
    selected_track: tuple[int, str] | None,
    use_predicted: bool,
    root: str | Path | None,
) -> PrimitivesSession:
    if not selected_frames:
        raise ValueError("Selected xpkg video does not contain any labeled frames.")

    first_instance: Instance | PredictedInstance | None = None
    for frame in selected_frames:
        first_instance = _select_frame_instance(
            frame,
            selected_track=selected_track,
            use_predicted=use_predicted,
        )
        if first_instance is not None:
            break
    if first_instance is None:
        raise ValueError("Could not find any xpkg instances for the selected stream.")

    source_skeleton = first_instance.skeleton
    skeleton_definition = _skeleton_definition(source_skeleton)
    bodyparts = skeleton_definition.bodyparts
    frame_count = _frame_count_for_video(selected_video, selected_frames)
    coords_xy = np.full((frame_count, len(bodyparts), 2), np.nan, dtype=np.float32)
    likelihoods = np.full((frame_count, len(bodyparts)), np.nan, dtype=np.float32)

    for frame in selected_frames:
        instance = _select_frame_instance(
            frame,
            selected_track=selected_track,
            use_predicted=use_predicted,
        )
        if instance is None:
            continue
        if instance.skeleton != source_skeleton:
            raise ValueError("Selected xpkg stream changes skeleton across frames.")

        points = _instance_xy_score_array(instance)
        if points.shape[0] != len(bodyparts):
            raise ValueError(
                "Instance keypoint count does not match selected skeleton: "
                f"{points.shape[0]} != {len(bodyparts)}"
            )

        frame_idx = int(frame.frame_idx)
        if frame_idx < 0 or frame_idx >= frame_count:
            raise IndexError(f"xpkg frame index {frame_idx} is out of bounds.")

        coords_xy[frame_idx, :, :] = points[:, :2]
        likelihoods[frame_idx, :] = points[:, 2]

    selected_track_id = None if selected_track is None else selected_track[0]
    selected_track_name = None if selected_track is None else selected_track[1]
    fps_raw = float(getattr(selected_video, "fps", 0.0) or 0.0)
    return PrimitivesSession.from_keypoints(
        label=_video_label(selected_video, selected_video_index),
        modality="xpkg",
        root=_labels_root(labels, root),
        bodyparts=bodyparts,
        coords_xy=coords_xy,
        skeleton=skeleton_definition,
        likelihoods=likelihoods,
        videos=_video_streams(selected_video),
        fps=fps_raw if fps_raw > 0 else None,
        tags={"source": "xpkg"},
        extras={
            "xpkg": {
                "video_index": selected_video_index,
                "video_candidates": list(_video_candidates(selected_video)),
                "track_id": selected_track_id,
                "track_name": selected_track_name,
                "use_predicted": bool(use_predicted),
                "provenance": dict(labels.provenance),
                "session": dict(labels.session),
                "preferences": dict(labels.preferences),
            }
        },
    )


def labels_to_primitives_session(
    labels: Labels,
    *,
    video: VideoSelector = None,
    track: TrackSelector = None,
    use_predicted: bool = True,
    root: str | Path | None = None,
) -> PrimitivesSession:
    """Convert loaded xpkg labels into a ``primitives.PrimitivesSession``.

    This is an in-memory adapter: it reads already materialized ``Labels`` and
    selected ``Instance`` point arrays. It does not validate or hydrate a project
    payload on its own.
    """
    selected_video_index, selected_video = _select_video(labels, video)
    selected_frames = [
        frame for frame in labels.labeled_frames if frame.video is selected_video
    ]
    selected_track = _select_track_key(selected_frames, track)
    return _build_session(
        labels,
        selected_video_index=selected_video_index,
        selected_video=selected_video,
        selected_frames=selected_frames,
        selected_track=selected_track,
        use_predicted=use_predicted,
        root=root,
    )


def project_to_primitives_session(
    project: str | PathLike[str] | Any,
    *,
    video: VideoSelector = None,
    track: TrackSelector = None,
    use_predicted: bool = True,
) -> PrimitivesSession:
    """Load xpkg project labels and convert them into a primitives session."""
    from xpkg.services import ProjectService

    if isinstance(project, str):
        service = ProjectService.open(project)
    elif isinstance(project, PathLike):
        project_path = fspath(project)
        if not isinstance(project_path, str):
            raise TypeError("project path must resolve to a string path.")
        service = ProjectService.open(project_path)
    elif hasattr(project, "load_labels"):
        service = project
    else:
        raise TypeError(
            "project must be a path-like xpkg project root or a ProjectService-like object."
        )

    labels = service.load_labels()
    return labels_to_primitives_session(
        labels,
        video=video,
        track=track,
        use_predicted=use_predicted,
        root=getattr(service, "project_root", None),
    )


__all__ = ["labels_to_primitives_session", "project_to_primitives_session"]
