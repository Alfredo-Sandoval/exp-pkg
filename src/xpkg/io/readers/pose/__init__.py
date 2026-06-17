"""Generic dispatchers for low-level external pose readers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from xpkg.io.readers.pose import dlc, mediapipe_pose_landmarks, mmpose, sleap_analysis_h5
from xpkg.io.readers.pose._common import PoseTrack, with_pose_track_metadata

_SLEAP_FILE_TYPES = {"h5", "hdf5"}
_DLC_FILE_TYPES = {"csv", "h5", "hdf5"}
_LIGHTNING_POSE_FILE_TYPES = {"csv"}
_MEDIAPIPE_FILE_TYPES = {"json"}
_MMPOSE_FILE_TYPES = {"json"}
_LIGHTNING_POSE_ALIASES = {
    "LIGHTNINGPOSE",
    "LIGHTNING_POSE",
    "LIGHTNING-POSE",
    "LIGHTNING POSE",
    "LP",
}


def _normalize_software(software: str) -> str:
    normalized = str(software).strip().upper()
    if not normalized:
        raise ValueError("software must be a non-empty string.")
    return normalized


def _normalize_file_type(file_type: str) -> str:
    normalized = str(file_type).strip().lower()
    if not normalized:
        raise ValueError("file_type must be a non-empty string.")
    return normalized


def _resolve_reader(software: str, file_type: str) -> tuple[str, str]:
    normalized_software = _normalize_software(software)
    normalized_file_type = _normalize_file_type(file_type)

    if normalized_software == "SLEAP":
        if normalized_file_type not in _SLEAP_FILE_TYPES:
            raise ValueError(
                f"Unsupported SLEAP file_type {file_type!r}. Expected one of ['h5', 'hdf5']."
            )
        return normalized_software, normalized_file_type

    if normalized_software == "DLC":
        if normalized_file_type not in _DLC_FILE_TYPES:
            raise ValueError(
                f"Unsupported DLC file_type {file_type!r}. Expected one of ['csv', 'h5', 'hdf5']."
            )
        return normalized_software, normalized_file_type

    if normalized_software in _LIGHTNING_POSE_ALIASES:
        if normalized_file_type not in _LIGHTNING_POSE_FILE_TYPES:
            raise ValueError(
                f"Unsupported Lightning Pose file_type {file_type!r}. Expected one of ['csv']."
            )
        return "LIGHTNING_POSE", normalized_file_type

    if normalized_software == "MEDIAPIPE":
        if normalized_file_type not in _MEDIAPIPE_FILE_TYPES:
            raise ValueError(
                f"Unsupported MEDIAPIPE file_type {file_type!r}. Expected one of ['json']."
            )
        return normalized_software, normalized_file_type

    if normalized_software == "MMPOSE":
        if normalized_file_type not in _MMPOSE_FILE_TYPES:
            raise ValueError(
                f"Unsupported MMPose file_type {file_type!r}. Expected one of ['json']."
            )
        return normalized_software, normalized_file_type

    raise ValueError(
        "Unsupported software "
        f"{software!r}. Expected one of ['DLC', 'LightningPose', 'MEDIAPIPE', 'MMPose', 'SLEAP']."
    )


def _source_type_for_reader(software: str, file_type: str) -> str:
    if software == "SLEAP":
        return "sleap_analysis_h5"
    if software == "DLC":
        return "dlc_h5" if file_type in _DLC_FILE_TYPES - {"csv"} else "dlc_csv"
    if software == "LIGHTNING_POSE":
        return "lightning_pose_csv"
    if software == "MEDIAPIPE":
        return "mediapipe_pose_landmarks_json"
    if software == "MMPOSE":
        return "mmpose_topdown_json"
    raise ValueError(f"Unsupported normalized pose software {software!r}.")


def _dispatch_metadata(
    *,
    path: Path,
    software: str,
    file_type: str,
    track_index: int,
) -> dict[str, object]:
    return {
        "source": {"type": _source_type_for_reader(software, file_type), "path": str(path)},
        "software": software,
        "file_type": file_type,
        "track_index": int(track_index),
    }


def read_pose_track(
    path: Path | str,
    *,
    software: str,
    file_type: str,
    track_index: int = 0,
) -> PoseTrack:
    """Read a pose track from a supported external tracking export."""

    normalized_software, normalized_file_type = _resolve_reader(software, file_type)
    resolved_path = Path(path)

    if normalized_software == "SLEAP":
        track = sleap_analysis_h5.read_track(resolved_path, track_index=track_index)
        return with_pose_track_metadata(
            track,
            _dispatch_metadata(
                path=resolved_path,
                software=normalized_software,
                file_type=normalized_file_type,
                track_index=track_index,
            ),
        )
    if normalized_software == "MEDIAPIPE":
        track = mediapipe_pose_landmarks.read_track(resolved_path, track_index=track_index)
        return with_pose_track_metadata(
            track,
            _dispatch_metadata(
                path=resolved_path,
                software=normalized_software,
                file_type=normalized_file_type,
                track_index=track_index,
            ),
        )
    if normalized_software == "MMPOSE":
        track = mmpose.read_track(resolved_path, track_index=track_index)
        return with_pose_track_metadata(
            track,
            _dispatch_metadata(
                path=resolved_path,
                software=normalized_software,
                file_type=normalized_file_type,
                track_index=track_index,
            ),
        )
    track = dlc.read_track(
        resolved_path,
        file_type=normalized_file_type,
        track_index=track_index,
    )
    return with_pose_track_metadata(
        track,
        _dispatch_metadata(
            path=resolved_path,
            software=normalized_software,
            file_type=normalized_file_type,
            track_index=track_index,
        ),
    )


def read_pose_node_names(
    path: Path | str,
    *,
    software: str,
    file_type: str,
) -> list[str]:
    """Read node names from a supported external tracking export."""

    normalized_software, normalized_file_type = _resolve_reader(software, file_type)
    resolved_path = Path(path)

    if normalized_software == "SLEAP":
        return sleap_analysis_h5.read_node_names(resolved_path)
    if normalized_software == "MEDIAPIPE":
        return mediapipe_pose_landmarks.read_node_names(resolved_path)
    if normalized_software == "MMPOSE":
        return mmpose.read_node_names(resolved_path)
    return dlc.read_node_names(resolved_path, file_type=normalized_file_type)


def resolve_pose_node_indices(
    path: Path | str,
    *,
    software: str,
    file_type: str,
    target_names: Sequence[str],
) -> list[int]:
    """Resolve requested node names to source indices for a supported pose export."""

    normalized_software, normalized_file_type = _resolve_reader(software, file_type)
    resolved_path = Path(path)

    if normalized_software == "SLEAP":
        return sleap_analysis_h5.resolve_node_indices(resolved_path, target_names)
    if normalized_software == "MEDIAPIPE":
        return mediapipe_pose_landmarks.resolve_node_indices(resolved_path, target_names)
    if normalized_software == "MMPOSE":
        return mmpose.resolve_node_indices(resolved_path, list(target_names))
    return dlc.resolve_node_indices(
        resolved_path,
        file_type=normalized_file_type,
        target_names=target_names,
    )


__all__ = [
    "PoseTrack",
    "read_pose_node_names",
    "read_pose_track",
    "resolve_pose_node_indices",
]
