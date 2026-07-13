"""Versioned JSON exchange for the canonical recording-session ontology."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from xpkg._core.json_utils import load_json_dict, write_json
from xpkg.io.pose_json import pose_labels_from_payload, pose_labels_payload
from xpkg.model.behavior import BehaviorLabels
from xpkg.model.calibration import Calibration
from xpkg.model.events import Event, EventTable, SyncEvent
from xpkg.model.metadata import AcquisitionMetadata, CameraMetadata
from xpkg.model.session import (
    CalibrationCameraLink,
    RecordingSession,
    SessionBehavior,
    SessionCalibration,
    SessionPose,
    SessionSignal,
    SessionVideo,
    SynchronizationMethod,
    TimebaseAlignment,
)
from xpkg.model.signals import (
    PhotometryChannel,
    PhotometryRecording,
    SignalChannel,
    TimeSeries,
)
from xpkg.model.time import Timebase, Timeline
from xpkg.pose.trajectory import (
    CoordinateFrameKind,
    PoseCoordinateFrame,
    PoseTrajectory,
)

RECORDING_SESSION_FORMAT = "xpkg.recording-session"
RECORDING_SESSION_SCHEMA_VERSION = 2


def recording_session_document(
    session: RecordingSession,
    *,
    document_metadata: Mapping[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Serialize one valid recording session into the versioned JSON document."""
    if not isinstance(session, RecordingSession):
        raise TypeError(f"session must be a RecordingSession, got {session!r}.")
    return {
        "format": RECORDING_SESSION_FORMAT,
        "schema_version": RECORDING_SESSION_SCHEMA_VERSION,
        "payload": {
            "session": recording_session_payload(
                session,
                project_root=None if project_root is None else Path(project_root),
            ),
            "metadata": dict(document_metadata or {}),
        },
    }


def recording_session_from_document(
    document: Mapping[str, Any], *, project_root: str | Path | None = None
) -> RecordingSession:
    """Parse a versioned document into a valid recording-session object."""
    payload = _document_payload(document)
    raw_session = _required_mapping(payload, "session", context="recording payload")
    return recording_session_from_payload(raw_session, project_root=project_root)


def recording_session_from_payload(
    raw_session: Mapping[str, Any], *, project_root: str | Path | None = None
) -> RecordingSession:
    """Parse one recording-session payload nested in a larger ontology document."""

    acquisition = _acquisition_from_payload(raw_session.get("acquisition"))
    return RecordingSession(
        session_id=_required_str(raw_session, "session_id", context="recording session"),
        title=_optional_str(raw_session.get("title"), name="recording session title"),
        acquisition=acquisition,
        timebase=_timebase_from_payload(
            _required_mapping(raw_session, "timebase", context="recording session")
        ),
        signals=tuple(
            _signal_from_payload(item)
            for item in _required_mapping_sequence(raw_session, "signals", "recording session")
        ),
        videos=tuple(
            _video_from_payload(item, acquisition)
            for item in _required_mapping_sequence(raw_session, "videos", "recording session")
        ),
        poses=tuple(
            _pose_from_payload(
                item,
                project_root=None if project_root is None else Path(project_root),
            )
            for item in _required_mapping_sequence(raw_session, "poses", "recording session")
        ),
        behaviors=tuple(
            _behavior_from_payload(item)
            for item in _required_mapping_sequence(
                raw_session, "behaviors", "recording session"
            )
        ),
        calibrations=tuple(
            _calibration_from_payload(item, acquisition)
            for item in _required_mapping_sequence(
                raw_session, "calibrations", "recording session"
            )
        ),
        alignments=tuple(
            _alignment_from_payload(item)
            for item in _required_mapping_sequence(
                raw_session, "alignments", "recording session"
            )
        ),
        events=_event_table_from_payload(
            _required_mapping(raw_session, "events", context="recording session")
        ),
        metadata=dict(_required_mapping(raw_session, "metadata", context="recording session")),
    )


def write_recording_session_json(
    path: str | Path,
    session: RecordingSession,
    *,
    document_metadata: Mapping[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> Path:
    """Write one recording-session document atomically."""
    target = Path(path)
    write_json(
        target,
        recording_session_document(
            session,
            document_metadata=document_metadata,
            project_root=project_root,
        ),
    )
    return target


def read_recording_session_json(
    path: str | Path, *, project_root: str | Path | None = None
) -> RecordingSession:
    """Read one recording-session document from disk."""
    return recording_session_from_document(load_json_dict(path), project_root=project_root)


def read_recording_session_metadata_json(path: str | Path) -> dict[str, Any]:
    """Read session metadata without hydrating dense modality payloads."""
    payload = _document_payload(load_json_dict(path))
    raw_session = _required_mapping(payload, "session", context="recording payload")
    return dict(_required_mapping(raw_session, "metadata", context="recording session"))


def _document_payload(document: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(document, Mapping):
        raise TypeError("recording-session document must be an object.")
    if document.get("format") != RECORDING_SESSION_FORMAT:
        raise ValueError(
            f"recording-session format must be {RECORDING_SESSION_FORMAT!r}, "
            f"got {document.get('format')!r}."
        )
    if document.get("schema_version") != RECORDING_SESSION_SCHEMA_VERSION:
        raise ValueError(
            "recording-session schema_version must be "
            f"{RECORDING_SESSION_SCHEMA_VERSION}, got {document.get('schema_version')!r}."
        )
    return _required_mapping(document, "payload", context="recording-session document")


def recording_session_payload(
    session: RecordingSession, *, project_root: str | Path | None = None
) -> dict[str, Any]:
    """Serialize one recording session for nesting in a larger ontology document."""

    root = None if project_root is None else Path(project_root)
    return {
        "session_id": session.session_id,
        "title": session.title,
        "acquisition": (
            None if session.acquisition is None else session.acquisition.to_dict()
        ),
        "timebase": _timebase_payload(session.timebase),
        "signals": [_signal_payload(link) for link in session.signals],
        "videos": [_video_payload(video) for video in session.videos],
        "poses": [_pose_payload(link, project_root=root) for link in session.poses],
        "behaviors": [_behavior_payload(link) for link in session.behaviors],
        "calibrations": [_calibration_payload(link) for link in session.calibrations],
        "alignments": [_alignment_payload(link) for link in session.alignments],
        "events": _event_table_payload(session.events),
        "metadata": dict(session.metadata),
    }


def _signal_payload(link: SessionSignal) -> dict[str, Any]:
    recording = link.recording
    if isinstance(recording, PhotometryRecording):
        return {
            "name": link.name,
            "recording_type": "photometry",
            "series": _time_series_payload(recording.series),
            "signal_channel": recording.signal_channel,
            "reference_channel": recording.reference_channel,
            "metadata": dict(recording.metadata),
        }
    return {
        "name": link.name,
        "recording_type": "time_series",
        "series": _time_series_payload(recording),
    }


def _signal_from_payload(payload: Mapping[str, Any]) -> SessionSignal:
    name = _required_str(payload, "name", context="session signal")
    recording_type = _required_str(payload, "recording_type", context="session signal")
    series = _time_series_from_payload(
        _required_mapping(payload, "series", context="session signal")
    )
    if recording_type == "time_series":
        return SessionSignal(name=name, recording=series)
    if recording_type == "photometry":
        recording = PhotometryRecording(
            series=series,
            signal_channel=_optional_str(
                payload.get("signal_channel"), name="photometry signal_channel"
            ),
            reference_channel=_optional_str(
                payload.get("reference_channel"), name="photometry reference_channel"
            ),
            metadata=dict(_required_mapping(payload, "metadata", context="photometry")),
        )
        return SessionSignal(name=name, recording=recording)
    raise ValueError(f"Unsupported session signal recording_type: {recording_type!r}.")


def _time_series_payload(series: TimeSeries) -> dict[str, Any]:
    return {
        "name": series.name,
        "timeline": {
            "timestamps_s": series.timeline.timestamps_s.tolist(),
            "timebase": _timebase_payload(series.timeline.timebase),
            "sample_rate_hz": series.timeline.sample_rate_hz,
        },
        "channels": [_channel_payload(channel) for channel in series.channels],
        "values": series.values.tolist(),
        "provenance": dict(series.provenance),
    }


def _time_series_from_payload(payload: Mapping[str, Any]) -> TimeSeries:
    timeline_payload = _required_mapping(payload, "timeline", context="time series")
    raw_timestamps = _required_number_sequence(timeline_payload, "timestamps_s", context="timeline")
    raw_values = _required_sequence(payload, "values", context="time series")
    return TimeSeries(
        name=_required_str(payload, "name", context="time series"),
        timeline=Timeline(
            timestamps_s=np.asarray(raw_timestamps, dtype=np.float64),
            timebase=_timebase_from_payload(
                _required_mapping(timeline_payload, "timebase", context="timeline")
            ),
            sample_rate_hz=_optional_float(
                timeline_payload.get("sample_rate_hz"), name="timeline sample_rate_hz"
            ),
        ),
        channels=tuple(
            _channel_from_payload(item)
            for item in _required_mapping_sequence(payload, "channels", "time series")
        ),
        values=np.asarray(raw_values, dtype=np.float64),
        provenance=dict(_required_mapping(payload, "provenance", context="time series")),
    )


def _channel_payload(channel: SignalChannel) -> dict[str, Any]:
    payload = {
        "channel_type": "photometry" if isinstance(channel, PhotometryChannel) else "signal",
        "name": channel.name,
        "unit": channel.unit,
        "description": channel.description,
        "metadata": dict(channel.metadata),
    }
    if isinstance(channel, PhotometryChannel):
        payload["excitation"] = channel.excitation
    return payload


def _channel_from_payload(payload: Mapping[str, Any]) -> SignalChannel:
    kwargs = {
        "name": _required_str(payload, "name", context="signal channel"),
        "unit": _required_str(payload, "unit", context="signal channel", allow_empty=True),
        "description": _required_str(
            payload, "description", context="signal channel", allow_empty=True
        ),
        "metadata": dict(_required_mapping(payload, "metadata", context="signal channel")),
    }
    channel_type = _required_str(payload, "channel_type", context="signal channel")
    if channel_type == "signal":
        return SignalChannel(**kwargs)
    if channel_type == "photometry":
        return PhotometryChannel(
            **kwargs,
            excitation=_required_str(
                payload, "excitation", context="photometry channel", allow_empty=True
            ),
        )
    raise ValueError(f"Unsupported signal channel_type: {channel_type!r}.")


def _event_table_payload(table: EventTable) -> dict[str, Any]:
    return {
        "timebase": _timebase_payload(table.timebase),
        "events": [_event_payload(event) for event in table],
        "metadata": dict(table.metadata),
    }


def _event_payload(event: Event) -> dict[str, Any]:
    payload = event.to_dict()
    payload["event_type"] = "sync" if isinstance(event, SyncEvent) else "event"
    if isinstance(event, SyncEvent):
        payload["source"] = event.source
    return payload


def _event_table_from_payload(payload: Mapping[str, Any]) -> EventTable:
    return EventTable(
        timebase=_timebase_from_payload(
            _required_mapping(payload, "timebase", context="event table")
        ),
        events=tuple(
            _event_from_payload(item)
            for item in _required_mapping_sequence(payload, "events", "event table")
        ),
        metadata=dict(_required_mapping(payload, "metadata", context="event table")),
    )


def _video_payload(video: SessionVideo) -> dict[str, Any]:
    return {
        "role": video.role,
        "path": video.path.as_posix(),
        "camera_id": None if video.camera is None else video.camera.camera_id,
        "timebase": _timebase_payload(video.timebase),
        "frame_rate_hz": video.frame_rate_hz,
        "frame_count": video.frame_count,
        "metadata": dict(video.metadata),
    }


def _pose_payload(link: SessionPose, *, project_root: Path | None) -> dict[str, Any]:
    payload = {
        "name": link.name,
        "video_roles": list(link.video_roles),
        "calibration_name": link.calibration_name,
        "provenance": None if link.provenance is None else link.provenance.to_dict(),
        "metadata": dict(link.metadata),
    }
    if isinstance(link.data, PoseTrajectory):
        payload["data_type"] = "trajectory"
        payload["data"] = _trajectory_payload(link.data)
    else:
        payload["data_type"] = "labels"
        payload["data"] = pose_labels_payload(link.data, project_root=project_root)
    return payload


def _pose_from_payload(
    payload: Mapping[str, Any], *, project_root: Path | None
) -> SessionPose:
    raw_roles = _required_sequence(payload, "video_roles", context="session pose")
    if any(not isinstance(role, str) for role in raw_roles):
        raise TypeError("session pose.video_roles must contain only strings.")
    data_type = _required_str(payload, "data_type", context="session pose")
    raw_data = _required_mapping(payload, "data", context="session pose")
    if data_type == "labels":
        data = pose_labels_from_payload(raw_data, project_root=project_root)
    elif data_type == "trajectory":
        data = _trajectory_from_payload(raw_data)
    else:
        raise ValueError(f"Unsupported session pose data_type: {data_type!r}.")
    return SessionPose(
        name=_required_str(payload, "name", context="session pose"),
        data=data,
        video_roles=tuple(cast("Sequence[str]", raw_roles)),
        calibration_name=_optional_str(
            payload.get("calibration_name"), name="session pose calibration_name"
        ),
        provenance=_pose_provenance_from_payload(payload.get("provenance")),
        metadata=dict(_required_mapping(payload, "metadata", context="session pose")),
    )


def _pose_provenance_from_payload(value: Any):
    if value is None:
        return None
    from xpkg.model.metadata import PoseModelProvenance

    if not isinstance(value, Mapping):
        raise TypeError("session pose.provenance must be an object or null.")
    return PoseModelProvenance.from_dict(value)


def _behavior_payload(link: SessionBehavior) -> dict[str, Any]:
    return {
        "name": link.name,
        "video_role": link.video_role,
        "labels": link.labels.to_dict(),
    }


def _behavior_from_payload(payload: Mapping[str, Any]) -> SessionBehavior:
    return SessionBehavior(
        name=_required_str(payload, "name", context="session behavior"),
        video_role=_optional_str(
            payload.get("video_role"), name="session behavior video_role"
        ),
        labels=BehaviorLabels.from_dict(
            _required_mapping(payload, "labels", context="session behavior")
        ),
    )


def _calibration_payload(link: SessionCalibration) -> dict[str, Any]:
    return {
        "name": link.name,
        "calibration": link.calibration.to_dict(),
        "camera_links": [
            {
                "camera_id": camera_link.camera.camera_id,
                "calibrated_camera_name": camera_link.calibrated_camera.name,
            }
            for camera_link in link.camera_links
        ],
    }


def _calibration_from_payload(
    payload: Mapping[str, Any], acquisition: AcquisitionMetadata | None
) -> SessionCalibration:
    calibration = Calibration.from_dict(
        _required_mapping(payload, "calibration", context="session calibration")
    )
    return SessionCalibration(
        name=_required_str(payload, "name", context="session calibration"),
        calibration=calibration,
        camera_links=tuple(
            _calibration_camera_link_from_payload(item, acquisition, calibration)
            for item in _required_mapping_sequence(
                payload, "camera_links", "session calibration"
            )
        ),
    )


def _alignment_payload(link: TimebaseAlignment) -> dict[str, Any]:
    return {
        "name": link.name,
        "source": _timebase_payload(link.source),
        "target": _timebase_payload(link.target),
        "method": link.method.value,
        "scale": link.scale,
        "offset_s": link.offset_s,
        "residual_s": link.residual_s,
        "evidence": [_event_payload(event) for event in link.evidence],
        "metadata": dict(link.metadata),
    }


def _alignment_from_payload(payload: Mapping[str, Any]) -> TimebaseAlignment:
    raw_method = _required_str(payload, "method", context="timebase alignment")
    try:
        method = SynchronizationMethod(raw_method)
    except ValueError as exc:
        raise ValueError(f"Unsupported synchronization method: {raw_method!r}.") from exc
    evidence = tuple(
        _event_from_payload(item)
        for item in _required_mapping_sequence(payload, "evidence", "timebase alignment")
    )
    if any(not isinstance(event, SyncEvent) for event in evidence):
        raise TypeError("timebase alignment evidence must contain sync events.")
    return TimebaseAlignment(
        name=_required_str(payload, "name", context="timebase alignment"),
        source=_timebase_from_payload(
            _required_mapping(payload, "source", context="timebase alignment")
        ),
        target=_timebase_from_payload(
            _required_mapping(payload, "target", context="timebase alignment")
        ),
        method=method,
        scale=_required_float(payload, "scale", context="timebase alignment"),
        offset_s=_required_float(payload, "offset_s", context="timebase alignment"),
        residual_s=_optional_float(
            payload.get("residual_s"), name="timebase alignment residual_s"
        ),
        evidence=cast("tuple[SyncEvent, ...]", evidence),
        metadata=dict(_required_mapping(payload, "metadata", context="timebase alignment")),
    )


def _event_from_payload(payload: Mapping[str, Any]) -> Event:
    kwargs = {
        "kind": _required_str(payload, "kind", context="event"),
        "start_s": _required_float(payload, "start_s", context="event"),
        "duration_s": _required_float(payload, "duration_s", context="event"),
        "label": _optional_str(payload.get("label"), name="event label"),
        "metadata": dict(_optional_mapping(payload.get("metadata"), name="event metadata")),
    }
    event_type = _required_str(payload, "event_type", context="event")
    if event_type == "event":
        return Event(**kwargs)
    if event_type == "sync":
        return SyncEvent(
            **kwargs,
            source=_required_str(payload, "source", context="sync event"),
        )
    raise ValueError(f"Unsupported event_type: {event_type!r}.")


def _video_from_payload(
    payload: Mapping[str, Any], acquisition: AcquisitionMetadata | None
) -> SessionVideo:
    camera_id = _optional_str(payload.get("camera_id"), name="session video camera_id")
    return SessionVideo(
        role=_required_str(payload, "role", context="session video"),
        path=Path(_required_str(payload, "path", context="session video")),
        camera=_acquisition_camera(acquisition, camera_id),
        timebase=_timebase_from_payload(
            _required_mapping(payload, "timebase", context="session video")
        ),
        frame_rate_hz=_optional_float(
            payload.get("frame_rate_hz"), name="session video frame_rate_hz"
        ),
        frame_count=_optional_int(payload.get("frame_count"), name="session video frame_count"),
        metadata=dict(_required_mapping(payload, "metadata", context="session video")),
    )


def _acquisition_from_payload(value: object) -> AcquisitionMetadata | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("recording session.acquisition must be an object or null.")
    return AcquisitionMetadata.from_dict(cast("Mapping[str, Any]", value))


def _acquisition_camera(
    acquisition: AcquisitionMetadata | None, camera_id: str | None
) -> CameraMetadata | None:
    if camera_id is None:
        return None
    if acquisition is None:
        raise ValueError(f"Video references camera {camera_id!r} without acquisition metadata.")
    for camera in acquisition.cameras:
        if camera.camera_id == camera_id:
            return camera
    raise ValueError(f"Video references unknown acquisition camera {camera_id!r}.")


def _calibration_camera_link_from_payload(
    payload: Mapping[str, Any],
    acquisition: AcquisitionMetadata | None,
    calibration: Calibration,
) -> CalibrationCameraLink:
    camera_id = _required_str(payload, "camera_id", context="calibration camera link")
    camera = _acquisition_camera(acquisition, camera_id)
    if camera is None:
        raise AssertionError("required calibration camera resolution returned None.")
    calibrated_name = _required_str(
        payload, "calibrated_camera_name", context="calibration camera link"
    )
    return CalibrationCameraLink(
        camera=camera,
        calibrated_camera=calibration.camera_by_name(calibrated_name),
    )


def _trajectory_payload(trajectory: PoseTrajectory) -> dict[str, Any]:
    frame = trajectory.coordinate_frame
    return {
        "fps": trajectory.fps,
        "track_ids": list(trajectory.track_ids),
        "keypoint_names": list(trajectory.keypoint_names),
        "positions": trajectory.positions.tolist(),
        "valid": trajectory.valid.tolist(),
        "confidence": (
            None if trajectory.confidence is None else trajectory.confidence.tolist()
        ),
        "dims": trajectory.dims,
        "coordinate_frame": {
            "kind": frame.kind.value,
            "units": frame.units,
            "name": frame.name,
            "axis_convention": frame.axis_convention,
            "description": frame.description,
        },
        "frame_offset": trajectory.frame_offset,
        "skeleton_edges": [list(edge) for edge in trajectory.skeleton_edges],
        "source_kind": trajectory.source_kind,
        "source_path": (
            None if trajectory.source_path is None else trajectory.source_path.as_posix()
        ),
        "metadata": dict(trajectory.metadata),
    }


def _trajectory_from_payload(payload: Mapping[str, Any]) -> PoseTrajectory:
    frame = _required_mapping(payload, "coordinate_frame", context="pose trajectory")
    raw_kind = _required_str(frame, "kind", context="pose coordinate frame")
    try:
        kind = CoordinateFrameKind(raw_kind)
    except ValueError as exc:
        raise ValueError(f"Unsupported pose coordinate-frame kind: {raw_kind!r}.") from exc
    return PoseTrajectory(
        fps=_required_float(payload, "fps", context="pose trajectory"),
        track_ids=_required_str_tuple(payload, "track_ids", "pose trajectory"),
        keypoint_names=_required_str_tuple(payload, "keypoint_names", "pose trajectory"),
        positions=np.asarray(
            _required_sequence(payload, "positions", context="pose trajectory"),
            dtype=np.float64,
        ),
        valid=np.asarray(
            _required_sequence(payload, "valid", context="pose trajectory"), dtype=bool
        ),
        confidence=_optional_number_array(payload.get("confidence")),
        dims=cast(
            "Literal[2, 3]",
            _required_int(payload, "dims", context="pose trajectory"),
        ),
        coordinate_frame=PoseCoordinateFrame(
            kind=kind,
            units=_required_str(frame, "units", context="pose coordinate frame"),
            name=_optional_str(frame.get("name"), name="pose coordinate frame name"),
            axis_convention=_optional_str(
                frame.get("axis_convention"), name="pose axis_convention"
            ),
            description=_optional_str(
                frame.get("description"), name="pose frame description"
            ),
        ),
        frame_offset=_required_int(payload, "frame_offset", context="pose trajectory"),
        skeleton_edges=_required_edge_tuple(payload, "skeleton_edges"),
        source_kind=_required_str(
            payload, "source_kind", context="pose trajectory", allow_empty=True
        ),
        source_path=(
            None
            if payload.get("source_path") is None
            else Path(_required_str(payload, "source_path", context="pose trajectory"))
        ),
        metadata=dict(_required_mapping(payload, "metadata", context="pose trajectory")),
    )


def _timebase_payload(timebase: Timebase) -> dict[str, Any]:
    return {"name": timebase.name, "unit": timebase.unit, "offset_s": timebase.offset_s}


def _timebase_from_payload(payload: Mapping[str, Any]) -> Timebase:
    return Timebase(
        name=_required_str(payload, "name", context="timebase"),
        unit=_required_str(payload, "unit", context="timebase"),
        offset_s=_required_float(payload, "offset_s", context="timebase"),
    )


def _required_mapping(payload: Mapping[str, Any], key: str, *, context: str) -> Mapping[str, Any]:
    return _optional_mapping(payload.get(key), name=f"{context}.{key}", required=True)


def _optional_mapping(value: object, *, name: str, required: bool = False) -> Mapping[str, Any]:
    if value is None and not required:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object.")
    return cast("Mapping[str, Any]", value)


def _required_sequence(payload: Mapping[str, Any], key: str, *, context: str) -> Sequence[Any]:
    value = payload.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise TypeError(f"{context}.{key} must be an array.")
    return cast("Sequence[Any]", value)


def _required_mapping_sequence(
    payload: Mapping[str, Any], key: str, context: str
) -> tuple[Mapping[str, Any], ...]:
    values = _required_sequence(payload, key, context=context)
    out: list[Mapping[str, Any]] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise TypeError(f"{context}.{key}[{index}] must be an object.")
        out.append(cast("Mapping[str, Any]", value))
    return tuple(out)


def _required_number_sequence(
    payload: Mapping[str, Any], key: str, *, context: str
) -> tuple[float, ...]:
    values = _required_sequence(payload, key, context=context)
    try:
        return tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context}.{key} must contain only numbers.") from exc


def _required_str_tuple(
    payload: Mapping[str, Any], key: str, context: str
) -> tuple[str, ...]:
    values = _required_sequence(payload, key, context=context)
    if any(not isinstance(value, str) for value in values):
        raise TypeError(f"{context}.{key} must contain only strings.")
    return tuple(cast("Sequence[str]", values))


def _required_edge_tuple(
    payload: Mapping[str, Any], key: str
) -> tuple[tuple[str, str], ...]:
    values = _required_sequence(payload, key, context="pose trajectory")
    edges: list[tuple[str, str]] = []
    for index, value in enumerate(values):
        if (
            not isinstance(value, Sequence)
            or isinstance(value, str | bytes | bytearray)
            or len(value) != 2
            or any(not isinstance(item, str) for item in value)
        ):
            raise TypeError(f"pose trajectory.{key}[{index}] must be a string pair.")
        edges.append((cast("str", value[0]), cast("str", value[1])))
    return tuple(edges)


def _optional_number_array(value: object) -> np.ndarray | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise TypeError("pose trajectory.confidence must be an array or null.")
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("pose trajectory.confidence must contain only numbers.") from exc


def _required_str(
    payload: Mapping[str, Any],
    key: str,
    *,
    context: str,
    allow_empty: bool = False,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{context}.{key} must be a string.")
    if not allow_empty and not value:
        raise ValueError(f"{context}.{key} must not be empty.")
    return value


def _optional_str(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or null.")
    return value


def _required_float(payload: Mapping[str, Any], key: str, *, context: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{context}.{key} must be a number.")
    return float(value)


def _required_int(payload: Mapping[str, Any], key: str, *, context: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context}.{key} must be an integer.")
    return value


def _optional_float(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number or null.")
    return float(value)


def _optional_int(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer or null.")
    return value


__all__ = [
    "RECORDING_SESSION_FORMAT",
    "RECORDING_SESSION_SCHEMA_VERSION",
    "read_recording_session_json",
    "read_recording_session_metadata_json",
    "recording_session_document",
    "recording_session_from_document",
    "recording_session_from_payload",
    "recording_session_payload",
    "write_recording_session_json",
]
