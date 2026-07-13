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
from xpkg.model.emg import EMGProcessingState, EMGSide, EMGSignalData
from xpkg.model.events import Event, EventTable, SyncEvent
from xpkg.model.force import ForcePlateData
from xpkg.model.metadata import AcquisitionMetadata, CameraMetadata, SourceProvenance
from xpkg.model.session import (
    AlignmentModel,
    CalibrationCameraLink,
    RecordingSession,
    SessionBehavior,
    SessionCalibration,
    SessionEventStream,
    SessionPose,
    SessionSignal,
    SessionVideo,
    SynchronizationMethod,
    TimebaseAlignment,
    TimebaseCorrespondence,
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
    PoseTrack,
    PoseTrajectory,
)

RECORDING_SESSION_FORMAT = "xpkg.recording-session"
RECORDING_SESSION_SCHEMA_VERSION = 4


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
    timebases = tuple(
        _timebase_from_payload(item)
        for item in _required_mapping_sequence(raw_session, "timebases", "recording session")
    )
    session_timebase_name = _required_str(
        raw_session, "session_timebase_name", context="recording session"
    )
    session_timebase = _named_timebase(timebases, session_timebase_name)
    root = None if project_root is None else Path(project_root)
    videos = tuple(
        _video_from_payload(item, acquisition)
        for item in _required_mapping_sequence(raw_session, "videos", "recording session")
    )
    calibrations = tuple(
        _calibration_from_payload(item, acquisition)
        for item in _required_mapping_sequence(raw_session, "calibrations", "recording session")
    )
    poses = tuple(
        _pose_from_payload(item, videos=videos, calibrations=calibrations, project_root=root)
        for item in _required_mapping_sequence(raw_session, "poses", "recording session")
    )
    return RecordingSession(
        session_id=_required_str(raw_session, "session_id", context="recording session"),
        title=_optional_str(raw_session.get("title"), name="recording session title"),
        acquisition=acquisition,
        timebase=session_timebase,
        timebases=timebases,
        signals=tuple(
            _signal_from_payload(item)
            for item in _required_mapping_sequence(raw_session, "signals", "recording session")
        ),
        videos=videos,
        poses=poses,
        behaviors=tuple(
            _behavior_from_payload(item, videos=videos, poses=poses)
            for item in _required_mapping_sequence(raw_session, "behaviors", "recording session")
        ),
        calibrations=calibrations,
        alignments=tuple(
            _alignment_from_payload(item)
            for item in _required_mapping_sequence(raw_session, "alignments", "recording session")
        ),
        event_streams=tuple(
            _event_stream_from_payload(item)
            for item in _required_mapping_sequence(
                raw_session, "event_streams", "recording session"
            )
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
        "acquisition": (None if session.acquisition is None else session.acquisition.to_dict()),
        "session_timebase_name": session.timebase.name,
        "timebases": [_timebase_payload(timebase) for timebase in session.timebases],
        "signals": [_signal_payload(link) for link in session.signals],
        "videos": [_video_payload(video) for video in session.videos],
        "poses": [_pose_payload(link, project_root=root) for link in session.poses],
        "behaviors": [_behavior_payload(link) for link in session.behaviors],
        "calibrations": [_calibration_payload(link) for link in session.calibrations],
        "alignments": [_alignment_payload(link) for link in session.alignments],
        "event_streams": [_event_stream_payload(stream) for stream in session.event_streams],
        "metadata": dict(session.metadata),
    }


def _signal_payload(link: SessionSignal) -> dict[str, Any]:
    recording = link.recording
    provenance = _source_provenance_payload(link.provenance)
    if isinstance(recording, PhotometryRecording):
        return {
            "name": link.name,
            "recording_type": "photometry",
            "provenance": provenance,
            "series": _time_series_payload(recording.series),
            "signal_channel": recording.signal_channel,
            "reference_channel": recording.reference_channel,
            "metadata": dict(recording.metadata),
        }
    return {
        "name": link.name,
        "provenance": provenance,
        **_sampled_signal_payload(recording),
    }


def _sampled_signal_payload(
    recording: TimeSeries | EMGSignalData | ForcePlateData,
) -> dict[str, Any]:
    if isinstance(recording, TimeSeries):
        return {"recording_type": "time_series", "series": _time_series_payload(recording)}
    if isinstance(recording, EMGSignalData):
        return {"recording_type": "emg", "emg": _emg_payload(recording)}
    return {"recording_type": "force_plate", "force_plate": _force_payload(recording)}


def _signal_from_payload(payload: Mapping[str, Any]) -> SessionSignal:
    name = _required_str(payload, "name", context="session signal")
    recording_type = _required_str(payload, "recording_type", context="session signal")
    provenance = _source_provenance_from_payload(payload.get("provenance"))
    if recording_type == "time_series":
        return SessionSignal(
            name=name,
            recording=_time_series_from_payload(
                _required_mapping(payload, "series", context="session signal")
            ),
            provenance=provenance,
        )
    if recording_type == "photometry":
        series = _time_series_from_payload(
            _required_mapping(payload, "series", context="session signal")
        )
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
        return SessionSignal(name=name, recording=recording, provenance=provenance)
    if recording_type == "emg":
        return SessionSignal(
            name=name,
            recording=_emg_from_payload(
                _required_mapping(payload, "emg", context="session signal")
            ),
            provenance=provenance,
        )
    if recording_type == "force_plate":
        return SessionSignal(
            name=name,
            recording=_force_from_payload(
                _required_mapping(payload, "force_plate", context="session signal")
            ),
            provenance=provenance,
        )
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


def _event_stream_payload(stream: SessionEventStream) -> dict[str, Any]:
    return {
        "name": stream.name,
        "events": _event_table_payload(stream.events),
        "provenance": _source_provenance_payload(stream.provenance),
    }


def _event_stream_from_payload(payload: Mapping[str, Any]) -> SessionEventStream:
    return SessionEventStream(
        name=_required_str(payload, "name", context="session event stream"),
        events=_event_table_from_payload(
            _required_mapping(payload, "events", context="session event stream")
        ),
        provenance=_source_provenance_from_payload(payload.get("provenance")),
    )


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
        "video_roles": [video.role for video in link.videos],
        "calibration_name": None if link.calibration is None else link.calibration.name,
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
    payload: Mapping[str, Any],
    *,
    videos: tuple[SessionVideo, ...],
    calibrations: tuple[SessionCalibration, ...],
    project_root: Path | None,
) -> SessionPose:
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
        videos=_resolve_video_roles(
            _required_str_tuple(payload, "video_roles", "session pose"), videos
        ),
        calibration=_resolve_optional_calibration(
            _optional_str(payload.get("calibration_name"), name="session pose calibration_name"),
            calibrations,
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
        "video_roles": [video.role for video in link.videos],
        "pose_names": [pose.name for pose in link.poses],
        "provenance": _source_provenance_payload(link.provenance),
        "labels": link.labels.to_dict(),
    }


def _behavior_from_payload(
    payload: Mapping[str, Any],
    *,
    videos: tuple[SessionVideo, ...],
    poses: tuple[SessionPose, ...],
) -> SessionBehavior:
    return SessionBehavior(
        name=_required_str(payload, "name", context="session behavior"),
        videos=_resolve_video_roles(
            _required_str_tuple(payload, "video_roles", "session behavior"), videos
        ),
        poses=_resolve_pose_names(
            _required_str_tuple(payload, "pose_names", "session behavior"), poses
        ),
        provenance=_source_provenance_from_payload(payload.get("provenance")),
        labels=BehaviorLabels.from_dict(
            _required_mapping(payload, "labels", context="session behavior")
        ),
    )


def _resolve_video_roles(
    roles: tuple[str, ...], videos: tuple[SessionVideo, ...]
) -> tuple[SessionVideo, ...]:
    lookup = {video.role: video for video in videos}
    missing = sorted(set(roles) - lookup.keys())
    if missing:
        raise ValueError(f"Session link references unknown video roles: {', '.join(missing)}.")
    return tuple(lookup[role] for role in roles)


def _resolve_pose_names(
    names: tuple[str, ...], poses: tuple[SessionPose, ...]
) -> tuple[SessionPose, ...]:
    lookup = {pose.name: pose for pose in poses}
    missing = sorted(set(names) - lookup.keys())
    if missing:
        raise ValueError(f"Session behavior references unknown poses: {', '.join(missing)}.")
    return tuple(lookup[name] for name in names)


def _resolve_optional_calibration(
    name: str | None, calibrations: tuple[SessionCalibration, ...]
) -> SessionCalibration | None:
    if name is None:
        return None
    for calibration in calibrations:
        if calibration.name == name:
            return calibration
    raise ValueError(f"Session pose references unknown calibration: {name!r}.")


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
            for item in _required_mapping_sequence(payload, "camera_links", "session calibration")
        ),
    )


def _alignment_payload(link: TimebaseAlignment) -> dict[str, Any]:
    return {
        "name": link.name,
        "source": _timebase_payload(link.source),
        "target": _timebase_payload(link.target),
        "model": link.model.value,
        "method": link.method.value,
        "scale": link.scale,
        "offset_s": link.offset_s,
        "residual_s": link.residual_s,
        "evidence": [_correspondence_payload(item) for item in link.evidence],
        "provenance": _source_provenance_payload(link.provenance),
        "metadata": dict(link.metadata),
    }


def _alignment_from_payload(payload: Mapping[str, Any]) -> TimebaseAlignment:
    raw_model = _required_str(payload, "model", context="timebase alignment")
    try:
        model = AlignmentModel(raw_model)
    except ValueError as exc:
        raise ValueError(f"Unsupported alignment model: {raw_model!r}.") from exc
    raw_method = _required_str(payload, "method", context="timebase alignment")
    try:
        method = SynchronizationMethod(raw_method)
    except ValueError as exc:
        raise ValueError(f"Unsupported synchronization method: {raw_method!r}.") from exc
    evidence = tuple(
        _correspondence_from_payload(item)
        for item in _required_mapping_sequence(payload, "evidence", "timebase alignment")
    )
    alignment = TimebaseAlignment(
        name=_required_str(payload, "name", context="timebase alignment"),
        source=_timebase_from_payload(
            _required_mapping(payload, "source", context="timebase alignment")
        ),
        target=_timebase_from_payload(
            _required_mapping(payload, "target", context="timebase alignment")
        ),
        model=model,
        method=method,
        scale=_required_float(payload, "scale", context="timebase alignment"),
        offset_s=_required_float(payload, "offset_s", context="timebase alignment"),
        evidence=evidence,
        provenance=_source_provenance_from_payload(payload.get("provenance")),
        metadata=dict(_required_mapping(payload, "metadata", context="timebase alignment")),
    )
    stored_residual = _optional_float(
        payload.get("residual_s"), name="timebase alignment residual_s"
    )
    if stored_residual != alignment.residual_s:
        raise ValueError(
            "timebase alignment residual_s does not match its correspondence evidence."
        )
    return alignment


def _correspondence_payload(item: TimebaseCorrespondence) -> dict[str, Any]:
    return {
        "source_time_s": item.source_time_s,
        "target_time_s": item.target_time_s,
        "correspondence_id": item.correspondence_id,
        "metadata": dict(item.metadata),
    }


def _correspondence_from_payload(payload: Mapping[str, Any]) -> TimebaseCorrespondence:
    return TimebaseCorrespondence(
        source_time_s=_required_float(payload, "source_time_s", context="timebase correspondence"),
        target_time_s=_required_float(payload, "target_time_s", context="timebase correspondence"),
        correspondence_id=_optional_str(
            payload.get("correspondence_id"),
            name="timebase correspondence correspondence_id",
        ),
        metadata=dict(_required_mapping(payload, "metadata", context="timebase correspondence")),
    )


def _event_from_payload(payload: Mapping[str, Any]) -> Event:
    kwargs = {
        "event_id": _required_str(payload, "event_id", context="event"),
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
        "tracks": [
            {
                "track_id": track.track_id,
                "name": track.name,
                "metadata": dict(track.metadata),
            }
            for track in trajectory.tracks
        ],
        "keypoint_names": list(trajectory.keypoint_names),
        "positions": trajectory.positions.tolist(),
        "valid": trajectory.valid.tolist(),
        "confidence": (None if trajectory.confidence is None else trajectory.confidence.tolist()),
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
        tracks=tuple(
            PoseTrack(
                track_id=_required_str(item, "track_id", context="pose trajectory track"),
                name=_optional_str(item.get("name"), name="pose trajectory track name"),
                metadata=dict(
                    _required_mapping(item, "metadata", context="pose trajectory track")
                ),
            )
            for item in _required_mapping_sequence(payload, "tracks", "pose trajectory")
        ),
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
            description=_optional_str(frame.get("description"), name="pose frame description"),
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


def _emg_payload(recording: EMGSignalData) -> dict[str, Any]:
    return {
        "sample_times_s": recording.sample_times_s.tolist(),
        "signals": recording.signals.tolist(),
        "channel_names": list(recording.channel_names),
        "muscle_names": list(recording.muscle_names),
        "sides": [side.value for side in recording.sides],
        "sample_rate_hz": recording.sample_rate_hz,
        "units": [list(pair) for pair in recording.units],
        "processing_state": recording.processing_state.value,
        "timebase": _timebase_payload(recording.timebase),
    }


def _emg_from_payload(payload: Mapping[str, Any]) -> EMGSignalData:
    raw_sides = _required_str_tuple(payload, "sides", "EMG signal")
    try:
        sides = tuple(EMGSide(value) for value in raw_sides)
        processing_state = EMGProcessingState(
            _required_str(payload, "processing_state", context="EMG signal")
        )
    except ValueError as exc:
        raise ValueError("EMG signal contains an unsupported enum value.") from exc
    return EMGSignalData(
        sample_times_s=np.asarray(
            _required_sequence(payload, "sample_times_s", context="EMG signal"),
            dtype=np.float64,
        ),
        signals=np.asarray(
            _required_sequence(payload, "signals", context="EMG signal"),
            dtype=np.float64,
        ),
        channel_names=_required_str_tuple(payload, "channel_names", "EMG signal"),
        muscle_names=_required_str_tuple(payload, "muscle_names", "EMG signal"),
        sides=sides,
        sample_rate_hz=_required_float(payload, "sample_rate_hz", context="EMG signal"),
        units=_required_text_pairs(payload, "units", context="EMG signal"),
        processing_state=processing_state,
        timebase=_timebase_from_payload(
            _required_mapping(payload, "timebase", context="EMG signal")
        ),
    )


def _force_payload(recording: ForcePlateData) -> dict[str, Any]:
    return {
        "sample_times_s": recording.sample_times_s.tolist(),
        "force_xyz_N": recording.force_xyz_N.tolist(),
        "plate_names": list(recording.plate_names),
        "valid_mask": recording.valid_mask.tolist(),
        "sample_rate_hz": recording.sample_rate_hz,
        "units": [list(pair) for pair in recording.units],
        "axis_convention": [list(pair) for pair in recording.axis_convention],
        "moment_xyz_Nm": (
            None if recording.moment_xyz_Nm is None else recording.moment_xyz_Nm.tolist()
        ),
        "cop_xyz_m": None if recording.cop_xyz_m is None else recording.cop_xyz_m.tolist(),
        "timebase": _timebase_payload(recording.timebase),
    }


def _force_from_payload(payload: Mapping[str, Any]) -> ForcePlateData:
    return ForcePlateData(
        sample_times_s=np.asarray(
            _required_sequence(payload, "sample_times_s", context="force plate"),
            dtype=np.float64,
        ),
        force_xyz_N=np.asarray(
            _required_sequence(payload, "force_xyz_N", context="force plate"),
            dtype=np.float64,
        ),
        plate_names=_required_str_tuple(payload, "plate_names", "force plate"),
        valid_mask=np.asarray(
            _required_sequence(payload, "valid_mask", context="force plate"),
            dtype=bool,
        ),
        sample_rate_hz=_required_float(payload, "sample_rate_hz", context="force plate"),
        units=_required_text_pairs(payload, "units", context="force plate"),
        axis_convention=_required_text_pairs(payload, "axis_convention", context="force plate"),
        moment_xyz_Nm=_optional_array(payload.get("moment_xyz_Nm"), name="force moment"),
        cop_xyz_m=_optional_array(payload.get("cop_xyz_m"), name="force center of pressure"),
        timebase=_timebase_from_payload(
            _required_mapping(payload, "timebase", context="force plate")
        ),
    )


def _source_provenance_payload(value: SourceProvenance | None) -> dict[str, Any] | None:
    return None if value is None else value.to_dict()


def _source_provenance_from_payload(value: object) -> SourceProvenance | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("modality provenance must be an object or null.")
    return SourceProvenance.from_dict(cast("Mapping[str, Any]", value))


def _timebase_payload(timebase: Timebase) -> dict[str, Any]:
    return {"name": timebase.name, "unit": timebase.unit, "offset_s": timebase.offset_s}


def _timebase_from_payload(payload: Mapping[str, Any]) -> Timebase:
    return Timebase(
        name=_required_str(payload, "name", context="timebase"),
        unit=_required_str(payload, "unit", context="timebase"),
        offset_s=_required_float(payload, "offset_s", context="timebase"),
    )


def _named_timebase(timebases: tuple[Timebase, ...], name: str) -> Timebase:
    for timebase in timebases:
        if timebase.name == name:
            return timebase
    raise ValueError(f"session_timebase_name references unknown timebase {name!r}.")


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


def _required_str_tuple(payload: Mapping[str, Any], key: str, context: str) -> tuple[str, ...]:
    values = _required_sequence(payload, key, context=context)
    if any(not isinstance(value, str) for value in values):
        raise TypeError(f"{context}.{key} must contain only strings.")
    return tuple(cast("Sequence[str]", values))


def _required_text_pairs(
    payload: Mapping[str, Any], key: str, *, context: str
) -> tuple[tuple[str, str], ...]:
    values = _required_sequence(payload, key, context=context)
    pairs: list[tuple[str, str]] = []
    for index, value in enumerate(values):
        if (
            not isinstance(value, Sequence)
            or isinstance(value, str | bytes | bytearray)
            or len(value) != 2
            or any(not isinstance(item, str) for item in value)
        ):
            raise TypeError(f"{context}.{key}[{index}] must be a string pair.")
        pairs.append((cast("str", value[0]), cast("str", value[1])))
    return tuple(pairs)


def _optional_array(value: object, *, name: str) -> np.ndarray | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise TypeError(f"{name} must be an array or null.")
    return np.asarray(value, dtype=np.float64)


def _required_edge_tuple(payload: Mapping[str, Any], key: str) -> tuple[tuple[str, str], ...]:
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
