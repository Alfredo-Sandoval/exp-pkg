"""Ontology objects for a multimodal recording session."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, cast

from xpkg.io.labels.model import Labels
from xpkg.model._metadata_validation import (
    finite_float,
    metadata_dict,
)
from xpkg.model._metadata_validation import (
    strict_required_text as _required_text,
)
from xpkg.model.behavior import BehaviorLabels
from xpkg.model.calibration import Calibration, Camera
from xpkg.model.events import EventTable, SyncEvent
from xpkg.model.metadata import AcquisitionMetadata, CameraMetadata, PoseModelProvenance
from xpkg.model.signals import PhotometryRecording, TimeSeries
from xpkg.model.time import Timebase, TimeRange
from xpkg.pose.trajectory import CoordinateFrameKind, PoseTrajectory

SessionSignalValue = TimeSeries | PhotometryRecording
SessionPoseData = Labels | PoseTrajectory


class SynchronizationMethod(StrEnum):
    """Evidence method used to align two named timebases."""

    OFFSET = "offset"
    AFFINE = "affine"
    PULSES = "pulses"
    TIMESTAMPS = "timestamps"
    MANUAL = "manual"


class _NamedLink(Protocol):
    @property
    def name(self) -> str: ...


@dataclass(frozen=True, slots=True)
class SessionSignal:
    """Named link from a recording session to one sampled signal object."""

    name: str
    recording: SessionSignalValue

    def __post_init__(self) -> None:
        name = _required_text(self.name, name="session signal name")
        if not isinstance(self.recording, TimeSeries | PhotometryRecording):
            raise TypeError(
                "session signal recording must be a TimeSeries or PhotometryRecording; "
                f"got {self.recording!r}."
            )
        object.__setattr__(self, "name", name)


@dataclass(frozen=True, slots=True)
class SessionVideo:
    """Named link from a recording session to one source video path."""

    role: str
    path: Path
    camera: CameraMetadata | None = None
    timebase: Timebase = field(default_factory=Timebase)
    frame_rate_hz: float | None = None
    frame_count: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        role = _required_text(self.role, name="session video role")
        path = Path(self.path)
        if not path.name:
            raise ValueError("session video path must identify a file.")
        if not isinstance(self.timebase, Timebase):
            raise TypeError(f"session video timebase must be a Timebase, got {self.timebase!r}.")
        if self.camera is not None and not isinstance(self.camera, CameraMetadata):
            raise TypeError("session video camera must be CameraMetadata or None.")
        frame_rate_hz = self.frame_rate_hz
        if frame_rate_hz is not None:
            frame_rate_hz = finite_float(frame_rate_hz, name="session video frame_rate_hz")
            if frame_rate_hz <= 0.0:
                raise ValueError("session video frame_rate_hz must be positive.")
        frame_count = self.frame_count
        if frame_count is not None:
            if isinstance(frame_count, bool) or not isinstance(frame_count, int):
                raise TypeError("session video frame_count must be an integer or null.")
            if frame_count < 0:
                raise ValueError("session video frame_count must be non-negative.")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "frame_rate_hz", frame_rate_hz)
        object.__setattr__(self, "frame_count", frame_count)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(metadata_dict(self.metadata, name="session video metadata")),
        )


@dataclass(frozen=True, slots=True)
class SessionPose:
    """Named link from a session to 2D labels or a 2D/3D trajectory."""

    name: str
    data: SessionPoseData
    video_roles: tuple[str, ...] = ()
    calibration_name: str | None = None
    provenance: PoseModelProvenance | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = _required_text(self.name, name="session pose name")
        if not isinstance(self.data, Labels | PoseTrajectory):
            raise TypeError(
                "session pose data must be Labels or PoseTrajectory; "
                f"got {self.data!r}."
            )
        if isinstance(self.data, Labels):
            self.data.validate()
        video_roles = _unique_text_tuple(self.video_roles, name="session pose video_roles")
        calibration_name = (
            None
            if self.calibration_name is None
            else _required_text(self.calibration_name, name="session pose calibration_name")
        )
        if self.provenance is not None and not isinstance(
            self.provenance, PoseModelProvenance
        ):
            raise TypeError("session pose provenance must be PoseModelProvenance or None.")
        if isinstance(self.data, PoseTrajectory):
            needs_calibration = self.data.coordinate_frame.kind in {
                CoordinateFrameKind.CAMERA,
                CoordinateFrameKind.CALIBRATION_WORLD,
            }
            if needs_calibration and calibration_name is None:
                raise ValueError(
                    "camera-frame and calibration-world pose trajectories require "
                    "calibration_name."
                )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "video_roles", video_roles)
        object.__setattr__(self, "calibration_name", calibration_name)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(metadata_dict(self.metadata, name="session pose metadata")),
        )


@dataclass(frozen=True, slots=True)
class SessionBehavior:
    """Named link from a session to canonical temporal behavior labels."""

    name: str
    labels: BehaviorLabels
    video_role: str | None = None

    def __post_init__(self) -> None:
        name = _required_text(self.name, name="session behavior name")
        if not isinstance(self.labels, BehaviorLabels):
            raise TypeError(
                f"session behavior labels must be BehaviorLabels, got {self.labels!r}."
            )
        video_role = (
            None
            if self.video_role is None
            else _required_text(self.video_role, name="session behavior video_role")
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "video_role", video_role)


@dataclass(frozen=True, slots=True)
class CalibrationCameraLink:
    """Mapping between one acquisition camera and one calibrated camera."""

    camera: CameraMetadata
    calibrated_camera: Camera

    def __post_init__(self) -> None:
        if not isinstance(self.camera, CameraMetadata):
            raise TypeError("calibration camera link camera must be CameraMetadata.")
        if not isinstance(self.calibrated_camera, Camera):
            raise TypeError("calibration camera link calibrated_camera must be Camera.")


@dataclass(frozen=True, slots=True)
class SessionCalibration:
    """Named link from a session to one camera calibration."""

    name: str
    calibration: Calibration
    camera_links: tuple[CalibrationCameraLink, ...] = ()

    def __post_init__(self) -> None:
        name = _required_text(self.name, name="session calibration name")
        if not isinstance(self.calibration, Calibration):
            raise TypeError(
                f"session calibration must be a Calibration, got {self.calibration!r}."
            )
        object.__setattr__(self, "name", name)
        links = tuple(self.camera_links)
        _require_unique_calibration_camera_links(links)
        if any(
            link.calibrated_camera not in self.calibration.cameras for link in links
        ):
            raise ValueError("calibration camera link references a camera outside its calibration.")
        object.__setattr__(self, "camera_links", links)


@dataclass(frozen=True, slots=True)
class TimebaseAlignment:
    """First-class evidence-backed affine alignment between two timebases."""

    name: str
    source: Timebase
    target: Timebase
    method: SynchronizationMethod
    scale: float = 1.0
    offset_s: float = 0.0
    residual_s: float | None = None
    evidence: tuple[SyncEvent, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = _required_text(self.name, name="timebase alignment name")
        if not isinstance(self.source, Timebase) or not isinstance(self.target, Timebase):
            raise TypeError("timebase alignment source and target must be Timebase objects.")
        method = self.method
        if not isinstance(method, SynchronizationMethod):
            raise TypeError("timebase alignment method must be a SynchronizationMethod.")
        scale = finite_float(self.scale, name="timebase alignment scale")
        if scale <= 0.0:
            raise ValueError("timebase alignment scale must be positive.")
        offset_s = finite_float(self.offset_s, name="timebase alignment offset_s")
        residual_s = self.residual_s
        if residual_s is not None:
            residual_s = finite_float(residual_s, name="timebase alignment residual_s")
            if residual_s < 0.0:
                raise ValueError("timebase alignment residual_s must be non-negative.")
        evidence = tuple(self.evidence)
        if any(not isinstance(event, SyncEvent) for event in evidence):
            raise TypeError("timebase alignment evidence must contain SyncEvent objects.")
        if method is SynchronizationMethod.PULSES and not evidence:
            raise ValueError("pulse-based timebase alignment requires sync-event evidence.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "offset_s", offset_s)
        object.__setattr__(self, "residual_s", residual_s)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(metadata_dict(self.metadata, name="timebase alignment metadata")),
        )

    def map_time(self, value_s: float) -> float:
        """Map one source-time value into the target timebase."""
        value = finite_float(value_s, name="source time")
        return float((value * self.scale) + self.offset_s)


@dataclass(frozen=True, slots=True)
class RecordingSession:
    """Multimodal experiment session with explicit typed modality links."""

    session_id: str
    title: str | None = None
    acquisition: AcquisitionMetadata | None = None
    timebase: Timebase = field(default_factory=Timebase)
    signals: tuple[SessionSignal, ...] = ()
    videos: tuple[SessionVideo, ...] = ()
    poses: tuple[SessionPose, ...] = ()
    behaviors: tuple[SessionBehavior, ...] = ()
    calibrations: tuple[SessionCalibration, ...] = ()
    alignments: tuple[TimebaseAlignment, ...] = ()
    events: EventTable = field(default_factory=EventTable)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        session_id = _required_text(self.session_id, name="session_id")
        title = None if self.title is None else _required_text(self.title, name="session title")
        if not isinstance(self.timebase, Timebase):
            raise TypeError(f"session timebase must be a Timebase, got {self.timebase!r}.")
        if self.acquisition is not None and not isinstance(
            self.acquisition, AcquisitionMetadata
        ):
            raise TypeError("session acquisition must be AcquisitionMetadata or None.")
        signals = tuple(self.signals)
        videos = tuple(self.videos)
        poses = tuple(self.poses)
        behaviors = tuple(self.behaviors)
        calibrations = tuple(self.calibrations)
        alignments = tuple(self.alignments)
        _require_unique_signals(signals)
        _require_unique_videos(videos)
        _require_unique_links(poses, SessionPose, name="pose")
        _require_unique_links(behaviors, SessionBehavior, name="behavior")
        _require_unique_links(calibrations, SessionCalibration, name="calibration")
        _require_unique_links(alignments, TimebaseAlignment, name="alignment")
        _require_video_links(videos, poses, behaviors)
        _require_acquisition_links(self.acquisition, videos, calibrations)
        _require_pose_calibrations(poses, calibrations)
        if not isinstance(self.events, EventTable):
            raise TypeError(f"session events must be an EventTable, got {self.events!r}.")
        metadata = MappingProxyType(metadata_dict(self.metadata, name="session metadata"))
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "signals", signals)
        object.__setattr__(self, "videos", videos)
        object.__setattr__(self, "poses", poses)
        object.__setattr__(self, "behaviors", behaviors)
        object.__setattr__(self, "calibrations", calibrations)
        object.__setattr__(self, "alignments", alignments)
        object.__setattr__(self, "metadata", metadata)

    @property
    def modality_names(self) -> tuple[str, ...]:
        """Return modality groups currently represented in this session."""
        names: list[str] = []
        if self.acquisition is not None:
            names.append("acquisition")
        if self.videos:
            names.append("videos")
        if self.signals:
            names.append("signals")
        if self.poses:
            names.append("pose")
        if self.behaviors:
            names.append("behavior")
        if self.calibrations:
            names.append("calibration")
        if len(self.events) > 0:
            names.append("events")
        if self.alignments:
            names.append("synchronization")
        return tuple(names)

    @property
    def time_range(self) -> TimeRange | None:
        """Return the broadest available time range across timed modalities."""
        ranges = [link.recording.timeline.time_range for link in self.signals]
        ranges.extend(event.time_range for event in self.events)
        if not ranges:
            return None
        return TimeRange(
            min(time_range.start_s for time_range in ranges),
            max(time_range.end_s for time_range in ranges),
        )

    @property
    def signal_names(self) -> tuple[str, ...]:
        """Return signal link names in storage order."""
        return tuple(link.name for link in self.signals)

    def signal(self, name: str) -> SessionSignalValue:
        """Return the signal linked under ``name`` or raise with session context."""
        key = _required_text(name, name="session signal name")
        for link in self.signals:
            if link.name == key:
                return link.recording
        raise KeyError(f"Recording session {self.session_id!r} has no signal named {key!r}.")

    def video(self, role: str) -> SessionVideo:
        """Return the video link for ``role`` or raise with session context."""
        key = _required_text(role, name="session video role")
        for link in self.videos:
            if link.role == key:
                return link
        raise KeyError(f"Recording session {self.session_id!r} has no video role {key!r}.")

    def pose(self, name: str = "pose") -> SessionPoseData:
        """Return pose data linked under ``name``."""
        return _named_link_value(self.session_id, self.poses, name, "pose", "data")

    def behavior(self, name: str) -> BehaviorLabels:
        """Return behavior labels linked under ``name``."""
        return _named_link_value(self.session_id, self.behaviors, name, "behavior", "labels")

    def calibration(self, name: str) -> Calibration:
        """Return the calibration linked under ``name``."""
        return _named_link_value(
            self.session_id, self.calibrations, name, "calibration", "calibration"
        )


def _require_unique_signals(links: tuple[object, ...]) -> None:
    values: list[str] = []
    for link in links:
        if not isinstance(link, SessionSignal):
            raise TypeError(f"session signal entries must be SessionSignal objects, got {link!r}.")
        if link.name in values:
            raise ValueError(f"Duplicate session signal name: {link.name!r}.")
        values.append(link.name)


def _require_unique_videos(links: tuple[object, ...]) -> None:
    values: list[str] = []
    for link in links:
        if not isinstance(link, SessionVideo):
            raise TypeError(f"session video entries must be SessionVideo objects, got {link!r}.")
        if link.role in values:
            raise ValueError(f"Duplicate session video role: {link.role!r}.")
        values.append(link.role)


def _unique_text_tuple(values: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    result = tuple(_required_text(value, name=name) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique.")
    return result


def _require_unique_links(links: tuple[object, ...], cls: type, *, name: str) -> None:
    names: list[str] = []
    for link in links:
        if not isinstance(link, cls):
            raise TypeError(f"session {name} entries must be {cls.__name__} objects.")
        link_name = cast("_NamedLink", link).name
        if link_name in names:
            raise ValueError(f"Duplicate session {name} name: {link_name!r}.")
        names.append(link_name)


def _require_video_links(
    videos: tuple[SessionVideo, ...],
    poses: tuple[SessionPose, ...],
    behaviors: tuple[SessionBehavior, ...],
) -> None:
    roles = {video.role for video in videos}
    referenced = [role for pose in poses for role in pose.video_roles]
    referenced.extend(
        behavior.video_role for behavior in behaviors if behavior.video_role is not None
    )
    missing = sorted(set(referenced) - roles)
    if missing:
        raise ValueError(f"Session links reference unknown video roles: {', '.join(missing)}.")


def _require_unique_calibration_camera_links(
    links: tuple[CalibrationCameraLink, ...],
) -> None:
    camera_ids: set[str] = set()
    calibration_names: set[str] = set()
    for link in links:
        if not isinstance(link, CalibrationCameraLink):
            raise TypeError(
                "session calibration camera_links must contain CalibrationCameraLink objects."
            )
        if link.camera.camera_id in camera_ids:
            raise ValueError(f"Duplicate calibration camera_id: {link.camera.camera_id!r}.")
        if link.calibrated_camera.name in calibration_names:
            raise ValueError(
                f"Duplicate calibrated camera name: {link.calibrated_camera.name!r}."
            )
        camera_ids.add(link.camera.camera_id)
        calibration_names.add(link.calibrated_camera.name)


def _require_acquisition_links(
    acquisition: AcquisitionMetadata | None,
    videos: tuple[SessionVideo, ...],
    calibrations: tuple[SessionCalibration, ...],
) -> None:
    linked_cameras = [video.camera for video in videos if video.camera is not None]
    linked_cameras.extend(link.camera for item in calibrations for link in item.camera_links)
    if linked_cameras and acquisition is None:
        raise ValueError("Session camera links require acquisition metadata.")
    if acquisition is None:
        return
    registered = {camera.camera_id: camera for camera in acquisition.cameras}
    for camera in linked_cameras:
        if registered.get(camera.camera_id) != camera:
            raise ValueError(
                f"Session references unregistered acquisition camera {camera.camera_id!r}."
            )


def _require_pose_calibrations(
    poses: tuple[SessionPose, ...], calibrations: tuple[SessionCalibration, ...]
) -> None:
    names = {link.name for link in calibrations}
    missing = sorted(
        {
            pose.calibration_name
            for pose in poses
            if pose.calibration_name is not None and pose.calibration_name not in names
        }
    )
    if missing:
        raise ValueError(
            f"Session poses reference unknown calibrations: {', '.join(missing)}."
        )


def _named_link_value(
    session_id: str,
    links: Sequence[_NamedLink],
    name: str,
    kind: str,
    attribute: str,
):
    key = _required_text(name, name=f"session {kind} name")
    for link in links:
        if link.name == key:
            return getattr(link, attribute)
    raise KeyError(f"Recording session {session_id!r} has no {kind} named {key!r}.")


__all__ = [
    "CalibrationCameraLink",
    "RecordingSession",
    "SessionBehavior",
    "SessionCalibration",
    "SessionPose",
    "SessionPoseData",
    "SessionSignal",
    "SessionSignalValue",
    "SessionVideo",
    "SynchronizationMethod",
    "TimebaseAlignment",
]
