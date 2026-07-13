from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from xpkg.io.session_json import (
    recording_session_document,
    recording_session_from_document,
)
from xpkg.model import (
    CoordinateFrameKind,
    Event,
    EventTable,
    PhotometryChannel,
    PhotometryRecording,
    PoseCoordinateFrame,
    PoseTrajectory,
    RecordingSession,
    SessionPose,
    SessionSignal,
    SessionVideo,
    SyncEvent,
    Timeline,
    TimeSeries,
)

_JSON_SCALAR = (
    st.none()
    | st.booleans()
    | st.integers(min_value=-(2**53), max_value=2**53)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=40)
)
_JSON_VALUE = st.recursive(
    _JSON_SCALAR,
    lambda children: st.lists(children, max_size=6)
    | st.dictionaries(st.text(max_size=20), children, max_size=6),
    max_leaves=30,
)


@given(
    sample_count=st.integers(min_value=1, max_value=30),
    channel_count=st.integers(min_value=1, max_value=4),
    start_s=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
    sample_rate_hz=st.floats(
        min_value=0.1,
        max_value=1000.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_recording_session_document_roundtrips_sampled_signals(
    sample_count: int,
    channel_count: int,
    start_s: float,
    sample_rate_hz: float,
) -> None:
    values = np.arange(sample_count * channel_count, dtype=np.float64).reshape(
        sample_count, channel_count
    )
    channels = tuple(
        PhotometryChannel(name=f"channel_{index}", unit="a.u.", excitation=f"{index} nm")
        for index in range(channel_count)
    )
    series = TimeSeries(
        values=values,
        channels=channels,
        timeline=Timeline.from_sample_rate(
            n_samples=sample_count,
            sample_rate_hz=sample_rate_hz,
            start_s=start_s,
        ),
        name="photometry",
        provenance={"source": {"path": "Media/signals/source.csv"}},
    )
    session = RecordingSession(
        session_id="session-1",
        title="Session 1",
        signals=(
            SessionSignal(
                name="photometry",
                recording=PhotometryRecording(
                    series=series,
                    signal_channel="channel_0",
                    metadata={"instrument": "test"},
                ),
            ),
        ),
        videos=(SessionVideo(role="behavior", path=Path("Media/behavior.mp4")),),
        events=EventTable(
            events=(
                Event(kind="trial", start_s=start_s, duration_s=0.5),
                SyncEvent(kind="pulse", start_s=start_s + 1.0, source="daq"),
            )
        ),
        metadata={"subject": "mouse-1"},
    )

    restored = recording_session_from_document(recording_session_document(session))

    assert restored.session_id == session.session_id
    assert restored.title == session.title
    assert restored.videos == session.videos
    assert dict(restored.metadata) == dict(session.metadata)
    assert isinstance(restored.events.events[1], SyncEvent)
    assert restored.events.events[1].source == "daq"
    restored_recording = restored.signal("photometry")
    assert isinstance(restored_recording, PhotometryRecording)
    np.testing.assert_array_equal(restored_recording.series.values, values)
    np.testing.assert_allclose(
        restored_recording.timeline.timestamps_s,
        series.timeline.timestamps_s,
        rtol=0.0,
        atol=0.0,
    )
    assert restored_recording.timeline.sample_rate_hz == sample_rate_hz
    assert restored_recording.series.channels == channels


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format", "legacy.recording", "recording-session format"),
        ("schema_version", 1, "recording-session schema_version"),
    ],
)
def test_recording_session_document_rejects_unknown_contract(
    field: str,
    value: object,
    message: str,
) -> None:
    document = recording_session_document(RecordingSession(session_id="session-1"))
    document[field] = value

    with pytest.raises(ValueError, match=message):
        recording_session_from_document(document)


def test_recording_session_document_rejects_non_object_signal() -> None:
    document = recording_session_document(RecordingSession(session_id="session-1"))
    document["payload"]["session"]["signals"] = ["not-an-object"]

    with pytest.raises(TypeError, match=r"recording session\.signals\[0\]"):
        recording_session_from_document(document)


@given(
    n_frames=st.integers(min_value=1, max_value=12),
    n_tracks=st.integers(min_value=1, max_value=4),
    n_keypoints=st.integers(min_value=1, max_value=6),
    dims=st.sampled_from((2, 3)),
)
def test_recording_session_document_roundtrips_pose_trajectories(
    n_frames: int, n_tracks: int, n_keypoints: int, dims: Literal[2, 3]
) -> None:
    kind = (
        CoordinateFrameKind.IMAGE_PIXEL
        if dims == 2
        else CoordinateFrameKind.LIFTED_MODEL
    )
    trajectory = PoseTrajectory(
        fps=29.97,
        track_ids=tuple(f"track-{index}" for index in range(n_tracks)),
        keypoint_names=tuple(f"point_{index}" for index in range(n_keypoints)),
        positions=np.arange(
            n_frames * n_tracks * n_keypoints * dims, dtype=np.float64
        ).reshape(
            n_frames, n_tracks, n_keypoints, dims
        ),
        valid=np.ones((n_frames, n_tracks, n_keypoints), dtype=bool),
        confidence=np.full(
            (n_frames, n_tracks, n_keypoints), 0.75, dtype=np.float64
        ),
        dims=dims,
        coordinate_frame=PoseCoordinateFrame(kind=kind, units="px" if dims == 2 else "a.u."),
    )
    session = RecordingSession(
        session_id="trajectory-session",
        poses=(SessionPose(name="pose", data=trajectory),),
    )

    restored = recording_session_from_document(recording_session_document(session))

    restored_trajectory = restored.pose()
    assert isinstance(restored_trajectory, PoseTrajectory)
    assert restored_trajectory.coordinate_frame == trajectory.coordinate_frame
    np.testing.assert_array_equal(restored_trajectory.positions, trajectory.positions)
    np.testing.assert_array_equal(restored_trajectory.valid, trajectory.valid)
    np.testing.assert_array_equal(restored_trajectory.confidence, trajectory.confidence)


@given(document=st.dictionaries(st.text(max_size=20), _JSON_VALUE, max_size=8))
def test_recording_session_parser_fails_with_declared_errors_only(
    document: dict[str, object],
) -> None:
    try:
        restored = recording_session_from_document(document)
    except (TypeError, ValueError):
        return

    assert isinstance(restored, RecordingSession)
