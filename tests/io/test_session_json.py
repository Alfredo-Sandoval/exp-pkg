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
    AlignmentModel,
    CoordinateFrameKind,
    EMGProcessingState,
    EMGSide,
    EMGSignalData,
    Event,
    EventTable,
    ForcePlateData,
    PhotometryChannel,
    PhotometryRecording,
    PoseCoordinateFrame,
    PoseTrack,
    PoseTrajectory,
    RecordingSession,
    SessionEventStream,
    SessionPose,
    SessionSignal,
    SessionVideo,
    SourceProvenance,
    SyncEvent,
    SynchronizationMethod,
    Timebase,
    TimebaseCorrespondence,
    Timeline,
    TimeSeries,
    fit_timebase_alignment,
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
        timebases=(Timebase(name="camera"), Timebase(name="daq")),
        alignments=(
            fit_timebase_alignment(
                name="camera-to-daq",
                source=Timebase(name="camera"),
                target=Timebase(name="daq"),
                model=AlignmentModel.OFFSET,
                method=SynchronizationMethod.PULSES,
                evidence=(
                    TimebaseCorrespondence(0.0, 0.25, correspondence_id="pulse-1"),
                    TimebaseCorrespondence(1.0, 1.25, correspondence_id="pulse-2"),
                ),
            ),
        ),
        event_streams=(
            SessionEventStream(
                "trials",
                EventTable(
                    events=(
                        Event(
                            event_id="trial-1",
                            kind="trial",
                            start_s=start_s,
                            duration_s=0.5,
                        ),
                        SyncEvent(
                            event_id="pulse-1",
                            kind="pulse",
                            start_s=start_s + 1.0,
                            source="daq",
                        ),
                    )
                ),
            ),
        ),
        metadata={"subject": "mouse-1"},
    )

    restored = recording_session_from_document(recording_session_document(session))

    assert restored.session_id == session.session_id
    assert restored.title == session.title
    assert restored.videos == session.videos
    assert dict(restored.metadata) == dict(session.metadata)
    restored_events = restored.event_stream("trials")
    assert isinstance(restored_events.events[1], SyncEvent)
    assert restored_events.events[1].source == "daq"
    assert restored.alignment("camera-to-daq").evidence[0].correspondence_id == "pulse-1"
    assert restored.alignment("camera-to-daq").offset_s == pytest.approx(0.25)
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
        tracks=tuple(PoseTrack(f"track-{index}") for index in range(n_tracks)),
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


def test_recording_session_document_roundtrips_emg_force_and_provenance() -> None:
    daq = Timebase(name="daq")
    source = SourceProvenance(
        source_type="csv",
        source_path="Media/signals/source.csv",
        size_bytes=128,
        sha256="a" * 64,
        tool="xpkg",
        tool_version="0.1.0",
    )
    emg = EMGSignalData(
        sample_times_s=np.array([0.0, 0.001]),
        signals=np.array([[1.0, 2.0], [3.0, 4.0]]),
        channel_names=("left-ta", "right-ta"),
        muscle_names=("tibialis anterior", "tibialis anterior"),
        sides=(EMGSide.LEFT, EMGSide.RIGHT),
        sample_rate_hz=1000.0,
        units=(("amplitude", "mV"),),
        processing_state=EMGProcessingState.RAW,
        timebase=daq,
    )
    force = ForcePlateData(
        sample_times_s=np.array([0.0, 0.001]),
        force_xyz_N=np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]),
        plate_names=("plate-1",),
        valid_mask=np.ones((2, 1), dtype=bool),
        sample_rate_hz=1000.0,
        units=(("force", "N"),),
        axis_convention=(("x", "forward"), ("y", "lateral"), ("z", "up")),
        timebase=daq,
    )
    session = RecordingSession(
        session_id="biomechanics",
        timebases=(daq,),
        signals=(
            SessionSignal("emg", emg, provenance=source),
            SessionSignal("force", force, provenance=source),
        ),
    )

    restored = recording_session_from_document(recording_session_document(session))

    restored_emg = restored.signal("emg")
    restored_force = restored.signal("force")
    assert isinstance(restored_emg, EMGSignalData)
    assert isinstance(restored_force, ForcePlateData)
    np.testing.assert_array_equal(restored_emg.signals, emg.signals)
    np.testing.assert_array_equal(restored_force.force_xyz_N, force.force_xyz_N)
    assert restored.signals[0].provenance == source
    assert restored_emg.timebase == daq
    assert restored_force.timebase == daq


@given(document=st.dictionaries(st.text(max_size=20), _JSON_VALUE, max_size=8))
def test_recording_session_parser_fails_with_declared_errors_only(
    document: dict[str, object],
) -> None:
    try:
        restored = recording_session_from_document(document)
    except (TypeError, ValueError):
        return

    assert isinstance(restored, RecordingSession)
