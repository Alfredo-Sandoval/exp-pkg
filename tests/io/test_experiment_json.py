from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from xpkg.io.experiment_json import experiment_document, experiment_from_document
from xpkg.io.session_json import (
    recording_session_document,
    recording_session_from_document,
)
from xpkg.model import (
    BehaviorInterval,
    BehaviorLabels,
    BehaviorSubjectLink,
    CoordinateFrameKind,
    DatasetShareMetadata,
    Event,
    EventRelationship,
    EventRelationshipKind,
    EventTable,
    Experiment,
    ExperimentalCondition,
    ExperimentSessionLink,
    PoseCoordinateFrame,
    PoseTrack,
    PoseTrajectory,
    Protocol,
    RecordingSession,
    SessionBehavior,
    SessionConditionLink,
    SessionEventStream,
    SessionPose,
    SessionProtocolLink,
    SessionSignal,
    SessionSubjectLink,
    Subject,
    SubjectTrackAssignment,
    TimeSeries,
    add_experiment_session,
    replace_experiment_session,
)


def _session(session_id: str, values: list[float]) -> RecordingSession:
    return RecordingSession(
        session_id=session_id,
        signals=(
            SessionSignal(
                name="signal",
                recording=TimeSeries.from_samples(
                    values,
                    sample_rate_hz=10.0,
                    channel_names=["signal"],
                ),
            ),
        ),
    )


def test_experiment_document_roundtrips_multiple_session_links() -> None:
    subject = Subject(subject_id="mouse-1", species="Mus musculus")
    protocol = Protocol(protocol_id="open-field-v1", name="Open field", version="1")
    condition = ExperimentalCondition(condition_id="vehicle", name="Vehicle")
    experiment = Experiment(
        experiment_id="experiment-1",
        title="Longitudinal experiment",
        subjects=(subject,),
        protocols=(protocol,),
        conditions=(condition,),
        session_links=(
            ExperimentSessionLink(
                session=_session("baseline", [1.0, 2.0]),
                subjects=(SessionSubjectLink(subject=subject, role="focal"),),
                protocols=(SessionProtocolLink(protocol=protocol, role="behavior"),),
                conditions=(
                    SessionConditionLink(condition=condition, subjects=(subject,)),
                ),
                metadata={"role": "baseline", "day": 0},
            ),
            ExperimentSessionLink(
                session=_session("follow-up", [3.0, 4.0]),
                metadata={"role": "follow-up", "day": 7},
            ),
        ),
        dataset_share=DatasetShareMetadata(
            title="Longitudinal experiment",
            creators=("A. Researcher",),
        ),
        metadata={"cohort": "pilot"},
    )

    restored = experiment_from_document(experiment_document(experiment))

    assert restored.experiment_id == "experiment-1"
    assert restored.session_ids == ("baseline", "follow-up")
    assert dict(restored.session_links[1].metadata) == {"role": "follow-up", "day": 7}
    restored_signal = restored.session("baseline").signal("signal")
    assert isinstance(restored_signal, TimeSeries)
    assert restored_signal.values[:, 0].tolist() == [1.0, 2.0]
    assert restored.subjects == (subject,)
    assert restored.session_links[0].subjects[0].subject is restored.subjects[0]
    assert restored.session_links[0].protocols[0].protocol is restored.protocols[0]
    assert restored.session_links[0].conditions[0].condition is restored.conditions[0]
    assert restored.dataset_share is not None
    assert restored.dataset_share.title == "Longitudinal experiment"
    assert dict(restored.metadata) == {"cohort": "pilot"}


def test_experiment_roundtrips_subject_to_multi_animal_pose_tracks() -> None:
    first = Subject(subject_id="mouse-1")
    second = Subject(subject_id="mouse-2")
    trajectory = PoseTrajectory(
        fps=30.0,
        tracks=(PoseTrack("track-a"), PoseTrack("track-b")),
        keypoint_names=("nose", "tail"),
        positions=np.zeros((3, 2, 2, 3), dtype=np.float64),
        valid=np.ones((3, 2, 2), dtype=bool),
        dims=3,
        coordinate_frame=PoseCoordinateFrame(
            kind=CoordinateFrameKind.LIFTED_MODEL,
            units="mm",
        ),
    )
    session = RecordingSession(
        session_id="social-interaction",
        poses=(SessionPose(name="pose-3d", data=trajectory),),
    )
    pose = session.poses[0]
    experiment = Experiment(
        experiment_id="experiment-1",
        title="Social interaction",
        subjects=(first, second),
        session_links=(
            ExperimentSessionLink(
                session=session,
                subjects=(SessionSubjectLink(first), SessionSubjectLink(second)),
                subject_track_assignments=(
                    SubjectTrackAssignment(
                        first,
                        pose=pose,
                        track=trajectory.track("track-a"),
                        start_frame=0,
                        end_frame=2,
                    ),
                    SubjectTrackAssignment(
                        second,
                        pose=pose,
                        track=trajectory.track("track-b"),
                        start_frame=0,
                        end_frame=2,
                    ),
                ),
            ),
        ),
    )

    restored = experiment_from_document(experiment_document(experiment))

    links = restored.session_links[0].subject_track_assignments
    assert [(link.subject_id, link.track_id) for link in links] == [
        ("mouse-1", "track-a"),
        ("mouse-2", "track-b"),
    ]
    assert links[0].subject is restored.subjects[0]

    replacement = recording_session_from_document(recording_session_document(session))
    replaced = replace_experiment_session(experiment, replacement)
    replaced_assignment = replaced.session_links[0].subject_track_assignments[0]
    assert replaced_assignment.pose is replacement.poses[0]
    assert replaced_assignment.track is replacement.poses[0].data.tracks[0]


def test_experiment_roundtrips_behavior_attribution_and_event_relationships() -> None:
    subject = Subject(subject_id="mouse-1")
    behavior = SessionBehavior(
        name="ethogram",
        labels=BehaviorLabels(
            source_type="manual",
            intervals=(BehaviorInterval(label="approach", start_s=1.0, end_s=2.0),),
        ),
    )
    stimulus = Event(event_id="stimulus-1", kind="stimulus", start_s=0.5)
    response = Event(event_id="response-1", kind="response", start_s=1.0)
    stimuli = SessionEventStream("stimuli", EventTable(events=(stimulus,)))
    responses = SessionEventStream("responses", EventTable(events=(response,)))
    session = RecordingSession(
        session_id="session-1",
        behaviors=(behavior,),
        event_streams=(stimuli, responses),
    )
    experiment = Experiment(
        experiment_id="experiment-1",
        title="Stimulus response",
        subjects=(subject,),
        session_links=(
            ExperimentSessionLink(
                session=session,
                subjects=(SessionSubjectLink(subject),),
                behavior_subjects=(BehaviorSubjectLink(behavior, subject),),
                event_relationships=(
                    EventRelationship(
                        source_stream=stimuli,
                        source_event=stimulus,
                        target_stream=responses,
                        target_event=response,
                        kind=EventRelationshipKind.TRIGGERS,
                    ),
                ),
            ),
        ),
    )

    restored = experiment_from_document(experiment_document(experiment))

    restored_link = restored.session_links[0]
    assert restored_link.behavior_subjects[0].subject is restored.subjects[0]
    assert restored_link.behavior_subjects[0].behavior is restored_link.session.behaviors[0]
    relationship = restored_link.event_relationships[0]
    assert relationship.kind is EventRelationshipKind.TRIGGERS
    assert relationship.source_event.event_id == "stimulus-1"
    assert relationship.target_event.event_id == "response-1"

    replacement = recording_session_from_document(recording_session_document(session))
    replaced = replace_experiment_session(experiment, replacement)
    replaced_link = replaced.session_links[0]
    assert replaced_link.behavior_subjects[0].behavior is replacement.behaviors[0]
    assert replaced_link.event_relationships[0].source_stream is replacement.event_streams[0]
    replaced_relationship = replaced_link.event_relationships[0]
    assert replaced_relationship.source_event is replacement.event_streams[0].events.events[0]


def test_subject_track_reassignment_requires_nonoverlapping_frame_ranges() -> None:
    first = Subject(subject_id="mouse-1")
    second = Subject(subject_id="mouse-2")
    trajectory = PoseTrajectory(
        fps=30.0,
        tracks=(PoseTrack("track-a"),),
        keypoint_names=("nose",),
        positions=np.zeros((10, 1, 1, 2), dtype=np.float64),
        valid=np.ones((10, 1, 1), dtype=bool),
        dims=2,
        coordinate_frame=PoseCoordinateFrame(
            kind=CoordinateFrameKind.IMAGE_PIXEL,
            units="px",
        ),
    )
    session = RecordingSession(
        session_id="session-1",
        poses=(SessionPose(name="pose", data=trajectory),),
    )
    pose = session.poses[0]
    assignments = (
        SubjectTrackAssignment(first, pose, trajectory.track("track-a"), 0, 4),
        SubjectTrackAssignment(second, pose, trajectory.track("track-a"), 5, 9),
    )

    link = ExperimentSessionLink(
        session=session,
        subjects=(SessionSubjectLink(first), SessionSubjectLink(second)),
        subject_track_assignments=assignments,
    )
    assert link.subject_track_assignments == assignments

    with pytest.raises(ValueError, match="cannot overlap"):
        ExperimentSessionLink(
            session=session,
            subjects=(SessionSubjectLink(first), SessionSubjectLink(second)),
            subject_track_assignments=(
                assignments[0],
                SubjectTrackAssignment(second, pose, trajectory.track("track-a"), 4, 9),
            ),
        )


def test_experiment_rejects_subject_link_to_unknown_pose_track() -> None:
    subject = Subject(subject_id="mouse-1")
    trajectory = PoseTrajectory(
        fps=30.0,
        tracks=(PoseTrack("track-a"),),
        keypoint_names=("nose",),
        positions=np.zeros((3, 1, 1, 2), dtype=np.float64),
        valid=np.ones((3, 1, 1), dtype=bool),
        dims=2,
        coordinate_frame=PoseCoordinateFrame(
            kind=CoordinateFrameKind.IMAGE_PIXEL,
            units="px",
        ),
    )

    pose = SessionPose(name="pose", data=trajectory)
    with pytest.raises(ValueError, match="track outside its pose"):
        ExperimentSessionLink(
            session=RecordingSession(
                session_id="session-1",
                poses=(pose,),
            ),
            subjects=(SessionSubjectLink(subject),),
            subject_track_assignments=(
                SubjectTrackAssignment(
                    subject,
                    pose=pose,
                    track=PoseTrack("track-b"),
                    start_frame=0,
                    end_frame=2,
                ),
            ),
        )


def test_experiment_rejects_duplicate_session_identity() -> None:
    session = RecordingSession(session_id="duplicate")

    with pytest.raises(ValueError, match="Duplicate experiment session session_id"):
        Experiment(
            experiment_id="experiment-1",
            title="Experiment",
            session_links=(
                ExperimentSessionLink(session=session),
                ExperimentSessionLink(session=session),
            ),
        )


def test_replace_experiment_session_preserves_link_metadata_and_order() -> None:
    experiment = Experiment(
        experiment_id="experiment-1",
        title="Experiment",
        session_links=(
            ExperimentSessionLink(
                session=_session("baseline", [1.0]), metadata={"role": "baseline"}
            ),
            ExperimentSessionLink(session=_session("follow-up", [2.0])),
        ),
    )

    replaced = replace_experiment_session(
        experiment,
        _session("baseline", [10.0, 11.0]),
    )

    assert replaced.session_ids == ("baseline", "follow-up")
    assert dict(replaced.session_links[0].metadata) == {"role": "baseline"}
    replaced_signal = replaced.session("baseline").signal("signal")
    assert isinstance(replaced_signal, TimeSeries)
    assert replaced_signal.n_samples == 2


def test_add_experiment_session_rejects_existing_identity() -> None:
    experiment = Experiment(
        experiment_id="experiment-1",
        title="Experiment",
        session_links=(ExperimentSessionLink(session=_session("baseline", [1.0])),),
    )

    with pytest.raises(ValueError, match="already has session 'baseline'"):
        add_experiment_session(experiment, _session("baseline", [2.0]))


_JSON_VALUE = st.recursive(
    st.none() | st.booleans() | st.integers() | st.text(max_size=20),
    lambda child: st.lists(child, max_size=5)
    | st.dictionaries(st.text(max_size=20), child, max_size=5),
    max_leaves=20,
)


@given(document=st.dictionaries(st.text(max_size=20), _JSON_VALUE, max_size=8))
def test_experiment_parser_returns_valid_objects_or_declared_errors(
    document: dict[str, object],
) -> None:
    try:
        experiment = experiment_from_document(document)
    except (TypeError, ValueError):
        return

    assert isinstance(experiment, Experiment)
