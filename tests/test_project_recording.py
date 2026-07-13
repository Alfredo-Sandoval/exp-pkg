from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xpkg.model import (
    AlignmentModel,
    RecordingSession,
    SessionSignal,
    SynchronizationMethod,
    Timebase,
    TimeSeries,
)
from xpkg.project import (
    init_project,
    load_project_session,
    load_project_summary,
    pack_project,
    save_project_labels,
    unpack_project,
    validate_project,
)
from xpkg.services import ProjectService


def _write_photometry_csv(path: Path) -> None:
    path.write_text("time,green,reference\n0.0,1.0,3.0\n0.1,2.0,4.0\n", encoding="utf-8")


def _write_events_csv(path: Path) -> None:
    path.write_text(
        "time,kind,label,duration\n0.5,stimulus,tone,0.1\n1.5,reward,pellet,0.0\n",
        encoding="utf-8",
    )


def _write_behavior_csv(path: Path) -> None:
    path.write_text(
        "onset_s,offset_s,behavior\n0.25,0.75,rear\n1.0,1.4,groom\n",
        encoding="utf-8",
    )


def _write_synchronization_csv(path: Path) -> None:
    path.write_text(
        "pulse_id,source_time_s,target_time_s\np1,0.0,0.25\np2,10.0,10.35\n",
        encoding="utf-8",
    )


def test_photometry_csv_import_persists_typed_recording_session(tmp_path: Path) -> None:
    source = tmp_path / "photometry.csv"
    _write_photometry_csv(source)
    project = ProjectService.create(tmp_path / "Recording Project")

    state_path = project.import_signals(
        "photometry-csv",
        path=source,
        session_id="mouse-1-day-1",
    )

    session = project.load_session()
    recording = session.signal("photometry")
    assert state_path.is_file()
    assert session.session_id == "mouse-1-day-1"
    assert recording.channel_names == ("green", "reference")
    np.testing.assert_array_equal(recording.series.values, [[1.0, 3.0], [2.0, 4.0]])
    source_provenance = recording.series.provenance["source"]
    assert source_provenance["path"] == "Media/signals/photometry.csv"
    assert len(source_provenance["sha256"]) == 64
    assert (project.project_root / source_provenance["path"]).is_file()

    summary = load_project_summary(project.project_root)
    assert summary.state_kind == "experiment"
    assert summary.session_ids == ("mouse-1-day-1",)
    assert summary.state_summary == {
        "experiment_id": project.descriptor().project_id,
            "session_count": 1,
            "subject_count": 0,
            "protocol_count": 0,
            "condition_count": 0,
            "acquisition_session_count": 0,
            "has_dataset_share": False,
        "signal_count": 1,
        "channel_count": 2,
        "sample_count": 2,
        "video_count": 0,
        "event_count": 0,
        "pose_count": 0,
        "label_frame_count": 0,
            "prediction_frame_count": 0,
            "trajectory_frame_count": 0,
        "behavior_count": 0,
        "behavior_interval_count": 0,
        "calibration_count": 0,
        "alignment_count": 0,
        "start_s": 0.0,
        "end_s": 0.1,
    }
    assert summary.modalities == ("signals",)
    validate_project(project.project_root)


def test_recording_session_survives_pack_unpack_roundtrip(tmp_path: Path) -> None:
    source = tmp_path / "photometry.csv"
    _write_photometry_csv(source)
    project = ProjectService.create(tmp_path / "Portable Recording")
    project.import_signals("photometry-csv", path=source, session_id="portable-session")

    artifact = pack_project(project.project_root)
    restored_root = unpack_project(artifact, tmp_path / "Restored Recording")
    restored = load_project_session(restored_root)

    assert restored.session_id == "portable-session"
    assert restored.signal_names == ("photometry",)
    assert restored.signal("photometry").series.provenance["source"]["path"] == (
        "Media/signals/photometry.csv"
    )
    validate_project(restored_root)


def test_event_and_behavior_imports_persist_typed_portable_session_state(
    tmp_path: Path,
) -> None:
    event_source = tmp_path / "events.csv"
    behavior_source = tmp_path / "behavior.csv"
    _write_events_csv(event_source)
    _write_behavior_csv(behavior_source)
    project = ProjectService.create(tmp_path / "Multimodal Recording")

    project.import_events("events-csv", path=event_source, session_id="session-1")
    project.import_behavior(
        "behavior-csv",
        path=behavior_source,
        behavior_name="manual-observations",
        session_id="session-1",
    )

    events = project.load_events()
    behavior = project.load_behavior(behavior_name="manual-observations")
    session = project.load_session()
    assert [(event.kind, event.label) for event in events] == [
        ("stimulus", "tone"),
        ("reward", "pellet"),
    ]
    assert events.metadata["source"]["path"] == "Media/events/events.csv"
    assert len(events.metadata["source"]["sha256"]) == 64
    assert behavior.label_names == ("groom", "rear")
    assert behavior.metadata["source"]["path"] == "Media/behavior/behavior.csv"
    assert behavior.media_path is None
    assert session.modality_names == ("behavior", "events")
    validate_project(project.project_root)

    artifact = project.pack()
    restored = ProjectService.unpack(artifact, tmp_path / "Restored Multimodal")
    assert len(restored.load_events()) == 2
    assert restored.load_behavior(behavior_name="manual-observations").label_names == (
        "groom",
        "rear",
    )
    validate_project(restored.project_root)


def test_event_and_behavior_imports_require_force_to_replace(tmp_path: Path) -> None:
    event_source = tmp_path / "events.csv"
    behavior_source = tmp_path / "behavior.csv"
    _write_events_csv(event_source)
    _write_behavior_csv(behavior_source)
    project = ProjectService.create(tmp_path / "Replacement Rules")
    project.import_events("events-csv", path=event_source, session_id="session-1")
    project.import_behavior("behavior-csv", path=behavior_source, session_id="session-1")

    with pytest.raises(FileExistsError, match="already has events"):
        project.import_events("events-csv", path=event_source, session_id="session-1")
    with pytest.raises(FileExistsError, match="already has behavior"):
        project.import_behavior("behavior-csv", path=behavior_source, session_id="session-1")

    project.import_events(
        "events-csv",
        path=event_source,
        session_id="session-1",
        force=True,
    )
    project.import_behavior(
        "behavior-csv",
        path=behavior_source,
        session_id="session-1",
        force=True,
    )
    assert len(project.load_events()) == 2
    assert project.load_behavior().label_names == ("groom", "rear")
    assert [path.name for path in (project.project_root / "Media/events").iterdir()] == [
        "events.csv"
    ]
    assert [
        path.name for path in (project.project_root / "Media/behavior").iterdir()
    ] == ["behavior.csv"]


def test_synchronization_import_persists_paired_evidence_and_survives_pack(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sync.csv"
    _write_synchronization_csv(source)
    project = ProjectService.create(tmp_path / "Synchronized Recording")

    project.import_synchronization(
        "synchronization-csv",
        path=source,
        source_timebase=Timebase(name="camera"),
        target_timebase=Timebase(name="daq"),
        model=AlignmentModel.AFFINE,
        method=SynchronizationMethod.PULSES,
        session_id="session-1",
    )

    alignment = project.load_alignment(alignment_name="camera-to-daq")
    assert alignment.scale == pytest.approx(1.01)
    assert alignment.offset_s == pytest.approx(0.25)
    assert alignment.evidence[0].correspondence_id == "p1"
    assert alignment.metadata["source"]["path"] == (
        "Media/synchronization/sync.csv"
    )
    assert len(alignment.metadata["source"]["sha256"]) == 64
    assert project.load_session().modality_names == ("synchronization",)

    restored = ProjectService.unpack(project.pack(), tmp_path / "Restored Sync")
    restored_alignment = restored.load_alignment(alignment_name="camera-to-daq")
    assert restored_alignment.evidence == alignment.evidence
    assert restored_alignment.residual_s == pytest.approx(alignment.residual_s)
    validate_project(restored.project_root)


def test_empty_event_import_fails_without_copying_source(tmp_path: Path) -> None:
    source = tmp_path / "empty-events.csv"
    source.write_text("time,kind\n", encoding="utf-8")
    project = ProjectService.create(tmp_path / "No Empty Events")

    with pytest.raises(ValueError, match="contains no events"):
        project.import_events("events-csv", path=source, session_id="session-1")

    assert not (project.project_root / "Media" / "events").exists()


@pytest.mark.parametrize(
    ("format_name", "suffix", "content", "expected_source_type"),
    [
        pytest.param(
            "behavior-json",
            ".json",
            '{"behaviorEvents":[{"label":"rear","startTimeSec":1.0,"endTimeSec":2.0}]}',
            "behavior_events_json",
            id="behavior-json",
        ),
        pytest.param(
            "boris-csv",
            ".csv",
            "Behavior,Start (seconds),Stop (seconds)\nrear,1.0,2.0\n",
            "boris",
            id="boris",
        ),
        pytest.param(
            "bsoid-csv",
            ".csv",
            "frame,cluster_id,probability\n3,2,0.94\n",
            "bsoid",
            id="bsoid",
        ),
        pytest.param(
            "simba-csv",
            ".csv",
            "Frame,Attack,Probability_Attack\n0,1,0.91\n",
            "simba",
            id="simba",
        ),
        pytest.param(
            "keypoint-moseq-csv",
            ".csv",
            "frame_index,motif,score\n10,5,0.83\n",
            "keypoint_moseq",
            id="keypoint-moseq",
        ),
    ],
)
def test_project_behavior_import_dispatches_every_specialized_reader(
    tmp_path: Path,
    format_name: str,
    suffix: str,
    content: str,
    expected_source_type: str,
) -> None:
    source = tmp_path / f"labels{suffix}"
    source.write_text(content, encoding="utf-8")
    project = ProjectService.create(tmp_path / f"Project {format_name}")

    project.import_behavior(format_name, path=source, session_id="session-1")

    labels = project.load_behavior()
    assert labels.source_type == expected_source_type
    assert labels.metadata["source"]["path"].startswith("Media/behavior/")
    assert len(labels.metadata["source"]["sha256"]) == 64


def test_recording_state_cache_rebuilds_from_durable_head(tmp_path: Path) -> None:
    source = tmp_path / "photometry.csv"
    _write_photometry_csv(source)
    project = ProjectService.create(tmp_path / "Cache Rebuild")
    state_path = project.import_signals("photometry-csv", path=source)
    state_path.unlink()

    validate_project(project.project_root)

    assert state_path.is_file()
    assert project.load_session().signal_names == ("photometry",)


def test_recording_project_inspection_is_shallow(tmp_path: Path) -> None:
    source = tmp_path / "photometry.csv"
    _write_photometry_csv(source)
    project = ProjectService.create(tmp_path / "Shallow Recording")
    project.import_signals("photometry-csv", path=source, session_id="shallow-session")

    inspection = project.inspect()

    assert inspection.state_kind == "experiment"
    assert inspection.summary.session_ids == ("shallow-session",)
    assert inspection.summary.state_summary["sample_count"] == 2
    assert inspection.is_valid is True


def test_pose_and_signals_coexist_in_one_session_state(tmp_path: Path) -> None:
    from tests.factories import make_labels

    project = tmp_path / "Recording State"
    init_project(project)
    service = ProjectService.open(project)
    service.save_session(
        RecordingSession(
            session_id="session-1",
            signals=(
                SessionSignal(
                    name="signal",
                    recording=TimeSeries.from_samples(
                        [1.0, 2.0],
                        sample_rate_hz=10.0,
                        channel_names=["signal"],
                    ),
                ),
            ),
        )
    )
    labels = make_labels(tmp_path, x=1.0, y=2.0)

    save_project_labels(project, labels)

    session = service.load_session()
    restored_labels = service.load_labels()
    assert session.signal_names == ("signal",)
    assert len(session.poses) == 1
    assert restored_labels.user_instances[0]["nose"].x == pytest.approx(1.0)


def test_session_save_rejects_absolute_signal_provenance_path(tmp_path: Path) -> None:
    project = ProjectService.create(tmp_path / "Portable Paths")
    source = tmp_path / "external.csv"
    _write_photometry_csv(source)
    session = RecordingSession(
        session_id="session-1",
        signals=(
            SessionSignal(
                name="signal",
                recording=TimeSeries.from_samples(
                    [1.0, 2.0],
                    sample_rate_hz=10.0,
                    channel_names=["signal"],
                    provenance={"source": {"path": str(source.resolve())}},
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="Session source paths must be project-relative"):
        project.save_session(session)


def test_cli_import_signals_uses_project_service_action(tmp_path: Path, capsys) -> None:
    from xpkg.cli import main

    source = tmp_path / "photometry.csv"
    project = tmp_path / "CLI Recording"
    _write_photometry_csv(source)

    code = main(
        [
            "import",
            "signals",
            "photometry-csv",
            "--path",
            str(source),
            "--out",
            str(project),
            "--session-id",
            "cli-session",
        ]
    )

    assert code == 0
    assert load_project_session(project).session_id == "cli-session"
    assert "Imported photometry CSV" in capsys.readouterr().out


def test_cli_imports_events_and_behavior_through_project_service(tmp_path: Path, capsys) -> None:
    from xpkg.cli import main

    event_source = tmp_path / "events.csv"
    behavior_source = tmp_path / "behavior.csv"
    project = tmp_path / "CLI Multimodal"
    _write_events_csv(event_source)
    _write_behavior_csv(behavior_source)

    event_code = main(
        [
            "import",
            "events",
            "events-csv",
            "--path",
            str(event_source),
            "--out",
            str(project),
            "--session-id",
            "cli-session",
        ]
    )
    behavior_code = main(
        [
            "import",
            "behavior",
            "behavior-csv",
            "--path",
            str(behavior_source),
            "--out",
            str(project),
            "--session-id",
            "cli-session",
        ]
    )

    session = load_project_session(project)
    assert event_code == 0
    assert behavior_code == 0
    assert len(session.events) == 2
    assert session.behavior("behavior").label_names == ("groom", "rear")
    output = capsys.readouterr().out
    assert "Imported event CSV" in output
    assert "Imported behavior labels" in output


def test_cli_imports_paired_synchronization_evidence(tmp_path: Path, capsys) -> None:
    from xpkg.cli import main

    source = tmp_path / "sync.csv"
    project = tmp_path / "CLI Synchronization"
    _write_synchronization_csv(source)

    code = main(
        [
            "import",
            "synchronization",
            "synchronization-csv",
            "--path",
            str(source),
            "--out",
            str(project),
            "--source-timebase",
            "camera",
            "--target-timebase",
            "daq",
            "--session-id",
            "cli-session",
        ]
    )

    alignment = ProjectService.open(project).load_alignment(
        alignment_name="camera-to-daq"
    )
    assert code == 0
    assert alignment.method is SynchronizationMethod.PULSES
    assert len(alignment.evidence) == 2
    assert "Imported timebase alignment" in capsys.readouterr().out


def test_project_owns_multiple_sessions_and_requires_explicit_selection(
    tmp_path: Path,
) -> None:
    from tests.factories import make_labels

    project = ProjectService.create(tmp_path / "Multi Session", title="Multi Session")
    project.save_session(
        RecordingSession(
            session_id="baseline",
            signals=(
                SessionSignal(
                    name="signal",
                    recording=TimeSeries.from_samples(
                        [1.0, 2.0],
                        sample_rate_hz=10.0,
                        channel_names=["signal"],
                    ),
                ),
            ),
        )
    )
    project.save_session(
        RecordingSession(
            session_id="follow-up",
            signals=(
                SessionSignal(
                    name="signal",
                    recording=TimeSeries.from_samples(
                        [3.0, 4.0, 5.0],
                        sample_rate_hz=10.0,
                        channel_names=["signal"],
                    ),
                ),
            ),
        )
    )

    experiment = project.load_experiment()
    summary = project.describe().summary
    assert experiment.session_ids == ("baseline", "follow-up")
    assert summary.session_ids == ("baseline", "follow-up")
    assert summary.state_summary["session_count"] == 2
    assert summary.state_summary["sample_count"] == 5
    assert project.load_session(session_id="follow-up").signal("signal").n_samples == 3
    with pytest.raises(ValueError, match="session_id is required"):
        project.load_session()
    with pytest.raises(ValueError, match="session_id is required"):
        project.save_labels(make_labels(tmp_path, x=1.0, y=2.0))

    project.save_labels(
        make_labels(tmp_path, x=1.0, y=2.0),
        session_id="follow-up",
    )
    updated = project.load_experiment()
    assert updated.session_ids == ("baseline", "follow-up")
    assert len(updated.session("baseline").poses) == 0
    assert len(updated.session("follow-up").poses) == 1


def test_multi_session_experiment_survives_pack_unpack(tmp_path: Path) -> None:
    project = ProjectService.create(tmp_path / "Portable Experiment")
    project.save_session(RecordingSession(session_id="day-1"))
    project.save_session(RecordingSession(session_id="day-2"))

    artifact = project.pack()
    restored = ProjectService.unpack(artifact, tmp_path / "Restored Experiment")

    assert restored.load_experiment().session_ids == ("day-1", "day-2")
    assert restored.load_session(session_id="day-2").session_id == "day-2"
    with pytest.raises(ValueError, match="session_id is required"):
        restored.load_session()
