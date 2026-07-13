from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xpkg.model import RecordingSession, SessionSignal, TimeSeries
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
