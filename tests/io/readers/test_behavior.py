from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpkg.io.readers import (
    read_behavior_events_csv,
    read_behavior_events_json,
    read_boris_csv,
    read_bsoid_csv,
    read_keypoint_moseq_syllables_csv,
    read_simba_csv,
)
from xpkg.model import BehaviorLabels

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def test_read_behavior_events_json_maps_human_annotation_intervals(tmp_path) -> None:
    path = tmp_path / "behavior_annotation.json"
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "projectName": "RAT_1_FR1_15MIN_04_05_2026",
                    "videoFileName": "RAT_1_FR1_15MIN_04_05_2026.mkv",
                    "annotatorName": "Dara Freemon",
                },
                "behaviorEvents": [
                    {
                        "behaviorEventId": "behavior-1",
                        "label": "Rearing #1",
                        "startTimeSec": 35.734,
                        "endTimeSec": 37.588,
                        "durationSec": 1.854,
                        "confidence": "high",
                        "notes": "clean bout",
                    }
                ],
                "bodyParts": [{"name": "Nose"}],
                "frameAnnotations": [],
                "skeletonConnections": [[1, 2]],
            }
        ),
        encoding="utf-8",
    )

    labels = read_behavior_events_json(path)

    assert isinstance(labels, BehaviorLabels)
    assert labels.source_type == "behavior_events_json"
    assert labels.media_path == "RAT_1_FR1_15MIN_04_05_2026.mkv"
    assert labels.annotator == "Dara Freemon"
    assert labels.metadata["body_part_count"] == 1
    assert labels.metadata["frame_annotation_count"] == 0
    assert labels.intervals[0].label == "Rearing #1"
    assert labels.intervals[0].source_id == "behavior-1"
    assert labels.intervals[0].confidence == "high"
    assert labels.intervals[0].duration_s == pytest.approx(1.854)
    assert labels.to_event_table().events[0].metadata["notes"] == "clean bout"


@pytest.mark.parametrize(
    ("reader", "source_type"),
    [
        (read_behavior_events_json, " behavior_events_json"),
        (read_behavior_events_csv, " behavior_csv"),
    ],
)
def test_behavior_event_readers_reject_unclean_source_type_before_file_access(
    tmp_path,
    reader,
    source_type: str,
) -> None:
    with pytest.raises(ValueError, match="source_type must not contain surrounding whitespace"):
        reader(tmp_path / "missing-source.csv", source_type=source_type)


def test_read_behavior_events_csv_accepts_interval_labels_from_package_exports(tmp_path) -> None:
    path = tmp_path / "bsoid_bouts.csv"
    path.write_text(
        "\n".join(
            [
                "onset_s,offset_s,behavior,probability,start_frame,end_frame",
                "1.0,2.5,rear,0.93,30,75",
                "4.0,4.5,groom,0.81,120,135",
            ]
        ),
        encoding="utf-8",
    )

    labels = read_behavior_events_csv(path, source_type="bsoid")

    assert labels.source_type == "bsoid"
    assert labels.label_names == ("groom", "rear")
    assert [item.label for item in labels.intervals] == ["rear", "groom"]
    assert labels.intervals[0].start_frame == 30
    assert labels.intervals[0].end_frame == 75
    assert labels.intervals[0].score == pytest.approx(0.93)
    assert labels.to_event_table().query(label="groom")[0].start_s == pytest.approx(4.0)


def test_read_boris_csv_maps_aggregated_event_exports(tmp_path) -> None:
    path = tmp_path / "boris_events.csv"
    path.write_text(
        "\n".join(
            [
                "Observation id,Subject,Behavior,Behavioral category,Behavior type,"
                "Start (seconds),Stop (seconds),Duration (seconds),Media file name,"
                "Comment start,Comment stop",
                "obs-1,rat-1,rear,posture,STATE,1.25,2.75,1.5,trial.mp4,clean start,",
                "obs-1,rat-1,groom,maintenance,STATE,4.0,5.0,1.0,trial.mp4,,clean stop",
            ]
        ),
        encoding="utf-8",
    )

    labels = read_boris_csv(path)

    assert labels.source_type == "boris"
    assert labels.media_path == "trial.mp4"
    assert labels.metadata["source"]["format"] == "aggregated_events_csv"
    assert [item.label for item in labels.intervals] == ["rear", "groom"]
    assert labels.intervals[0].source_id == "obs-1"
    assert labels.intervals[0].start_s == pytest.approx(1.25)
    assert labels.intervals[0].end_s == pytest.approx(2.75)
    assert labels.intervals[0].duration_s == pytest.approx(1.5)
    assert labels.intervals[0].metadata["Subject"] == "rat-1"
    assert labels.intervals[0].metadata["Behavior type"] == "STATE"
    assert labels.intervals[0].metadata["Comment start"] == "clean start"
    assert labels.to_event_table().query(label="groom")[0].end_s == pytest.approx(5.0)


def test_read_bsoid_csv_maps_framewise_cluster_exports(tmp_path) -> None:
    path = tmp_path / "bsoid_predictions.csv"
    path.write_text(
        "\n".join(
            [
                "frame,cluster_id,probability,uncertainty,source_file",
                "3,2,0.94,0.06,trial_pose.csv",
                "4,5,0.72,0.28,trial_pose.csv",
            ]
        ),
        encoding="utf-8",
    )

    labels = read_bsoid_csv(path, media_path="trial.mp4")

    assert labels.source_type == "bsoid"
    assert labels.media_path == "trial.mp4"
    assert labels.metadata["source"]["format"] == "bsoid_csv"
    assert labels.metadata["source"]["label_column"] == "cluster_id"
    assert [(item.frame_index, item.label) for item in labels.frame_labels] == [
        (3, "2"),
        (4, "5"),
    ]
    assert labels.frame_labels[0].score == pytest.approx(0.94)
    assert labels.frame_labels[0].metadata["source_file"] == "trial_pose.csv"
    assert labels.frame_labels[0].metadata["uncertainty"] == pytest.approx(0.06)


def test_read_bsoid_csv_accepts_interval_bout_exports(tmp_path) -> None:
    path = tmp_path / "bsoid_bouts.csv"
    path.write_text(
        "\n".join(
            [
                "start_frame,end_frame,behavior,confidence,bout_id",
                "10,16,rear,high,bout-1",
                "20,24,groom,medium,bout-2",
            ]
        ),
        encoding="utf-8",
    )

    labels = read_bsoid_csv(path)

    assert [item.label for item in labels.intervals] == ["rear", "groom"]
    assert labels.intervals[0].start_frame == 10
    assert labels.intervals[0].end_frame == 16
    assert labels.intervals[0].confidence == "high"
    assert labels.intervals[0].metadata["bout_id"] == "bout-1"


def test_read_simba_csv_maps_machine_results_frame_labels(tmp_path) -> None:
    path = tmp_path / "trial_machine_results.csv"
    path.write_text(
        "\n".join(
            [
                "Frame,Time,Attack,Probability_Attack,Confidence_Attack,"
                "Sniffing,Probability_Sniffing,nose_x",
                "0,0.000,0,0.12,low,1,0.88,23.1",
                "1,0.033,1,0.91,high,0,0.22,24.0",
            ]
        ),
        encoding="utf-8",
    )

    labels = read_simba_csv(path, media_path="trial.mp4")

    assert labels.source_type == "simba"
    assert labels.media_path == "trial.mp4"
    assert labels.metadata["source"]["format"] == "framewise_classifier_csv"
    assert labels.metadata["source"]["behavior_columns"] == ["Attack", "Sniffing"]
    assert labels.metadata["source"]["probability_columns"] == [
        "Probability_Attack",
        "Probability_Sniffing",
    ]
    assert [(item.frame_index, item.label) for item in labels.frame_labels] == [
        (0, "Sniffing"),
        (1, "Attack"),
    ]
    assert labels.frame_labels[0].score == pytest.approx(0.88)
    assert labels.frame_labels[0].metadata["time_s"] == pytest.approx(0.0)
    assert labels.frame_labels[0].metadata["Probability_Attack"] == pytest.approx(0.12)
    assert labels.frame_labels[0].metadata["nose_x"] == pytest.approx(23.1)
    assert labels.frame_labels[1].score == pytest.approx(0.91)
    assert labels.frame_labels[1].confidence == "high"
    assert labels.frame_labels[1].metadata["Probability_Sniffing"] == pytest.approx(0.22)


def test_read_simba_csv_accepts_probability_only_validation_outputs(tmp_path) -> None:
    path = tmp_path / "trial_validation.csv"
    path.write_text(
        "\n".join(
            [
                "Probability_Rearing,velocity",
                "0.2,1.4",
                "0.7,1.9",
            ]
        ),
        encoding="utf-8",
    )

    labels = read_simba_csv(path, probability_threshold=0.5)

    assert [(item.frame_index, item.label) for item in labels.frame_labels] == [(1, "Rearing")]
    assert labels.frame_labels[0].score == pytest.approx(0.7)
    assert labels.frame_labels[0].metadata["frame_index_source"] == "row_index"
    assert labels.frame_labels[0].metadata["velocity"] == pytest.approx(1.9)


def test_read_keypoint_moseq_syllables_csv_maps_row_indexed_exports() -> None:
    path = FIXTURES / "keypoint_moseq_syllables.csv"

    labels = read_keypoint_moseq_syllables_csv(path, media_path="trial.mp4")

    assert labels.source_type == "keypoint_moseq"
    assert labels.media_path == "trial.mp4"
    assert labels.metadata["source"]["format"] == "syllable_csv"
    assert labels.metadata["source"]["frame_index_source"] == "row_index"
    assert labels.metadata["source"]["syllable_column"] == "syllable"
    assert labels.metadata["source"]["recording_name"] == "keypoint_moseq_syllables"
    assert labels.metadata["source"]["uncertainty_columns"] == ["uncertainty"]
    assert [(item.frame_index, item.label) for item in labels.frame_labels] == [
        (0, "syllable_2"),
        (1, "syllable_2"),
        (2, "syllable_4"),
    ]
    assert labels.frame_labels[0].confidence == "high"
    assert labels.frame_labels[0].metadata["frame_index_source"] == "row_index"
    assert labels.frame_labels[0].metadata["recording_name"] == "keypoint_moseq_syllables"
    assert labels.frame_labels[0].metadata["source_label"] == 2
    assert labels.frame_labels[0].metadata["source_confidence"] == "high"
    assert labels.frame_labels[0].metadata["centroid x"] == pytest.approx(10.5)
    assert labels.frame_labels[0].metadata["latent_state 0"] == pytest.approx(0.1)
    assert labels.frame_labels[0].metadata["uncertainty"] == pytest.approx(0.03)
    assert labels.frame_labels[0].metadata["onset"] is True


def test_read_keypoint_moseq_syllables_csv_accepts_framewise_motif_outputs(tmp_path) -> None:
    path = tmp_path / "moseq_df.csv"
    path.write_text(
        "\n".join(
            [
                "frame_index,motif,score,entropy,name,time_s",
                "10,5,0.83,0.12,trial-a,0.333",
                "12,motif_7,0.62,0.31,trial-a,0.400",
            ]
        ),
        encoding="utf-8",
    )

    labels = read_keypoint_moseq_syllables_csv(path)

    assert labels.metadata["source"]["frame_column"] == "frame_index"
    assert labels.metadata["source"]["recording_column"] == "name"
    assert labels.metadata["source"]["recording_name"] == "trial-a"
    assert [(item.frame_index, item.label) for item in labels.frame_labels] == [
        (10, "motif_5"),
        (12, "motif_7"),
    ]
    assert labels.frame_labels[0].score == pytest.approx(0.83)
    assert labels.frame_labels[0].metadata["recording_name"] == "trial-a"
    assert labels.frame_labels[0].metadata["time_s"] == pytest.approx(0.333)
    assert labels.frame_labels[0].metadata["source_score"] == pytest.approx(0.83)
    assert labels.frame_labels[0].metadata["entropy"] == pytest.approx(0.12)


def test_read_behavior_events_csv_accepts_framewise_motif_labels(tmp_path) -> None:
    path = tmp_path / "keypoint_moseq_syllables.csv"
    path.write_text(
        "\n".join(
            [
                "frame,syllable,confidence",
                "11,syllable_2,high",
                "10,syllable_1,medium",
            ]
        ),
        encoding="utf-8",
    )

    labels = read_behavior_events_csv(path, source_type="keypoint_moseq")

    assert labels.intervals == ()
    assert [item.frame_index for item in labels.frame_labels] == [10, 11]
    assert labels.frame_labels[0].label == "syllable_1"
    assert labels.frame_labels[0].confidence == "medium"
    assert len(labels.to_event_table()) == 0


def test_read_behavior_events_csv_requires_labels_and_timing(tmp_path) -> None:
    missing_label = tmp_path / "missing_label.csv"
    missing_label.write_text("timestamp\n1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="behavior label column"):
        read_behavior_events_csv(missing_label)

    missing_time = tmp_path / "missing_time.csv"
    missing_time.write_text("behavior\nrear\n", encoding="utf-8")

    with pytest.raises(ValueError, match="time or frame index"):
        read_behavior_events_csv(missing_time)


@pytest.mark.parametrize(
    ("kwargs", "exc_type", "message"),
    [
        (
            {"label_column": ""},
            ValueError,
            "label_column must be a non-empty string",
        ),
        (
            {"start_column": " start_s"},
            ValueError,
            "start_column must not contain surrounding whitespace",
        ),
        (
            {"end_column": 0},
            TypeError,
            "end_column must be a string",
        ),
        (
            {"duration_column": " duration"},
            ValueError,
            "duration_column must not contain surrounding whitespace",
        ),
        (
            {"frame_column": " frame"},
            ValueError,
            "frame_column must not contain surrounding whitespace",
        ),
        (
            {"start_frame_column": " start_frame"},
            ValueError,
            "start_frame_column must not contain surrounding whitespace",
        ),
        (
            {"end_frame_column": " end_frame"},
            ValueError,
            "end_frame_column must not contain surrounding whitespace",
        ),
        (
            {"score_column": " score"},
            ValueError,
            "score_column must not contain surrounding whitespace",
        ),
        (
            {"confidence_column": " confidence"},
            ValueError,
            "confidence_column must not contain surrounding whitespace",
        ),
        (
            {"source_id_column": " source_id"},
            ValueError,
            "source_id_column must not contain surrounding whitespace",
        ),
    ],
)
def test_read_behavior_events_csv_rejects_unclean_explicit_column_selectors(
    tmp_path,
    kwargs: dict[str, object],
    exc_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exc_type, match=message):
        read_behavior_events_csv(tmp_path / "missing.csv", **kwargs)


# --- Real-format contract tests (byte-faithful fixtures) ---------------------
# The fixtures below reproduce the exact layouts the real tools write, derived
# from each tool's writer source (not copied datasets), so the readers stay
# pinned to real exports rather than synthetic conveniences.


def test_read_boris_csv_reads_real_tabular_export() -> None:
    # Default "Export events > Tabular": metadata preamble, then START/STOP/POINT
    # rows in a Status column with a single Time column.
    labels = read_boris_csv(FIXTURES / "boris_tabular_events.csv")

    assert labels.source_type == "boris"
    assert labels.metadata["source"]["format"] == "tabular_events_csv"
    assert labels.media_path == "trial.mp4"
    by_label = {interval.label: interval for interval in labels.intervals}
    assert by_label["rear"].start_s == pytest.approx(1.25)
    assert by_label["rear"].end_s == pytest.approx(2.75)
    assert by_label["groom"].start_s == pytest.approx(4.0)
    # POINT row becomes a zero-duration interval.
    assert by_label["blink"].start_s == pytest.approx(6.5)
    assert by_label["blink"].duration_s == pytest.approx(0.0)
    assert by_label["rear"].metadata["behavior_type"] == "STATE"
    assert by_label["blink"].metadata["behavior_type"] == "POINT"


def test_read_boris_csv_reads_legacy_aggregated_media_file() -> None:
    # Legacy aggregated export uses the singular "Media file" header and the
    # "Start (s)"/"Stop (s)" naming; POINT rows carry Duration (s) = NA.
    labels = read_boris_csv(FIXTURES / "boris_aggregated_legacy.csv")

    assert labels.metadata["source"]["format"] == "aggregated_events_csv"
    assert labels.media_path == "trial.mp4"
    by_label = {interval.label: interval for interval in labels.intervals}
    assert by_label["rear"].start_s == pytest.approx(1.8)
    assert by_label["rear"].end_s == pytest.approx(8.125)
    # POINT row: NA duration parses to a zero-length interval at its onset.
    assert by_label["groom"].start_s == pytest.approx(32.825)
    assert by_label["groom"].end_s == pytest.approx(32.825)


def test_read_bsoid_csv_reads_real_per_frame_labels() -> None:
    # Real B-SOiD "labels_pose" export: leading index column + "B-SOiD labels".
    labels = read_bsoid_csv(FIXTURES / "bsoid_labels_pose.csv")

    assert labels.metadata["source"]["format"] == "bsoid_labels_csv"
    assert labels.metadata["source"]["label_column"] == "B-SOiD labels"
    assert [(item.frame_index, item.label) for item in labels.frame_labels] == [
        (0, "4"),
        (1, "4"),
        (2, "2"),
    ]


def test_read_bsoid_csv_reads_real_run_length_bouts() -> None:
    # Real B-SOiD run-length table: "Start time (frames)" + "Run lengths".
    labels = read_bsoid_csv(FIXTURES / "bsoid_runlen.csv")

    assert labels.metadata["source"]["format"] == "bsoid_runlen_csv"
    intervals = {interval.label: interval for interval in labels.intervals}
    assert intervals["4"].start_frame == 0
    assert intervals["4"].end_frame == 2  # frames 0..2 (run length 3)
    assert intervals["7"].start_frame == 3
    assert intervals["7"].end_frame == 4  # frames 3..4 (run length 2)


def test_read_simba_csv_reads_real_machine_results() -> None:
    # Real SimBA machine_results: empty-header index column, then for each
    # classifier a Probability_<Name> float and a <Name> binary column.
    labels = read_simba_csv(FIXTURES / "simba_machine_results.csv")

    assert labels.source_type == "simba"
    positives = {(item.frame_index, item.label) for item in labels.frame_labels}
    assert (0, "Attack") in positives
    assert (1, "Sniffing") in positives
    # Frame 2 is below every classifier's 0/1 flag, so it yields no label.
    assert all(item.frame_index != 2 for item in labels.frame_labels)


def test_read_boris_tabular_skips_nonfinite_time_rows(tmp_path) -> None:
    # A single corrupt (inf/NaN) Time cell must skip that row, not abort the
    # whole parse. The inf STOP below is dropped, so the START at 1.0 pairs with
    # the valid STOP at 2.0.
    path = tmp_path / "boris_tab_corrupt.csv"
    path.write_text(
        "\r\n".join(
            [
                "Time,Media file path,Subject,Behavior,Behavioral category,Comment,Status",
                "1.000,v.mp4,s1,rear,posture,,START",
                "inf,v.mp4,s1,rear,posture,,STOP",
                "2.000,v.mp4,s1,rear,posture,,STOP",
            ]
        )
        + "\r\n",
        encoding="utf-8",
    )

    labels = read_boris_csv(path)

    assert labels.metadata["source"]["format"] == "tabular_events_csv"
    assert len(labels.intervals) == 1
    assert labels.intervals[0].start_s == pytest.approx(1.0)
    assert labels.intervals[0].end_s == pytest.approx(2.0)
