from __future__ import annotations

import json

import pytest

from xpkg.io.readers import read_behavior_events_csv, read_behavior_events_json
from xpkg.model import BehaviorLabels


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
