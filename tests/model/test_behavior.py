from __future__ import annotations

import pytest

from xpkg.model import (
    BehaviorEmbedding,
    BehaviorFrameLabel,
    BehaviorInterval,
    BehaviorLabels,
    Event,
    EventTable,
)


def test_behavior_labels_round_trip_intervals_frames_and_embeddings() -> None:
    labels = BehaviorLabels(
        source_type="keypoint_moseq",
        media_path="Media/session.mkv",
        intervals=(
            BehaviorInterval(
                label="syllable_4",
                start_s=1.0,
                end_s=1.5,
                start_frame=30,
                end_frame=45,
                score=0.88,
                source_id="bout-1",
            ),
        ),
        frame_labels=(
            BehaviorFrameLabel(frame_index=31, label="syllable_4", score=0.91),
            BehaviorFrameLabel(frame_index=30, label="syllable_4", score=0.90),
        ),
        embeddings=(BehaviorEmbedding(frame_index=30, values=(0.1, -0.2, 0.3)),),
        metadata={"producer": {"name": "keypoint-moseq"}},
    )

    payload = labels.to_dict()
    hydrated = BehaviorLabels.from_dict(payload)

    assert payload["schema"] == "xpkg.behavior_labels.v1"
    assert hydrated.to_dict() == payload
    assert hydrated.label_names == ("syllable_4",)
    assert [item.frame_index for item in hydrated.frame_labels] == [30, 31]
    assert hydrated.embeddings[0].values == (0.1, -0.2, 0.3)


@pytest.mark.parametrize(
    ("source_type", "exc_type", "message"),
    [
        (" human_annotation", ValueError, "source_type must not contain surrounding whitespace"),
        ("", ValueError, "source_type must be a non-empty string"),
        (0, TypeError, "source_type must be a string"),
    ],
)
def test_behavior_labels_rejects_unclean_source_type(
    source_type: object,
    exc_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exc_type, match=message):
        BehaviorLabels(
            source_type=source_type,
            intervals=(BehaviorInterval(label="rear", start_s=1.0),),
        )


def test_behavior_labels_from_dict_rejects_unclean_source_type() -> None:
    labels = BehaviorLabels(
        source_type="human_annotation",
        intervals=(BehaviorInterval(label="rear", start_s=1.0),),
    )
    payload = labels.to_dict()
    payload["source_type"] = " human_annotation"

    with pytest.raises(ValueError, match="source_type must not contain surrounding whitespace"):
        BehaviorLabels.from_dict(payload)


def test_behavior_labels_from_dict_rejects_unclean_timebase_text() -> None:
    labels = BehaviorLabels(
        source_type="human_annotation",
        intervals=(BehaviorInterval(label="rear", start_s=1.0),),
    )
    payload = labels.to_dict()
    payload["timebase"]["unit"] = " s"

    with pytest.raises(ValueError, match="timebase unit must not contain surrounding whitespace"):
        BehaviorLabels.from_dict(payload)


def test_behavior_labels_project_time_intervals_to_event_table() -> None:
    labels = BehaviorLabels(
        source_type="human_annotation",
        intervals=(
            BehaviorInterval(label="rear", start_s=3.0, end_s=5.0, confidence="high"),
            BehaviorInterval(label="frame_only", start_frame=10, end_frame=12),
        ),
    )

    events = labels.to_event_table()

    assert len(events) == 1
    assert events.events[0].kind == "behavior"
    assert events.events[0].label == "rear"
    assert events.events[0].start_s == pytest.approx(3.0)
    assert events.events[0].duration_s == pytest.approx(2.0)
    assert events.events[0].metadata["source_type"] == "human_annotation"
    assert events.events[0].metadata["confidence"] == "high"


def test_behavior_labels_can_hydrate_from_generic_event_table() -> None:
    events = EventTable.from_events([Event(kind="cue", start_s=1.0, duration_s=0.25, label="tone")])

    labels = BehaviorLabels.from_event_table(events, source_type="events_csv")

    assert labels.source_type == "events_csv"
    assert labels.intervals[0].label == "tone"
    assert labels.intervals[0].start_s == pytest.approx(1.0)
    assert labels.intervals[0].end_s == pytest.approx(1.25)
    assert labels.to_event_table().events[0].label == "tone"


def test_behavior_interval_rejects_missing_coordinates_and_bad_order() -> None:
    with pytest.raises(ValueError, match="requires start_s or start_frame"):
        BehaviorInterval(label="rear")

    with pytest.raises(ValueError, match="end_s must be >= start_s"):
        BehaviorInterval(label="rear", start_s=2.0, end_s=1.0)

    with pytest.raises(ValueError, match="non-negative"):
        BehaviorFrameLabel(frame_index=-1, label="rear")
