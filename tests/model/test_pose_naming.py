from __future__ import annotations

import pytest

from xpkg.model.pose_naming import (
    lookup_unique_label_or_marker,
    normalize_event_side,
    normalize_event_type,
    normalize_marker_name,
    normalize_source_label,
    strip_subject_prefix,
)


def test_subject_prefix_stripping_and_marker_normalization() -> None:
    assert strip_subject_prefix("Mouse1:L_Hip") == "L_Hip"
    assert normalize_marker_name("Mouse1:L_Hip") == "l_hip"


def test_duplicate_canonical_marker_labels_require_exact_lookup() -> None:
    labels = ("Mouse1:L_Hip", "Mouse2:L_Hip")

    assert lookup_unique_label_or_marker(labels, "Mouse2:L_Hip", kind="Marker") == 1
    with pytest.raises(KeyError, match="ambiguous"):
        lookup_unique_label_or_marker(labels, "L_Hip", kind="Marker")


def test_source_label_normalization_preserves_namespace() -> None:
    assert normalize_source_label(" Mouse1:L_Hip ") == "mouse1:l_hip"


def test_event_type_normalization() -> None:
    assert normalize_event_type(" Foot-Strike ") == "foot_strike"
    assert normalize_event_type(" @@@ ") == "event"


def test_event_side_normalization() -> None:
    assert normalize_event_side(" Left ") == "left"
    assert normalize_event_side("Right") == "right"
    assert normalize_event_side("General") is None

