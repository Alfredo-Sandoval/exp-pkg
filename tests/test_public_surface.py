from __future__ import annotations

import posetta
from posetta.adapters import (
    ConversionResult,
    convert_dlc_csv,
    convert_dlc_h5,
    convert_dlc_project,
    convert_sleap_package,
)
from posetta.formats import (
    LazyDatasetHandle,
    MaxInstancesExceededError,
    PredictionAppendItem,
    SerializerPredictedInstance,
    append_predictions_siesta,
    merge_predictions_siesta,
    read_labels_json_payload,
    read_metrics_table,
    read_siesta,
    summarize_project,
    update_labels_siesta,
    validate_project,
    write_labels_json,
    write_metrics_table,
    write_siesta,
)
from posetta.model import (
    Instance,
    Keypoint,
    KPFlag,
    LabeledFrame,
    Labels,
    Point,
    PointArray,
    PredictedInstance,
    PredictedPoint,
    PredictedPointArray,
    Skeleton,
    SuggestionFrame,
    Track,
    Video,
    build_keypoint_skeleton,
    is_predicted_instance,
    load_skeleton,
    load_skeleton_dlc,
    load_skeleton_siesta_json,
    load_skeleton_sleap,
    load_skeleton_ultralytics,
)


def test_public_exports_are_callable() -> None:
    assert posetta.__version__
    assert posetta.adapters is not None
    assert posetta.formats is not None
    assert posetta.model is not None
    assert ConversionResult is not None
    assert LazyDatasetHandle is not None
    assert PredictionAppendItem is not None
    assert SerializerPredictedInstance is not None
    assert MaxInstancesExceededError is not None
    assert callable(append_predictions_siesta)
    assert callable(merge_predictions_siesta)
    assert callable(read_labels_json_payload)
    assert callable(read_metrics_table)
    assert callable(read_siesta)
    assert callable(summarize_project)
    assert callable(update_labels_siesta)
    assert callable(validate_project)
    assert callable(write_labels_json)
    assert callable(write_metrics_table)
    assert callable(write_siesta)
    assert callable(convert_dlc_csv)
    assert callable(convert_dlc_h5)
    assert callable(convert_dlc_project)
    assert callable(convert_sleap_package)


def test_model_exports_are_available() -> None:
    assert Labels is not None
    assert SuggestionFrame is not None
    assert Skeleton is not None
    assert Keypoint is not None
    assert Track is not None
    assert LabeledFrame is not None
    assert Instance is not None
    assert PredictedInstance is not None
    assert Point is not None
    assert PredictedPoint is not None
    assert PointArray is not None
    assert PredictedPointArray is not None
    assert Video is not None
    assert KPFlag is not None
    assert callable(build_keypoint_skeleton)
    assert callable(is_predicted_instance)
    assert callable(load_skeleton)
    assert callable(load_skeleton_dlc)
    assert callable(load_skeleton_siesta_json)
    assert callable(load_skeleton_sleap)
    assert callable(load_skeleton_ultralytics)
