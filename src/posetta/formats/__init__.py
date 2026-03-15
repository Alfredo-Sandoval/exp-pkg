"""Public format entry points."""

from __future__ import annotations

from posetta.formats.siesta import (
    LazyDatasetHandle,
    MaxInstancesExceededError,
    PredictionAppendItem,
    SerializerPredictedInstance,
    append_predictions_siesta,
    merge_predictions_siesta,
    read_metrics_table,
    read_siesta,
    summarize_project,
    update_labels_siesta,
    validate_project,
    write_metrics_table,
    write_siesta,
)
from posetta.formats.siesta_store import (
    SiestaStore,
    create_store_from_archive,
    create_store_from_sta,
    open_store,
)
from posetta.io.labels.json_format import read_labels_json_payload, write_labels_json

__all__ = [
    "LazyDatasetHandle",
    "MaxInstancesExceededError",
    "PredictionAppendItem",
    "SerializerPredictedInstance",
    "SiestaStore",
    "append_predictions_siesta",
    "create_store_from_archive",
    "merge_predictions_siesta",
    "create_store_from_sta",
    "open_store",
    "read_metrics_table",
    "read_labels_json_payload",
    "read_siesta",
    "summarize_project",
    "update_labels_siesta",
    "validate_project",
    "write_labels_json",
    "write_metrics_table",
    "write_siesta",
]
