"""Public API for the unified `.siesta` serializer."""

from posetta.io.siesta_format.append_ops import append_predictions_siesta, merge_predictions_siesta
from posetta.io.siesta_format.predictions_datasets import (
    MaxInstancesExceededError,
    PredictionAppendItem,
    SerializerPredictedInstance,
)
from posetta.io.siesta_format.reader import (
    LazyDatasetHandle,
    read_siesta,
    summarize_project,
    validate_project,
)
from posetta.io.siesta_format.writer_core import update_labels_siesta, write_siesta

__all__ = [
    "LazyDatasetHandle",
    "MaxInstancesExceededError",
    "PredictionAppendItem",
    "SerializerPredictedInstance",
    "append_predictions_siesta",
    "merge_predictions_siesta",
    "read_siesta",
    "summarize_project",
    "update_labels_siesta",
    "validate_project",
    "write_siesta",
]
