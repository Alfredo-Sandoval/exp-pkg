"""Public API for the unified `.sta` serializer."""

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
from posetta.io.siesta_format.segmentation_hdf5 import (
    SEGMENTATION_SCHEMA_VERSION,
    read_segmentation_group,
    write_segmentation_group,
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
    "SEGMENTATION_SCHEMA_VERSION",
    "read_segmentation_group",
    "write_segmentation_group",
]
