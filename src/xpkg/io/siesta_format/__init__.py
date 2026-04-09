"""Public API for the native `.siesta` serializer."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "LazyDatasetHandle": ("xpkg.io.siesta_format.reader", "LazyDatasetHandle"),
    "MaxInstancesExceededError": (
        "xpkg.io.siesta_format.predictions_datasets",
        "MaxInstancesExceededError",
    ),
    "PredictionAppendItem": (
        "xpkg.io.siesta_format.predictions_datasets",
        "PredictionAppendItem",
    ),
    "SEGMENTATION_SCHEMA_VERSION": (
        "xpkg.io.siesta_format.segmentation_hdf5",
        "SEGMENTATION_SCHEMA_VERSION",
    ),
    "SerializerPredictedInstance": (
        "xpkg.io.siesta_format.predictions_datasets",
        "SerializerPredictedInstance",
    ),
    "append_predictions_siesta": (
        "xpkg.io.siesta_format.append_ops",
        "append_predictions_siesta",
    ),
    "merge_predictions_siesta": (
        "xpkg.io.siesta_format.append_ops",
        "merge_predictions_siesta",
    ),
    "read_segmentation_group": (
        "xpkg.io.siesta_format.segmentation_hdf5",
        "read_segmentation_group",
    ),
    "read_siesta": ("xpkg.io.siesta_format.reader", "read_siesta"),
    "summarize_project": ("xpkg.io.siesta_format.reader", "summarize_project"),
    "update_labels_siesta": ("xpkg.io.siesta_format.writer_core", "update_labels_siesta"),
    "validate_project": ("xpkg.io.siesta_format.reader", "validate_project"),
    "write_segmentation_group": (
        "xpkg.io.siesta_format.segmentation_hdf5",
        "write_segmentation_group",
    ),
    "write_siesta": ("xpkg.io.siesta_format.writer_core", "write_siesta"),
}

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


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
