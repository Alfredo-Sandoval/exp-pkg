"""Compatibility surface for canonical `.xpkg` archives and edge helpers."""

from __future__ import annotations

import warnings
from typing import Any

from xpkg.io.archive_io import (
    LazyDatasetHandle,
    MaxInstancesExceededError,
    PredictionAppendItem,
    SerializerPredictedInstance,
    read_metrics_table,
    write_metrics_table,
)
from xpkg.io.archive_io import (
    append_predictions_archive as _append_predictions_archive,
)
from xpkg.io.archive_io import (
    merge_predictions_archive as _merge_predictions_archive,
)
from xpkg.io.archive_io import (
    read_archive as _read_archive,
)
from xpkg.io.archive_io import (
    summarize_archive as _summarize_archive,
)
from xpkg.io.archive_io import (
    update_labels_archive as _update_labels_archive,
)
from xpkg.io.archive_io import (
    validate_archive as _validate_archive,
)
from xpkg.io.archive_io import (
    write_archive as _write_archive,
)
from xpkg.io.archive_store import (
    ArchiveStore,
    create_xpkg_store,
)
from xpkg.io.archive_store import (
    create_archive_store as _create_archive_store,
)
from xpkg.io.archive_store import (
    open_archive_store as _open_archive_store,
)

read_xpkg = _read_archive
write_xpkg = _write_archive
update_labels_xpkg = _update_labels_archive
append_predictions_xpkg = _append_predictions_archive
merge_predictions_xpkg = _merge_predictions_archive
summarize_xpkg = _summarize_archive
validate_xpkg = _validate_archive

create_store_from_xpkg = create_xpkg_store
open_store = _open_archive_store

_LEGACY_EXPORTS: dict[str, tuple[str, Any]] = {
    "append_predictions_archive": ("append_predictions_xpkg", _append_predictions_archive),
    "create_archive_store": ("create_store_from_xpkg", _create_archive_store),
    "create_store_from_archive": ("create_store_from_xpkg", _create_archive_store),
    "merge_predictions_archive": ("merge_predictions_xpkg", _merge_predictions_archive),
    "open_archive_store": ("open_store", _open_archive_store),
    "read_archive": ("read_xpkg", _read_archive),
    "summarize_archive": ("summarize_xpkg", _summarize_archive),
    "summarize_project": ("summarize_xpkg", _summarize_archive),
    "update_labels_archive": ("update_labels_xpkg", _update_labels_archive),
    "validate_archive": ("validate_xpkg", _validate_archive),
    "validate_project": ("validate_xpkg", _validate_archive),
    "write_archive": ("write_xpkg", _write_archive),
}

__all__ = [
    "ArchiveStore",
    "LazyDatasetHandle",
    "MaxInstancesExceededError",
    "PredictionAppendItem",
    "SerializerPredictedInstance",
    "append_predictions_xpkg",
    "create_store_from_xpkg",
    "create_xpkg_store",
    "merge_predictions_xpkg",
    "open_store",
    "read_metrics_table",
    "read_xpkg",
    "summarize_xpkg",
    "update_labels_xpkg",
    "validate_xpkg",
    "write_metrics_table",
    "write_xpkg",
]


def __getattr__(name: str) -> Any:
    legacy_export = _LEGACY_EXPORTS.get(name)
    if legacy_export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    canonical_name, value = legacy_export
    warnings.warn(
        f"xpkg.compat.{name} is a legacy alias; use xpkg.compat.{canonical_name}",
        DeprecationWarning,
        stacklevel=2,
    )
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
