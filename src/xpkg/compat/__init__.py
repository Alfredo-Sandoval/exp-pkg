"""Explicit `.xpkg` edge helpers that survive the workspace-first cutover.

New integrations should start with ``xpkg.services`` and ``xpkg.formats``.
This module remains available for direct archive IO, fixtures, migration, and
other deliberate edge workflows that still need explicit `.xpkg` handling.
"""

from __future__ import annotations

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
    load_archive_metadata_field as _load_archive_metadata_field,
)
from xpkg.io.archive_io import (
    merge_predictions_archive as _merge_predictions_archive,
)
from xpkg.io.archive_io import (
    read_archive as _read_archive,
)
from xpkg.io.archive_io import (
    save_archive_metadata_field as _save_archive_metadata_field,
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
    open_archive_store,
)
from xpkg.io.archive_store import (
    create_xpkg_store as _create_xpkg_store,
)

read_xpkg = _read_archive
write_xpkg = _write_archive
update_labels_xpkg = _update_labels_archive
append_predictions_xpkg = _append_predictions_archive
merge_predictions_xpkg = _merge_predictions_archive
summarize_xpkg = _summarize_archive
validate_xpkg = _validate_archive
load_archive_metadata_field = _load_archive_metadata_field
save_archive_metadata_field = _save_archive_metadata_field

create_store_from_xpkg = _create_xpkg_store
open_store = open_archive_store

__all__ = [
    "ArchiveStore",
    "LazyDatasetHandle",
    "MaxInstancesExceededError",
    "PredictionAppendItem",
    "SerializerPredictedInstance",
    "append_predictions_xpkg",
    "create_store_from_xpkg",
    "load_archive_metadata_field",
    "merge_predictions_xpkg",
    "open_store",
    "read_metrics_table",
    "read_xpkg",
    "save_archive_metadata_field",
    "summarize_xpkg",
    "update_labels_xpkg",
    "validate_xpkg",
    "write_metrics_table",
    "write_xpkg",
]
