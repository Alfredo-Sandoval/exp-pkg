"""Project save/import boundary backed by the private ``.xpkg`` store.

The package gathers storage, cache, conversion, and importer implementations
from focused submodules:

- :mod:`xpkg.project.store._helpers` — small lifecycle/lookup helpers
- :mod:`xpkg.project.store.payloads` — state ↔ public payload converters
- :mod:`xpkg.project.store.provenance` — pose / prediction provenance helpers
- :mod:`xpkg.project.store.media` — managed-media file copying and rebasing
- :mod:`xpkg.project.store.cache` — state cache, commit, and write helpers
- :mod:`xpkg.project.store.conversion` — converter-result → project orchestration
- :mod:`xpkg.project.store.imports` — per-format importer implementations
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from xpkg._core.json_utils import parse_json
from xpkg._core.path_registry import ensure_dir
from xpkg.io.labels.json_format import XPKG_LABELS_JSON_FORMAT
from xpkg.project.layout import (
    load_project_descriptor,
    project_current_state_path,
    project_exports_root,
    project_media_root,
    project_store_root,
    resolve_project_root,
)
from xpkg.project.state_io import (
    predictions_payload_from_labels,
    read_project_state,
    rewrite_project_metadata_paths,
)
from xpkg.project.store._helpers import (
    ProjectStore as ProjectStore,
)
from xpkg.project.store._helpers import (
    _clone_metadata,
    _project_store,
    _touch_descriptor,
)
from xpkg.project.store._helpers import (
    current_project_commit_id as current_project_commit_id,
)
from xpkg.project.store._helpers import (
    ensure_project as ensure_project,
)
from xpkg.project.store._helpers import (
    init_project as init_project,
)
from xpkg.project.store.cache import (
    _commit_labels_to_project,
    _commit_state_metadata_to_project,
    _current_project_state_payload,
    _project_state_components,
)
from xpkg.project.store.cache import (
    ensure_current_project_state_cache as ensure_current_project_state_cache,
)
from xpkg.project.store.cache import (
    rebuild_project_state_cache as rebuild_project_state_cache,
)
from xpkg.project.store.media import (
    rebase_project_payload_videos as rebase_project_payload_videos,
)
from xpkg.project.store.payloads import (
    _predictions_committed_length,
    _predictions_payload_from_state_payload,
    _public_payload_from_state_labels,
    _state_metadata_from_state_payload,
)

if TYPE_CHECKING:
    from xpkg.model import Labels


_PREDICTIONS_STATE_MARKER = ',"predictions":'
_STATE_PREFIX_READ_BYTES = 64 * 1024


def _labels_state_payload_from_document(document: object) -> dict[str, Any] | None:
    if not isinstance(document, Mapping):
        raise TypeError("Project state document must contain a mapping")
    document_mapping = cast(Mapping[str, object], document)
    raw_format = str(document_mapping.get("format", "")).strip()
    if raw_format != XPKG_LABELS_JSON_FORMAT:
        return None
    payload = document_mapping.get("payload")
    if not isinstance(payload, Mapping):
        return None
    return dict(cast(Mapping[str, Any], payload))


def _read_labels_state_payload_without_predictions(state_path: Path) -> dict[str, Any] | None:
    text = ""
    with state_path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(_STATE_PREFIX_READ_BYTES)
            if not chunk:
                return _labels_state_payload_from_document(parse_json(text))
            text += chunk
            marker_index = text.find(_PREDICTIONS_STATE_MARKER)
            if marker_index >= 0:
                return _labels_state_payload_from_document(parse_json(text[:marker_index] + "}}"))


def _current_labels_state_payload_without_predictions(root: Path) -> dict[str, Any] | None:
    state_path = current_project_state_path(root)
    if not state_path.exists():
        return None
    return _read_labels_state_payload_without_predictions(state_path)


def load_project_payload(path: str | Path, *, include_predictions: bool = True) -> dict[str, Any]:
    """Return the current committed project payload on the public project surface."""
    root = resolve_project_root(path)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg project: {path}")
    state: tuple[dict[str, Any], str] | None
    if include_predictions:
        state = _current_project_state_payload(root)
    else:
        labels_payload = _current_labels_state_payload_without_predictions(root)
        state = None if labels_payload is None else (labels_payload, "state_labels")
    if state is None:
        return {"metadata": {}}
    payload, source_kind = state
    metadata = _state_metadata_from_state_payload(payload) or {}
    if source_kind == "state_labels":
        public_payload = _public_payload_from_state_labels(payload, metadata=metadata)
        rebase_project_payload_videos(public_payload, root)
        return public_payload
    raise ValueError(f"Unsupported project state source: {source_kind!r}")


def current_project_state_path(path: str | Path) -> Path:
    """Return the project state cache path under `.xpkg/state/current.json`."""

    return project_current_state_path(path)


def load_project_metadata(project: str | Path) -> dict[str, Any]:
    """Return the current project metadata payload from the managed head."""

    root = resolve_project_root(project)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg project: {project}")

    store = _project_store(root)
    labels_state_payload = (
        _read_labels_state_payload_without_predictions(store.current_state_path())
        if store.has_current_state()
        else _current_labels_state_payload_without_predictions(root)
    )
    if labels_state_payload is not None:
        return _clone_metadata(_state_metadata_from_state_payload(labels_state_payload))

    current_state = _current_project_state_payload(root)
    if current_state is None:
        return {}

    state_payload, source_kind = current_state
    if source_kind != "state_labels":
        raise ValueError(f"Unsupported project state source: {source_kind!r}")
    metadata = _state_metadata_from_state_payload(state_payload)
    return _clone_metadata(metadata)


def save_project_metadata(
    project: str | Path,
    metadata: Mapping[str, Any] | None,
    *,
    reason: str = "project.save.metadata",
) -> Path:
    """Commit updated metadata onto the current project head."""

    root = resolve_project_root(project)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg project: {project}")

    descriptor = load_project_descriptor(root)
    descriptor.validate()
    ensure_dir(project_store_root(root))
    ensure_dir(project_media_root(root))
    ensure_dir(project_exports_root(root))

    current_state = _current_project_state_payload(root)
    if current_state is None:
        raise FileNotFoundError(f"Project has no committed state: {root}")

    normalized_metadata = rewrite_project_metadata_paths(
        None if metadata is None else dict(metadata),
        project_root=root,
    )
    state_payload, source_kind = current_state
    if source_kind == "state_labels":
        state_path = _commit_state_metadata_to_project(
            root,
            state_payload=state_payload,
            metadata=normalized_metadata,
            reason=reason,
        )
        _touch_descriptor(root)
        return state_path

    raise ValueError(f"Unsupported project state source: {source_kind!r}")


def save_project_labels(
    project: str | Path,
    labels: Labels,
    *,
    metadata: dict[str, Any] | None = None,
    journal: bool = True,
    regenerate_predictions: bool = False,
) -> Path:
    """Commit a label save into the project's private store."""
    root = resolve_project_root(project)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg project: {project}")

    descriptor = load_project_descriptor(root)
    descriptor.validate()
    ensure_dir(project_store_root(root))
    ensure_dir(project_media_root(root))
    ensure_dir(project_exports_root(root))

    state_path = current_project_state_path(root)
    current_state = _current_project_state_payload(root)
    has_state_cache = state_path.exists()
    has_committed_state = current_state is not None
    if metadata is not None and (has_state_cache or has_committed_state):
        raise ValueError(
            "Project saves with existing history do not accept metadata overrides. "
            "Update project metadata through a dedicated metadata API."
        )

    if not has_state_cache and not has_committed_state:
        initial_metadata = rewrite_project_metadata_paths(
            metadata,
            project_root=root,
        )
        initial_predictions = predictions_payload_from_labels(labels)
        state_path = _commit_labels_to_project(
            root,
            labels=labels,
            metadata=initial_metadata,
            predictions=initial_predictions,
            reason="project.save.init",
        )
        from xpkg.project.summary import labels_media_summary, labels_state_summary

        _touch_descriptor(
            root,
            state_summary=labels_state_summary(labels, initial_predictions),
            media_summary=labels_media_summary(labels, initial_predictions, project_root=root),
        )
        labels.path = root
        return state_path

    del journal
    state_metadata: dict[str, Any] | None = None
    predictions: dict[str, Any] | None = None
    if current_state is not None:
        state_payload, source_kind = current_state
        state_metadata, predictions = _project_state_components(
            state_payload,
            source_kind=source_kind,
        )
    elif has_state_cache:
        state_payload = read_project_state(state_path)
        state_metadata = _state_metadata_from_state_payload(state_payload)
        predictions = _predictions_payload_from_state_payload(state_payload)

    candidate_predictions = predictions_payload_from_labels(labels)
    if regenerate_predictions:
        predictions = candidate_predictions
    elif (
        _predictions_committed_length(predictions) <= 0
        and _predictions_committed_length(candidate_predictions) > 0
    ):
        predictions = candidate_predictions

    state_path = _commit_labels_to_project(
        root,
        labels=labels,
        metadata=state_metadata,
        predictions=predictions,
        reason="project.save",
    )
    from xpkg.project.summary import labels_media_summary, labels_state_summary

    _touch_descriptor(
        root,
        state_summary=labels_state_summary(labels, predictions),
        media_summary=labels_media_summary(labels, predictions, project_root=root),
    )
    labels.path = root
    return state_path
