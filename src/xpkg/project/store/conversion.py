"""Conversion-result → project import orchestration."""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xpkg.project.layout import (
    resolve_project_root,
)
from xpkg.project.state_io import predictions_payload_from_labels
from xpkg.project.store._helpers import (
    _ensure_project_for_import,
    _stage_project_parent,
    _touch_descriptor,
)
from xpkg.project.store.cache import _commit_labels_to_project
from xpkg.project.store.provenance import (
    _attach_prediction_provenance,
    _persist_pose_provenance,
)

if TYPE_CHECKING:
    from xpkg.model import Labels, PoseModelProvenance


def _import_project_from_conversion(
    project: str | Path,
    *,
    force: bool,
    reason: str,
    convert: Callable[[Path], Any],
    prediction_provenance: Mapping[str, Any] | None = None,
) -> Path:
    root = _ensure_project_for_import(
        project,
        force=force,
    )
    stage_parent = _stage_project_parent(root)
    with tempfile.TemporaryDirectory(
        prefix=".project_import_",
        dir=str(stage_parent),
    ) as tmp_dir:
        result = convert(Path(tmp_dir))
        result.metadata = dict(result.metadata)
        _attach_prediction_provenance(
            result.labels,
            result.metadata,
            prediction_provenance,
        )
        state_path = _commit_labels_to_project(
            root,
            labels=result.labels,
            metadata=result.metadata,
            reason=reason,
        )
    from xpkg.project.summary import labels_media_summary, labels_state_summary

    predictions = predictions_payload_from_labels(result.labels)
    _touch_descriptor(
        root,
        state_summary=labels_state_summary(result.labels, predictions),
        media_summary=labels_media_summary(result.labels, predictions, project_root=root),
    )
    return state_path


def _unify_matching_skeletons(base_labels: Labels, new_labels: Labels) -> None:
    mapping: dict[int, Any] = {}
    for skeleton in new_labels.skeletons:
        target = next(
            (existing for existing in base_labels.skeletons if existing.matches(skeleton)),
            None,
        )
        if target is not None:
            mapping[id(skeleton)] = target

    if not mapping:
        return

    for labeled_frame in new_labels.labeled_frames:
        for instance in labeled_frame.instances:
            target = mapping.get(id(instance.skeleton))
            if target is None or instance.skeleton is target:
                continue
            instance.skeleton = target
            instance.realign_points()

    deduped_skeletons: list[Any] = []
    seen_ids: set[int] = set()
    for skeleton in new_labels.skeletons:
        target = mapping.get(id(skeleton), skeleton)
        target_id = id(target)
        if target_id in seen_ids:
            continue
        seen_ids.add(target_id)
        deduped_skeletons.append(target)
    new_labels.skeletons = deduped_skeletons


def _merge_labels_for_import(
    merged_labels: Labels | None,
    new_labels: Labels,
) -> Labels:
    if merged_labels is None:
        return new_labels

    _unify_matching_skeletons(merged_labels, new_labels)
    merged_labels.extend_from(new_labels, unify=False)
    return merged_labels


def _import_pose_project(
    project: str | Path,
    *,
    force: bool,
    reason: str,
    convert: Callable[[Path], Any],
    prediction_provenance: Mapping[str, Any] | None,
    provenance: PoseModelProvenance | Mapping[str, Any] | None,
    default_tool: str,
    source_path: str | Path,
) -> Path:
    state_path = _import_project_from_conversion(
        project,
        force=force,
        reason=reason,
        convert=convert,
        prediction_provenance=prediction_provenance,
    )
    if provenance is not None:
        root = resolve_project_root(project)
        if root is None:
            root = state_path.parents[2]
        _persist_pose_provenance(
            root,
            provenance,
            default_tool=default_tool,
            source_path=source_path,
        )
    return state_path

