"""Conversion-result → project import orchestration."""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xpkg.project.recording import save_project_labels
from xpkg.project.store._helpers import (
    _ensure_project_for_import,
    _stage_project_parent,
)
from xpkg.project.store.provenance import (
    _attach_prediction_provenance,
    _pose_provenance_record,
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
    provenance: PoseModelProvenance | None = None,
    session_id: str | None = None,
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
        state_path = save_project_labels(
            root,
            labels=result.labels,
            pose_metadata=result.metadata,
            provenance=provenance,
            session_id=session_id,
            reason=reason,
            replace_existing=force,
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
    session_id: str | None = None,
) -> Path:
    record = _pose_provenance_record(
        provenance,
        default_tool=default_tool,
        source_path=source_path,
    )
    return _import_project_from_conversion(
        project,
        force=force,
        reason=reason,
        convert=convert,
        prediction_provenance=prediction_provenance,
        provenance=record,
        session_id=session_id,
    )
