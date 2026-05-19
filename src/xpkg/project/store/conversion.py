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
from xpkg.project.store.cache import (
    _commit_labels_to_project,
    _commit_vicon_to_project,
)
from xpkg.project.store.media import (
    _copy_vicon_import_bundle,
)
from xpkg.project.store.provenance import (
    _attach_prediction_provenance,
    _persist_pose_provenance,
)

from ..._core.path_registry import resolve_path

if TYPE_CHECKING:
    from xpkg.model import Labels, ViconRecording


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
    from xpkg.project.summary import labels_state_summary

    _touch_descriptor(
        root,
        state_summary=labels_state_summary(
            result.labels,
            predictions_payload_from_labels(result.labels),
        ),
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
    provenance: Any,
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


def _import_vicon_project_recording(
    recording_path: str | Path,
    project: str | Path,
    *,
    force: bool,
    reason: str,
    progress_callback: Any | None,
    reader: Callable[[str | Path], ViconRecording],
    source_name: str,
) -> Path:
    root = _ensure_project_for_import(
        project,
        force=force,
    )
    if progress_callback is not None:
        progress_callback(f"Reading {source_name} recording")
    recording = reader(recording_path)
    if progress_callback is not None:
        progress_callback("Copying Vicon recording bundle into project store")
    managed_recording = _copy_vicon_import_bundle(recording, root)
    metadata = {
        "source": source_name,
        "source_recording": resolve_path(recording_path).as_posix(),
    }
    if recording.xcp_path is not None:
        metadata["source_xcp"] = resolve_path(recording.xcp_path).as_posix()
    if recording.vsk_path is not None:
        metadata["source_vsk"] = resolve_path(recording.vsk_path).as_posix()
    state_path = _commit_vicon_to_project(
        root,
        recording=managed_recording,
        metadata=metadata,
        reason=reason,
    )
    from xpkg.project.summary import vicon_state_summary

    _touch_descriptor(root, state_summary=vicon_state_summary(managed_recording))
    return state_path
