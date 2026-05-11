"""Pose / prediction provenance helpers."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xpkg.project.layout import (
    _now_utc_iso,
)

from ..._core.hashing import sha256_file
from ..._core.path_registry import resolve_path

if TYPE_CHECKING:
    from xpkg.model import Labels


_POSE_PREDICTION_TOOLS: dict[str, tuple[str, str]] = {
    "dlc_csv_import": ("DeepLabCut", "csv"),
    "dlc_h5_import": ("DeepLabCut", "h5"),
    "dlc_project_import": ("DeepLabCut", "project"),
    "lightning_pose_csv_import": ("Lightning Pose", "csv"),
    "mediapipe_pose_landmarks_json_import": ("MediaPipe", "json"),
    "mmpose_topdown_json_import": ("MMPose", "json"),
    "sleap_h5_import": ("SLEAP", "h5"),
    "sleap_pkg_import": ("SLEAP", "pkg.slp"),
}


def _merge_metadata_dict(base: dict[str, Any], extra: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    if extra is None:
        return merged
    for key, value in extra.items():
        key_text = str(key)
        if (
            isinstance(value, Mapping)
            and isinstance(merged.get(key_text), dict)
        ):
            merged[key_text] = _merge_metadata_dict(merged[key_text], value)
            continue
        merged[key_text] = value
    return merged


def _config_snapshot_payload(path: str | Path) -> dict[str, str]:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Prediction provenance config snapshot not found: {resolved}")
    return {
        "path": resolved.as_posix(),
        "sha256": sha256_file(resolved),
    }


def _source_inputs_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in {"source", "project_name"}:
            continue
        if key.startswith("source"):
            inputs[key] = deepcopy(value)
    return inputs


def _normalized_prediction_provenance(
    metadata: Mapping[str, Any],
    extra: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source_label = str(metadata.get("source") or "unknown_pose_import")
    tool_name, source_format = _POSE_PREDICTION_TOOLS.get(
        source_label,
        ("unknown", "unknown"),
    )
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "importer": source_label,
        "tool": {"name": tool_name},
        "source_format": source_format,
        "inputs": _source_inputs_from_metadata(metadata),
    }
    if extra is None:
        return provenance

    extra_payload = dict(extra)
    tool_payload = dict(provenance["tool"])
    model_payload: dict[str, Any] = {}
    metadata_payload: dict[str, Any] = {}

    for flat_key, target_key in (
        ("tool_name", "name"),
        ("tool_version", "version"),
        ("framework_version", "framework_version"),
    ):
        value = extra_payload.pop(flat_key, None)
        if value is not None:
            tool_payload[target_key] = value
    for flat_key, target_key in (
        ("model_name", "name"),
        ("model_version", "version"),
        ("training_set", "training_set"),
        ("training_set_ref", "training_set"),
    ):
        value = extra_payload.pop(flat_key, None)
        if value is not None:
            model_payload[target_key] = value

    config_path = extra_payload.pop("config_snapshot_path", None)
    if config_path is None:
        config_path = extra_payload.pop("config_path", None)
    if config_path is not None:
        extra_payload["config_snapshot"] = _merge_metadata_dict(
            _config_snapshot_payload(config_path),
            extra_payload.get("config_snapshot")
            if isinstance(extra_payload.get("config_snapshot"), Mapping)
            else None,
        )

    nested_tool = extra_payload.pop("tool", None)
    if isinstance(nested_tool, Mapping):
        tool_payload = _merge_metadata_dict(tool_payload, nested_tool)
    nested_model = extra_payload.pop("model", None)
    if isinstance(nested_model, Mapping):
        model_payload = _merge_metadata_dict(model_payload, nested_model)
    nested_metadata = extra_payload.pop("metadata", None)
    if isinstance(nested_metadata, Mapping):
        metadata_payload = _merge_metadata_dict(metadata_payload, nested_metadata)

    provenance["tool"] = tool_payload
    if model_payload:
        provenance["model"] = model_payload
    if metadata_payload:
        provenance["metadata"] = metadata_payload
    return _merge_metadata_dict(provenance, extra_payload)


def _attach_prediction_provenance(
    labels: Labels,
    metadata: dict[str, Any],
    extra: Mapping[str, Any] | None,
) -> dict[str, Any]:
    provenance = _normalized_prediction_provenance(metadata, extra)
    labels.provenance = dict(labels.provenance)
    labels.provenance["pose_prediction"] = provenance
    metadata["prediction_provenance"] = provenance
    return provenance


def _persist_pose_provenance(
    root: Path,
    provenance: Any,
    *,
    default_tool: str,
    source_path: str | Path,
) -> None:
    if provenance is None:
        return
    from xpkg.model import PoseModelProvenance
    from xpkg.project.metadata import save_project_pose_provenance

    if isinstance(provenance, PoseModelProvenance):
        record = provenance
    elif isinstance(provenance, Mapping):
        payload = dict(provenance)
        if not str(payload.get("tool", "")).strip():
            payload["tool"] = default_tool
        record = PoseModelProvenance.from_dict(payload)
    else:
        raise TypeError(
            f"provenance must be PoseModelProvenance or mapping, got {provenance!r}."
        )

    fields: dict[str, Any] = {}
    if record.imported_from is None:
        fields["imported_from"] = resolve_path(source_path).as_posix()
    if record.imported_at is None:
        fields["imported_at"] = _now_utc_iso()
    if fields:
        record = PoseModelProvenance.from_dict({**record.to_dict(), **fields})
    save_project_pose_provenance(root, record)


