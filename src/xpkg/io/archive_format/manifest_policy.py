"""Manifest/provenance policy helpers for `.xpkg` writer flows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from xpkg.core.json_utils import parse_json
from xpkg.core.path_registry import resolve_path
from xpkg.io.manifest import AssetType, ProjectManifest, resolve_project_path
from xpkg.io.archive_format.shared import _mapping_to_str_key_dict


def load_manifest_from_metadata(metadata_group: h5py.Group) -> ProjectManifest:
    manifest_raw = metadata_group.attrs.get("manifest_json")
    if manifest_raw:
        if isinstance(manifest_raw, bytes | bytearray | np.bytes_):
            manifest_raw = manifest_raw.decode("utf-8")
        manifest_payload = parse_json(str(manifest_raw))
        if not isinstance(manifest_payload, Mapping):
            raise TypeError("manifest_json must decode to a mapping")
        return ProjectManifest.from_dict(
            _mapping_to_str_key_dict(manifest_payload, name="manifest_json")
        )
    return ProjectManifest()


def register_videos(
    manifest: ProjectManifest,
    *,
    videos: Sequence[Any],
    project_root: Path,
) -> None:
    for idx, video in enumerate(videos):
        raw_filename = str(video.filename or "").strip()
        if raw_filename:
            _register_file_video(
                manifest,
                video=video,
                project_root=project_root,
                index=idx,
                raw_filename=raw_filename,
            )
            continue
        _register_image_sequence_video(
            manifest,
            video=video,
            project_root=project_root,
            index=idx,
        )


def _register_file_video(
    manifest: ProjectManifest,
    *,
    video: Any,
    project_root: Path,
    index: int,
    raw_filename: str,
) -> None:
    raw_path = Path(raw_filename)
    _, resolved_path = resolve_project_path(raw_path, project_root=project_root)
    if not raw_path.is_absolute() and not resolved_path.exists():
        raise FileNotFoundError(f"Relative video path not found under project root: {raw_filename}")
    metadata_entry: dict[str, Any] = {
        "index": index,
        "backend": str(video.backend or ""),
    }
    sha256_value = video.sha256
    if sha256_value:
        metadata_entry["sha256"] = str(sha256_value)
    manifest.register(
        raw_filename,
        AssetType.VIDEO,
        project_root=project_root,
        metadata=metadata_entry,
    )


def _register_image_sequence_video(
    manifest: ProjectManifest,
    *,
    video: Any,
    project_root: Path,
    index: int,
) -> None:
    image_filenames = _image_sequence_filenames(video)
    sequence_root = _image_sequence_root(image_filenames)
    manifest.register(
        sequence_root,
        AssetType.VIDEO,
        project_root=project_root,
        metadata={
            "index": index,
            "backend": "images",
            "role": "image_sequence",
            "frame_count": len(image_filenames),
        },
    )


def _image_sequence_filenames(video: Any) -> list[str]:
    filenames = [str(path).strip() for path in video.image_filenames or [] if str(path).strip()]
    if not filenames:
        raise ValueError("Image-sequence video is missing image filenames for manifest emission")
    return filenames


def _image_sequence_root(image_filenames: Sequence[str]) -> Path:
    parents = {Path(path).resolve().parent for path in image_filenames}
    if len(parents) != 1:
        raise ValueError("Image-sequence video frames must share exactly one parent directory")
    sequence_root = next(iter(parents))
    if not sequence_root.is_dir():
        raise FileNotFoundError(f"Image-sequence directory not found: {sequence_root}")
    return sequence_root


def register_archive(manifest: ProjectManifest, archive_path: Path) -> None:
    manifest.register(
        archive_path,
        AssetType.PREDICTIONS,
        project_root=archive_path.parent,
        metadata={"role": "archive"},
    )


def register_bundle(manifest: ProjectManifest, bundle_path: Path) -> None:
    """Legacy alias for `register_archive`."""
    register_archive(manifest, bundle_path)


def register_metadata_assets(
    manifest: ProjectManifest,
    *,
    bundle_path: Path,
    metadata_input: Mapping[str, Any],
) -> None:
    def _coerce_path(path_val: Any) -> Path | None:
        if isinstance(path_val, Path):
            val = str(path_val).strip()
            if not val:
                return None
            candidate = path_val
        elif isinstance(path_val, str):
            val = path_val.strip()
            if not val:
                return None
            candidate = Path(val)
        else:
            return None

        if not candidate.is_absolute():
            candidate = resolve_path(bundle_path.parent / candidate)
        return candidate

    def _register_asset(path_val: Any, asset_type: AssetType, *, role: str | None = None) -> None:
        path_obj = _coerce_path(path_val)
        if path_obj is None:
            return
        meta: dict[str, Any] = {}
        if role is not None:
            meta["role"] = role
        manifest.register(path_obj, asset_type, project_root=bundle_path.parent, metadata=meta)

    def _lookup(mapping: Mapping[str, Any], dotted_key: str) -> Any:
        if dotted_key in mapping:
            return mapping[dotted_key]
        current: Any = mapping
        for part in dotted_key.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return None
            current = current[part]
        return current

    metadata_asset_fields: tuple[tuple[str, AssetType, str | None], ...] = (
        ("pose.predictions_path", AssetType.PREDICTIONS, "inference_output"),
        ("pose.predictions_h5_path", AssetType.PREDICTIONS, "inference_output"),
        ("pose.export_path", AssetType.PREDICTIONS, "inference_output"),
    )

    runtime_asset_fields: tuple[tuple[str, AssetType, str | None], ...] = (
        ("predictions_path", AssetType.PREDICTIONS, "inference_output"),
    )

    runtime_cfg = metadata_input.get("runtime_config")

    for key, asset_type, role in metadata_asset_fields:
        _register_asset(_lookup(metadata_input, key), asset_type, role=role)

    if isinstance(runtime_cfg, Mapping):
        for key, asset_type, role in runtime_asset_fields:
            _register_asset(_lookup(runtime_cfg, key), asset_type, role=role)


__all__ = [
    "load_manifest_from_metadata",
    "register_bundle",
    "register_metadata_assets",
    "register_videos",
]
