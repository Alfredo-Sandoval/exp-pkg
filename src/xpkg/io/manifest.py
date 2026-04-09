"""
Project manifest for tracking assets with stable identifiers.

This module provides a centralized registry for project assets (videos, models,
skeletons, archives) that replaces scattered file searching with O(1) lookups.
The manifest persists in the compatibility archive payload and uses portable
path IDs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import h5py

from xpkg.core.path_registry import PathId, normalize_separators, slugify_path_component


class AssetType(StrEnum):
    """Supported asset types for the manifest."""

    VIDEO = "video"
    MODEL = "model"
    CHECKPOINT = "checkpoint"
    SKELETON = "skeleton"
    PREDICTIONS = "predictions"
    CONFIG = "config"
    OTHER = "other"


def coerce_asset_type(asset_type: AssetType | str) -> AssetType:
    """Normalize asset type inputs into the local AssetType."""
    if isinstance(asset_type, AssetType):
        return asset_type
    if isinstance(asset_type, str):
        return AssetType(asset_type)
    raise TypeError("asset_type must be an AssetType or str")


def resolve_project_path(raw_path: Path | str, *, project_root: Path | None) -> tuple[str, Path]:
    """Return `(stored_path, resolved_path)` using relative storage when possible.

    Args:
        raw_path: The path to resolve.
        project_root: The root directory of the project.

    Returns:
        tuple[str, Path]: Stored path string plus resolved filesystem path.

    Raises:
        ValueError: If the path is empty or escapes the project root.
    """
    raw_str = str(raw_path).strip()
    if not raw_str:
        raise ValueError("Asset path is empty")

    raw = Path(raw_str)
    root = Path(project_root).resolve() if project_root is not None else None

    if raw.is_absolute():
        resolved = raw.resolve()
    else:
        if root is None:
            raise ValueError(f"Project root is required for relative path: {raw_str}")
        resolved = (root / raw).resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"Relative path escapes project root: {raw_str}")

    try:
        stored_path = (
            resolved.relative_to(root).as_posix()
            if root is not None
            else normalize_separators(str(resolved))
        )
    except ValueError:
        stored_path = normalize_separators(str(resolved))
    return stored_path, resolved


def _make_manifest_path_id(
    resolved_path: Path,
    *,
    prefix: str,
    project_root: Path | None,
) -> PathId:
    root = Path(project_root).resolve() if project_root is not None else None
    try:
        identity_path = (
            resolved_path.relative_to(root).as_posix()
            if root is not None
            else normalize_separators(str(resolved_path))
        )
    except ValueError:
        identity_path = normalize_separators(str(resolved_path))
    digest = hashlib.sha1(identity_path.encode("utf-8")).hexdigest()[:8]
    return PathId(
        id=f"{prefix}_{digest}",
        label=slugify_path_component(resolved_path),
        path=identity_path,
    )


@dataclass(slots=True)
class AssetEntry:
    """
    A single asset tracked by the manifest.

    Attributes:
        id: Stable PathId hash for consistent referencing.
        label: Human-readable display name.
        path: Stored absolute path string.
        asset_type: Category of asset (video, model, etc.).
        exists: Cached existence status (updated on refresh).
        modified_at: ISO timestamp of last known modification.
        metadata: Optional key-value pairs for asset-specific data.
    """

    id: str
    label: str
    path: str
    asset_type: AssetType
    exists: bool = True
    modified_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        asset_type: AssetType | str,
        *,
        project_root: Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AssetEntry:
        """Create an AssetEntry from a filesystem path.

        Args:
            path: The filesystem path to the asset.
            asset_type: The category of the asset.
            project_root: Optional project root for relative path resolution.
            metadata: Optional metadata dictionary.

        Returns:
            AssetEntry: The initialized asset entry.
        """
        resolved_asset_type = coerce_asset_type(asset_type)
        stored_path, resolved_path = resolve_project_path(path, project_root=project_root)
        path_id = _make_manifest_path_id(
            resolved_path,
            prefix=resolved_asset_type.value,
            project_root=project_root,
        )
        exists = resolved_path.exists()
        modified_at = ""
        if exists:
            mtime = resolved_path.stat().st_mtime
            modified_at = datetime.fromtimestamp(mtime, tz=UTC).isoformat()
        return cls(
            id=path_id.id,
            label=path_id.label,
            path=stored_path,
            asset_type=resolved_asset_type,
            exists=exists,
            modified_at=modified_at,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        d = asdict(self)
        d["asset_type"] = self.asset_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssetEntry:
        """Deserialize from dictionary."""
        data = dict(data)
        data["asset_type"] = AssetType(data["asset_type"])
        return cls(**data)

    def refresh(self, *, project_root: Path | None = None) -> None:
        """Update existence and modification time from filesystem."""
        if project_root is None and not Path(self.path).is_absolute():
            p = Path(self.path)
        else:
            _, p = resolve_project_path(self.path, project_root=project_root)
        self.exists = p.exists()
        if self.exists:
            mtime = p.stat().st_mtime
            self.modified_at = datetime.fromtimestamp(mtime, tz=UTC).isoformat()
        else:
            self.modified_at = ""


class ProjectManifest:
    """
    Centralized registry for project assets.

    The manifest tracks all assets by their stable PathId, enabling fast lookups
    without recursive filesystem searches. Assets are organized by type for
    efficient filtering.

    Example:
        manifest = ProjectManifest()
        entry = manifest.register(Path("video.mp4"), AssetType.VIDEO)
        found = manifest.get(entry.id)
        videos = manifest.find_by_type(AssetType.VIDEO)
    """

    def __init__(self) -> None:
        self._entries: dict[str, AssetEntry] = {}

    def register(
        self,
        path: Path | str,
        asset_type: AssetType | str,
        *,
        project_root: Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AssetEntry:
        """
        Register an asset in the manifest.

        If an asset with the same PathId already exists, it is updated with
        the new metadata (existing metadata keys are preserved unless overwritten).

        Args:
            path: Filesystem path to the asset.
            asset_type: Category of the asset.
            metadata: Optional key-value pairs for asset-specific data.

        Returns:
            The registered or updated AssetEntry.
        """
        entry = AssetEntry.from_path(
            path,
            asset_type,
            project_root=project_root,
            metadata=metadata,
        )
        if entry.id in self._entries:
            existing = self._entries[entry.id]

            merged_meta = {**existing.metadata, **(metadata or {})}
            entry.metadata = merged_meta
        self._entries[entry.id] = entry
        return entry

    def get(self, asset_id: str) -> AssetEntry | None:
        """Get an asset by its PathId hash."""
        return self._entries.get(asset_id)

    def get_by_path(
        self,
        path: Path | str,
        asset_type: AssetType | str = AssetType.OTHER,
        *,
        project_root: Path | None = None,
    ) -> AssetEntry | None:
        """Get an asset by its filesystem path."""
        resolved_asset_type = coerce_asset_type(asset_type)
        _, resolved_path = resolve_project_path(path, project_root=project_root)
        path_id = _make_manifest_path_id(
            resolved_path,
            prefix=resolved_asset_type.value,
            project_root=project_root,
        )
        return self._entries.get(path_id.id)

    def has_entry(
        self,
        path: Path | str,
        asset_types: AssetType | str | list[AssetType | str],
        *,
        role: str | None = None,
        project_root: Path | None = None,
    ) -> bool:
        """Check if a path is registered in the manifest.

        Args:
            path: Filesystem path to check.
            asset_types: Single type or list of types to check.
            role: If provided, only match entries with this role in metadata.

        Returns:
            True if the path is registered with any of the given types (and role if specified).
        """
        if not path:
            return False
        types = asset_types if isinstance(asset_types, list) else [asset_types]
        for asset_type in types:
            resolved_type = coerce_asset_type(asset_type)
            entry = self.get_by_path(path, resolved_type, project_root=project_root)
            if entry is None:
                continue
            if role is None:
                return True
            meta = entry.metadata or {}
            if meta.get("role") == role:
                return True
        return False

    def remove(self, asset_id: str) -> bool:
        """Remove an asset from the manifest.

        Args:
            asset_id: The unique ID of the asset to remove.

        Returns:
            bool: True if the asset was removed, False if not found.
        """
        if asset_id in self._entries:
            del self._entries[asset_id]
            return True
        return False

    def find_by_type(self, asset_type: AssetType | str) -> list[AssetEntry]:
        """Find all assets of a given type.

        Args:
            asset_type: The type of assets to find.

        Returns:
            list[AssetEntry]: A list of matching asset entries.
        """
        resolved_type = coerce_asset_type(asset_type)
        return [e for e in self._entries.values() if e.asset_type == resolved_type]

    def find_by_label(self, label: str, *, exact: bool = True) -> list[AssetEntry]:
        """Find assets by label.

        Args:
            label: The label to search for.
            exact: If True, match exactly. If False, match as substring.

        Returns:
            list[AssetEntry]: A list of matching asset entries.
        """
        if exact:
            return [e for e in self._entries.values() if e.label == label]
        label_lower = label.lower()
        return [e for e in self._entries.values() if label_lower in e.label.lower()]

    def find_by_run_id(self, run_id: str) -> list[AssetEntry]:
        """Find all assets linked to a specific training run.

        Args:
            run_id: The training run ID.

        Returns:
            list[AssetEntry]: A list of matching asset entries.
        """
        return [e for e in self._entries.values() if e.metadata.get("run_id") == run_id]

    def all_entries(self) -> list[AssetEntry]:
        """Return all registered assets.

        Returns:
            list[AssetEntry]: A list of all asset entries in the manifest.
        """
        return list(self._entries.values())

    def refresh(self, *, project_root: Path | None = None) -> int:
        """Refresh existence status for all assets.

        Returns:
            int: Number of assets whose status changed.
        """
        changed = 0
        for entry in self._entries.values():
            old_exists = entry.exists
            entry.refresh(project_root=project_root)
            if entry.exists != old_exists:
                changed += 1
        return changed

    def to_dict(self) -> dict[str, Any]:
        """Serialize manifest to dictionary for JSON storage."""
        return {
            "version": 1,
            "entries": [e.to_dict() for e in self._entries.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectManifest:
        """Deserialize manifest from dictionary."""
        manifest = cls()
        version = data.get("version", 1)
        if version != 1:
            raise ValueError(f"Unsupported manifest version: {version}")
        for entry_data in data.get("entries", []):
            entry = AssetEntry.from_dict(entry_data)
            manifest._entries[entry.id] = entry
        return manifest

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, asset_id: str) -> bool:
        return asset_id in self._entries


def coerce_manifest(
    manifest: object,
) -> ProjectManifest | None:
    """
    Normalize manifest inputs into a ProjectManifest instance.

    Args:
        manifest: Either a ProjectManifest, a dict-like payload, or None.

    Returns:
        ProjectManifest or None when no manifest was provided.

    Raises:
        TypeError: If the manifest input is neither a ProjectManifest nor a Mapping.
    """
    if manifest is None:
        return None
    if isinstance(manifest, ProjectManifest):
        return manifest
    if isinstance(manifest, Mapping):
        payload: dict[str, Any] = {}
        for key, value in manifest.items():
            if not isinstance(key, str):
                raise TypeError("manifest mapping keys must be strings")
            payload[key] = value
        return ProjectManifest.from_dict(payload)
    raise TypeError("manifest must be a ProjectManifest, mapping, or None")


def resolve_asset_path(
    raw_path: Path | str,
    *,
    asset_type: AssetType | str,
    manifest: ProjectManifest | Mapping[str, Any] | None = None,
    project_root: Path | None = None,
    strict: bool = True,
) -> Path:
    """
    Resolve an asset path through the manifest, with optional permissive fallback.
    """
    resolved_asset_type = coerce_asset_type(asset_type)
    _, resolved_raw = resolve_project_path(raw_path, project_root=project_root)

    manifest_obj = coerce_manifest(manifest)
    if manifest_obj is None:
        if strict:
            raise ValueError("Manifest is required to resolve asset paths")
        return resolved_raw

    candidate_id = _make_manifest_path_id(
        resolved_raw,
        prefix=resolved_asset_type.value,
        project_root=project_root,
    )
    entry = manifest_obj.get(candidate_id.id)
    if entry is None:
        if strict:
            raise FileNotFoundError(f"Asset not found in manifest: {resolved_raw}")
        return resolved_raw

    _, resolved_entry = resolve_project_path(entry.path, project_root=project_root)
    if not resolved_entry.exists():
        if strict:
            raise FileNotFoundError(f"Manifest entry missing on disk: {resolved_entry}")
        return resolved_entry
    return resolved_entry


def persist_manifest(
    archive_path: Path | str, manifest: ProjectManifest | Mapping[str, Any]
) -> None:
    """Write the provided manifest into the project's compatibility archive."""
    from xpkg.io.archive_format.transaction import ArchiveFileLock

    archive = Path(archive_path).resolve()
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")
    if isinstance(manifest, ProjectManifest):
        manifest_obj = manifest
    elif isinstance(manifest, Mapping):
        manifest_obj = ProjectManifest.from_dict(dict(manifest))

    from xpkg.io.archive_format.shared import _serialize_json

    manifest_json = _serialize_json(manifest_obj.to_dict())

    with ArchiveFileLock(archive):
        with h5py.File(str(archive), "r+") as h5file:
            meta_group = h5file.get("project_metadata")
            if meta_group is None:
                raise ValueError("Archive is missing the project_metadata group")
            if not isinstance(meta_group, h5py.Group):
                raise TypeError("project_metadata must be an h5py Group")
            meta_group.attrs["manifest_json"] = manifest_json
            h5file.flush()
