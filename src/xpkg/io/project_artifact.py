"""Workspace pack/unpack and validation helpers."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from xpkg.core.path_registry import resolve_path
from xpkg.io.archive_format import read_archive
from xpkg.io.project_layout import (
    CANONICAL_ARCHIVE_SUFFIX,
    EXPKG_SUFFIX,
    PROJECT_DESCRIPTOR_FILENAME,
    ProjectDescriptor,
    _candidate_workspace_root,
    _now_utc_iso,
    default_expkg_path,
    load_project_descriptor,
    project_descriptor_path,
    resolve_workspace_root,
    workspace_media_root,
    workspace_store_root,
    write_project_descriptor,
)


def _iter_workspace_files(workspace_root: Path) -> list[Path]:
    files: list[Path] = []
    descriptor_path = project_descriptor_path(workspace_root)
    if descriptor_path.is_file():
        files.append(descriptor_path)
    for root_dir in (workspace_store_root(workspace_root), workspace_media_root(workspace_root)):
        if not root_dir.exists():
            continue
        for candidate in sorted(root_dir.rglob("*")):
            if candidate.is_file():
                files.append(candidate)
    return files


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _validate_vicon_bundle_paths(workspace_root: Path, recording: Any) -> None:
    store_root = workspace_store_root(workspace_root).resolve()
    managed_paths = {
        "recording": recording.path,
        "xcp": recording.xcp_path,
        "vsk": recording.vsk_path,
    }
    for label, raw_path in managed_paths.items():
        if raw_path is None:
            continue
        resolved = resolve_path(raw_path)
        if not resolved.is_file():
            raise FileNotFoundError(f"Workspace Vicon {label} file missing: {resolved}")
        if not _is_within(resolved, store_root):
            raise ValueError(
                "Workspace Vicon bundle files must live under the workspace store. "
                f"Found unmanaged {label} path: {resolved}"
            )


def _workspace_media_violations(workspace_root: Path) -> list[str]:
    from xpkg.io.project_workspace import current_project_state_path
    from xpkg.io.workspace_state import workspace_state_kind
    from xpkg.model import Labels

    state_path = current_project_state_path(workspace_root)
    if not state_path.exists():
        return []
    if state_path.suffix.lower() == ".json" and workspace_state_kind(state_path) == "vicon":
        return []
    labels = Labels.load_file(workspace_root.as_posix())
    media_root = workspace_media_root(workspace_root).resolve()
    violations: list[str] = []

    for video in labels.videos:
        filename = getattr(video, "filename", None)
        if filename:
            resolved = resolve_path(filename)
            if not _is_within(resolved, media_root):
                violations.append(str(filename))
        for frame_path in getattr(video, "image_filenames", []) or []:
            resolved = resolve_path(frame_path)
            if not _is_within(resolved, media_root):
                violations.append(str(frame_path))

    return list(dict.fromkeys(violations))


def pack_project(
    workspace: str | Path,
    *,
    out: str | Path | None = None,
    mode: str | None = None,
    overwrite: bool = False,
) -> Path:
    root = resolve_workspace_root(workspace)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg workspace: {workspace}")
    validate_workspace(root)

    descriptor = load_project_descriptor(root)
    selected_mode = descriptor.default_pack_mode if mode is None else mode
    if selected_mode not in {"portable", "snapshot"}:
        raise ValueError(f"Unsupported pack mode: {selected_mode!r}")
    if selected_mode == "portable":
        violations = _workspace_media_violations(root)
        if violations:
            joined = ", ".join(violations[:5])
            if len(violations) > 5:
                joined += ", ..."
            raise ValueError(
                "Portable pack requires all referenced media to live under Media/. "
                f"Found external or unmanaged references: {joined}"
            )

    out_path = resolve_path(out) if out is not None else default_expkg_path(root)
    if out_path.suffix.lower() != EXPKG_SUFFIX:
        raise ValueError(f"Packed project artifacts must use {EXPKG_SUFFIX}: {out_path}")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output artifact already exists: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    members = _iter_workspace_files(root)
    with tempfile.NamedTemporaryFile(
        prefix=f".{out_path.stem}_",
        suffix=".tmp",
        dir=str(out_path.parent),
        delete=False,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        with zipfile.ZipFile(tmp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for source_path in members:
                relative_name = source_path.relative_to(root).as_posix()
                archive.write(source_path, arcname=relative_name)
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return out_path


def _validated_zip_member(name: str) -> PurePosixPath:
    member = PurePosixPath(name)
    if member.is_absolute():
        raise ValueError(f"Packed artifact contains an absolute path: {name!r}")
    if ".." in member.parts:
        raise ValueError(f"Packed artifact contains a parent traversal path: {name!r}")
    return member


def unpack_project(
    artifact: str | Path,
    out: str | Path,
    *,
    force: bool = False,
    rename_title: str | None = None,
) -> Path:
    artifact_path = resolve_path(artifact)
    if artifact_path.suffix.lower() != EXPKG_SUFFIX:
        raise ValueError(f"Packed project artifacts must use {EXPKG_SUFFIX}: {artifact_path}")
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Packed project artifact not found: {artifact_path}")

    out_root = _candidate_workspace_root(out)
    if out_root.exists() and not out_root.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {out_root}")
    if out_root.exists():
        entries = list(out_root.iterdir())
        if entries and not force:
            raise FileExistsError(f"Output directory is not empty: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(artifact_path, mode="r") as archive:
        for info in archive.infolist():
            member = _validated_zip_member(info.filename)
            target = out_root.joinpath(*member.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, mode="r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)

    descriptor = load_project_descriptor(out_root)
    (out_root / descriptor.store_path).mkdir(parents=True, exist_ok=True)
    (out_root / descriptor.media_root).mkdir(parents=True, exist_ok=True)
    (out_root / descriptor.exports_root).mkdir(parents=True, exist_ok=True)
    if rename_title is not None and rename_title.strip():
        descriptor.title = rename_title.strip()
        descriptor.updated_at = _now_utc_iso()
        write_project_descriptor(out_root, descriptor)
    validate_workspace(out_root)
    return out_root


def validate_workspace(workspace: str | Path) -> None:
    root = resolve_workspace_root(workspace)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg workspace: {workspace}")
    descriptor = load_project_descriptor(root)
    descriptor.validate()

    store_root = root / descriptor.store_path
    if not store_root.is_dir():
        raise FileNotFoundError(f"Workspace store directory missing: {store_root}")

    media_root = root / descriptor.media_root
    exports_root = root / descriptor.exports_root
    if media_root.exists() and not media_root.is_dir():
        raise ValueError(f"Workspace media root is not a directory: {media_root}")
    if exports_root.exists() and not exports_root.is_dir():
        raise ValueError(f"Workspace exports root is not a directory: {exports_root}")

    from xpkg.io.project_workspace import (
        current_project_state_path,
        ensure_current_workspace_snapshot_cache,
        load_workspace_vicon_recording,
    )
    from xpkg.io.workspace_state import workspace_state_kind
    from xpkg.model import Labels

    snapshot_path = ensure_current_workspace_snapshot_cache(root)
    state_path = snapshot_path if snapshot_path is not None else current_project_state_path(root)
    if not state_path.exists():
        return
    if state_path.suffix.lower() == ".json":
        if workspace_state_kind(state_path) == "vicon":
            recording = load_workspace_vicon_recording(root)
            _validate_vicon_bundle_paths(root, recording)
            return
    labels = Labels.load_file(root.as_posix())
    labels.validate()


def validate_expkg(artifact: str | Path) -> None:
    artifact_path = resolve_path(artifact)
    if artifact_path.suffix.lower() != EXPKG_SUFFIX:
        raise ValueError(f"Packed project artifacts must use {EXPKG_SUFFIX}: {artifact_path}")
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Packed project artifact not found: {artifact_path}")

    with zipfile.ZipFile(artifact_path, mode="r") as archive:
        descriptor_data: dict[str, Any] | None = None
        for info in archive.infolist():
            member = _validated_zip_member(info.filename)
            if member.as_posix() != PROJECT_DESCRIPTOR_FILENAME:
                continue
            with archive.open(info, mode="r") as handle:
                raw = handle.read().decode("utf-8")
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise TypeError("Packed PROJECT.json must contain a JSON object")
            descriptor_data = parsed
            break
    if descriptor_data is None:
        raise FileNotFoundError("Packed project artifact is missing PROJECT.json")
    ProjectDescriptor.from_dict(descriptor_data)


def validate_artifact(path: str | Path) -> None:
    resolved = resolve_path(path)
    if resolved.is_dir() or resolved.name == PROJECT_DESCRIPTOR_FILENAME:
        validate_workspace(resolved)
        return
    if resolved.suffix.lower() == EXPKG_SUFFIX:
        validate_expkg(resolved)
        return
    if resolved.suffix.lower() == CANONICAL_ARCHIVE_SUFFIX:
        read_archive(resolved, lazy=False)
        return
    raise ValueError(f"Unsupported artifact path: {resolved}")


__all__ = [
    "pack_project",
    "unpack_project",
    "validate_artifact",
    "validate_expkg",
    "validate_workspace",
]
