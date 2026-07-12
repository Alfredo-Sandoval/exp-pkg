"""Generic project artifact manifests, indexes, validation, and figure records."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from xpkg._core.json_utils import load_json_dict, write_json
from xpkg._core.path_registry import ensure_dir, resolve_path, slugify_path_component
from xpkg._core.time import now_utc_iso
from xpkg.project.layout import (
    project_artifacts_root,
    project_store_root,
)
from xpkg.project.layout import (
    require_project_root as _project_root,
)

ARTIFACT_MANIFEST_FILENAME = "manifest.json"
ARTIFACT_INDEX_FILENAME = "index.json"
ARTIFACT_SCHEMA_VERSION = "1.0.0"
ArtifactOutputSpec = Sequence[str | Path] | Mapping[str, str | Path]
FIGURES_DIRNAME = "figures"
FIGURE_MANIFEST_FILENAME = ARTIFACT_MANIFEST_FILENAME
FIGURE_ARTIFACT_SCHEMA_VERSION = ARTIFACT_SCHEMA_VERSION
FIGURE_ARTIFACT_TYPE = "figure"
FigureOutputSpec = ArtifactOutputSpec

_KNOWN_ARTIFACT_DIRS = {
    "analysis": "analyses",
    "figure": "figures",
    "model": "models",
    "report": "reports",
    "stats-report": "stats-reports",
    "table": "tables",
}
_KNOWN_DIR_ARTIFACT_TYPES = {value: key for key, value in _KNOWN_ARTIFACT_DIRS.items()}


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    """Checksum-bearing record for one file referenced by an artifact manifest."""

    role: str
    path: str
    sha256: str | None = None
    size_bytes: int | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ArtifactFile:
        """Create an artifact file record from a manifest payload."""
        role = str(data.get("role", "")).strip()
        path = str(data.get("path", "")).strip()
        if not role:
            raise ValueError("Artifact file record is missing role")
        if not path:
            raise ValueError("Artifact file record is missing path")
        size_value = data.get("size_bytes")
        return cls(
            role=role,
            path=path,
            sha256=str(data["sha256"]) if data.get("sha256") is not None else None,
            size_bytes=int(size_value) if size_value is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the portable JSON payload for this file record."""
        return {
            "role": self.role,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """Portable manifest for one saved project artifact."""

    artifact_type: str
    artifact_id: str
    title: str
    outputs: tuple[str, ...]
    inputs: tuple[str, ...]
    producer: dict[str, Any]
    stats: tuple[str, ...]
    metadata: dict[str, Any]
    namespace: str
    files: tuple[ArtifactFile, ...]
    manifest_path: Path
    artifact_root: Path
    created_at: str
    updated_at: str
    schema_version: str = ARTIFACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the portable JSON manifest payload."""
        return {
            "schema_version": self.schema_version,
            "artifact_type": self.artifact_type,
            "artifact_id": self.artifact_id,
            "namespace": self.namespace,
            "title": self.title,
            "inputs": list(self.inputs),
            "producer": dict(self.producer),
            "outputs": list(self.outputs),
            "stats": list(self.stats),
            "metadata": dict(self.metadata),
            "files": [file_record.to_dict() for file_record in self.files],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class ArtifactIndexEntry:
    """Compact artifact index entry used for project-wide discovery."""

    artifact_type: str
    artifact_id: str
    namespace: str
    title: str
    manifest_path: str
    outputs: tuple[str, ...]
    updated_at: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ArtifactIndexEntry:
        """Create an index entry from the project artifact index payload."""
        return cls(
            artifact_type=_artifact_type(str(data.get("artifact_type", ""))),
            artifact_id=str(data.get("artifact_id", "")).strip(),
            namespace=str(data.get("namespace", "") or ""),
            title=str(data.get("title", "")),
            manifest_path=str(data.get("manifest_path", "")).strip(),
            outputs=tuple(str(item) for item in data.get("outputs", []) or []),
            updated_at=str(data.get("updated_at", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the portable JSON payload for this index entry."""
        return {
            "artifact_type": self.artifact_type,
            "artifact_id": self.artifact_id,
            "namespace": self.namespace,
            "title": self.title,
            "manifest_path": self.manifest_path,
            "outputs": list(self.outputs),
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class FigureArtifact:
    """Portable manifest for one saved figure artifact."""

    artifact_id: str
    title: str
    outputs: tuple[str, ...]
    inputs: tuple[str, ...]
    producer: dict[str, Any]
    stats: tuple[str, ...]
    metadata: dict[str, Any]
    namespace: str
    manifest_path: Path
    artifact_root: Path
    created_at: str
    updated_at: str
    files: tuple[ArtifactFile, ...] = ()
    artifact_type: str = FIGURE_ARTIFACT_TYPE
    schema_version: str = FIGURE_ARTIFACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the portable JSON manifest payload."""
        return {
            "schema_version": self.schema_version,
            "artifact_type": self.artifact_type,
            "artifact_id": self.artifact_id,
            "namespace": self.namespace,
            "title": self.title,
            "inputs": list(self.inputs),
            "producer": dict(self.producer),
            "outputs": list(self.outputs),
            "stats": list(self.stats),
            "metadata": dict(self.metadata),
            "files": [file_record.to_dict() for file_record in self.files],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def _artifact_id(value: str) -> str:
    if not str(value).strip():
        raise ValueError("artifact_id cannot be empty")
    artifact_id = slugify_path_component(value)
    if not artifact_id:
        raise ValueError("artifact_id cannot be empty")
    return artifact_id


def _artifact_type(value: str) -> str:
    if not str(value).strip():
        raise ValueError("artifact_type cannot be empty")
    artifact_type = slugify_path_component(value)
    if not artifact_type:
        raise ValueError("artifact_type cannot be empty")
    if artifact_type == "stats-report":
        return artifact_type
    return artifact_type


def _namespace(value: str | None) -> str:
    if value is None or not value.strip():
        return ""
    return slugify_path_component(value)


def artifact_kind_dir(artifact_type: str) -> str:
    """Return the storage directory name for an artifact type."""
    normalized_type = _artifact_type(artifact_type)
    return _KNOWN_ARTIFACT_DIRS.get(normalized_type, normalized_type)


def _artifact_type_from_kind_dir(kind_dir: str) -> str:
    return _KNOWN_DIR_ARTIFACT_TYPES.get(kind_dir, kind_dir)


def _infer_namespace(artifact_root: Path) -> str:
    type_root = artifact_root.parent
    container = type_root.parent
    if container.name == "artifacts":
        return ""
    return container.name


def _relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Artifact path must live inside the project: {path}") from exc


def _portable_path(value: str | Path, project_root: Path) -> str:
    raw = Path(value)
    if raw.is_absolute():
        return _relative_to_project(raw, project_root)
    portable = PurePosixPath(raw.as_posix())
    if portable.is_absolute() or ".." in portable.parts:
        raise ValueError(f"Artifact paths must be project-relative: {value}")
    return portable.as_posix()


def _portable_paths(values: Sequence[str | Path], project_root: Path) -> tuple[str, ...]:
    if isinstance(values, str | bytes | bytearray):
        raise TypeError("paths must be a sequence, not a string")
    return tuple(_portable_path(item, project_root) for item in values)


def _project_file_path(project_root: Path, portable_path: str) -> Path:
    portable = PurePosixPath(portable_path)
    if portable.is_absolute() or ".." in portable.parts:
        raise ValueError(f"Artifact paths must be project-relative: {portable_path}")
    return project_root.joinpath(*portable.parts)


def _validate_target_name(name: str) -> PurePosixPath:
    target = PurePosixPath(str(name))
    if target.is_absolute() or ".." in target.parts:
        raise ValueError(f"Artifact output target must be relative: {name!r}")
    if not target.name:
        raise ValueError(f"Artifact output target must name a file: {name!r}")
    return target


def _iter_output_specs(outputs: ArtifactOutputSpec) -> list[tuple[PurePosixPath, Path]]:
    if isinstance(outputs, Mapping):
        items = [
            (
                _validate_target_name(str(target_name)),
                resolve_path(cast(str | Path, source)),
            )
            for target_name, source in outputs.items()
        ]
    elif isinstance(outputs, Sequence) and not isinstance(outputs, str | bytes | bytearray):
        items = [
            (_validate_target_name(Path(source).name), resolve_path(source))
            for source in outputs
        ]
    else:
        raise TypeError("outputs must be a sequence of paths or a mapping of target names")

    if not items:
        raise ValueError("At least one artifact output is required")
    return items


def _copy_outputs(
    outputs: ArtifactOutputSpec,
    *,
    artifact_root: Path,
    project_root: Path,
    overwrite: bool,
) -> tuple[str, ...]:
    copied_paths: list[str] = []
    for target_name, source_path in _iter_output_specs(outputs):
        if not source_path.is_file():
            raise FileNotFoundError(f"Artifact output file not found: {source_path}")
        target_path = artifact_root.joinpath(*target_name.parts)
        if target_path.exists() and not overwrite:
            raise FileExistsError(f"Artifact output already exists: {target_path}")
        ensure_dir(target_path.parent)
        if source_path.resolve() != target_path.resolve():
            shutil.copy2(source_path, target_path)
        copied_paths.append(_relative_to_project(target_path, project_root))
    return tuple(copied_paths)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(role: str, portable_path: str, project_root: Path) -> ArtifactFile:
    path = _project_file_path(project_root, portable_path)
    if not path.is_file():
        return ArtifactFile(role=role, path=portable_path)
    stat = path.stat()
    return ArtifactFile(
        role=role,
        path=portable_path,
        sha256=_sha256_file(path),
        size_bytes=int(stat.st_size),
    )


def _file_records(
    *,
    outputs: Sequence[str],
    inputs: Sequence[str],
    stats: Sequence[str],
    project_root: Path,
) -> tuple[ArtifactFile, ...]:
    files: list[ArtifactFile] = []
    files.extend(_file_record("output", path, project_root) for path in outputs)
    files.extend(_file_record("input", path, project_root) for path in inputs)
    files.extend(_file_record("stat", path, project_root) for path in stats)
    return tuple(files)


def _load_existing_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return load_json_dict(path)


def _artifact_files_from_payload(payload: Mapping[str, Any]) -> tuple[ArtifactFile, ...]:
    raw_files = payload.get("files", []) or []
    if not isinstance(raw_files, Sequence) or isinstance(raw_files, str | bytes | bytearray):
        raise TypeError("Artifact manifest files must be a list")
    files: list[ArtifactFile] = []
    for raw_file in raw_files:
        if not isinstance(raw_file, Mapping):
            raise TypeError("Artifact manifest file records must be objects")
        files.append(ArtifactFile.from_dict(raw_file))
    return tuple(files)


def _artifact_from_manifest(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
    artifact_root: Path,
) -> ArtifactManifest:
    schema_version = str(payload.get("schema_version", ""))
    if schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported artifact schema_version: {schema_version!r}")
    artifact_type = _artifact_type(str(payload.get("artifact_type", "")))
    artifact_id = str(payload.get("artifact_id", "")).strip()
    if not artifact_id:
        raise ValueError("Artifact manifest is missing artifact_id")

    producer = payload.get("producer")
    metadata = payload.get("metadata")
    return ArtifactManifest(
        schema_version=schema_version,
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        namespace=str(payload.get("namespace", _infer_namespace(artifact_root)) or ""),
        title=str(payload.get("title", "")),
        inputs=tuple(str(item) for item in payload.get("inputs", []) or []),
        producer=dict(producer) if isinstance(producer, Mapping) else {},
        outputs=tuple(str(item) for item in payload.get("outputs", []) or []),
        stats=tuple(str(item) for item in payload.get("stats", []) or []),
        metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        files=_artifact_files_from_payload(payload),
        created_at=str(payload.get("created_at", "")),
        updated_at=str(payload.get("updated_at", "")),
        manifest_path=manifest_path,
        artifact_root=artifact_root,
    )


def _figure_from_artifact(artifact: ArtifactManifest) -> FigureArtifact:
    if artifact.artifact_type != FIGURE_ARTIFACT_TYPE:
        raise ValueError(f"Expected figure artifact manifest, got {artifact.artifact_type!r}")
    return FigureArtifact(
        artifact_id=artifact.artifact_id,
        title=artifact.title,
        inputs=artifact.inputs,
        producer=dict(artifact.producer),
        outputs=artifact.outputs,
        stats=artifact.stats,
        metadata=dict(artifact.metadata),
        namespace=artifact.namespace,
        files=artifact.files,
        manifest_path=artifact.manifest_path,
        artifact_root=artifact.artifact_root,
        created_at=artifact.created_at,
        updated_at=artifact.updated_at,
        artifact_type=artifact.artifact_type,
        schema_version=artifact.schema_version,
    )


def project_artifact_type_root(
    project: str | Path,
    artifact_type: str,
    *,
    namespace: str | None = None,
) -> Path:
    """Return the private project root for one artifact type."""
    normalized_namespace = _namespace(namespace)
    kind_dir = artifact_kind_dir(artifact_type)
    if normalized_namespace:
        return project_store_root(project) / normalized_namespace / kind_dir
    return project_artifacts_root(project) / kind_dir


def project_figures_root(project: str | Path, *, namespace: str | None = None) -> Path:
    """Return the private project figure artifact root."""
    return project_artifact_type_root(
        project,
        FIGURE_ARTIFACT_TYPE,
        namespace=namespace,
    )


def project_artifact_root(
    project: str | Path,
    artifact_id: str,
    artifact_type: str,
    *,
    namespace: str | None = None,
) -> Path:
    """Return the private project root for one artifact."""
    return project_artifact_type_root(
        project,
        artifact_type,
        namespace=namespace,
    ) / _artifact_id(artifact_id)


def project_figure_root(
    project: str | Path,
    figure_id: str,
    *,
    namespace: str | None = None,
) -> Path:
    """Return the private root for one figure artifact."""
    return project_artifact_root(
        project,
        figure_id,
        FIGURE_ARTIFACT_TYPE,
        namespace=namespace,
    )


def project_artifact_index_path(project: str | Path) -> Path:
    """Return the project-wide artifact index path."""
    return project_artifacts_root(project) / ARTIFACT_INDEX_FILENAME


def _manifest_path(artifact_root: Path) -> Path:
    return artifact_root / ARTIFACT_MANIFEST_FILENAME


def _generic_manifest_paths(
    project_root: Path,
    *,
    artifact_type: str | None,
    artifact_id: str | None,
) -> list[Path]:
    if artifact_type is None:
        if artifact_id is None:
            return sorted(
                project_artifacts_root(project_root).glob(
                    f"*/*/{ARTIFACT_MANIFEST_FILENAME}"
                )
            )
        return sorted(
            project_artifacts_root(project_root).glob(
                f"*/{_artifact_id(artifact_id)}/{ARTIFACT_MANIFEST_FILENAME}"
            )
        )
    type_root = project_artifact_type_root(project_root, artifact_type)
    if artifact_id is None:
        return sorted(type_root.glob(f"*/{ARTIFACT_MANIFEST_FILENAME}"))
    return [_manifest_path(type_root / _artifact_id(artifact_id))]


def _namespaced_manifest_paths(
    project_root: Path,
    *,
    artifact_type: str | None,
    artifact_id: str | None,
    namespace: str | None,
) -> list[Path]:
    store_root = project_store_root(project_root)
    normalized_namespace = _namespace(namespace)
    if normalized_namespace:
        namespace_root = store_root / normalized_namespace
        if artifact_type is None:
            if artifact_id is None:
                return sorted(namespace_root.glob(f"*/*/{ARTIFACT_MANIFEST_FILENAME}"))
            return sorted(
                namespace_root.glob(
                    f"*/{_artifact_id(artifact_id)}/{ARTIFACT_MANIFEST_FILENAME}"
                )
            )
        type_root = project_artifact_type_root(
            project_root,
            artifact_type,
            namespace=normalized_namespace,
        )
        if artifact_id is None:
            return sorted(type_root.glob(f"*/{ARTIFACT_MANIFEST_FILENAME}"))
        return [_manifest_path(type_root / _artifact_id(artifact_id))]

    if artifact_type is None:
        pattern = f"*/*/*/{ARTIFACT_MANIFEST_FILENAME}"
    else:
        kind_dir = artifact_kind_dir(artifact_type)
        pattern = (
            f"*/{kind_dir}/*/{ARTIFACT_MANIFEST_FILENAME}"
            if artifact_id is None
            else f"*/{kind_dir}/{_artifact_id(artifact_id)}/{ARTIFACT_MANIFEST_FILENAME}"
        )
    paths = []
    for path in sorted(store_root.glob(pattern)):
        try:
            relative_parts = path.relative_to(store_root).parts
        except ValueError:
            continue
        if relative_parts and relative_parts[0] == "artifacts":
            continue
        paths.append(path)
    return paths


def _candidate_manifest_paths(
    project: str | Path,
    artifact_id: str | None = None,
    *,
    artifact_type: str | None,
    namespace: str | None,
) -> list[Path]:
    root = _project_root(project)
    if namespace is not None and not _namespace(namespace):
        paths = _generic_manifest_paths(
            root,
            artifact_type=artifact_type,
            artifact_id=artifact_id,
        )
    elif namespace is not None:
        paths = _namespaced_manifest_paths(
            root,
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            namespace=namespace,
        )
    else:
        paths = [
            *_generic_manifest_paths(
                root,
                artifact_type=artifact_type,
                artifact_id=artifact_id,
            ),
            *_namespaced_manifest_paths(
                root,
                artifact_type=artifact_type,
                artifact_id=artifact_id,
                namespace=None,
            ),
        ]
    return list(dict.fromkeys(path for path in paths if path.is_file()))


def _manifest_matches_type(path: Path, artifact_type: str | None) -> bool:
    if artifact_type is None:
        return True
    kind_dir = path.parent.parent.name
    return _artifact_type_from_kind_dir(kind_dir) == _artifact_type(artifact_type)


def _entry_from_artifact(
    artifact: ArtifactManifest,
    *,
    project_root: Path,
) -> ArtifactIndexEntry:
    return ArtifactIndexEntry(
        artifact_type=artifact.artifact_type,
        artifact_id=artifact.artifact_id,
        namespace=artifact.namespace,
        title=artifact.title,
        manifest_path=_relative_to_project(artifact.manifest_path, project_root),
        outputs=artifact.outputs,
        updated_at=artifact.updated_at,
    )


def _load_index_entries(path: Path) -> list[ArtifactIndexEntry]:
    if not path.is_file():
        return []
    payload = load_json_dict(path)
    raw_entries = payload.get("artifacts", []) or []
    if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, str | bytes | bytearray):
        raise TypeError("Artifact index artifacts must be a list")
    entries: list[ArtifactIndexEntry] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            raise TypeError("Artifact index entries must be objects")
        entry = ArtifactIndexEntry.from_dict(raw_entry)
        if entry.artifact_id and entry.manifest_path:
            entries.append(entry)
    return entries


def _write_index_entries(
    project_root: Path,
    entries: Sequence[ArtifactIndexEntry],
) -> None:
    index_path = project_artifact_index_path(project_root)
    ensure_dir(index_path.parent)
    sorted_entries = sorted(
        entries,
        key=lambda entry: (
            entry.namespace,
            entry.artifact_type,
            entry.artifact_id,
            entry.manifest_path,
        ),
    )
    write_json(
        index_path,
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "updated_at": now_utc_iso(drop_microseconds=True),
            "artifacts": [entry.to_dict() for entry in sorted_entries],
        },
        indent=2,
        sort_keys=False,
        ensure_ascii=True,
    )
    from xpkg.project.summary import refresh_project_summary

    refresh_project_summary(project_root)


def _upsert_artifact_index(project_root: Path, artifact: ArtifactManifest) -> None:
    index_path = project_artifact_index_path(project_root)
    entries = _load_index_entries(index_path)
    new_entry = _entry_from_artifact(artifact, project_root=project_root)
    entries = [
        entry
        for entry in entries
        if not (
            entry.artifact_type == new_entry.artifact_type
            and entry.artifact_id == new_entry.artifact_id
            and entry.namespace == new_entry.namespace
        )
    ]
    entries.append(new_entry)
    _write_index_entries(project_root, entries)


def save_project_artifact(
    project: str | Path,
    *,
    artifact_id: str,
    artifact_type: str,
    outputs: ArtifactOutputSpec,
    title: str = "",
    inputs: Sequence[str | Path] = (),
    producer: Mapping[str, Any] | None = None,
    stats: Sequence[str | Path] = (),
    metadata: Mapping[str, Any] | None = None,
    namespace: str | None = None,
    overwrite: bool = True,
) -> ArtifactManifest:
    """Copy artifact outputs into the project and write a portable manifest."""
    root = _project_root(project)
    normalized_type = _artifact_type(artifact_type)
    normalized_id = _artifact_id(artifact_id)
    normalized_namespace = _namespace(namespace)
    artifact_root = project_artifact_root(
        root,
        normalized_id,
        normalized_type,
        namespace=normalized_namespace,
    )
    ensure_dir(artifact_root)
    manifest_path = _manifest_path(artifact_root)
    existing_manifest = _load_existing_manifest(manifest_path)
    now = now_utc_iso(drop_microseconds=True)
    created_at = (
        str(existing_manifest.get("created_at", ""))
        if existing_manifest is not None
        else now
    )

    copied_outputs = _copy_outputs(
        outputs,
        artifact_root=artifact_root,
        project_root=root,
        overwrite=overwrite,
    )
    portable_inputs = _portable_paths(inputs, root)
    portable_stats = _portable_paths(stats, root)
    artifact = ArtifactManifest(
        artifact_type=normalized_type,
        artifact_id=normalized_id,
        title=str(title or normalized_id),
        inputs=portable_inputs,
        producer=dict(producer or {}),
        outputs=copied_outputs,
        stats=portable_stats,
        metadata=dict(metadata or {}),
        namespace=normalized_namespace,
        files=_file_records(
            outputs=copied_outputs,
            inputs=portable_inputs,
            stats=portable_stats,
            project_root=root,
        ),
        manifest_path=manifest_path,
        artifact_root=artifact_root,
        created_at=created_at or now,
        updated_at=now,
    )
    write_json(
        manifest_path,
        artifact.to_dict(),
        indent=2,
        sort_keys=False,
        ensure_ascii=True,
    )
    validate_project_artifact(
        root,
        normalized_id,
        artifact_type=normalized_type,
        namespace=normalized_namespace,
    )
    _upsert_artifact_index(root, artifact)
    return artifact


def delete_project_artifact(
    project: str | Path,
    artifact_id: str,
    *,
    artifact_type: str,
    namespace: str | None = None,
    missing_ok: bool = False,
) -> bool:
    """Delete one saved artifact and remove it from the compact index."""

    root = _project_root(project)
    normalized_id = _artifact_id(artifact_id)
    normalized_type = _artifact_type(artifact_type)
    normalized_namespace = _namespace(namespace)
    artifact_root = project_artifact_root(
        root,
        normalized_id,
        normalized_type,
        namespace=normalized_namespace,
    )
    if not artifact_root.exists():
        if missing_ok:
            return False
        raise FileNotFoundError(f"Project artifact does not exist: {artifact_root}")
    if not artifact_root.is_dir():
        raise ValueError(f"Project artifact root is not a directory: {artifact_root}")

    shutil.rmtree(artifact_root)
    entries = _load_index_entries(project_artifact_index_path(root))
    entries = [
        entry
        for entry in entries
        if not (
            entry.artifact_id == normalized_id
            and entry.artifact_type == normalized_type
            and entry.namespace == normalized_namespace
        )
    ]
    _write_index_entries(root, entries)
    return True


def save_project_figure(
    project: str | Path,
    *,
    figure_id: str,
    outputs: FigureOutputSpec,
    title: str = "",
    inputs: Sequence[str | Path] = (),
    producer: Mapping[str, Any] | None = None,
    stats: Sequence[str | Path] = (),
    metadata: Mapping[str, Any] | None = None,
    namespace: str | None = None,
    overwrite: bool = True,
) -> FigureArtifact:
    """Copy figure outputs into the project and write a portable manifest."""
    return _figure_from_artifact(
        save_project_artifact(
            project,
            artifact_id=figure_id,
            artifact_type=FIGURE_ARTIFACT_TYPE,
            outputs=outputs,
            title=title,
            inputs=inputs,
            producer=producer,
            stats=stats,
            metadata=metadata,
            namespace=namespace,
            overwrite=overwrite,
        )
    )


def load_project_artifact(
    project: str | Path,
    artifact_id: str,
    *,
    artifact_type: str | None = None,
    namespace: str | None = None,
) -> ArtifactManifest:
    """Load one project artifact manifest by id."""
    root = _project_root(project)
    manifests = [
        path
        for path in _candidate_manifest_paths(
            root,
            artifact_id,
            artifact_type=artifact_type,
            namespace=namespace,
        )
        if _manifest_matches_type(path, artifact_type)
    ]
    if not manifests:
        raise FileNotFoundError(f"Artifact not found: {artifact_id}")
    if len(manifests) > 1:
        raise ValueError(
            f"Artifact {artifact_id!r} exists in multiple locations or multiple namespaces; "
            "pass artifact_type=... and namespace=... to choose one"
        )
    manifest_path = manifests[0]
    return _artifact_from_manifest(
        load_json_dict(manifest_path),
        manifest_path=manifest_path,
        artifact_root=manifest_path.parent,
    )


def load_project_figure(
    project: str | Path,
    figure_id: str,
    *,
    namespace: str | None = None,
) -> FigureArtifact:
    """Load one project figure artifact manifest."""
    return _figure_from_artifact(
        load_project_artifact(
            project,
            figure_id,
            artifact_type=FIGURE_ARTIFACT_TYPE,
            namespace=namespace,
        )
    )


def list_project_artifacts(
    project: str | Path,
    *,
    artifact_type: str | None = None,
    namespace: str | None = None,
) -> list[ArtifactManifest]:
    """List saved project artifact manifests."""
    root = _project_root(project)
    manifest_paths = [
        path
        for path in _candidate_manifest_paths(
            root,
            artifact_type=artifact_type,
            namespace=namespace,
        )
        if _manifest_matches_type(path, artifact_type)
    ]
    if not manifest_paths:
        return []
    artifacts: list[ArtifactManifest] = []
    for manifest_path in sorted(manifest_paths):
        artifacts.append(
            _artifact_from_manifest(
                load_json_dict(manifest_path),
                manifest_path=manifest_path,
                artifact_root=manifest_path.parent,
            )
        )
    return artifacts


def list_project_figures(
    project: str | Path,
    *,
    namespace: str | None = None,
) -> list[FigureArtifact]:
    """List saved project figure artifacts."""
    return [
        _figure_from_artifact(artifact)
        for artifact in list_project_artifacts(
            project,
            artifact_type=FIGURE_ARTIFACT_TYPE,
            namespace=namespace,
        )
    ]


def _filter_index_entries(
    entries: Sequence[ArtifactIndexEntry],
    *,
    artifact_type: str | None,
    namespace: str | None,
) -> list[ArtifactIndexEntry]:
    normalized_type = _artifact_type(artifact_type) if artifact_type is not None else None
    normalized_namespace = _namespace(namespace) if namespace is not None else None
    return [
        entry
        for entry in entries
        if (normalized_type is None or entry.artifact_type == normalized_type)
        and (normalized_namespace is None or entry.namespace == normalized_namespace)
    ]


def list_project_artifact_index(
    project: str | Path,
    *,
    artifact_type: str | None = None,
    namespace: str | None = None,
) -> list[ArtifactIndexEntry]:
    """List the compact project artifact index, rebuilding it if missing."""
    root = _project_root(project)
    index_path = project_artifact_index_path(root)
    entries = _load_index_entries(index_path)
    if not index_path.is_file():
        entries = rebuild_project_artifact_index(root)
    return _filter_index_entries(entries, artifact_type=artifact_type, namespace=namespace)


def validate_project_artifact(
    project: str | Path,
    artifact_id: str,
    *,
    artifact_type: str | None = None,
    namespace: str | None = None,
) -> ArtifactManifest:
    """Validate one artifact manifest and referenced project-local files."""
    root = _project_root(project)
    artifact = load_project_artifact(
        root,
        artifact_id,
        artifact_type=artifact_type,
        namespace=namespace,
    )
    for path_value in (*artifact.outputs, *artifact.inputs, *artifact.stats):
        path = _project_file_path(root, path_value)
        if not path.is_file():
            raise FileNotFoundError(
                f"Artifact {artifact.artifact_id!r} references missing file: {path_value}"
            )

    for file_record in artifact.files:
        if file_record.sha256 is None:
            continue
        path = _project_file_path(root, file_record.path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Artifact {artifact.artifact_id!r} references missing file: "
                f"{file_record.path}"
            )
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != file_record.sha256:
            raise ValueError(
                f"Artifact {artifact.artifact_id!r} checksum mismatch for "
                f"{file_record.path}"
            )
        if file_record.size_bytes is not None and path.stat().st_size != file_record.size_bytes:
            raise ValueError(
                f"Artifact {artifact.artifact_id!r} size mismatch for {file_record.path}"
            )
    return artifact


def validate_project_figure(
    project: str | Path,
    figure_id: str,
    *,
    namespace: str | None = None,
) -> FigureArtifact:
    """Validate one figure manifest and referenced project-local files."""
    return _figure_from_artifact(
        validate_project_artifact(
            project,
            figure_id,
            artifact_type=FIGURE_ARTIFACT_TYPE,
            namespace=namespace,
        )
    )


def validate_project_artifacts(
    project: str | Path,
    *,
    artifact_type: str | None = None,
    namespace: str | None = None,
) -> list[ArtifactManifest]:
    """Validate every saved project artifact manifest."""
    artifacts = list_project_artifacts(
        project,
        artifact_type=artifact_type,
        namespace=namespace,
    )
    root = _project_root(project)
    return [
        validate_project_artifact(
            root,
            artifact.artifact_id,
            artifact_type=artifact.artifact_type,
            namespace=artifact.namespace,
        )
        for artifact in artifacts
    ]


def validate_project_figures(
    project: str | Path,
    *,
    namespace: str | None = None,
) -> list[FigureArtifact]:
    """Validate every saved project figure artifact."""
    return [
        _figure_from_artifact(artifact)
        for artifact in validate_project_artifacts(
            project,
            artifact_type=FIGURE_ARTIFACT_TYPE,
            namespace=namespace,
        )
    ]


def rebuild_project_artifact_index(project: str | Path) -> list[ArtifactIndexEntry]:
    """Rebuild and write the project-wide artifact index from manifests."""
    root = _project_root(project)
    artifacts = list_project_artifacts(root)
    entries = [_entry_from_artifact(artifact, project_root=root) for artifact in artifacts]
    _write_index_entries(root, entries)
    return sorted(
        entries,
        key=lambda entry: (
            entry.namespace,
            entry.artifact_type,
            entry.artifact_id,
            entry.manifest_path,
        ),
    )


__all__ = [
    "ARTIFACT_INDEX_FILENAME",
    "ARTIFACT_MANIFEST_FILENAME",
    "ARTIFACT_SCHEMA_VERSION",
    "ArtifactFile",
    "ArtifactIndexEntry",
    "ArtifactManifest",
    "ArtifactOutputSpec",
    "FIGURES_DIRNAME",
    "FIGURE_ARTIFACT_SCHEMA_VERSION",
    "FIGURE_ARTIFACT_TYPE",
    "FIGURE_MANIFEST_FILENAME",
    "FigureArtifact",
    "FigureOutputSpec",
    "artifact_kind_dir",
    "delete_project_artifact",
    "list_project_figures",
    "list_project_artifact_index",
    "list_project_artifacts",
    "load_project_figure",
    "load_project_artifact",
    "rebuild_project_artifact_index",
    "save_project_figure",
    "save_project_artifact",
    "validate_project_figure",
    "validate_project_figures",
    "validate_project_artifact",
    "validate_project_artifacts",
    "project_artifact_index_path",
    "project_figure_root",
    "project_figures_root",
    "project_artifact_root",
    "project_artifact_type_root",
]
