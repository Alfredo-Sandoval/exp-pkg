"""Workspace figure artifact manifests and file registration."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from xpkg.core.json_utils import load_json_dict, write_json
from xpkg.core.path_registry import ensure_dir, resolve_path, slugify_path_component
from xpkg.io.project_layout import (
    _now_utc_iso,
    resolve_workspace_root,
    workspace_artifacts_root,
    workspace_store_root,
)

FIGURES_DIRNAME = "figures"
FIGURE_MANIFEST_FILENAME = "manifest.json"
FIGURE_ARTIFACT_SCHEMA_VERSION = "1.0.0"
FIGURE_ARTIFACT_TYPE = "figure"
FigureOutputSpec = Sequence[str | Path] | Mapping[str, str | Path]


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
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def _workspace_root(path: str | Path) -> Path:
    root = resolve_workspace_root(path)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg workspace: {path}")
    return root


def _figure_id(value: str) -> str:
    figure_id = slugify_path_component(value)
    if not figure_id:
        raise ValueError("figure_id cannot be empty")
    return figure_id


def _namespace(value: str | None) -> str:
    if value is None or not value.strip():
        return ""
    return slugify_path_component(value)


def _infer_namespace(artifact_root: Path) -> str:
    figures_root = artifact_root.parent
    container = figures_root.parent
    if container.name == "artifacts":
        return ""
    return container.name


def workspace_figures_root(workspace: str | Path, *, namespace: str | None = None) -> Path:
    """Return the private workspace figure artifact root."""
    normalized_namespace = _namespace(namespace)
    if normalized_namespace:
        return workspace_store_root(workspace) / normalized_namespace / FIGURES_DIRNAME
    return workspace_artifacts_root(workspace) / FIGURES_DIRNAME


def workspace_figure_root(
    workspace: str | Path,
    figure_id: str,
    *,
    namespace: str | None = None,
) -> Path:
    """Return the private root for one figure artifact."""
    return workspace_figures_root(workspace, namespace=namespace) / _figure_id(figure_id)


def _manifest_path(figure_root: Path) -> Path:
    return figure_root / FIGURE_MANIFEST_FILENAME


def _relative_to_workspace(path: Path, workspace_root: Path) -> str:
    try:
        return path.resolve().relative_to(workspace_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Artifact path must live inside the workspace: {path}") from exc


def _portable_path(value: str | Path, workspace_root: Path) -> str:
    raw = Path(value)
    if raw.is_absolute():
        return _relative_to_workspace(raw, workspace_root)
    portable = PurePosixPath(raw.as_posix())
    if portable.is_absolute() or ".." in portable.parts:
        raise ValueError(f"Artifact paths must be workspace-relative: {value}")
    return portable.as_posix()


def _portable_paths(values: Sequence[str | Path], workspace_root: Path) -> tuple[str, ...]:
    if isinstance(values, str | bytes | bytearray):
        raise TypeError("paths must be a sequence, not a string")
    return tuple(_portable_path(item, workspace_root) for item in values)


def _validate_target_name(name: str) -> PurePosixPath:
    target = PurePosixPath(str(name))
    if target.is_absolute() or ".." in target.parts:
        raise ValueError(f"Figure output target must be relative: {name!r}")
    if not target.name:
        raise ValueError(f"Figure output target must name a file: {name!r}")
    return target


def _iter_output_specs(outputs: FigureOutputSpec) -> list[tuple[PurePosixPath, Path]]:
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
        raise ValueError("At least one figure output is required")
    return items


def _copy_outputs(
    outputs: FigureOutputSpec,
    *,
    figure_root: Path,
    workspace_root: Path,
    overwrite: bool,
) -> tuple[str, ...]:
    copied_paths: list[str] = []
    for target_name, source_path in _iter_output_specs(outputs):
        if not source_path.is_file():
            raise FileNotFoundError(f"Figure output file not found: {source_path}")
        target_path = figure_root.joinpath(*target_name.parts)
        if target_path.exists() and not overwrite:
            raise FileExistsError(f"Figure output already exists: {target_path}")
        ensure_dir(target_path.parent)
        if source_path.resolve() != target_path.resolve():
            shutil.copy2(source_path, target_path)
        copied_paths.append(_relative_to_workspace(target_path, workspace_root))
    return tuple(copied_paths)


def _load_existing_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return load_json_dict(path)


def _artifact_from_manifest(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
    artifact_root: Path,
) -> FigureArtifact:
    artifact_type = str(payload.get("artifact_type", ""))
    if artifact_type != FIGURE_ARTIFACT_TYPE:
        raise ValueError(f"Expected figure artifact manifest, got {artifact_type!r}")
    schema_version = str(payload.get("schema_version", ""))
    if schema_version != FIGURE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported figure artifact schema_version: {schema_version!r}")
    artifact_id = str(payload.get("artifact_id", "")).strip()
    if not artifact_id:
        raise ValueError("Figure artifact manifest is missing artifact_id")

    producer = payload.get("producer")
    metadata = payload.get("metadata")
    return FigureArtifact(
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
        created_at=str(payload.get("created_at", "")),
        updated_at=str(payload.get("updated_at", "")),
        manifest_path=manifest_path,
        artifact_root=artifact_root,
    )


def _candidate_manifest_paths(
    workspace: str | Path,
    figure_id: str,
    *,
    namespace: str | None,
) -> list[Path]:
    root = _workspace_root(workspace)
    normalized_id = _figure_id(figure_id)
    if namespace is not None:
        return [_manifest_path(workspace_figure_root(root, normalized_id, namespace=namespace))]

    candidates = [
        _manifest_path(workspace_figure_root(root, normalized_id)),
        *sorted(
            workspace_store_root(root).glob(
                f"*/{FIGURES_DIRNAME}/{normalized_id}/{FIGURE_MANIFEST_FILENAME}"
            )
        ),
    ]
    return list(dict.fromkeys(candidates))


def load_workspace_figure(
    workspace: str | Path,
    figure_id: str,
    *,
    namespace: str | None = None,
) -> FigureArtifact:
    """Load one workspace figure artifact manifest."""
    root = _workspace_root(workspace)
    manifests = [
        path
        for path in _candidate_manifest_paths(root, figure_id, namespace=namespace)
        if path.is_file()
    ]
    if not manifests:
        raise FileNotFoundError(f"Figure artifact not found: {figure_id}")
    if len(manifests) > 1:
        raise ValueError(
            f"Figure artifact {figure_id!r} exists in multiple namespaces; "
            "pass namespace=... to choose one"
        )
    manifest_path = manifests[0]
    figure_root = manifest_path.parent
    payload = load_json_dict(manifest_path)
    return _artifact_from_manifest(
        payload,
        manifest_path=manifest_path,
        artifact_root=figure_root,
    )


def list_workspace_figures(
    workspace: str | Path,
    *,
    namespace: str | None = None,
) -> list[FigureArtifact]:
    """List saved workspace figure artifacts."""
    root = _workspace_root(workspace)
    if namespace is None:
        manifest_paths = [
            *sorted(workspace_figures_root(root).glob(f"*/{FIGURE_MANIFEST_FILENAME}")),
            *sorted(
                workspace_store_root(root).glob(
                    f"*/{FIGURES_DIRNAME}/*/{FIGURE_MANIFEST_FILENAME}"
                )
            ),
        ]
    else:
        figures_root = workspace_figures_root(root, namespace=namespace)
        manifest_paths = sorted(figures_root.glob(f"*/{FIGURE_MANIFEST_FILENAME}"))
    manifest_paths = list(dict.fromkeys(path for path in manifest_paths if path.is_file()))
    if not manifest_paths:
        return []
    artifacts: list[FigureArtifact] = []
    for manifest_path in sorted(manifest_paths):
        artifacts.append(
            _artifact_from_manifest(
                load_json_dict(manifest_path),
                manifest_path=manifest_path,
                artifact_root=manifest_path.parent,
            )
        )
    return artifacts


def save_workspace_figure(
    workspace: str | Path,
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
    """Copy figure outputs into the workspace and write a portable manifest."""
    root = _workspace_root(workspace)
    normalized_id = _figure_id(figure_id)
    normalized_namespace = _namespace(namespace)
    figure_root = workspace_figure_root(root, normalized_id, namespace=normalized_namespace)
    ensure_dir(figure_root)
    manifest_path = _manifest_path(figure_root)
    existing_manifest = _load_existing_manifest(manifest_path)
    now = _now_utc_iso()
    created_at = (
        str(existing_manifest.get("created_at", ""))
        if existing_manifest is not None
        else now
    )

    copied_outputs = _copy_outputs(
        outputs,
        figure_root=figure_root,
        workspace_root=root,
        overwrite=overwrite,
    )
    artifact = FigureArtifact(
        artifact_id=normalized_id,
        title=str(title or normalized_id),
        inputs=_portable_paths(inputs, root),
        producer=dict(producer or {}),
        outputs=copied_outputs,
        stats=_portable_paths(stats, root),
        metadata=dict(metadata or {}),
        namespace=normalized_namespace,
        manifest_path=manifest_path,
        artifact_root=figure_root,
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
    validate_workspace_figure(root, normalized_id, namespace=normalized_namespace)
    return artifact


def validate_workspace_figure(
    workspace: str | Path,
    figure_id: str,
    *,
    namespace: str | None = None,
) -> FigureArtifact:
    """Validate one figure manifest and referenced workspace-local files."""
    root = _workspace_root(workspace)
    artifact = load_workspace_figure(root, figure_id, namespace=namespace)
    for path_value in (*artifact.outputs, *artifact.inputs, *artifact.stats):
        path = root / path_value
        if not path.is_file():
            raise FileNotFoundError(
                f"Figure artifact {artifact.artifact_id!r} references missing file: "
                f"{path_value}"
            )
    return artifact


def validate_workspace_figures(
    workspace: str | Path,
    *,
    namespace: str | None = None,
) -> list[FigureArtifact]:
    """Validate every saved workspace figure artifact."""
    artifacts = list_workspace_figures(workspace, namespace=namespace)
    root = _workspace_root(workspace)
    return [
        validate_workspace_figure(
            root,
            artifact.artifact_id,
            namespace=artifact.namespace,
        )
        for artifact in artifacts
    ]


__all__ = [
    "FIGURE_ARTIFACT_SCHEMA_VERSION",
    "FIGURE_ARTIFACT_TYPE",
    "FIGURE_MANIFEST_FILENAME",
    "FIGURES_DIRNAME",
    "FigureArtifact",
    "FigureOutputSpec",
    "list_workspace_figures",
    "load_workspace_figure",
    "save_workspace_figure",
    "validate_workspace_figure",
    "validate_workspace_figures",
    "workspace_figure_root",
    "workspace_figures_root",
]
