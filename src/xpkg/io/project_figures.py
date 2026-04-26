"""Workspace figure artifact manifests and file registration."""

from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xpkg.io.project_artifacts import (
    ARTIFACT_MANIFEST_FILENAME,
    ARTIFACT_SCHEMA_VERSION,
    ArtifactFile,
    ArtifactManifest,
    ArtifactOutputSpec,
    list_workspace_artifacts,
    load_workspace_artifact,
    save_workspace_artifact,
    validate_workspace_artifact,
    validate_workspace_artifacts,
    workspace_artifact_root,
    workspace_artifact_type_root,
)

FIGURES_DIRNAME = "figures"
FIGURE_MANIFEST_FILENAME = ARTIFACT_MANIFEST_FILENAME
FIGURE_ARTIFACT_SCHEMA_VERSION = ARTIFACT_SCHEMA_VERSION
FIGURE_ARTIFACT_TYPE = "figure"
FigureOutputSpec = ArtifactOutputSpec


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


def workspace_figures_root(workspace: str | Path, *, namespace: str | None = None) -> Path:
    """Return the private workspace figure artifact root."""
    return workspace_artifact_type_root(
        workspace,
        FIGURE_ARTIFACT_TYPE,
        namespace=namespace,
    )


def workspace_figure_root(
    workspace: str | Path,
    figure_id: str,
    *,
    namespace: str | None = None,
) -> Path:
    """Return the private root for one figure artifact."""
    return workspace_artifact_root(
        workspace,
        figure_id,
        FIGURE_ARTIFACT_TYPE,
        namespace=namespace,
    )


def load_workspace_figure(
    workspace: str | Path,
    figure_id: str,
    *,
    namespace: str | None = None,
) -> FigureArtifact:
    """Load one workspace figure artifact manifest."""
    return _figure_from_artifact(
        load_workspace_artifact(
            workspace,
            figure_id,
            artifact_type=FIGURE_ARTIFACT_TYPE,
            namespace=namespace,
        )
    )


def list_workspace_figures(
    workspace: str | Path,
    *,
    namespace: str | None = None,
) -> builtins.list[FigureArtifact]:
    """List saved workspace figure artifacts."""
    return [
        _figure_from_artifact(artifact)
        for artifact in list_workspace_artifacts(
            workspace,
            artifact_type=FIGURE_ARTIFACT_TYPE,
            namespace=namespace,
        )
    ]


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
    return _figure_from_artifact(
        save_workspace_artifact(
            workspace,
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


def validate_workspace_figure(
    workspace: str | Path,
    figure_id: str,
    *,
    namespace: str | None = None,
) -> FigureArtifact:
    """Validate one figure manifest and referenced workspace-local files."""
    return _figure_from_artifact(
        validate_workspace_artifact(
            workspace,
            figure_id,
            artifact_type=FIGURE_ARTIFACT_TYPE,
            namespace=namespace,
        )
    )


def validate_workspace_figures(
    workspace: str | Path,
    *,
    namespace: str | None = None,
) -> builtins.list[FigureArtifact]:
    """Validate every saved workspace figure artifact."""
    return [
        _figure_from_artifact(artifact)
        for artifact in validate_workspace_artifacts(
            workspace,
            artifact_type=FIGURE_ARTIFACT_TYPE,
            namespace=namespace,
        )
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
