"""Service-bound figure artifact API."""

from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

from xpkg.workspace.figures import (
    FigureArtifact,
    FigureOutputSpec,
    list_workspace_figures,
    load_workspace_figure,
    save_workspace_figure,
    validate_workspace_figure,
    validate_workspace_figures,
)


@dataclass(frozen=True, slots=True)
class WorkspaceFigures:
    """Workspace-bound helpers for figure output manifests."""

    workspace_root: Path

    def save(
        self,
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
        """Copy figure outputs into the workspace and write a manifest."""
        return save_workspace_figure(
            self.workspace_root,
            figure_id=figure_id,
            outputs=outputs,
            title=title,
            inputs=inputs,
            producer=producer,
            stats=stats,
            metadata=metadata,
            namespace=namespace,
            overwrite=overwrite,
        )

    def load(self, figure_id: str, *, namespace: str | None = None) -> FigureArtifact:
        """Load one saved figure manifest."""
        return load_workspace_figure(self.workspace_root, figure_id, namespace=namespace)

    def list(self, *, namespace: str | None = None) -> builtins.list[FigureArtifact]:
        """List saved figure manifests."""
        return list_workspace_figures(self.workspace_root, namespace=namespace)

    @overload
    def validate(self, figure_id: str, *, namespace: str | None = None) -> FigureArtifact: ...

    @overload
    def validate(
        self,
        figure_id: None = None,
        *,
        namespace: str | None = None,
    ) -> builtins.list[FigureArtifact]: ...

    def validate(
        self,
        figure_id: str | None = None,
        *,
        namespace: str | None = None,
    ) -> FigureArtifact | builtins.list[FigureArtifact]:
        """Validate one saved figure, or every saved figure when omitted."""
        if figure_id is None:
            return validate_workspace_figures(self.workspace_root, namespace=namespace)
        return validate_workspace_figure(
            self.workspace_root,
            figure_id,
            namespace=namespace,
        )


__all__ = [
    "WorkspaceFigures",
]
