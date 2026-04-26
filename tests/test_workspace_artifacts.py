from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from xpkg.formats import (
    list_workspace_artifact_index,
    list_workspace_artifacts,
    load_workspace_artifact,
    pack_project,
    rebuild_workspace_artifact_index,
    save_workspace_artifact,
    validate_workspace_artifact,
)
from xpkg.services import WorkspaceService


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_workspace_artifacts_register_table_manifest_and_index(tmp_path: Path) -> None:
    workspace = WorkspaceService.create(tmp_path / "Artifact Project")
    table = _write_text(tmp_path / "source_data.csv", "x,y\n1,2\n")
    events = _write_text(
        workspace.workspace_root / ".xpkg/analysis/events/session_001/events.csv",
        "event,time\nresponse,1.0\n",
    )

    artifact = workspace.artifacts.register(
        artifact_id="source_data_table",
        artifact_type="table",
        title="Source data table",
        outputs={"source_data.csv": table},
        inputs=[events],
        producer={
            "package": "analysis-toolkit",
            "command": "analysis-toolkit make-tables",
        },
        metadata={"unit": "event"},
    )

    assert artifact.artifact_type == "table"
    assert artifact.artifact_id == "source-data-table"
    assert artifact.outputs == (
        ".xpkg/artifacts/tables/source-data-table/source_data.csv",
    )
    assert artifact.inputs == (".xpkg/analysis/events/session_001/events.csv",)
    assert {file.role for file in artifact.files} == {"input", "output"}
    assert all(file.sha256 for file in artifact.files)

    loaded = workspace.artifacts.load("source_data_table", kind="table")
    assert loaded.to_dict() == artifact.to_dict()
    assert workspace.artifacts.list(kind="table")[0].artifact_id == artifact.artifact_id
    assert workspace.artifacts.index(kind="table")[0].manifest_path == (
        ".xpkg/artifacts/tables/source-data-table/manifest.json"
    )
    assert workspace.artifacts.validate("source_data_table", kind="table").artifact_id == (
        artifact.artifact_id
    )


def test_workspace_artifacts_support_namespaces_without_known_package_names(
    tmp_path: Path,
) -> None:
    workspace = WorkspaceService.create(tmp_path / "Shared Workspace")
    output = _write_text(tmp_path / "report.md", "# Summary\n")

    artifact = workspace.artifacts.register(
        artifact_id="session_summary",
        artifact_type="report",
        namespace="arbitrary-downstream-tool",
        outputs={"report.md": output},
        producer={"package": "arbitrary-downstream-tool"},
    )

    assert artifact.namespace == "arbitrary-downstream-tool"
    assert artifact.outputs == (
        ".xpkg/arbitrary-downstream-tool/reports/session-summary/report.md",
    )
    assert workspace.artifacts.load(
        "session_summary",
        kind="report",
        namespace="arbitrary-downstream-tool",
    ).namespace == "arbitrary-downstream-tool"


def test_workspace_artifact_free_functions_and_pack_include_index(
    tmp_path: Path,
) -> None:
    workspace = WorkspaceService.create(tmp_path / "Packed Artifacts")
    table = _write_text(tmp_path / "summary.csv", "x\n1\n")

    artifact = save_workspace_artifact(
        workspace.workspace_root,
        artifact_id="summary",
        artifact_type="table",
        outputs=[table],
        producer={"package": "tests"},
    )

    assert load_workspace_artifact(
        workspace.workspace_root,
        "summary",
        artifact_type="table",
    ).outputs == artifact.outputs
    assert list_workspace_artifacts(
        workspace.workspace_root,
        artifact_type="table",
    )[0].artifact_id == "summary"
    assert validate_workspace_artifact(
        workspace.workspace_root,
        "summary",
        artifact_type="table",
    ).artifact_id == "summary"

    packed = pack_project(workspace.workspace_root, out=tmp_path / "artifacts.expkg")
    with zipfile.ZipFile(packed) as archive:
        names = set(archive.namelist())

    assert ".xpkg/artifacts/index.json" in names
    assert ".xpkg/artifacts/tables/summary/manifest.json" in names
    assert ".xpkg/artifacts/tables/summary/summary.csv" in names


def test_workspace_artifact_validation_detects_checksum_mismatch(tmp_path: Path) -> None:
    workspace = WorkspaceService.create(tmp_path / "Checksum Workspace")
    table = _write_text(tmp_path / "summary.csv", "x\n1\n")

    artifact = workspace.artifacts.register(
        artifact_id="summary",
        artifact_type="table",
        outputs=[table],
    )

    stored_output = workspace.workspace_root / artifact.outputs[0]
    stored_output.write_text("x\n2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        workspace.artifacts.validate("summary", kind="table")


def test_workspace_artifact_rebuild_index_discovers_existing_manifests(
    tmp_path: Path,
) -> None:
    workspace = WorkspaceService.create(tmp_path / "Index Workspace")
    figure = _write_text(tmp_path / "figure.svg", "<svg></svg>\n")
    workspace.figures.save(figure_id="summary", outputs=[figure])

    index_path = workspace.workspace_root / ".xpkg/artifacts/index.json"
    index_path.unlink()

    entries = rebuild_workspace_artifact_index(workspace.workspace_root)
    assert [entry.artifact_id for entry in entries] == ["summary"]
    assert list_workspace_artifact_index(workspace.workspace_root)[0].artifact_type == "figure"
