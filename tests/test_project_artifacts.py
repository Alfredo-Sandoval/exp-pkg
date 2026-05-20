from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from xpkg.project import (
    delete_project_artifact,
    list_project_artifact_index,
    list_project_artifacts,
    load_project_artifact,
    pack_project,
    rebuild_project_artifact_index,
    save_project_artifact,
    validate_project_artifact,
)
from xpkg.services import ProjectService


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_project_artifacts_register_table_manifest_and_index(tmp_path: Path) -> None:
    project = ProjectService.create(tmp_path / "Artifact Project")
    table = _write_text(tmp_path / "source_data.csv", "x,y\n1,2\n")
    events = _write_text(
        project.project_root / ".xpkg/analysis/events/session_001/events.csv",
        "event,time\nresponse,1.0\n",
    )

    artifact = project.artifacts.register(
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

    loaded = project.artifacts.load("source_data_table", kind="table")
    assert loaded.to_dict() == artifact.to_dict()
    assert project.artifacts.list(kind="table")[0].artifact_id == artifact.artifact_id
    assert project.artifacts.index(kind="table")[0].manifest_path == (
        ".xpkg/artifacts/tables/source-data-table/manifest.json"
    )
    assert project.artifacts.validate("source_data_table", kind="table").artifact_id == (
        artifact.artifact_id
    )


def test_project_artifacts_support_namespaces_without_known_package_names(
    tmp_path: Path,
) -> None:
    project = ProjectService.create(tmp_path / "Shared Project")
    output = _write_text(tmp_path / "report.md", "# Summary\n")

    artifact = project.artifacts.register(
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
    assert project.artifacts.load(
        "session_summary",
        kind="report",
        namespace="arbitrary-downstream-tool",
    ).namespace == "arbitrary-downstream-tool"


def test_project_artifacts_delete_namespaced_custom_kind(tmp_path: Path) -> None:
    project = ProjectService.create(tmp_path / "OpenOperant Project")
    alignment = _write_text(tmp_path / "alignment.json", '{"offset": 4.25}\n')

    artifact = project.artifacts.register(
        artifact_id="session_a_medpc_alignment",
        artifact_type="medpc-alignment",
        namespace="openoperant",
        outputs={"openoperant_manifest.json": alignment},
        metadata={"video_offset_seconds": 4.25},
    )

    assert artifact.artifact_type == "medpc-alignment"
    assert artifact.namespace == "openoperant"
    assert (project.project_root / artifact.manifest_path).exists()
    assert len(project.artifacts.index(kind="medpc-alignment")) == 1

    assert project.artifacts.delete(
        "session_a_medpc_alignment",
        kind="medpc-alignment",
        namespace="openoperant",
    )
    assert not (project.project_root / artifact.artifact_root).exists()
    assert project.artifacts.index(kind="medpc-alignment") == []
    assert not delete_project_artifact(
        project.project_root,
        "session_a_medpc_alignment",
        artifact_type="medpc-alignment",
        namespace="openoperant",
        missing_ok=True,
    )
    with pytest.raises(FileNotFoundError, match="Project artifact does not exist"):
        project.artifacts.delete(
            "session_a_medpc_alignment",
            kind="medpc-alignment",
            namespace="openoperant",
        )


def test_project_artifact_free_functions_and_pack_include_index(
    tmp_path: Path,
) -> None:
    project = ProjectService.create(tmp_path / "Packed Artifacts")
    table = _write_text(tmp_path / "summary.csv", "x\n1\n")

    artifact = save_project_artifact(
        project.project_root,
        artifact_id="summary",
        artifact_type="table",
        outputs=[table],
        producer={"package": "tests"},
    )

    assert load_project_artifact(
        project.project_root,
        "summary",
        artifact_type="table",
    ).outputs == artifact.outputs
    assert list_project_artifacts(
        project.project_root,
        artifact_type="table",
    )[0].artifact_id == "summary"
    assert validate_project_artifact(
        project.project_root,
        "summary",
        artifact_type="table",
    ).artifact_id == "summary"

    packed = pack_project(project.project_root, out=tmp_path / "artifacts.expkg")
    with zipfile.ZipFile(packed) as archive:
        names = set(archive.namelist())

    assert ".xpkg/artifacts/index.json" in names
    assert ".xpkg/artifacts/tables/summary/manifest.json" in names
    assert ".xpkg/artifacts/tables/summary/summary.csv" in names


def test_project_artifact_validation_detects_checksum_mismatch(tmp_path: Path) -> None:
    project = ProjectService.create(tmp_path / "Checksum Project")
    table = _write_text(tmp_path / "summary.csv", "x\n1\n")

    artifact = project.artifacts.register(
        artifact_id="summary",
        artifact_type="table",
        outputs=[table],
    )

    stored_output = project.project_root / artifact.outputs[0]
    stored_output.write_text("x\n2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        project.artifacts.validate("summary", kind="table")


def test_project_artifact_rebuild_index_discovers_existing_manifests(
    tmp_path: Path,
) -> None:
    project = ProjectService.create(tmp_path / "Index Project")
    figure = _write_text(tmp_path / "figure.svg", "<svg></svg>\n")
    project.figures.save(figure_id="summary", outputs=[figure])

    index_path = project.project_root / ".xpkg/artifacts/index.json"
    index_path.unlink()

    entries = rebuild_project_artifact_index(project.project_root)
    assert [entry.artifact_id for entry in entries] == ["summary"]
    assert list_project_artifact_index(project.project_root)[0].artifact_type == "figure"
