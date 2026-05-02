from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from xpkg.services import WorkspaceService
from xpkg.workspace import (
    list_workspace_artifact_index,
    list_workspace_figures,
    load_workspace_figure,
    pack_project,
    save_workspace_figure,
    validate_expkg,
    validate_workspace_figure,
)


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_workspace_figures_save_manifest_and_outputs(tmp_path: Path) -> None:
    workspace = WorkspaceService.create(tmp_path / "Figure Project")
    source_dir = tmp_path / "source"
    svg = _write_text(source_dir / "validation.svg", "<svg></svg>\n")
    pdf = _write_text(source_dir / "validation.pdf", "%PDF-1.7\n")
    source_data = _write_text(source_dir / "source_data.csv", "x,y\n1,2\n")
    event_table = _write_text(
        workspace.workspace_root / ".xpkg/analysis/events/session_001/final_events.csv",
        "event,time\nresponse,1.0\n",
    )
    stats_report = _write_text(
        workspace.workspace_root / ".xpkg/analysis/stats/session_001/stats_report.json",
        "{}\n",
    )

    artifact = workspace.figures.save(
        figure_id="Validation Figure 3",
        title="Validation against reviewer labels",
        outputs={
            "figure.svg": svg,
            "figure.pdf": pdf,
            "source_data.csv": source_data,
        },
        inputs=[event_table],
        stats=[stats_report],
        producer={
            "package": "analysis-toolkit",
            "module": "analysis_toolkit.figures.validation",
            "command": "analysis-toolkit make-figures --figure validation",
            "git_commit": "abc123",
        },
        metadata={"panel": "figure-3"},
    )

    assert artifact.artifact_id == "validation-figure-3"
    assert artifact.title == "Validation against reviewer labels"
    assert artifact.outputs == (
        ".xpkg/artifacts/figures/validation-figure-3/figure.svg",
        ".xpkg/artifacts/figures/validation-figure-3/figure.pdf",
        ".xpkg/artifacts/figures/validation-figure-3/source_data.csv",
    )
    assert artifact.inputs == (
        ".xpkg/analysis/events/session_001/final_events.csv",
    )
    assert artifact.stats == (
        ".xpkg/analysis/stats/session_001/stats_report.json",
    )
    assert {file.role for file in artifact.files} == {"input", "output", "stat"}
    assert artifact.manifest_path.is_file()
    assert (artifact.artifact_root / "figure.svg").read_text(encoding="utf-8") == "<svg></svg>\n"

    loaded = workspace.figures.load("validation-figure-3")
    assert loaded.to_dict() == artifact.to_dict()
    assert workspace.figures.list()[0].artifact_id == artifact.artifact_id
    assert workspace.figures.validate("validation-figure-3").artifact_id == (
        artifact.artifact_id
    )
    assert list_workspace_artifact_index(
        workspace.workspace_root,
        artifact_type="figure",
    )[0].artifact_id == artifact.artifact_id


def test_workspace_figure_free_functions_and_pack_include_artifacts(
    tmp_path: Path,
) -> None:
    workspace = WorkspaceService.create(tmp_path / "Packed Figures")
    png = _write_text(tmp_path / "figure.png", "png-bytes\n")
    artifact = save_workspace_figure(
        workspace.workspace_root,
        figure_id="summary",
        title="Summary",
        outputs=[png],
        producer={"package": "tests"},
    )

    assert load_workspace_figure(workspace.workspace_root, "summary").outputs == artifact.outputs
    assert list_workspace_figures(workspace.workspace_root)[0].artifact_id == "summary"
    assert validate_workspace_figure(workspace.workspace_root, "summary").artifact_id == "summary"

    packed = pack_project(workspace.workspace_root, out=tmp_path / "figures.expkg")
    validate_expkg(packed)
    with zipfile.ZipFile(packed) as archive:
        names = set(archive.namelist())

    assert ".xpkg/artifacts/figures/summary/manifest.json" in names
    assert ".xpkg/artifacts/figures/summary/figure.png" in names


def test_workspace_figures_save_into_app_namespace(tmp_path: Path) -> None:
    workspace = WorkspaceService.create(tmp_path / "Namespaced Workspace")
    svg = _write_text(tmp_path / "validation.svg", "<svg></svg>\n")
    event_table = _write_text(
        workspace.workspace_root / ".xpkg/analysis-app/events/session_001/final_events.csv",
        "event,time\nresponse,1.0\n",
    )

    artifact = workspace.figures.save(
        figure_id="validation_figure_3",
        title="Validation against reviewer labels",
        outputs={"figure.svg": svg},
        inputs=[event_table],
        producer={"package": "analysis-app"},
        namespace="analysis-app",
    )

    assert artifact.namespace == "analysis-app"
    assert artifact.outputs == (
        ".xpkg/analysis-app/figures/validation-figure-3/figure.svg",
    )
    assert workspace.figures.load(
        "validation_figure_3",
        namespace="analysis-app",
    ).namespace == "analysis-app"
    assert workspace.figures.list(namespace="analysis-app")[0].artifact_id == (
        "validation-figure-3"
    )
    assert workspace.figures.validate(
        "validation_figure_3",
        namespace="analysis-app",
    ).outputs == artifact.outputs


def test_workspace_figures_support_arbitrary_app_namespaces(
    tmp_path: Path,
) -> None:
    workspace = WorkspaceService.create(tmp_path / "Shared Workspace")
    namespaces = ("analysis-a", "analysis-b", "review-ui", "qc-runner", "report-builder")

    for namespace in namespaces:
        output = _write_text(tmp_path / f"{namespace}.svg", f"<svg>{namespace}</svg>\n")
        workspace.figures.save(
            figure_id="summary",
            title=f"{namespace} summary",
            outputs={"figure.svg": output},
            producer={"package": namespace},
            namespace=namespace,
        )

    artifacts = workspace.figures.list()
    assert {artifact.namespace for artifact in artifacts} == set(namespaces)
    assert workspace.figures.load("summary", namespace="review-ui").namespace == "review-ui"
    with pytest.raises(ValueError, match="multiple namespaces"):
        workspace.figures.load("summary")


def test_workspace_figure_manifest_rejects_external_input_paths(tmp_path: Path) -> None:
    workspace = WorkspaceService.create(tmp_path / "External Input")
    svg = _write_text(tmp_path / "plot.svg", "<svg></svg>\n")
    external_input = _write_text(tmp_path / "outside.csv", "x\n1\n")

    with pytest.raises(ValueError, match="inside the workspace"):
        workspace.figures.save(
            figure_id="external-input",
            outputs=[svg],
            inputs=[external_input.resolve()],
        )
