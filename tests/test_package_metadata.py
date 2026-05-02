from __future__ import annotations

import re
import tomllib
from pathlib import Path

from xpkg.version import __version__

ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_package_metadata_declares_public_identity() -> None:
    project = _pyproject()["project"]

    assert project["name"] == "exp-pkg"
    assert project["dynamic"] == ["version"]
    assert "version" not in project
    assert project["license"] == "BSD-3-Clause"
    assert project["license-files"] == ["LICENSE"]
    assert project["requires-python"] == ">=3.12"
    assert project["scripts"]["xpkg"] == "xpkg.cli:main"

    author_names = {author["name"] for author in project["authors"]}
    assert {"Alfredo Sandoval", "Joseph Sandoval"} <= author_names

    assert "neuroscience" in project["keywords"]
    assert "fiber-photometry" in project["keywords"]
    assert "Typing :: Typed" in project["classifiers"]


def test_version_comes_from_package_version_module() -> None:
    pyproject = _pyproject()

    assert pyproject["tool"]["hatch"]["version"]["path"] == "src/xpkg/version.py"
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[a-z]+\d*)?", __version__)


def test_package_declares_inline_typing_marker() -> None:
    assert (ROOT / "src" / "xpkg" / "py.typed").is_file()


def test_sdist_excludes_repo_only_planning_and_agent_files() -> None:
    exclude = set(_pyproject()["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"])

    assert "/AGENTS.md" in exclude
    assert "/plan.md" in exclude


def test_wheel_includes_public_project_schema() -> None:
    force_include = _pyproject()["tool"]["hatch"]["build"]["targets"]["wheel"][
        "force-include"
    ]

    assert force_include["schemas/project.schema.json"] == "xpkg/schemas/project.schema.json"
