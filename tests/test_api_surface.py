"""Smoke tests for the Posetta public API facade."""

from __future__ import annotations

import posetta.api as api


def test_posetta_api_exposes_io_contract() -> None:
    expected = {
        "Labels",
        "ProjectDescriptor",
        "ConversionResult",
        "WorkspaceLayout",
        "WorkspaceService",
        "validate_workspace",
        "pack_project",
        "convert_dlc_csv",
    }

    assert expected.issubset(set(api.__all__))
    assert api.Labels.__name__ == "Labels"
    assert api.ProjectDescriptor.__name__ == "ProjectDescriptor"
    assert api.WorkspaceService.__name__ == "WorkspaceService"
