"""Smoke tests for the Posetta public API facade."""

from __future__ import annotations

import posetta.api as api


def test_posetta_api_exposes_io_contract() -> None:
    expected = {
        "Labels",
        "ProjectDescriptor",
        "ConversionResult",
        "SleapTrack",
        "WorkspaceLayout",
        "WorkspaceService",
        "convert_dlc_csv",
        "pack_project",
        "read_sleap_node_names",
        "read_sleap_track",
        "resolve_sleap_node_indices",
        "validate_workspace",
    }

    assert expected.issubset(set(api.__all__))
    assert api.Labels.__name__ == "Labels"
    assert api.ProjectDescriptor.__name__ == "ProjectDescriptor"
    assert api.SleapTrack.__name__ == "SleapTrack"
    assert api.WorkspaceService.__name__ == "WorkspaceService"
    assert callable(api.read_sleap_node_names)
    assert callable(api.read_sleap_track)
    assert callable(api.resolve_sleap_node_indices)
