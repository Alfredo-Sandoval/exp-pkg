"""Importing a primitives-format YAML into an xpkg.Skeleton."""

from __future__ import annotations

from pathlib import Path

import pytest

from xpkg.io.skeleton_loaders import (
    build_skeleton_from_primitives,
    detect_yaml_skeleton_format,
    load_primitives_yaml_skeleton,
    load_skeleton,
)


def _write_primitives_yaml(tmp_path: Path) -> Path:
    text = (
        "bodyparts: [nose, ear_l, ear_r, tail_base]\n"
        "skeleton:\n"
        "  - [nose, ear_l]\n"
        "  - [nose, ear_r]\n"
        "  - [nose, tail_base]\n"
        "aliases:\n"
        "  left_ear: ear_l\n"
        "  right_ear: ear_r\n"
        "triads:\n"
        "  head: [ear_l, nose, ear_r]\n"
    )
    path = tmp_path / "skeleton_primitives.yaml"
    path.write_text(text)
    return path


def test_load_primitives_yaml_round_trips_into_xpkg_skeleton(tmp_path: Path) -> None:
    path = _write_primitives_yaml(tmp_path)

    skeleton = load_primitives_yaml_skeleton(path)

    assert skeleton.keypoint_names == ["nose", "ear_l", "ear_r", "tail_base"]
    assert sorted(skeleton.links_ids) == [(0, 1), (0, 2), (0, 3)]
    assert skeleton.aliases == {"left_ear": "ear_l", "right_ear": "ear_r"}
    assert skeleton.triads is not None
    assert skeleton.triads["head"] == ("ear_l", "nose", "ear_r")
    assert skeleton.extras["source_path"] == str(path)


def test_load_skeleton_auto_detects_primitives_yaml(tmp_path: Path) -> None:
    path = _write_primitives_yaml(tmp_path)

    assert detect_yaml_skeleton_format(path) == "primitives"

    skeleton = load_skeleton(path)

    assert skeleton.aliases == {"left_ear": "ear_l", "right_ear": "ear_r"}
    assert skeleton.triads is not None
    assert skeleton.triads["head"] == ("ear_l", "nose", "ear_r")


def test_as_definition_round_trip_preserves_bodyparts_and_edges(tmp_path: Path) -> None:
    path = _write_primitives_yaml(tmp_path)

    skeleton = load_primitives_yaml_skeleton(path)
    definition = skeleton.as_definition()

    assert definition.bodyparts == ("nose", "ear_l", "ear_r", "tail_base")
    assert set(definition.edges) == {
        ("nose", "ear_l"),
        ("nose", "ear_r"),
        ("nose", "tail_base"),
    }


def test_build_from_primitives_drops_edges_referencing_unknown_bodyparts() -> None:
    from primitives.skeletons.registry import SkeletonDefinition

    definition = SkeletonDefinition(
        name="partial",
        bodyparts=("nose", "tail"),
        edges=(("nose", "tail"), ("nose", "ghost")),
        path=Path("/tmp/anything.yaml"),
    )

    skeleton = build_skeleton_from_primitives(definition)

    assert skeleton.links_ids == [(0, 1)]


@pytest.mark.parametrize("name_override", [None, "custom_skeleton"])
def test_build_from_primitives_respects_name_override(name_override: str | None) -> None:
    from primitives.skeletons.registry import SkeletonDefinition

    definition = SkeletonDefinition(
        name="default_name",
        bodyparts=("a", "b"),
        edges=(("a", "b"),),
        path=Path("/tmp/x.yaml"),
    )

    skeleton = build_skeleton_from_primitives(definition, name=name_override)

    assert skeleton.name == (name_override or "default_name")


def test_skeleton_to_dict_round_trips_primitives_fields(tmp_path: Path) -> None:
    """aliases/triads/node_properties survive a to_dict/from_dict round trip."""
    from xpkg.pose.skeleton import Skeleton

    skeleton = load_primitives_yaml_skeleton(_write_primitives_yaml(tmp_path))
    payload = skeleton.to_dict()

    assert payload["aliases"] == {"left_ear": "ear_l", "right_ear": "ear_r"}
    assert payload["triads"]["head"] == ["ear_l", "nose", "ear_r"]

    rebuilt = Skeleton.from_dict(payload)
    assert rebuilt.aliases == skeleton.aliases
    assert rebuilt.triads == skeleton.triads
    assert rebuilt.node_properties == skeleton.node_properties
