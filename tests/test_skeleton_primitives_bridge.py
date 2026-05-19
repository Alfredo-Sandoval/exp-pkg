"""Bridge between xpkg.Skeleton and primitives.SkeletonDefinition."""

# ruff: noqa: E402, I001

from __future__ import annotations
import pytest

pytest.importorskip("primitives.skeletons.registry")
from primitives.skeletons.registry import SkeletonDefinition

from xpkg.pose.skeleton import Keypoint, Skeleton


def _build_skeleton() -> Skeleton:
    return Skeleton(
        name="mouse",
        keypoints=[
            Keypoint(id=0, name="nose"),
            Keypoint(id=1, name="ear_l"),
            Keypoint(id=2, name="ear_r"),
        ],
        links_ids=[(0, 1), (0, 2)],
    )


def test_as_definition_returns_primitives_view() -> None:
    skeleton = _build_skeleton()

    definition = skeleton.as_definition()

    assert isinstance(definition, SkeletonDefinition)
    assert definition.name == "mouse"
    assert definition.bodyparts == ("nose", "ear_l", "ear_r")
    assert definition.edges == (("nose", "ear_l"), ("nose", "ear_r"))


def test_as_definition_skips_links_referencing_missing_ids() -> None:
    skeleton = Skeleton(
        name="partial",
        keypoints=[Keypoint(id=0, name="nose"), Keypoint(id=1, name="ear_l")],
        links_ids=[(0, 1), (0, 99)],
    )

    definition = skeleton.as_definition()

    assert definition.edges == (("nose", "ear_l"),)


def test_as_definition_path_uses_extras_source_path() -> None:
    skeleton = _build_skeleton()
    skeleton.extras["source_path"] = "/tmp/skeleton_mouse.yaml"

    definition = skeleton.as_definition()

    assert str(definition.path) == "/tmp/skeleton_mouse.yaml"


def test_as_definition_path_empty_without_extras_source() -> None:
    skeleton = _build_skeleton()

    definition = skeleton.as_definition()

    assert str(definition.path) == "."
