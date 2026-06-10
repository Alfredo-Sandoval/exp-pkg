from __future__ import annotations

import pytest

from xpkg.pose.skeleton import Keypoint, Skeleton, base_lr_name, infer_side, to_snake


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("left_eye", "left_eye"),
        ("leftEye", "left_eye"),
        ("LeftEye", "left_eye"),
        ("left-eye", "left_eye"),
        ("left eye", "left_eye"),
        ("eye_l", "eye_left"),
        ("eye_r", "eye_right"),
        ("LEFT__EYE", "left_eye"),
    ],
)
def test_to_snake_normalizes_keypoint_names(raw_name: str, expected: str) -> None:
    assert to_snake(raw_name) == expected


def test_side_helpers_resolve_common_left_right_names() -> None:
    assert infer_side("eye_left") == "left"
    assert infer_side("eye_right") == "right"
    assert infer_side("nose") == "unknown"
    assert infer_side("spine", default="midline") == "midline"
    assert base_lr_name("left_eye") == ("eye", "left")
    assert base_lr_name("ear_right") == ("ear", "right")
    assert base_lr_name("nose") == ("nose", None)


def _simple_skeleton() -> Skeleton:
    return Skeleton(
        name="test",
        keypoints=[
            Keypoint(id=0, name="nose"),
            Keypoint(id=1, name="left_eye", side="left"),
            Keypoint(id=2, name="right_eye", side="right"),
        ],
        links_ids=[(0, 1), (0, 2)],
    )


def test_skeleton_basic_lookup_and_index_contracts() -> None:
    skeleton = _simple_skeleton()

    assert skeleton.n_keypoints == 3
    assert skeleton.keypoint_names == ["nose", "left_eye", "right_eye"]
    assert skeleton.name_to_id() == {"nose": 0, "left_eye": 1, "right_eye": 2}
    assert skeleton.id_to_name() == {0: "nose", 1: "left_eye", 2: "right_eye"}
    assert "nose" in skeleton
    assert skeleton.keypoints[0] in skeleton
    assert 2 in skeleton
    assert skeleton.keypoint_to_index("left_eye") == 1
    assert skeleton.keypoint_to_index(skeleton.keypoints[2]) == 2
    with pytest.raises(KeyError, match="unknown"):
        skeleton.keypoint_to_index("unknown")
    with pytest.raises(KeyError, match="100"):
        skeleton.keypoint_to_index(100)


def test_skeleton_from_dict_normalizes_and_validates_payloads() -> None:
    skeleton = Skeleton.from_dict(
        {
            "name": "test",
            "keypoints": ["LeftEye", "RightEye"],
            "links": [["LeftEye", "RightEye"]],
        }
    )

    assert skeleton.keypoint_names == ["left_eye", "right_eye"]
    assert skeleton.links_ids == [(0, 1)]
    with pytest.raises(ValueError, match="keypoints"):
        Skeleton.from_dict({"name": "test"})


def test_skeleton_hash_and_lr_partner_contracts_are_stable() -> None:
    skeleton = _simple_skeleton()

    # compute_hash is documented as stable across releases (it guards model
    # metadata against skeleton mismatch), so the exact value is the contract.
    assert skeleton.content_hash() == "b2f6689ce7700ab7"
    assert skeleton.compute_hash() == (
        "6f2256c8a95d8ff416b186cf1dbbaaa0ca9b86bb0a773bbe553978ed2825e897"
    )
    # Structurally equal skeletons built independently hash identically.
    assert _simple_skeleton().content_hash() == skeleton.content_hash()
    assert skeleton.lr_partner_map() == {0: None, 1: 2, 2: 1}


def test_skeleton_content_hash_changes_with_structure() -> None:
    renamed = Skeleton(
        name="test",
        keypoints=[
            Keypoint(id=0, name="nose"),
            Keypoint(id=1, name="left_ear", side="left"),
            Keypoint(id=2, name="right_eye", side="right"),
        ],
        links_ids=[(0, 1), (0, 2)],
    )
    assert renamed.content_hash() == "bdcbedbf2d607c5f"

    fewer_links = Skeleton(
        name="test",
        keypoints=[
            Keypoint(id=0, name="nose"),
            Keypoint(id=1, name="left_eye", side="left"),
            Keypoint(id=2, name="right_eye", side="right"),
        ],
        links_ids=[(0, 1)],
    )
    assert fewer_links.content_hash() == "be7aad36dae298e6"
