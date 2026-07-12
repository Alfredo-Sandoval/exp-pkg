from __future__ import annotations

import json
from pathlib import Path

import h5py
import pytest
import yaml

from xpkg.io.skeleton_io import (
    dump_skeleton,
    load_any_skeleton,
)
from xpkg.io.skeleton_io import (
    load_skeleton as load_json_skeleton,
)
from xpkg.io.skeleton_loaders import (
    detect_skeleton_format,
    detect_yaml_skeleton_format,
)
from xpkg.model import (
    Skeleton,
    load_skeleton,
    load_skeleton_dlc,
    load_skeleton_xpkg_json,
)


def _write_json_skeleton(
    path: Path,
    *,
    name: str = "test_skeleton",
    keypoints: list[str] | None = None,
    links: list[list[int | str]] | None = None,
) -> Path:
    payload = {
        "name": name,
        "keypoints": keypoints or ["nose", "tail"],
        "links": links or [[0, 1]],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_dlc_config(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "bodyparts": ["snout", "tailbase"],
                "skeleton": [["snout", "tailbase"]],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_sleap_yaml(path: Path, *, nodes: list[object] | None = None) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "name": "sleap_skeleton",
                "nodes": nodes or [{"name": "head"}, {"name": "thorax"}, {"name": "abdomen"}],
                "edges": [
                    {"source": {"name": "head"}, "destination": {"name": "thorax"}},
                    {"source": {"name": "thorax"}, "destination": {"name": "abdomen"}},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_ultralytics_yaml(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "kpt_shape": [3, 3],
                "keypoint_names": ["nose", "left_eye", "right_eye"],
                "skeleton": [[0, 1], [0, 2]],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_sleap_pkg(path: Path) -> Path:
    metadata = {
        "nodes": [{"name": "head"}, {"name": "tail"}],
        "skeletons": [
            {
                "nodes": [{"id": 0}, {"id": 1}],
                "links": [{"source": {"id": 0}, "target": {"id": 1}}],
            }
        ],
    }
    with h5py.File(str(path), "w") as handle:
        meta_group = handle.create_group("metadata")
        meta_group.attrs["json"] = json.dumps(metadata)
    return path


def test_dump_skeleton_writes_json(tmp_path: Path) -> None:
    skeleton = Skeleton.from_dict(
        {
            "name": "subject",
            "keypoints": ["nose", "tail_base"],
            "links": [["nose", "tail_base"]],
        }
    )
    output_path = tmp_path / "skeleton.json"

    dump_skeleton(skeleton, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["name"] == "subject"
    assert payload["keypoints"][0]["name"] == "nose"
    assert payload["links"] == [["nose", "tail_base"]]


def test_io_load_skeleton_reads_json(tmp_path: Path) -> None:
    skeleton = load_json_skeleton(_write_json_skeleton(tmp_path / "skeleton.json"))

    assert skeleton.name == "test_skeleton"
    assert skeleton.keypoint_names == ["nose", "tail"]
    assert skeleton.links_ids == [(0, 1)]


def test_io_load_skeleton_rejects_yaml_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="YAML skeletons are no longer supported"):
        load_json_skeleton(tmp_path / "skeleton.yaml")


def test_load_skeleton_xpkg_json(tmp_path: Path) -> None:
    skeleton = load_skeleton_xpkg_json(_write_json_skeleton(tmp_path / "skeleton.json"))

    assert skeleton.name == "test_skeleton"
    assert skeleton.keypoint_names == ["nose", "tail"]
    assert skeleton.links_ids == [(0, 1)]


def test_load_skeleton_dlc(tmp_path: Path) -> None:
    skeleton = load_skeleton_dlc(_write_dlc_config(tmp_path / "config.yaml"))

    assert skeleton.keypoint_names == ["snout", "tailbase"]
    assert skeleton.links_ids == [(0, 1)]


def test_load_skeleton_auto_detects_sleap_package(tmp_path: Path) -> None:
    skeleton = load_skeleton(_write_sleap_pkg(tmp_path / "sleap.pkg.slp"))

    assert skeleton.keypoint_names == ["head", "tail"]
    assert skeleton.links_ids == [(0, 1)]


def test_load_skeleton_auto_detects_sleap_yaml(tmp_path: Path) -> None:
    skeleton = load_skeleton(_write_sleap_yaml(tmp_path / "skeleton.yaml"))

    assert skeleton.name == "sleap_skeleton"
    assert skeleton.keypoint_names == ["head", "thorax", "abdomen"]
    assert skeleton.links_ids == [(0, 1), (1, 2)]


def test_load_skeleton_auto_detects_sleap_yaml_with_string_nodes(tmp_path: Path) -> None:
    skeleton = load_skeleton(_write_sleap_yaml(tmp_path / "strings.yaml", nodes=["head", "tail"]))

    assert skeleton.keypoint_names == ["head", "tail"]


def test_load_skeleton_auto_detects_ultralytics_yaml(tmp_path: Path) -> None:
    skeleton = load_skeleton(_write_ultralytics_yaml(tmp_path / "pose.yaml"))

    assert skeleton.keypoint_names == ["nose", "left_eye", "right_eye"]
    assert skeleton.links_ids == [(0, 1), (0, 2)]


def test_load_skeleton_rejects_plain_slp(tmp_path: Path) -> None:
    slp_file = tmp_path / "sample.slp"
    slp_file.write_text("invalid", encoding="utf-8")

    with pytest.raises(ValueError, match=r"pkg\.slp"):
        load_skeleton(slp_file)


def test_load_skeleton_rejects_h5(tmp_path: Path) -> None:
    h5_file = tmp_path / "sleap.h5"
    h5_file.write_text("invalid", encoding="utf-8")

    with pytest.raises(ValueError, match=r"pkg\.slp"):
        load_skeleton(h5_file)


def test_load_skeleton_rejects_unsupported_yaml_schema(tmp_path: Path) -> None:
    yaml_file = tmp_path / "unknown.yaml"
    yaml_file.write_text(yaml.safe_dump({"joints": ["nose", "tail"]}), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Unsupported YAML skeleton format; expected DLC, SLEAP, or Ultralytics schema.",
    ):
        load_skeleton(yaml_file)


def test_load_skeleton_rejects_primitives_yaml_schema(tmp_path: Path) -> None:
    yaml_file = tmp_path / "primitives.yaml"
    yaml_file.write_text(
        yaml.safe_dump(
            {
                "bodyparts": ["nose", "tail"],
                "aliases": {"tail": "tail_base"},
                "triads": {"spine": ["nose", "spine", "tail"]},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="primitives YAML is not supported"):
        load_skeleton(yaml_file)


def test_load_any_skeleton_uses_format_override(tmp_path: Path) -> None:
    skeleton = load_any_skeleton(_write_dlc_config(tmp_path / "config.yaml"), format="dlc")

    assert skeleton.keypoint_names == ["snout", "tailbase"]
    assert skeleton.links_ids == [(0, 1)]


def test_load_any_skeleton_accepts_yolo_alias(tmp_path: Path) -> None:
    skeleton = load_any_skeleton(_write_ultralytics_yaml(tmp_path / "pose.yaml"), format="yolo")

    assert skeleton.keypoint_names == ["nose", "left_eye", "right_eye"]
    assert skeleton.links_ids == [(0, 1), (0, 2)]


def test_load_any_skeleton_rejects_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown format: made_up"):
        load_any_skeleton(tmp_path / "config.yaml", format="made_up")


def test_detect_skeleton_format_reports_ultralytics_yaml(tmp_path: Path) -> None:
    pose_path = _write_ultralytics_yaml(tmp_path / "pose.yaml")

    assert detect_yaml_skeleton_format(pose_path) == "ultralytics"
    assert detect_skeleton_format(pose_path) == "ultralytics"
