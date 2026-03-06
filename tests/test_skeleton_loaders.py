from __future__ import annotations

import json
from pathlib import Path

import h5py
import yaml

from posetta.model import (
    load_skeleton,
    load_skeleton_dlc,
    load_skeleton_siesta_json,
)


def test_load_skeleton_siesta_json(tmp_path: Path) -> None:
    skeleton_file = tmp_path / "skeleton.json"
    skeleton_file.write_text(
        json.dumps(
            {
                "name": "test_skeleton",
                "keypoints": ["nose", "tail"],
                "links": [[0, 1]],
            }
        ),
        encoding="utf-8",
    )

    skeleton = load_skeleton_siesta_json(skeleton_file)

    assert skeleton.name == "test_skeleton"
    assert skeleton.keypoint_names == ["nose", "tail"]
    assert skeleton.links_ids == [(0, 1)]


def test_load_skeleton_dlc(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "bodyparts": ["snout", "tailbase"],
                "skeleton": [["snout", "tailbase"]],
            }
        ),
        encoding="utf-8",
    )

    skeleton = load_skeleton_dlc(config_file)

    assert skeleton.keypoint_names == ["snout", "tailbase"]
    assert skeleton.links_ids == [(0, 1)]


def test_load_skeleton_auto_detects_sleap_package(tmp_path: Path) -> None:
    pkg_file = tmp_path / "sleap.pkg.slp"
    metadata = {
        "nodes": [{"name": "head"}, {"name": "tail"}],
        "skeletons": [
            {
                "nodes": [{"id": 0}, {"id": 1}],
                "links": [{"source": {"id": 0}, "target": {"id": 1}}],
            }
        ],
    }
    with h5py.File(str(pkg_file), "w") as handle:
        meta_group = handle.create_group("metadata")
        meta_group.attrs["json"] = json.dumps(metadata)

    skeleton = load_skeleton(pkg_file)

    assert skeleton.keypoint_names == ["head", "tail"]
    assert skeleton.links_ids == [(0, 1)]
