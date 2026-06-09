"""Contract tests for normalized polygon sidecar labels and dataset YAML."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xpkg.segmentation.model import MaskType, SegmentationMask
from xpkg.segmentation.normalized_polygons import (
    NormalizedPolygonLabel,
    read_normalized_polygon_dataset_yaml,
    read_normalized_polygon_labels,
    read_normalized_polygon_rows,
    write_normalized_polygon_dataset_yaml,
    write_normalized_polygon_labels,
    write_normalized_polygon_rows,
)


def _write_label_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "frame_000.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_read_rows_parses_class_and_vertices_from_literal_file(tmp_path: Path) -> None:
    path = _write_label_file(tmp_path, "2 0.1 0.2 0.5 0.2 0.5 0.8\n\n0 0 0 1 0 1 1\n")

    rows = read_normalized_polygon_rows(path)

    assert len(rows) == 2
    assert rows[0].class_index == 2
    np.testing.assert_allclose(
        rows[0].vertices,
        np.array([[0.1, 0.2], [0.5, 0.2], [0.5, 0.8]], dtype=np.float32),
    )
    assert rows[1].class_index == 0
    np.testing.assert_allclose(
        rows[1].vertices,
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
    )


def test_read_rows_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert read_normalized_polygon_rows(tmp_path / "absent.txt") == []


@pytest.mark.parametrize(
    ("line", "message"),
    [
        pytest.param(
            "1 0.1 0.2 0.5 0.2",
            "must contain class index plus at least three points",
            id="too-few-values",
        ),
        pytest.param(
            "1 0.1 0.2 0.5 0.2 0.5 0.8 0.9",
            "odd number of coordinate values",
            id="odd-coordinates",
        ),
        pytest.param(
            "paw 0.1 0.2 0.5 0.2 0.5 0.8",
            "non-integer class index",
            id="non-integer-class",
        ),
        pytest.param(
            "-1 0.1 0.2 0.5 0.2 0.5 0.8",
            "class index must be non-negative",
            id="negative-class",
        ),
        pytest.param(
            "1 0.1 oops 0.5 0.2 0.5 0.8",
            "non-numeric coordinates",
            id="non-numeric-coordinate",
        ),
        pytest.param(
            "1 0.1 nan 0.5 0.2 0.5 0.8",
            "non-finite coordinates",
            id="non-finite-coordinate",
        ),
        pytest.param(
            "1 0.1 1.5 0.5 0.2 0.5 0.8",
            r"coordinates outside \[0, 1\]",
            id="out-of-bounds",
        ),
    ],
)
def test_read_rows_rejects_malformed_lines(tmp_path: Path, line: str, message: str) -> None:
    path = _write_label_file(tmp_path, "0 0 0 1 0 1 1\n" + line + "\n")

    with pytest.raises(ValueError, match=message) as excinfo:
        read_normalized_polygon_rows(path)
    assert str(excinfo.value).startswith("Line 2 ")


def test_read_rows_can_allow_out_of_bounds_coordinates(tmp_path: Path) -> None:
    path = _write_label_file(tmp_path, "1 -0.1 0.2 0.5 0.2 0.5 1.5\n")

    rows = read_normalized_polygon_rows(path, allow_out_of_bounds=True)

    assert rows[0].vertices[0, 0] == pytest.approx(-0.1)
    assert rows[0].vertices[2, 1] == pytest.approx(1.5)


def test_to_pixel_vertices_scales_by_image_size() -> None:
    row = NormalizedPolygonLabel(
        class_index=0,
        vertices=np.array([[0.1, 0.2], [0.5, 0.2], [0.5, 0.8]], dtype=np.float32),
    )

    pixels = row.to_pixel_vertices(image_width=640, image_height=480)

    np.testing.assert_allclose(
        pixels,
        np.array([[64.0, 96.0], [320.0, 96.0], [320.0, 384.0]], dtype=np.float32),
    )


def test_to_pixel_vertices_rejects_non_positive_image_size() -> None:
    row = NormalizedPolygonLabel(
        class_index=0,
        vertices=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="must be positive"):
        row.to_pixel_vertices(image_width=0, image_height=480)


def test_read_labels_resolves_class_names_from_sequence(tmp_path: Path) -> None:
    path = _write_label_file(tmp_path, "1 0 0 1 0 1 1\n5 0 0 1 0 1 1\n")

    masks = read_normalized_polygon_labels(
        path,
        image_width=10,
        image_height=10,
        class_names=["body", "paw"],
    )

    # Index 1 resolves to "paw"; index 5 is out of range and stays numeric.
    assert [mask.class_name for mask in masks] == ["paw", "5"]
    assert [mask.instance_ref for mask in masks] == [0, 1]


def test_read_labels_resolves_class_names_from_mapping(tmp_path: Path) -> None:
    path = _write_label_file(tmp_path, "3 0 0 1 0 1 1\n4 0 0 1 0 1 1\n")

    masks = read_normalized_polygon_labels(
        path,
        image_width=10,
        image_height=10,
        class_names={3: "tail"},
    )

    assert [mask.class_name for mask in masks] == ["tail", "4"]


def test_write_rows_strips_trailing_zeros_from_formatted_floats(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "frame.txt"

    write_normalized_polygon_rows(
        path,
        [
            NormalizedPolygonLabel(
                class_index=1,
                vertices=np.array([[0.5, 0.25], [1.0, 0.0], [0.125, 1.0]], dtype=np.float32),
            )
        ],
        precision=4,
    )

    assert path.read_text(encoding="utf-8") == "1 0.5 0.25 1 0 0.125 1\n"


def test_write_rows_with_no_rows_writes_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"

    write_normalized_polygon_rows(path, [])

    assert path.read_text(encoding="utf-8") == ""


def test_write_labels_accepts_numeric_class_name_without_mapping(tmp_path: Path) -> None:
    path = tmp_path / "frame.txt"
    mask = SegmentationMask.from_polygon(
        np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]], dtype=np.float32),
        class_name="3",
    )

    write_normalized_polygon_labels(path, [mask], image_width=10, image_height=10)

    assert path.read_text(encoding="utf-8") == "3 0 0 0.5 0 0.5 0.5\n"


def test_write_labels_rejects_unmapped_non_numeric_class(tmp_path: Path) -> None:
    mask = SegmentationMask.from_polygon(
        np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]], dtype=np.float32),
        class_name="paw",
    )

    with pytest.raises(ValueError, match="missing from class_name_to_id"):
        write_normalized_polygon_labels(
            tmp_path / "frame.txt",
            [mask],
            image_width=10,
            image_height=10,
        )


def test_write_labels_rejects_raster_masks_by_default(tmp_path: Path) -> None:
    binary = np.zeros((10, 10), dtype=np.uint8)
    binary[2:5, 3:7] = 1
    mask = SegmentationMask.from_binary_mask(binary, class_name="0")

    with pytest.raises(ValueError, match="allow_raster_to_polygon=True"):
        write_normalized_polygon_labels(
            tmp_path / "frame.txt",
            [mask],
            image_width=10,
            image_height=10,
        )


def test_write_labels_can_extract_contours_from_raster_masks(tmp_path: Path) -> None:
    binary = np.zeros((10, 10), dtype=np.uint8)
    binary[2:5, 3:7] = 1
    mask = SegmentationMask.from_binary_mask(binary, class_name="0")
    path = tmp_path / "frame.txt"

    write_normalized_polygon_labels(
        path,
        [mask],
        image_width=10,
        image_height=10,
        allow_raster_to_polygon=True,
    )
    rows = read_normalized_polygon_rows(path)

    assert len(rows) == 1
    assert rows[0].class_index == 0
    # The rectangle contour corners are (3,2), (6,2), (6,4), (3,4) in pixels.
    expected_corners = {(0.3, 0.2), (0.6, 0.2), (0.6, 0.4), (0.3, 0.4)}
    actual_corners = {
        (round(float(x), 6), round(float(y), 6)) for x, y in rows[0].vertices
    }
    assert actual_corners == expected_corners


def test_write_labels_rejects_polygon_holes(tmp_path: Path) -> None:
    mask = SegmentationMask(
        mask_type=MaskType.POLYGON,
        polygon_vertices=[
            np.array([[0.0, 0.0], [8.0, 0.0], [8.0, 8.0]], dtype=np.float32),
            np.array([[2.0, 2.0], [4.0, 2.0], [4.0, 4.0]], dtype=np.float32),
        ],
        class_name="0",
    )

    with pytest.raises(ValueError, match="cannot represent polygon holes"):
        write_normalized_polygon_labels(
            tmp_path / "frame.txt",
            [mask],
            image_width=10,
            image_height=10,
        )


def test_write_labels_rejects_vertices_outside_image(tmp_path: Path) -> None:
    mask = SegmentationMask.from_polygon(
        np.array([[0.0, 0.0], [12.0, 0.0], [12.0, 5.0]], dtype=np.float32),
        class_name="0",
    )

    with pytest.raises(ValueError, match=r"normalize outside \[0, 1\]"):
        write_normalized_polygon_labels(
            tmp_path / "frame.txt",
            [mask],
            image_width=10,
            image_height=10,
        )


def test_dataset_yaml_omits_test_split_when_unset(tmp_path: Path) -> None:
    yaml_path = tmp_path / "dataset.yaml"

    write_normalized_polygon_dataset_yaml(yaml_path, names=["body", "paw"])
    payload = read_normalized_polygon_dataset_yaml(yaml_path)

    assert payload == {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "names": {0: "body", 1: "paw"},
    }


def test_dataset_yaml_sorts_mapping_names_by_class_index(tmp_path: Path) -> None:
    yaml_path = tmp_path / "dataset.yaml"

    write_normalized_polygon_dataset_yaml(yaml_path, names={2: "tail", 0: "body"})

    assert (
        yaml_path.read_text(encoding="utf-8")
        == "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: body\n  2: tail\n"
    )


def test_dataset_yaml_read_returns_empty_dict_for_empty_file(tmp_path: Path) -> None:
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text("", encoding="utf-8")

    assert read_normalized_polygon_dataset_yaml(yaml_path) == {}


def test_dataset_yaml_read_rejects_non_mapping_payload(tmp_path: Path) -> None:
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text("- images/train\n- images/val\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Dataset YAML must contain a mapping"):
        read_normalized_polygon_dataset_yaml(yaml_path)
