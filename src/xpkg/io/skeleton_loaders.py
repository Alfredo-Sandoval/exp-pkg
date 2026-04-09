"""Unified skeleton loader supporting multiple external formats.

Imports skeletons from DeepLabCut, SLEAP, Ultralytics/YOLO, and xpkg archive
JSON. Returns xpkg.core.skeleton.Skeleton instances directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import h5py

from xpkg.config.loaders import load_yaml_file
from xpkg.core.json_utils import parse_json_dict
from xpkg.core.logging_utils import get_logger
from xpkg.core.skeleton import Skeleton

logger = get_logger(__name__)

EDGE_TYPE_BODY = 1
EDGE_TYPE_SYMMETRY = 2

type YamlSkeletonFormat = Literal["dlc", "sleap", "ultralytics"]
type SkeletonFormat = Literal["dlc", "xpkg_json", "sleap", "sleap_pkg_slp", "ultralytics"]


def build_sleap_skeleton(
    metadata: Mapping[str, Any], *, skeleton_name: str = "SLEAP"
) -> dict[str, Any]:
    """Normalize SLEAP metadata into a canonical xpkg skeleton definition.

    Args:
        metadata: JSON metadata extracted from a SLEAP `.pkg.slp` file (typically
            ``json.loads(h5_file["metadata"].attrs["json"])``). Only the first
            skeleton entry is considered, matching current export behaviour.
        skeleton_name: Name assigned to the resulting skeleton dictionary.

    Returns:
        dict: Dictionary with ``name``, ``keypoints`` and ``links`` entries suitable
            for ``xpkg.core.skeleton.Skeleton.from_dict``. Symmetry edges are
            dropped, edge types resolved via ``py/reduce`` or ``py/id`` payloads,
            and missing/invalid edge-type payloads fall back to BODY links.
    """

    node_id_to_name: dict[int, str] = {}
    if isinstance(metadata, Mapping):
        nodes = metadata.get("nodes") or []

    for idx, node in enumerate(nodes):
        if isinstance(node, Mapping):
            raw_name = node.get("name")
        else:
            raw_name = None
        if not raw_name:
            continue
        if isinstance(raw_name, int | float):
            name = str(int(raw_name))
        else:
            name = str(raw_name)
        if name:
            node_id_to_name[idx] = name

    skeletons = metadata.get("skeletons") if isinstance(metadata, Mapping) else None
    skeleton0 = skeletons[0] if isinstance(skeletons, Sequence) and skeletons else {}

    nodes_order = _extract_node_order(skeleton0)
    if nodes_order and all(idx in node_id_to_name for idx in nodes_order):
        keypoints = [node_id_to_name[idx] for idx in nodes_order]
    else:
        keypoints = [node_id_to_name[idx] for idx in sorted(node_id_to_name)]

    links = _extract_links(skeleton0, node_id_to_name)

    return {"name": skeleton_name, "keypoints": keypoints, "links": links}


def _extract_node_order(skeleton: Any) -> list[int]:
    order: list[int] = []
    if not isinstance(skeleton, Mapping):
        return order
    raw_nodes = skeleton.get("nodes") or []
    for entry in raw_nodes:
        node_id = _extract_int_id(entry)
        if node_id is not None:
            order.append(node_id)
    return order


def _extract_links(skeleton: Any, id_to_name: Mapping[int, str]) -> list[list[str]]:
    links: list[list[str]] = []
    if not isinstance(skeleton, Mapping):
        return links
    raw_links = skeleton.get("links") or []

    etype_id_to_code: dict[int, int] = {}
    next_type_id = 1

    for link in raw_links:
        if not isinstance(link, Mapping):
            continue

        et_code, next_type_id = _edge_type_code(link.get("type"), etype_id_to_code, next_type_id)
        if et_code == EDGE_TYPE_SYMMETRY:
            continue

        src_id = _extract_int_id(link.get("source"))
        dst_id = _extract_int_id(link.get("target"))
        if src_id is None or dst_id is None:
            continue
        if src_id not in id_to_name or dst_id not in id_to_name:
            continue
        links.append([id_to_name[src_id], id_to_name[dst_id]])

    return links


def _extract_int_id(value: Any) -> int | None:
    hops = 0
    current = value
    while isinstance(current, Mapping) and "id" in current and hops < 4:
        current = current.get("id")
        hops += 1
    if isinstance(current, int | str) and str(current).isdigit():
        return int(current)
    return None


def _edge_type_code(
    type_payload: Any, etype_id_to_code: dict[int, int], next_type_id: int
) -> tuple[int, int]:
    code = EDGE_TYPE_BODY

    if isinstance(type_payload, Mapping):
        if "py/reduce" in type_payload:
            reduce_payload = type_payload.get("py/reduce") or []
            for item in reduce_payload:
                if not isinstance(item, Mapping):
                    continue
                if "py/tuple" in item:
                    tup = item["py/tuple"]
                    if isinstance(tup, Sequence) and tup:
                        code = _require_int_payload(
                            tup[0], error_message="Invalid edge type definition in SLEAP metadata"
                        )
                        etype_id_to_code[next_type_id] = code
                        next_type_id += 1
                        break
        elif "py/id" in type_payload:
            ref = _require_int_payload(
                type_payload["py/id"], error_message="Invalid py/id in SLEAP metadata"
            )
            if ref not in etype_id_to_code:
                logger.error("Unknown edge type reference: %s", ref)
                raise ValueError(f"Unknown edge type reference: {ref}")
            code = int(etype_id_to_code[ref])

    if code not in (EDGE_TYPE_BODY, EDGE_TYPE_SYMMETRY):
        logger.error("Unknown edge type code: %s", code)
        raise ValueError(f"Unknown edge type code: {code}")
    return code, next_type_id


def _require_int_payload(value: Any, *, error_message: str) -> int:
    if isinstance(value, bool):
        logger.error(error_message)
        raise TypeError(error_message)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
    logger.error(error_message)
    raise ValueError(error_message)


def detect_yaml_skeleton_format(path: str | Path) -> YamlSkeletonFormat:
    """Classify a YAML skeleton file before dispatching to a concrete loader."""
    return _classify_yaml_skeleton_mapping(load_yaml_file(Path(path)))


def _classify_yaml_skeleton_mapping(data: Mapping[str, Any]) -> YamlSkeletonFormat:
    if "kpt_shape" in data or "keypoint_names" in data or "kpt_names" in data:
        return "ultralytics"
    if "nodes" in data and isinstance(data.get("nodes"), list):
        return "sleap"
    if "bodyparts" in data:
        return "dlc"
    raise ValueError(
        "Unsupported YAML skeleton format; expected DLC, SLEAP, or Ultralytics schema."
    )


def detect_skeleton_format(path: str | Path) -> SkeletonFormat:
    """Classify a supported skeleton file path before loading."""
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".json":
        return "xpkg_json"
    if ext == ".slp":
        if str(path).lower().endswith(".pkg.slp"):
            return "sleap_pkg_slp"
        raise ValueError("Only .pkg.slp SLEAP packages are supported for skeleton imports.")
    if ext in {".h5", ".hdf5"}:
        raise ValueError("SLEAP .h5 tracking files are not supported; export a .pkg.slp instead.")
    if ext in {".yaml", ".yml"}:
        return detect_yaml_skeleton_format(path)
    raise ValueError(f"Unsupported skeleton file: {ext}")


def load_skeleton(path: str | Path) -> Skeleton:
    """Auto-detect format and load skeleton into xpkg.core.skeleton.Skeleton.

    Supports:
    - .json: xpkg archive JSON format
    - .pkg.slp: SLEAP package files
    - .yaml/.yml: DLC config, SLEAP, or Ultralytics format (auto-detected)

    Returns:
        Skeleton: Normalized skeleton instance.

    Raises:
        ValueError: If file format is unsupported or parsing fails.
    """
    path = Path(path)
    detected_format = detect_skeleton_format(path)

    if detected_format == "xpkg_json":
        return load_skeleton_xpkg_json(path)
    if detected_format == "sleap_pkg_slp":
        return _load_skeleton_sleap_pkg_slp(path)
    if detected_format == "sleap":
        return _load_skeleton_sleap_yaml(path)
    if detected_format == "dlc":
        return load_skeleton_dlc(path)
    if detected_format == "ultralytics":
        return load_skeleton_ultralytics(path)
    raise AssertionError(f"Unhandled skeleton format: {detected_format}")


def load_skeleton_xpkg_json(path: str | Path, **kwargs: Any) -> Skeleton:
    """Load skeleton from canonical xpkg JSON / `.xpkg`-era format.

    Args:
        path: Path to JSON file with 'keypoints' and optional 'links'.

    Returns:
        Skeleton: Normalized skeleton instance.
    """
    return Skeleton.load(Path(path), **kwargs)


def load_skeleton_archive_json(path: str | Path, **kwargs: Any) -> Skeleton:
    """Compatibility alias for `load_skeleton_xpkg_json(...)`."""
    return load_skeleton_xpkg_json(path, **kwargs)


def load_skeleton_dlc(path: str | Path) -> Skeleton:
    """Load skeleton from DeepLabCut YAML config.

    Args:
        path: Path to DLC config.yaml.

    Returns:
        Skeleton: Skeleton with keypoints and edges from DLC.

    Raises:
        ValueError: If YAML parsing or conversion fails.
    """
    path = Path(path)
    data = load_yaml_file(path)

    bodyparts = list(data.get("bodyparts", []))
    skeleton_raw = data.get("skeleton", [])

    name_to_idx = {name: i for i, name in enumerate(bodyparts)}
    links: list[list[int]] = []

    for edge in skeleton_raw:
        if isinstance(edge, list | tuple) and len(edge) >= 2:
            a, b = edge[0], edge[1]
            if isinstance(a, str) and isinstance(b, str):
                ia = name_to_idx.get(a)
                ib = name_to_idx.get(b)
                if ia is not None and ib is not None:
                    links.append([ia, ib])
            elif isinstance(a, int) and isinstance(b, int):
                links.append([a, b])

    return Skeleton.from_dict(
        {
            "name": path.stem,
            "keypoints": bodyparts,
            "links": links,
        },
        normalize_names=True,
    )


def load_skeleton_sleap(path: str | Path) -> Skeleton:
    """Load skeleton from SLEAP .pkg.slp or YAML file.

    Args:
        path: Path to SLEAP file (.pkg.slp or .yaml/.yml).

    Returns:
        Skeleton: Skeleton extracted from SLEAP format.

    Raises:
        ValueError: If file type is unsupported or parsing fails.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".slp":
        if str(path).lower().endswith(".pkg.slp"):
            return _load_skeleton_sleap_pkg_slp(path)
        raise ValueError("Only .pkg.slp SLEAP packages are supported for skeleton imports.")
    if ext in {".yaml", ".yml"}:
        return _load_skeleton_sleap_yaml(path)
    if ext in {".h5", ".hdf5"}:
        raise ValueError("SLEAP .h5 tracking files are not supported for skeleton imports.")
    raise ValueError(f"Unsupported SLEAP file extension: {ext}")


def _load_skeleton_sleap_pkg_slp(path: Path) -> Skeleton:
    """Load skeleton from SLEAP .pkg.slp file (HDF5-based) without sleap-io.

    Args:
        path: Path to .pkg.slp file.

    Returns:
        Skeleton: Skeleton from SLEAP metadata.

    Raises:
        ValueError: If metadata is missing or malformed.
    """
    with h5py.File(str(path), "r") as f:
        if "metadata" not in f:
            raise ValueError("SLEAP .pkg.slp file has no metadata group")

        metadata_attrs = f["metadata"].attrs
        if "json" not in metadata_attrs:
            raise ValueError("SLEAP metadata has no JSON attribute")

        metadata_json_str = metadata_attrs["json"]
        metadata = parse_json_dict(metadata_json_str)

    skeleton_dict = build_sleap_skeleton(metadata, skeleton_name=path.stem)
    return Skeleton.from_dict(skeleton_dict, normalize_names=True)


def _load_skeleton_sleap_yaml(path: Path) -> Skeleton:
    """Load skeleton from SLEAP YAML format.

    Args:
        path: Path to SLEAP skeleton.yaml.

    Returns:
        Skeleton: Skeleton with nodes and edges.

    Raises:
        ValueError: If YAML structure is invalid.
    """
    data = load_yaml_file(path)

    nodes = data.get("nodes", [])
    keypoints = [
        n.get("name", f"node_{i}") if isinstance(n, dict) else str(n) for i, n in enumerate(nodes)
    ]

    name_to_idx = {name: i for i, name in enumerate(keypoints)}
    links: list[list[int]] = []

    for edge in data.get("edges", []):
        if isinstance(edge, dict):
            src = edge.get("source", {})
            dst = edge.get("destination", {})
            src_name = src.get("name") if isinstance(src, dict) else None
            dst_name = dst.get("name") if isinstance(dst, dict) else None
            if src_name and dst_name:
                ia = name_to_idx.get(src_name)
                ib = name_to_idx.get(dst_name)
                if ia is not None and ib is not None:
                    links.append([ia, ib])

    return Skeleton.from_dict(
        {
            "name": data.get("name", path.stem),
            "keypoints": keypoints,
            "links": links,
        },
        normalize_names=True,
    )


def load_skeleton_ultralytics(path: str | Path) -> Skeleton:
    """Load skeleton from Ultralytics/YOLO pose YAML config.

    Args:
        path: Path to Ultralytics YAML config.

    Returns:
        Skeleton: Skeleton with keypoints and skeleton edges.

    Raises:
        ValueError: If YAML structure is invalid.
    """
    path = Path(path)
    data = load_yaml_file(path)

    keypoints = list(data.get("keypoint_names", data.get("kpt_names", [])))

    if not keypoints:
        kpt_shape = data.get("kpt_shape", [])
        if isinstance(kpt_shape, list) and len(kpt_shape) >= 1:
            num_kpts = int(kpt_shape[0])
            keypoints = [f"keypoint_{i}" for i in range(num_kpts)]

    links: list[list[int]] = []
    for edge in data.get("skeleton", []):
        if isinstance(edge, list | tuple) and len(edge) >= 2:
            links.append([int(edge[0]), int(edge[1])])

    return Skeleton.from_dict(
        {
            "name": path.stem,
            "keypoints": keypoints,
            "links": links,
        },
        normalize_names=True,
    )


__all__ = [
    "detect_skeleton_format",
    "detect_yaml_skeleton_format",
    "load_skeleton",
    "load_skeleton_dlc",
    "load_skeleton_xpkg_json",
    "load_skeleton_archive_json",
    "load_skeleton_sleap",
    "load_skeleton_ultralytics",
]
