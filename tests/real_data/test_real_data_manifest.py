from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from xpkg.services import ProjectService

ROOT_ENV = "XPKG_REAL_DATA_ROOT"
MANIFEST_ENV = "XPKG_REAL_DATA_MANIFEST"
DEFAULT_MANIFEST = "xpkg-real-data.json"

pytestmark = pytest.mark.realdata


def _load_cases() -> list[dict[str, Any]]:
    root_raw = os.environ.get(ROOT_ENV)
    if not root_raw:
        return []

    root = Path(root_raw).expanduser().resolve()
    manifest_raw = os.environ.get(MANIFEST_ENV)
    manifest_path = (
        Path(manifest_raw).expanduser().resolve() if manifest_raw else root / DEFAULT_MANIFEST
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Real-data manifest not found: {manifest_path}. "
            f"Set {MANIFEST_ENV} or create {DEFAULT_MANIFEST} under {ROOT_ENV}."
        )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Real-data manifest must contain a JSON object: {manifest_path}")
    if payload.get("schema_version") != 1:
        raise ValueError(f"Real-data manifest schema_version must be 1: {manifest_path}")

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"Real-data manifest must contain a non-empty cases list: {manifest_path}")

    cases: list[dict[str, Any]] = []
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise TypeError(f"Real-data case {index} must be a JSON object: {manifest_path}")
        case = dict(raw_case)
        case["_root"] = root
        cases.append(case)
    return cases


def _case_id(case: Mapping[str, Any]) -> str:
    raw_id = case.get("id")
    if not isinstance(raw_id, str) or not raw_id.strip():
        raise ValueError("Each real-data case must define a non-empty string id.")
    return raw_id.strip()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "real-data"


def _case_kind(case: Mapping[str, Any]) -> str:
    raw_kind = case.get("kind")
    if not isinstance(raw_kind, str) or not raw_kind.strip():
        raise ValueError(f"Real-data case {_case_id(case)!r} must define a non-empty string kind.")
    return raw_kind.strip().lower()


def _case_root(case: Mapping[str, Any]) -> Path:
    root = case.get("_root")
    if not isinstance(root, Path):
        raise TypeError(f"Real-data case {_case_id(case)!r} is missing its resolved root.")
    return root


def _case_path(case: Mapping[str, Any], field: str) -> Path:
    value = case.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Real-data case {_case_id(case)!r} must define {field!r}.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = _case_root(case) / path
    return path.resolve()


def _require_existing_path(case: Mapping[str, Any], field: str) -> Path:
    path = _case_path(case, field)
    if not path.exists():
        raise FileNotFoundError(
            f"Real-data case {_case_id(case)!r} references missing {field}: {path}"
        )
    return path


def _expect(case: Mapping[str, Any]) -> Mapping[str, Any]:
    value = case.get("expect", {})
    if not isinstance(value, dict):
        raise TypeError(f"Real-data case {_case_id(case)!r} expect must be a JSON object.")
    return value


def _optional_str(case: Mapping[str, Any], field: str, default: str) -> str:
    value = case.get(field, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Real-data case {_case_id(case)!r} field {field!r} must be a string.")
    return value


def _optional_float(case: Mapping[str, Any], field: str, default: float) -> float:
    value = case.get(field, default)
    if not isinstance(value, int | float):
        raise TypeError(f"Real-data case {_case_id(case)!r} field {field!r} must be numeric.")
    return float(value)


def _optional_int(case: Mapping[str, Any], field: str, default: int) -> int:
    value = case.get(field, default)
    if not isinstance(value, int):
        raise TypeError(f"Real-data case {_case_id(case)!r} field {field!r} must be an integer.")
    return int(value)


def _optional_bool(case: Mapping[str, Any], field: str, default: bool) -> bool:
    value = case.get(field, default)
    if not isinstance(value, bool):
        raise TypeError(f"Real-data case {_case_id(case)!r} field {field!r} must be a boolean.")
    return bool(value)


def _import_case(case: Mapping[str, Any], project: ProjectService) -> None:
    kind = _case_kind(case)
    skeleton_name = _optional_str(case, "skeleton_name", "imported")
    threshold = _optional_float(case, "threshold", 0.0)

    if kind == "dlc":
        _import_dlc_case(case, project, skeleton_name=skeleton_name, threshold=threshold)
        return
    if kind in {"lightning_pose", "lightning-pose", "lightningpose"}:
        _import_lightning_pose_case(
            case,
            project,
            skeleton_name=skeleton_name,
            threshold=threshold,
        )
        return
    if kind == "sleap":
        _import_sleap_case(case, project, skeleton_name=skeleton_name, threshold=threshold)
        return
    if kind == "mmpose":
        project.import_pose(
            "mmpose-topdown-json",
            path=_require_existing_path(case, "json"),
            video=_require_existing_path(case, "video"),
            skeleton_name=skeleton_name,
            instance_index=_optional_int(case, "instance_index", 0),
            likelihood_threshold=threshold,
        )
        return
    if kind == "mediapipe":
        project.import_pose(
            "mediapipe-pose-landmarks-json",
            path=_require_existing_path(case, "json"),
            video=_require_existing_path(case, "video"),
            skeleton_name=skeleton_name,
            likelihood_threshold=threshold,
        )
        return
    raise ValueError(
        f"Unsupported real-data case kind {kind!r}. Expected one of "
        "'dlc', 'lightning_pose', 'sleap', 'mmpose', or 'mediapipe'."
    )


def _import_dlc_case(
    case: Mapping[str, Any],
    project: ProjectService,
    *,
    skeleton_name: str,
    threshold: float,
) -> None:
    has_project = isinstance(case.get("project"), str)
    has_tracking = isinstance(case.get("tracking"), str)
    if has_project == has_tracking:
        raise ValueError(
            f"Real-data DLC case {_case_id(case)!r} must define exactly one of "
            "'project' or 'tracking'."
        )

    if has_project:
        raw_skeleton_name = case.get("skeleton_name")
        project.import_pose(
            "dlc-project",
            path=_require_existing_path(case, "project"),
            skeleton_name=raw_skeleton_name if isinstance(raw_skeleton_name, str) else None,
            likelihood_threshold=threshold,
        )
        return

    tracking_path = _require_existing_path(case, "tracking")
    video_path = _require_existing_path(case, "video")
    suffix = tracking_path.suffix.lower()
    if suffix == ".csv":
        project.import_pose(
            "dlc-csv",
            path=tracking_path,
            video=video_path,
            skeleton_name=skeleton_name,
            likelihood_threshold=threshold,
        )
        return
    if suffix in {".h5", ".hdf5"}:
        project.import_pose(
            "dlc-h5",
            path=tracking_path,
            video=video_path,
            skeleton_name=skeleton_name,
            likelihood_threshold=threshold,
        )
        return
    raise ValueError(
        f"Real-data DLC case {_case_id(case)!r} tracking file must be .csv, .h5, "
        f"or .hdf5: {tracking_path}"
    )


def _import_sleap_case(
    case: Mapping[str, Any],
    project: ProjectService,
    *,
    skeleton_name: str,
    threshold: float,
) -> None:
    labels_path = _require_existing_path(case, "labels")
    suffixes = tuple(suffix.lower() for suffix in labels_path.suffixes)
    if labels_path.name.lower().endswith(".pkg.slp") or labels_path.suffix.lower() == ".slp":
        raw_encode_videos = case.get("encode_videos")
        project.import_pose(
            "sleap-package",
            path=labels_path,
            fps=_optional_int(case, "fps", 30),
            encode_videos=raw_encode_videos if isinstance(raw_encode_videos, bool) else None,
        )
        return
    if suffixes and suffixes[-1] in {".h5", ".hdf5"}:
        project.import_pose(
            "sleap-h5",
            path=labels_path,
            video=_require_existing_path(case, "video"),
            skeleton_name=skeleton_name,
            likelihood_threshold=threshold,
        )
        return
    raise ValueError(
        f"Real-data SLEAP case {_case_id(case)!r} labels file must be .slp, "
        f".pkg.slp, .h5, or .hdf5: {labels_path}"
    )


def _import_lightning_pose_case(
    case: Mapping[str, Any],
    project: ProjectService,
    *,
    skeleton_name: str,
    threshold: float,
) -> None:
    tracking_path = _require_existing_path(case, "tracking")
    if tracking_path.suffix.lower() != ".csv":
        raise ValueError(
            f"Real-data Lightning Pose case {_case_id(case)!r} tracking file must be "
            f".csv: {tracking_path}"
        )
    project.import_pose(
        "lightning-pose-csv",
        path=tracking_path,
        video=_require_existing_path(case, "video"),
        skeleton_name=skeleton_name,
        likelihood_threshold=threshold,
    )


def _assert_equal_if_present(actual: int, expected: Mapping[str, Any], key: str) -> None:
    value = expected.get(key)
    if value is None:
        return
    if not isinstance(value, int):
        raise TypeError(f"Expected value {key!r} must be an integer.")
    assert actual == value


def _assert_at_least_if_present(actual: int, expected: Mapping[str, Any], key: str) -> None:
    value = expected.get(key)
    if value is None:
        return
    if not isinstance(value, int):
        raise TypeError(f"Expected value {key!r} must be an integer.")
    assert actual >= value


def _assert_label_expectations(project: ProjectService, expected: Mapping[str, Any]) -> None:
    labels = project.load_labels()
    _assert_equal_if_present(len(labels.videos), expected, "videos")
    _assert_equal_if_present(len(labels.skeletons), expected, "skeletons")
    _assert_equal_if_present(len(labels.labeled_frames), expected, "labeled_frames")
    _assert_at_least_if_present(len(labels.labeled_frames), expected, "min_labeled_frames")
    if labels.skeletons:
        _assert_equal_if_present(len(labels.skeletons[0].keypoint_names), expected, "keypoints")


def _assert_case_expectations(case: Mapping[str, Any], project: ProjectService) -> None:
    expected = _expect(case)
    expected_state = expected.get("state", "labels")
    if expected_state == "labels":
        _assert_label_expectations(project, expected)
        return
    raise ValueError(
        f"Real-data case {_case_id(case)!r} expected state must be 'labels'."
    )


@pytest.mark.parametrize("case", _load_cases(), ids=_case_id)
def test_real_data_import_validate_pack_roundtrip(case: Mapping[str, Any], tmp_path: Path) -> None:
    case_id = _case_id(case)
    project = ProjectService.create(
        tmp_path / f"{_safe_name(case_id)}-project",
        title=case_id,
    )

    _import_case(case, project)

    layout = project.validate()
    assert layout.has_current_state
    _assert_case_expectations(case, project)

    if _optional_bool(case, "skip_pack", False):
        return

    artifact = project.pack(out=tmp_path / f"{_safe_name(case_id)}.expkg")
    assert artifact.is_file()

    unpacked = ProjectService.unpack(artifact, tmp_path / f"{_safe_name(case_id)}-unpacked")
    unpacked.validate()
    _assert_case_expectations(case, unpacked)
