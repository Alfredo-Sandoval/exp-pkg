from __future__ import annotations

from pathlib import Path

import pytest

from xpkg.pose.csv_matching import (
    DEFAULT_RAT_SKELETON_EDGES,
    expected_pose_csv_name,
    find_matching_pose_csvs,
    normalize_pose_session_name,
    pose_csv_hint_for_media,
    pose_csv_matches_media,
    pose_csv_session_name,
    resolve_skeleton_edges,
)


def test_pose_csv_session_name_accepts_prediction_and_plain_csv_names() -> None:
    assert pose_csv_session_name(Path("RAT_1__predictions.csv")) == "RAT_1"
    assert pose_csv_session_name(Path("FR1_15min_4_5_2026_RAT_10.csv")) == (
        "FR1_15min_4_5_2026_RAT_10"
    )


def test_pose_csv_matching_normalizes_bundle_and_labeled_video_suffixes() -> None:
    media_path = Path("RAT_5_FR1_15MIN_04_06_2026_labeled.mp4")
    pose_csv_path = Path("RAT_5_FR1_15MIN_04_06_2026-002__predictions.csv")

    assert normalize_pose_session_name("RAT_5_FR1_15MIN_04_06_2026-002") == (
        "RAT_5_FR1_15MIN_04_06_2026"
    )
    assert pose_csv_matches_media(pose_csv_path, media_path)
    assert expected_pose_csv_name(media_path) == (
        "RAT_5_FR1_15MIN_04_06_2026_labeled__predictions.csv"
    )
    assert pose_csv_hint_for_media(media_path) == (
        "RAT_5_FR1_15MIN_04_06_2026_labeled__predictions.csv "
        "or csv/RAT_5_FR1_15MIN_04_06_2026.csv"
    )


def test_find_matching_pose_csvs_returns_only_matching_files(tmp_path: Path) -> None:
    nested = tmp_path / "csv"
    nested.mkdir()
    match = nested / "SESSION_A.csv"
    miss = nested / "SESSION_B.csv"
    match.write_text("", encoding="utf-8")
    miss.write_text("", encoding="utf-8")

    assert find_matching_pose_csvs(tmp_path, Path("SESSION_A.mkv")) == (match.resolve(),)


def test_resolve_skeleton_edges_keeps_only_present_bodyparts() -> None:
    edges = resolve_skeleton_edges(("nose", "head", "spine1", "spine2", "frontpaw", "shoulder"))

    assert ("nose", "head") in edges
    assert ("head", "spine1") in edges
    assert ("spine1", "spine2") in edges
    assert ("shoulder", "frontpaw") in edges
    assert ("tail2", "tail_tip") not in edges
    assert len(edges) < len(DEFAULT_RAT_SKELETON_EDGES)


def test_resolve_skeleton_edges_accepts_spaced_bodypart_names() -> None:
    edges = resolve_skeleton_edges(("Nose", "Head", "Shoulder", "Front Paw"))

    assert ("Nose", "Head") in edges
    assert ("Shoulder", "Front Paw") in edges


def test_resolve_skeleton_edges_rejects_empty_matches() -> None:
    with pytest.raises(ValueError, match="No requested skeleton edges"):
        resolve_skeleton_edges(("nose",), requested_edges=(("tail1", "tail2"),))
