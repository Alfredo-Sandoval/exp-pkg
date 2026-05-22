from __future__ import annotations

import subprocess
import sys

import pytest

import xpkg.model as model
from xpkg.model import (
    IDENTITY_PROVENANCE_SCHEMA_VERSION,
    IdentityConfidenceSpan,
    IdentityEvent,
    IdentityProofreadingSpan,
    IdentityProvenanceRecord,
)
from xpkg.model.identity import identity_provenance_payload, identity_provenance_records


def test_identity_provenance_record_round_trips_json_payload() -> None:
    record = IdentityProvenanceRecord(
        track_id="0",
        track_name="mouse-a",
        source_tool="sleap",
        source_file="analysis.h5",
        identity_source="unknown",
        spans=(
            IdentityConfidenceSpan(
                video_id="video_0",
                start_frame=0,
                end_frame=99,
                identity_source="mot",
                confidence=0.91,
            ),
        ),
        events=(
            IdentityEvent(
                kind="identity_swap",
                video_id="video_0",
                frame=100,
                from_track_id="0",
                to_track_id="1",
            ),
        ),
        proofreading=(
            IdentityProofreadingSpan(
                video_id="video_0",
                start_frame=0,
                end_frame=120,
                reviewed=True,
                corrected=False,
                reviewer="manual",
            ),
        ),
        metadata={"source_track_index": 0},
    )

    payload = identity_provenance_payload([record])
    assert payload is not None
    assert payload["schema"] == IDENTITY_PROVENANCE_SCHEMA_VERSION
    assert identity_provenance_records(payload) == [record]
    assert IdentityProvenanceRecord.from_dict(record.to_dict()) == record


def test_identity_provenance_rejects_invalid_sources_and_frame_spans() -> None:
    with pytest.raises(ValueError, match="identity_source"):
        IdentityProvenanceRecord(track_id="0", identity_source="classifier")

    with pytest.raises(ValueError, match="end_frame"):
        IdentityConfidenceSpan(start_frame=10, end_frame=9)

    with pytest.raises(ValueError, match="between 0 and 1"):
        IdentityConfidenceSpan(start_frame=0, end_frame=1, confidence=1.5)


def test_identity_models_are_available_from_public_surface() -> None:
    assert model.IdentityProvenanceRecord is IdentityProvenanceRecord
    assert model.IdentityConfidenceSpan is IdentityConfidenceSpan


def test_labels_model_imports_without_identity_cycle() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from xpkg.io.labels.model import Labels; "
                "from xpkg.model.identity import IdentityProvenanceRecord; "
                "assert Labels and IdentityProvenanceRecord"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stderr == ""
