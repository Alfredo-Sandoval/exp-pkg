"""Round-trip and validation tests for Datasheet and ModelCard schemas."""

from __future__ import annotations

import pytest

from xpkg.model import (
    DatasetDatasheet,
    DatasheetCollection,
    DatasheetComposition,
    DatasheetDistribution,
    DatasheetMaintenance,
    DatasheetMotivation,
    DatasheetPreprocessing,
    DatasheetUses,
    ModelCard,
    ModelCardAnalysis,
    ModelCardData,
    ModelCardDetails,
    ModelCardFactors,
    ModelCardIntendedUse,
    ModelCardMetrics,
)


def _full_datasheet() -> DatasetDatasheet:
    return DatasetDatasheet(
        title="Mouse reach 2026",
        dataset_id="mouse-reach-2026",
        version="1.0",
        summary="Two-camera open-field mouse reach kinematics",
        motivation=DatasheetMotivation(
            purpose="Investigate VTA stimulation effects on reach kinematics.",
            creators=("Sandoval, A.", "Doe, J."),
            funders=("NIH MH128177",),
            notes="Pilot batch.",
        ),
        composition=DatasheetComposition(
            instances="One instance per 30-min reach session, per mouse.",
            instance_count=80,
            sampling="All sessions retained; no subsampling.",
            splits={"train": "60", "val": "10", "test": "10"},
            sensitive_content="No identifying human data.",
            confidentiality="None",
        ),
        collection=DatasheetCollection(
            process="Two BFLY cameras, 120 Hz GigE PoE, IR-illuminated arena.",
            collectors=("A. Sandoval",),
            timeframe="2026-01 to 2026-04",
            ethics_review="IACUC #2025-042",
            consent="N/A",
        ),
        preprocessing=DatasheetPreprocessing(
            steps="Frames trimmed to trial windows; raw retained.",
            raw_data_preserved=True,
            software=("ffmpeg 6.1", "xpkg 0.4"),
        ),
        uses=DatasheetUses(
            intended_uses="Pose-estimation training and benchmark.",
            prior_uses=("Sandoval 2026 (in prep)",),
            out_of_scope_uses="Not validated for non-open-field arenas.",
        ),
        distribution=DatasheetDistribution(
            distribution_plan="Public release with paper.",
            license="CC-BY-4.0",
            repository_url="https://github.com/example/mouse-reach",
            doi="10.5281/zenodo.1234",
        ),
        maintenance=DatasheetMaintenance(
            maintainer="Sandoval Lab",
            contact="sandoval@example.org",
            update_policy="Annual minor revisions.",
        ),
        metadata={"internal_ref": "BATCH-2026-A"},
    )


def test_datasheet_round_trips_full_payload() -> None:
    datasheet = _full_datasheet()
    payload = datasheet.to_dict()

    assert payload["title"] == "Mouse reach 2026"
    assert payload["motivation"]["creators"] == ["Sandoval, A.", "Doe, J."]
    assert payload["composition"]["instance_count"] == 80
    assert payload["composition"]["splits"] == {"train": "60", "val": "10", "test": "10"}
    assert payload["preprocessing"]["raw_data_preserved"] is True
    assert payload["distribution"]["doi"] == "10.5281/zenodo.1234"
    assert payload["metadata"] == {"internal_ref": "BATCH-2026-A"}

    assert DatasetDatasheet.from_dict(payload) == datasheet


def test_datasheet_omits_empty_sections() -> None:
    datasheet = DatasetDatasheet(title="Empty")
    payload = datasheet.to_dict()

    assert payload == {"title": "Empty"}
    assert DatasetDatasheet.from_dict(payload) == datasheet


def test_datasheet_requires_non_empty_title() -> None:
    with pytest.raises(ValueError, match="datasheet.title"):
        DatasetDatasheet(title="")


def test_datasheet_accepts_section_payload_dicts_directly() -> None:
    datasheet = DatasetDatasheet(
        title="From dict",
        motivation={"purpose": "Smoke test"},
        composition={"instance_count": 4},
    )
    assert datasheet.motivation.purpose == "Smoke test"
    assert datasheet.composition.instance_count == 4


def test_datasheet_rejects_invalid_instance_count() -> None:
    with pytest.raises(ValueError, match="composition.instance_count"):
        DatasheetComposition(instance_count=-1)


def _full_model_card() -> ModelCard:
    return ModelCard(
        details=ModelCardDetails(
            name="dlc-mouse-reach",
            version="2.3.4",
            date="2026-04-30",
            type="Pose estimation",
            architecture="ResNet-50 + transposed conv head",
            paper="https://doi.org/10.1038/s41593-018-0209-y",
            license="CC-BY-4.0",
            contact="sandoval@example.org",
            developers=("A. Sandoval",),
        ),
        intended_use=ModelCardIntendedUse(
            primary_uses="Mouse keypoint inference on open-field reach video.",
            primary_users=("Behavior labs",),
            out_of_scope_uses="Not for multi-animal scenes.",
        ),
        factors=ModelCardFactors(
            relevant=("Strain: C57BL/6J",),
            evaluation=("Lighting: IR vs visible",),
        ),
        metrics=ModelCardMetrics(
            measures=("RMSE px", "PCK@0.05"),
            decision_thresholds={"min_confidence": "0.4"},
            variation_approach="5-fold CV across mice.",
        ),
        evaluation_data=ModelCardData(
            description="Held-out 10 sessions.",
            source="In-lab acquisition rig.",
            preprocessing="Cropped to arena ROI.",
        ),
        training_data=ModelCardData(
            description="200 manually labeled frames.",
            source="In-lab acquisition rig.",
        ),
        quantitative_analyses=ModelCardAnalysis(
            unitary_results={"rmse_px": 3.7, "pck_0_05": 0.92},
            intersectional_results={"rmse_px_by_strain": {"C57": 3.7}},
        ),
        ethical_considerations="No human data; IACUC-approved animal use.",
        caveats=("Single-animal only.", "IR lighting required."),
        recommendations=("Re-evaluate for new arena geometry.",),
        metadata={"checkpoint_id": "snapshot-200000"},
    )


def test_model_card_round_trips_full_payload() -> None:
    card = _full_model_card()
    payload = card.to_dict()

    assert payload["details"]["name"] == "dlc-mouse-reach"
    assert payload["intended_use"]["primary_users"] == ["Behavior labs"]
    assert payload["metrics"]["decision_thresholds"] == {"min_confidence": "0.4"}
    assert payload["quantitative_analyses"]["unitary_results"]["rmse_px"] == 3.7
    assert payload["caveats"] == ["Single-animal only.", "IR lighting required."]
    assert payload["metadata"] == {"checkpoint_id": "snapshot-200000"}

    assert ModelCard.from_dict(payload) == card


def test_model_card_omits_empty_sections() -> None:
    card = ModelCard(details=ModelCardDetails(name="bare"))
    payload = card.to_dict()

    assert payload == {"details": {"name": "bare"}}
    assert ModelCard.from_dict(payload) == card


def test_model_card_requires_details_section() -> None:
    with pytest.raises(ValueError, match="details"):
        ModelCard.from_dict({"intended_use": {"primary_uses": "Anything"}})


def test_model_card_details_requires_name() -> None:
    with pytest.raises(ValueError, match="details.name"):
        ModelCardDetails(name="")


def test_model_card_accepts_section_payload_dicts_directly() -> None:
    card = ModelCard(
        details={"name": "from-dict-card", "version": "0.1"},
        metrics={"measures": ["accuracy"]},
    )
    assert card.details.name == "from-dict-card"
    assert card.metrics.measures == ("accuracy",)
