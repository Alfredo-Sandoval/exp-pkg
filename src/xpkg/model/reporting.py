"""Datasheet and Model Card schemas for FAIR reporting.

Implements the structures recommended in Luxem et al. (2023) "Open-source
tools for behavioral video analysis":

- ``DatasetDatasheet`` follows the seven-section template from Gebru et al.
  (2021), "Datasheets for Datasets" (Communications of the ACM 64(12)).
- ``ModelCard`` follows the nine-section template from Mitchell et al.
  (2019), "Model Cards for Model Reporting" (FAT* '19).

Both are dependency-light frozen dataclasses with ``from_dict`` / ``to_dict``
JSON round-tripping. Sub-sections accept either typed instances or plain
mapping payloads on construction so JSON loaded from disk can be passed
through unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

from xpkg.model._metadata_validation import (
    metadata_dict as _metadata,
    optional_bool as _optional_bool,
    optional_non_negative_int as _optional_int,
    optional_text as _optional_text,
    required_text as _required_text,
    text_mapping as _text_mapping,
    text_tuple as _text_tuple,
)


_TSection = TypeVar("_TSection")


def _coerce_section(
    value: Any,
    cls: type[_TSection],
    *,
    name: str,
) -> _TSection:
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        return cls.from_dict(value)  # type: ignore[attr-defined]
    raise TypeError(
        f"{name} must be {cls.__name__} or mapping, got {type(value).__name__}."
    )


# ---------------------------------------------------------------------------
# Datasheet for Datasets (Gebru et al. 2021)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DatasheetMotivation:
    """Why the dataset was created (Gebru §3.1)."""

    purpose: str | None = None
    creators: tuple[str, ...] = ()
    funders: tuple[str, ...] = ()
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "purpose", _optional_text(self.purpose, name="motivation.purpose"))
        object.__setattr__(self, "creators", _text_tuple(self.creators, name="motivation.creators"))
        object.__setattr__(self, "funders", _text_tuple(self.funders, name="motivation.funders"))
        object.__setattr__(self, "notes", _optional_text(self.notes, name="motivation.notes"))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.purpose is not None:
            payload["purpose"] = self.purpose
        if self.creators:
            payload["creators"] = list(self.creators)
        if self.funders:
            payload["funders"] = list(self.funders)
        if self.notes is not None:
            payload["notes"] = self.notes
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DatasheetMotivation:
        if not isinstance(payload, Mapping):
            raise TypeError("datasheet motivation payload must be a mapping.")
        return cls(
            purpose=payload.get("purpose"),
            creators=_text_tuple(payload.get("creators"), name="motivation.creators"),
            funders=_text_tuple(payload.get("funders"), name="motivation.funders"),
            notes=payload.get("notes"),
        )


@dataclass(frozen=True, slots=True)
class DatasheetComposition:
    """What instances represent and how the dataset is structured (Gebru §3.2)."""

    instances: str | None = None
    instance_count: int | None = None
    sampling: str | None = None
    splits: dict[str, str] = field(default_factory=dict)
    sensitive_content: str | None = None
    confidentiality: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "instances",
            _optional_text(self.instances, name="composition.instances"),
        )
        object.__setattr__(
            self,
            "instance_count",
            _optional_int(self.instance_count, name="composition.instance_count"),
        )
        object.__setattr__(self, "sampling", _optional_text(self.sampling, name="composition.sampling"))
        object.__setattr__(self, "splits", _text_mapping(self.splits, name="composition.splits"))
        object.__setattr__(
            self,
            "sensitive_content",
            _optional_text(self.sensitive_content, name="composition.sensitive_content"),
        )
        object.__setattr__(
            self,
            "confidentiality",
            _optional_text(self.confidentiality, name="composition.confidentiality"),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, name="composition.notes"))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key in ("instances", "sampling", "sensitive_content", "confidentiality", "notes"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.instance_count is not None:
            payload["instance_count"] = self.instance_count
        if self.splits:
            payload["splits"] = dict(self.splits)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DatasheetComposition:
        if not isinstance(payload, Mapping):
            raise TypeError("datasheet composition payload must be a mapping.")
        return cls(
            instances=payload.get("instances"),
            instance_count=payload.get("instance_count"),
            sampling=payload.get("sampling"),
            splits=_text_mapping(payload.get("splits"), name="composition.splits"),
            sensitive_content=payload.get("sensitive_content"),
            confidentiality=payload.get("confidentiality"),
            notes=payload.get("notes"),
        )


@dataclass(frozen=True, slots=True)
class DatasheetCollection:
    """How the dataset was collected (Gebru §3.3)."""

    process: str | None = None
    collectors: tuple[str, ...] = ()
    timeframe: str | None = None
    ethics_review: str | None = None
    consent: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "process", _optional_text(self.process, name="collection.process"))
        object.__setattr__(
            self,
            "collectors",
            _text_tuple(self.collectors, name="collection.collectors"),
        )
        object.__setattr__(
            self,
            "timeframe",
            _optional_text(self.timeframe, name="collection.timeframe"),
        )
        object.__setattr__(
            self,
            "ethics_review",
            _optional_text(self.ethics_review, name="collection.ethics_review"),
        )
        object.__setattr__(self, "consent", _optional_text(self.consent, name="collection.consent"))
        object.__setattr__(self, "notes", _optional_text(self.notes, name="collection.notes"))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key in ("process", "timeframe", "ethics_review", "consent", "notes"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.collectors:
            payload["collectors"] = list(self.collectors)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DatasheetCollection:
        if not isinstance(payload, Mapping):
            raise TypeError("datasheet collection payload must be a mapping.")
        return cls(
            process=payload.get("process"),
            collectors=_text_tuple(payload.get("collectors"), name="collection.collectors"),
            timeframe=payload.get("timeframe"),
            ethics_review=payload.get("ethics_review"),
            consent=payload.get("consent"),
            notes=payload.get("notes"),
        )


@dataclass(frozen=True, slots=True)
class DatasheetPreprocessing:
    """Preprocessing, cleaning, and labeling (Gebru §3.4)."""

    steps: str | None = None
    raw_data_preserved: bool | None = None
    software: tuple[str, ...] = ()
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", _optional_text(self.steps, name="preprocessing.steps"))
        object.__setattr__(
            self,
            "raw_data_preserved",
            _optional_bool(self.raw_data_preserved, name="preprocessing.raw_data_preserved"),
        )
        object.__setattr__(
            self,
            "software",
            _text_tuple(self.software, name="preprocessing.software"),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, name="preprocessing.notes"))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.steps is not None:
            payload["steps"] = self.steps
        if self.raw_data_preserved is not None:
            payload["raw_data_preserved"] = self.raw_data_preserved
        if self.software:
            payload["software"] = list(self.software)
        if self.notes is not None:
            payload["notes"] = self.notes
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DatasheetPreprocessing:
        if not isinstance(payload, Mapping):
            raise TypeError("datasheet preprocessing payload must be a mapping.")
        return cls(
            steps=payload.get("steps"),
            raw_data_preserved=payload.get("raw_data_preserved"),
            software=_text_tuple(payload.get("software"), name="preprocessing.software"),
            notes=payload.get("notes"),
        )


@dataclass(frozen=True, slots=True)
class DatasheetUses:
    """Intended and out-of-scope uses (Gebru §3.5)."""

    intended_uses: str | None = None
    prior_uses: tuple[str, ...] = ()
    out_of_scope_uses: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "intended_uses",
            _optional_text(self.intended_uses, name="uses.intended_uses"),
        )
        object.__setattr__(
            self,
            "prior_uses",
            _text_tuple(self.prior_uses, name="uses.prior_uses"),
        )
        object.__setattr__(
            self,
            "out_of_scope_uses",
            _optional_text(self.out_of_scope_uses, name="uses.out_of_scope_uses"),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, name="uses.notes"))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key in ("intended_uses", "out_of_scope_uses", "notes"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.prior_uses:
            payload["prior_uses"] = list(self.prior_uses)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DatasheetUses:
        if not isinstance(payload, Mapping):
            raise TypeError("datasheet uses payload must be a mapping.")
        return cls(
            intended_uses=payload.get("intended_uses"),
            prior_uses=_text_tuple(payload.get("prior_uses"), name="uses.prior_uses"),
            out_of_scope_uses=payload.get("out_of_scope_uses"),
            notes=payload.get("notes"),
        )


@dataclass(frozen=True, slots=True)
class DatasheetDistribution:
    """Distribution and licensing (Gebru §3.6)."""

    distribution_plan: str | None = None
    license: str | None = None
    repository_url: str | None = None
    doi: str | None = None
    embargo: str | None = None
    ip_terms: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "distribution_plan",
            _optional_text(self.distribution_plan, name="distribution.distribution_plan"),
        )
        object.__setattr__(self, "license", _optional_text(self.license, name="distribution.license"))
        object.__setattr__(
            self,
            "repository_url",
            _optional_text(self.repository_url, name="distribution.repository_url"),
        )
        object.__setattr__(self, "doi", _optional_text(self.doi, name="distribution.doi"))
        object.__setattr__(self, "embargo", _optional_text(self.embargo, name="distribution.embargo"))
        object.__setattr__(
            self,
            "ip_terms",
            _optional_text(self.ip_terms, name="distribution.ip_terms"),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, name="distribution.notes"))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key in (
            "distribution_plan",
            "license",
            "repository_url",
            "doi",
            "embargo",
            "ip_terms",
            "notes",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DatasheetDistribution:
        if not isinstance(payload, Mapping):
            raise TypeError("datasheet distribution payload must be a mapping.")
        return cls(
            distribution_plan=payload.get("distribution_plan"),
            license=payload.get("license"),
            repository_url=payload.get("repository_url"),
            doi=payload.get("doi"),
            embargo=payload.get("embargo"),
            ip_terms=payload.get("ip_terms"),
            notes=payload.get("notes"),
        )


@dataclass(frozen=True, slots=True)
class DatasheetMaintenance:
    """Maintenance, errata, and contact (Gebru §3.7)."""

    maintainer: str | None = None
    contact: str | None = None
    erratum: str | None = None
    update_policy: str | None = None
    contribution_policy: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "maintainer",
            _optional_text(self.maintainer, name="maintenance.maintainer"),
        )
        object.__setattr__(self, "contact", _optional_text(self.contact, name="maintenance.contact"))
        object.__setattr__(self, "erratum", _optional_text(self.erratum, name="maintenance.erratum"))
        object.__setattr__(
            self,
            "update_policy",
            _optional_text(self.update_policy, name="maintenance.update_policy"),
        )
        object.__setattr__(
            self,
            "contribution_policy",
            _optional_text(self.contribution_policy, name="maintenance.contribution_policy"),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, name="maintenance.notes"))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key in (
            "maintainer",
            "contact",
            "erratum",
            "update_policy",
            "contribution_policy",
            "notes",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DatasheetMaintenance:
        if not isinstance(payload, Mapping):
            raise TypeError("datasheet maintenance payload must be a mapping.")
        return cls(
            maintainer=payload.get("maintainer"),
            contact=payload.get("contact"),
            erratum=payload.get("erratum"),
            update_policy=payload.get("update_policy"),
            contribution_policy=payload.get("contribution_policy"),
            notes=payload.get("notes"),
        )


@dataclass(frozen=True, slots=True)
class DatasetDatasheet:
    """Datasheet for Datasets (Gebru et al. 2021) for an exp-pkg dataset.

    The datasheet wraps the seven Gebru sections plus a small set of
    identifying fields (``title``, ``dataset_id``, ``version``, ``summary``).
    Each section is optional; an empty section serializes to an absent key
    so a partially-filled datasheet round-trips losslessly.
    """

    title: str
    dataset_id: str | None = None
    version: str | None = None
    summary: str | None = None
    motivation: DatasheetMotivation = field(default_factory=DatasheetMotivation)
    composition: DatasheetComposition = field(default_factory=DatasheetComposition)
    collection: DatasheetCollection = field(default_factory=DatasheetCollection)
    preprocessing: DatasheetPreprocessing = field(default_factory=DatasheetPreprocessing)
    uses: DatasheetUses = field(default_factory=DatasheetUses)
    distribution: DatasheetDistribution = field(default_factory=DatasheetDistribution)
    maintenance: DatasheetMaintenance = field(default_factory=DatasheetMaintenance)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "title", _required_text(self.title, name="datasheet.title"))
        object.__setattr__(
            self, "dataset_id", _optional_text(self.dataset_id, name="datasheet.dataset_id")
        )
        object.__setattr__(self, "version", _optional_text(self.version, name="datasheet.version"))
        object.__setattr__(self, "summary", _optional_text(self.summary, name="datasheet.summary"))
        object.__setattr__(
            self,
            "motivation",
            _coerce_section(self.motivation, DatasheetMotivation, name="datasheet.motivation"),
        )
        object.__setattr__(
            self,
            "composition",
            _coerce_section(self.composition, DatasheetComposition, name="datasheet.composition"),
        )
        object.__setattr__(
            self,
            "collection",
            _coerce_section(self.collection, DatasheetCollection, name="datasheet.collection"),
        )
        object.__setattr__(
            self,
            "preprocessing",
            _coerce_section(
                self.preprocessing, DatasheetPreprocessing, name="datasheet.preprocessing"
            ),
        )
        object.__setattr__(
            self,
            "uses",
            _coerce_section(self.uses, DatasheetUses, name="datasheet.uses"),
        )
        object.__setattr__(
            self,
            "distribution",
            _coerce_section(self.distribution, DatasheetDistribution, name="datasheet.distribution"),
        )
        object.__setattr__(
            self,
            "maintenance",
            _coerce_section(self.maintenance, DatasheetMaintenance, name="datasheet.maintenance"),
        )
        object.__setattr__(
            self, "metadata", _metadata(self.metadata, name="datasheet.metadata")
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"title": self.title}
        for key in ("dataset_id", "version", "summary"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        for key in (
            "motivation",
            "composition",
            "collection",
            "preprocessing",
            "uses",
            "distribution",
            "maintenance",
        ):
            section_payload = getattr(self, key).to_dict()
            if section_payload:
                payload[key] = section_payload
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DatasetDatasheet:
        if not isinstance(payload, Mapping):
            raise TypeError("datasheet payload must be a mapping.")
        raw_metadata = payload.get("metadata")
        if raw_metadata is not None and not isinstance(raw_metadata, Mapping):
            raise TypeError("datasheet metadata must be a mapping when present.")
        return cls(
            title=payload.get("title", ""),
            dataset_id=payload.get("dataset_id"),
            version=payload.get("version"),
            summary=payload.get("summary"),
            motivation=DatasheetMotivation.from_dict(payload.get("motivation") or {}),
            composition=DatasheetComposition.from_dict(payload.get("composition") or {}),
            collection=DatasheetCollection.from_dict(payload.get("collection") or {}),
            preprocessing=DatasheetPreprocessing.from_dict(payload.get("preprocessing") or {}),
            uses=DatasheetUses.from_dict(payload.get("uses") or {}),
            distribution=DatasheetDistribution.from_dict(payload.get("distribution") or {}),
            maintenance=DatasheetMaintenance.from_dict(payload.get("maintenance") or {}),
            metadata=_metadata(raw_metadata, name="datasheet.metadata"),
        )


# ---------------------------------------------------------------------------
# Model Card (Mitchell et al. 2019)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelCardDetails:
    """Identifying details for the model (Mitchell §4.1)."""

    name: str
    version: str | None = None
    date: str | None = None
    type: str | None = None
    architecture: str | None = None
    paper: str | None = None
    license: str | None = None
    contact: str | None = None
    developers: tuple[str, ...] = ()
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required_text(self.name, name="details.name"))
        object.__setattr__(self, "version", _optional_text(self.version, name="details.version"))
        object.__setattr__(self, "date", _optional_text(self.date, name="details.date"))
        object.__setattr__(self, "type", _optional_text(self.type, name="details.type"))
        object.__setattr__(
            self,
            "architecture",
            _optional_text(self.architecture, name="details.architecture"),
        )
        object.__setattr__(self, "paper", _optional_text(self.paper, name="details.paper"))
        object.__setattr__(self, "license", _optional_text(self.license, name="details.license"))
        object.__setattr__(self, "contact", _optional_text(self.contact, name="details.contact"))
        object.__setattr__(
            self,
            "developers",
            _text_tuple(self.developers, name="details.developers"),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, name="details.notes"))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name}
        for key in (
            "version",
            "date",
            "type",
            "architecture",
            "paper",
            "license",
            "contact",
            "notes",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.developers:
            payload["developers"] = list(self.developers)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelCardDetails:
        if not isinstance(payload, Mapping):
            raise TypeError("model card details payload must be a mapping.")
        return cls(
            name=payload.get("name", ""),
            version=payload.get("version"),
            date=payload.get("date"),
            type=payload.get("type"),
            architecture=payload.get("architecture"),
            paper=payload.get("paper"),
            license=payload.get("license"),
            contact=payload.get("contact"),
            developers=_text_tuple(payload.get("developers"), name="details.developers"),
            notes=payload.get("notes"),
        )


@dataclass(frozen=True, slots=True)
class ModelCardIntendedUse:
    """Intended and out-of-scope uses (Mitchell §4.2)."""

    primary_uses: str | None = None
    primary_users: tuple[str, ...] = ()
    out_of_scope_uses: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "primary_uses",
            _optional_text(self.primary_uses, name="intended_use.primary_uses"),
        )
        object.__setattr__(
            self,
            "primary_users",
            _text_tuple(self.primary_users, name="intended_use.primary_users"),
        )
        object.__setattr__(
            self,
            "out_of_scope_uses",
            _optional_text(self.out_of_scope_uses, name="intended_use.out_of_scope_uses"),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, name="intended_use.notes"))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key in ("primary_uses", "out_of_scope_uses", "notes"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.primary_users:
            payload["primary_users"] = list(self.primary_users)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelCardIntendedUse:
        if not isinstance(payload, Mapping):
            raise TypeError("model card intended_use payload must be a mapping.")
        return cls(
            primary_uses=payload.get("primary_uses"),
            primary_users=_text_tuple(
                payload.get("primary_users"), name="intended_use.primary_users"
            ),
            out_of_scope_uses=payload.get("out_of_scope_uses"),
            notes=payload.get("notes"),
        )


@dataclass(frozen=True, slots=True)
class ModelCardFactors:
    """Relevant evaluation factors (Mitchell §4.3)."""

    relevant: tuple[str, ...] = ()
    evaluation: tuple[str, ...] = ()
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "relevant", _text_tuple(self.relevant, name="factors.relevant")
        )
        object.__setattr__(
            self, "evaluation", _text_tuple(self.evaluation, name="factors.evaluation")
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, name="factors.notes"))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.relevant:
            payload["relevant"] = list(self.relevant)
        if self.evaluation:
            payload["evaluation"] = list(self.evaluation)
        if self.notes is not None:
            payload["notes"] = self.notes
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelCardFactors:
        if not isinstance(payload, Mapping):
            raise TypeError("model card factors payload must be a mapping.")
        return cls(
            relevant=_text_tuple(payload.get("relevant"), name="factors.relevant"),
            evaluation=_text_tuple(payload.get("evaluation"), name="factors.evaluation"),
            notes=payload.get("notes"),
        )


@dataclass(frozen=True, slots=True)
class ModelCardMetrics:
    """Performance metrics, thresholds, and variation approach (Mitchell §4.4)."""

    measures: tuple[str, ...] = ()
    decision_thresholds: dict[str, str] = field(default_factory=dict)
    variation_approach: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "measures", _text_tuple(self.measures, name="metrics.measures")
        )
        object.__setattr__(
            self,
            "decision_thresholds",
            _text_mapping(self.decision_thresholds, name="metrics.decision_thresholds"),
        )
        object.__setattr__(
            self,
            "variation_approach",
            _optional_text(self.variation_approach, name="metrics.variation_approach"),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, name="metrics.notes"))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.measures:
            payload["measures"] = list(self.measures)
        if self.decision_thresholds:
            payload["decision_thresholds"] = dict(self.decision_thresholds)
        if self.variation_approach is not None:
            payload["variation_approach"] = self.variation_approach
        if self.notes is not None:
            payload["notes"] = self.notes
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelCardMetrics:
        if not isinstance(payload, Mapping):
            raise TypeError("model card metrics payload must be a mapping.")
        return cls(
            measures=_text_tuple(payload.get("measures"), name="metrics.measures"),
            decision_thresholds=_text_mapping(
                payload.get("decision_thresholds"),
                name="metrics.decision_thresholds",
            ),
            variation_approach=payload.get("variation_approach"),
            notes=payload.get("notes"),
        )


@dataclass(frozen=True, slots=True)
class ModelCardData:
    """Description of evaluation or training data (Mitchell §4.5/§4.6)."""

    description: str | None = None
    source: str | None = None
    preprocessing: str | None = None
    motivation: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "description", _optional_text(self.description, name="data.description"))
        object.__setattr__(self, "source", _optional_text(self.source, name="data.source"))
        object.__setattr__(
            self,
            "preprocessing",
            _optional_text(self.preprocessing, name="data.preprocessing"),
        )
        object.__setattr__(self, "motivation", _optional_text(self.motivation, name="data.motivation"))
        object.__setattr__(self, "notes", _optional_text(self.notes, name="data.notes"))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key in ("description", "source", "preprocessing", "motivation", "notes"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelCardData:
        if not isinstance(payload, Mapping):
            raise TypeError("model card data payload must be a mapping.")
        return cls(
            description=payload.get("description"),
            source=payload.get("source"),
            preprocessing=payload.get("preprocessing"),
            motivation=payload.get("motivation"),
            notes=payload.get("notes"),
        )


@dataclass(frozen=True, slots=True)
class ModelCardAnalysis:
    """Quantitative analysis results (Mitchell §4.7)."""

    unitary_results: dict[str, Any] = field(default_factory=dict)
    intersectional_results: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "unitary_results",
            _metadata(self.unitary_results, name="analysis.unitary_results"),
        )
        object.__setattr__(
            self,
            "intersectional_results",
            _metadata(self.intersectional_results, name="analysis.intersectional_results"),
        )
        object.__setattr__(self, "notes", _optional_text(self.notes, name="analysis.notes"))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.unitary_results:
            payload["unitary_results"] = dict(self.unitary_results)
        if self.intersectional_results:
            payload["intersectional_results"] = dict(self.intersectional_results)
        if self.notes is not None:
            payload["notes"] = self.notes
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelCardAnalysis:
        if not isinstance(payload, Mapping):
            raise TypeError("model card analysis payload must be a mapping.")
        return cls(
            unitary_results=_metadata(
                payload.get("unitary_results"), name="analysis.unitary_results"
            ),
            intersectional_results=_metadata(
                payload.get("intersectional_results"),
                name="analysis.intersectional_results",
            ),
            notes=payload.get("notes"),
        )


@dataclass(frozen=True, slots=True)
class ModelCard:
    """Model Card (Mitchell et al. 2019) for an exp-pkg model artifact.

    Designed for pose-estimation and behavior-classification models that
    produce outputs consumed by xpkg pipelines. The card carries the nine
    Mitchell sections plus an open ``metadata`` slot for tool-specific
    extensions.
    """

    details: ModelCardDetails
    intended_use: ModelCardIntendedUse = field(default_factory=ModelCardIntendedUse)
    factors: ModelCardFactors = field(default_factory=ModelCardFactors)
    metrics: ModelCardMetrics = field(default_factory=ModelCardMetrics)
    evaluation_data: ModelCardData = field(default_factory=ModelCardData)
    training_data: ModelCardData = field(default_factory=ModelCardData)
    quantitative_analyses: ModelCardAnalysis = field(default_factory=ModelCardAnalysis)
    ethical_considerations: str | None = None
    caveats: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "details",
            _coerce_section(self.details, ModelCardDetails, name="model_card.details"),
        )
        object.__setattr__(
            self,
            "intended_use",
            _coerce_section(
                self.intended_use, ModelCardIntendedUse, name="model_card.intended_use"
            ),
        )
        object.__setattr__(
            self,
            "factors",
            _coerce_section(self.factors, ModelCardFactors, name="model_card.factors"),
        )
        object.__setattr__(
            self,
            "metrics",
            _coerce_section(self.metrics, ModelCardMetrics, name="model_card.metrics"),
        )
        object.__setattr__(
            self,
            "evaluation_data",
            _coerce_section(
                self.evaluation_data, ModelCardData, name="model_card.evaluation_data"
            ),
        )
        object.__setattr__(
            self,
            "training_data",
            _coerce_section(
                self.training_data, ModelCardData, name="model_card.training_data"
            ),
        )
        object.__setattr__(
            self,
            "quantitative_analyses",
            _coerce_section(
                self.quantitative_analyses,
                ModelCardAnalysis,
                name="model_card.quantitative_analyses",
            ),
        )
        object.__setattr__(
            self,
            "ethical_considerations",
            _optional_text(
                self.ethical_considerations, name="model_card.ethical_considerations"
            ),
        )
        object.__setattr__(
            self, "caveats", _text_tuple(self.caveats, name="model_card.caveats")
        )
        object.__setattr__(
            self,
            "recommendations",
            _text_tuple(self.recommendations, name="model_card.recommendations"),
        )
        object.__setattr__(
            self, "metadata", _metadata(self.metadata, name="model_card.metadata")
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"details": self.details.to_dict()}
        for key in (
            "intended_use",
            "factors",
            "metrics",
            "evaluation_data",
            "training_data",
            "quantitative_analyses",
        ):
            section_payload = getattr(self, key).to_dict()
            if section_payload:
                payload[key] = section_payload
        if self.ethical_considerations is not None:
            payload["ethical_considerations"] = self.ethical_considerations
        if self.caveats:
            payload["caveats"] = list(self.caveats)
        if self.recommendations:
            payload["recommendations"] = list(self.recommendations)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelCard:
        if not isinstance(payload, Mapping):
            raise TypeError("model card payload must be a mapping.")
        raw_metadata = payload.get("metadata")
        if raw_metadata is not None and not isinstance(raw_metadata, Mapping):
            raise TypeError("model card metadata must be a mapping when present.")
        details_payload = payload.get("details")
        if details_payload is None:
            raise ValueError("model card payload requires a 'details' section.")
        return cls(
            details=ModelCardDetails.from_dict(details_payload),
            intended_use=ModelCardIntendedUse.from_dict(payload.get("intended_use") or {}),
            factors=ModelCardFactors.from_dict(payload.get("factors") or {}),
            metrics=ModelCardMetrics.from_dict(payload.get("metrics") or {}),
            evaluation_data=ModelCardData.from_dict(payload.get("evaluation_data") or {}),
            training_data=ModelCardData.from_dict(payload.get("training_data") or {}),
            quantitative_analyses=ModelCardAnalysis.from_dict(
                payload.get("quantitative_analyses") or {}
            ),
            ethical_considerations=payload.get("ethical_considerations"),
            caveats=_text_tuple(payload.get("caveats"), name="model_card.caveats"),
            recommendations=_text_tuple(
                payload.get("recommendations"), name="model_card.recommendations"
            ),
            metadata=_metadata(raw_metadata, name="model_card.metadata"),
        )


__all__ = [
    "DatasetDatasheet",
    "DatasheetCollection",
    "DatasheetComposition",
    "DatasheetDistribution",
    "DatasheetMaintenance",
    "DatasheetMotivation",
    "DatasheetPreprocessing",
    "DatasheetUses",
    "ModelCard",
    "ModelCardAnalysis",
    "ModelCardData",
    "ModelCardDetails",
    "ModelCardFactors",
    "ModelCardIntendedUse",
    "ModelCardMetrics",
]
