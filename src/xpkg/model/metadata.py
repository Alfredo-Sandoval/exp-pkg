"""Dependency-light metadata primitives for experiment datasets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from xpkg.model._metadata_validation import (
    metadata_dict as _metadata,
    optional_bool as _optional_bool,
    optional_text as _optional_text,
    required_text as _required_text,
    text_mapping as _text_mapping,
    text_tuple as _text_tuple,
)


def _positive_float(value: Any | None, *, name: str) -> float | None:
    if value is None:
        return None
    coerced = float(value)
    if coerced <= 0.0:
        raise ValueError(f"{name} must be positive when provided, got {coerced}.")
    return coerced


def _non_negative_float(value: Any | None, *, name: str) -> float | None:
    if value is None:
        return None
    coerced = float(value)
    if coerced < 0.0:
        raise ValueError(f"{name} must be non-negative when provided, got {coerced}.")
    return coerced


def _resolution(value: Iterable[Any] | None) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        raise TypeError("camera resolution_px must be a two-item iterable.")
    items = tuple(value)
    if len(items) != 2:
        raise ValueError("camera resolution_px must contain width and height.")
    width = int(items[0])
    height = int(items[1])
    if width <= 0 or height <= 0:
        raise ValueError("camera resolution_px width and height must be positive.")
    return (width, height)


def _camera_tuple(
    value: Iterable[CameraMetadata | Mapping[str, Any]] | None,
) -> tuple[CameraMetadata, ...]:
    if value is None:
        return ()
    cameras: list[CameraMetadata] = []
    for item in value:
        if isinstance(item, CameraMetadata):
            cameras.append(item)
        elif isinstance(item, Mapping):
            cameras.append(CameraMetadata.from_dict(item))
        else:
            raise TypeError(
                "acquisition cameras must contain CameraMetadata objects or mappings."
            )
    camera_ids = [camera.camera_id for camera in cameras]
    if len(set(camera_ids)) != len(camera_ids):
        raise ValueError("acquisition camera_id values must be unique.")
    return tuple(cameras)


@dataclass(frozen=True, slots=True)
class CameraMetadata:
    """Acquisition camera identity and capture settings."""

    camera_id: str
    name: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    serial_number: str | None = None
    lens: str | None = None
    distance_to_arena_mm: float | None = None
    frame_rate_hz: float | None = None
    resolution_px: tuple[int, int] | None = None
    exposure_ms: float | None = None
    gain: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera_id", _required_text(self.camera_id, name="camera_id"))
        object.__setattr__(self, "name", _optional_text(self.name, name="camera name"))
        object.__setattr__(
            self,
            "manufacturer",
            _optional_text(self.manufacturer, name="camera manufacturer"),
        )
        object.__setattr__(self, "model", _optional_text(self.model, name="camera model"))
        object.__setattr__(
            self,
            "serial_number",
            _optional_text(self.serial_number, name="camera serial_number"),
        )
        object.__setattr__(self, "lens", _optional_text(self.lens, name="camera lens"))
        object.__setattr__(
            self,
            "distance_to_arena_mm",
            _non_negative_float(
                self.distance_to_arena_mm,
                name="camera distance_to_arena_mm",
            ),
        )
        object.__setattr__(
            self,
            "frame_rate_hz",
            _positive_float(self.frame_rate_hz, name="camera frame_rate_hz"),
        )
        object.__setattr__(self, "resolution_px", _resolution(self.resolution_px))
        object.__setattr__(
            self,
            "exposure_ms",
            _non_negative_float(self.exposure_ms, name="camera exposure_ms"),
        )
        object.__setattr__(self, "gain", _non_negative_float(self.gain, name="camera gain"))
        object.__setattr__(
            self,
            "metadata",
            _metadata(self.metadata, name="camera metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly camera metadata payload."""
        payload: dict[str, Any] = {"camera_id": self.camera_id}
        for key in (
            "name",
            "manufacturer",
            "model",
            "serial_number",
            "lens",
            "distance_to_arena_mm",
            "frame_rate_hz",
            "exposure_ms",
            "gain",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.resolution_px is not None:
            payload["resolution_px"] = list(self.resolution_px)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CameraMetadata:
        """Hydrate camera metadata from a JSON-friendly payload."""
        if not isinstance(payload, Mapping):
            raise TypeError("camera metadata payload must be a mapping.")
        raw_metadata = payload.get("metadata")
        if raw_metadata is not None and not isinstance(raw_metadata, Mapping):
            raise TypeError("camera metadata must be a mapping when present.")
        return cls(
            camera_id=payload.get("camera_id", ""),
            name=payload.get("name"),
            manufacturer=payload.get("manufacturer"),
            model=payload.get("model"),
            serial_number=payload.get("serial_number"),
            lens=payload.get("lens"),
            distance_to_arena_mm=payload.get("distance_to_arena_mm"),
            frame_rate_hz=payload.get("frame_rate_hz"),
            resolution_px=payload.get("resolution_px"),
            exposure_ms=payload.get("exposure_ms"),
            gain=payload.get("gain"),
            metadata=_metadata(raw_metadata, name="camera metadata"),
        )


@dataclass(frozen=True, slots=True)
class AcquisitionMetadata:
    """Recording setup metadata for a session acquisition."""

    acquisition_id: str | None = None
    recorded_at: str | None = None
    experimenter: str | None = None
    site: str | None = None
    system: str | None = None
    cameras: tuple[CameraMetadata, ...] = ()
    arena_size: str | None = None
    arena_material: str | None = None
    arena_color: str | None = None
    lighting: str | None = None
    ir_lighting: bool | None = None
    software: dict[str, str] = field(default_factory=dict)
    hardware: dict[str, str] = field(default_factory=dict)
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "acquisition_id",
            _optional_text(self.acquisition_id, name="acquisition_id"),
        )
        object.__setattr__(
            self,
            "recorded_at",
            _optional_text(self.recorded_at, name="recorded_at"),
        )
        object.__setattr__(
            self,
            "experimenter",
            _optional_text(self.experimenter, name="experimenter"),
        )
        object.__setattr__(self, "site", _optional_text(self.site, name="site"))
        object.__setattr__(self, "system", _optional_text(self.system, name="system"))
        object.__setattr__(self, "cameras", _camera_tuple(self.cameras))
        object.__setattr__(
            self,
            "arena_size",
            _optional_text(self.arena_size, name="arena_size"),
        )
        object.__setattr__(
            self,
            "arena_material",
            _optional_text(self.arena_material, name="arena_material"),
        )
        object.__setattr__(
            self,
            "arena_color",
            _optional_text(self.arena_color, name="arena_color"),
        )
        object.__setattr__(self, "lighting", _optional_text(self.lighting, name="lighting"))
        object.__setattr__(
            self,
            "ir_lighting",
            _optional_bool(self.ir_lighting, name="ir_lighting"),
        )
        object.__setattr__(self, "software", _text_mapping(self.software, name="software"))
        object.__setattr__(self, "hardware", _text_mapping(self.hardware, name="hardware"))
        object.__setattr__(self, "notes", _optional_text(self.notes, name="notes"))
        object.__setattr__(
            self,
            "metadata",
            _metadata(self.metadata, name="acquisition metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly acquisition metadata payload."""
        payload: dict[str, Any] = {}
        for key in (
            "acquisition_id",
            "recorded_at",
            "experimenter",
            "site",
            "system",
            "arena_size",
            "arena_material",
            "arena_color",
            "lighting",
            "notes",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.ir_lighting is not None:
            payload["ir_lighting"] = self.ir_lighting
        if self.cameras:
            payload["cameras"] = [camera.to_dict() for camera in self.cameras]
        if self.software:
            payload["software"] = dict(self.software)
        if self.hardware:
            payload["hardware"] = dict(self.hardware)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AcquisitionMetadata:
        """Hydrate acquisition metadata from a JSON-friendly payload."""
        if not isinstance(payload, Mapping):
            raise TypeError("acquisition metadata payload must be a mapping.")
        raw_metadata = payload.get("metadata")
        if raw_metadata is not None and not isinstance(raw_metadata, Mapping):
            raise TypeError("acquisition metadata must be a mapping when present.")
        return cls(
            acquisition_id=payload.get("acquisition_id"),
            recorded_at=payload.get("recorded_at"),
            experimenter=payload.get("experimenter"),
            site=payload.get("site"),
            system=payload.get("system"),
            cameras=payload.get("cameras") or (),
            arena_size=payload.get("arena_size"),
            arena_material=payload.get("arena_material"),
            arena_color=payload.get("arena_color"),
            lighting=payload.get("lighting"),
            ir_lighting=payload.get("ir_lighting"),
            software=_text_mapping(payload.get("software"), name="software"),
            hardware=_text_mapping(payload.get("hardware"), name="hardware"),
            notes=payload.get("notes"),
            metadata=_metadata(raw_metadata, name="acquisition metadata"),
        )


@dataclass(frozen=True, slots=True)
class DatasetShareMetadata:
    """Dataset citation and sharing metadata for exported packages."""

    title: str
    creators: tuple[str, ...]
    dataset_id: str | None = None
    description: str | None = None
    license: str | None = None
    doi: str | None = None
    repository_url: str | None = None
    version: str | None = None
    funders: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    access: str | None = None
    related_publications: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        creators = _text_tuple(self.creators, name="creators")
        if not creators:
            raise ValueError("creators must contain at least one entry.")
        object.__setattr__(self, "title", _required_text(self.title, name="title"))
        object.__setattr__(self, "creators", creators)
        object.__setattr__(self, "dataset_id", _optional_text(self.dataset_id, name="dataset_id"))
        object.__setattr__(
            self,
            "description",
            _optional_text(self.description, name="description"),
        )
        object.__setattr__(self, "license", _optional_text(self.license, name="license"))
        object.__setattr__(self, "doi", _optional_text(self.doi, name="doi"))
        object.__setattr__(
            self,
            "repository_url",
            _optional_text(self.repository_url, name="repository_url"),
        )
        object.__setattr__(self, "version", _optional_text(self.version, name="version"))
        object.__setattr__(self, "funders", _text_tuple(self.funders, name="funders"))
        object.__setattr__(self, "keywords", _text_tuple(self.keywords, name="keywords"))
        object.__setattr__(self, "access", _optional_text(self.access, name="access"))
        object.__setattr__(
            self,
            "related_publications",
            _text_tuple(self.related_publications, name="related_publications"),
        )
        object.__setattr__(
            self,
            "metadata",
            _metadata(self.metadata, name="dataset share metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dataset sharing metadata payload."""
        payload: dict[str, Any] = {
            "title": self.title,
            "creators": list(self.creators),
        }
        for key in (
            "dataset_id",
            "description",
            "license",
            "doi",
            "repository_url",
            "version",
            "access",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.funders:
            payload["funders"] = list(self.funders)
        if self.keywords:
            payload["keywords"] = list(self.keywords)
        if self.related_publications:
            payload["related_publications"] = list(self.related_publications)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DatasetShareMetadata:
        """Hydrate dataset sharing metadata from a JSON-friendly payload."""
        if not isinstance(payload, Mapping):
            raise TypeError("dataset share metadata payload must be a mapping.")
        raw_metadata = payload.get("metadata")
        if raw_metadata is not None and not isinstance(raw_metadata, Mapping):
            raise TypeError("dataset share metadata must be a mapping when present.")
        raw_funders = payload.get("funders")
        if raw_funders is None and payload.get("funder") is not None:
            raw_funders = (payload["funder"],)
        return cls(
            title=payload.get("title", ""),
            creators=_text_tuple(payload.get("creators"), name="creators"),
            dataset_id=payload.get("dataset_id"),
            description=payload.get("description"),
            license=payload.get("license"),
            doi=payload.get("doi"),
            repository_url=payload.get("repository_url"),
            version=payload.get("version"),
            funders=_text_tuple(raw_funders, name="funders"),
            keywords=_text_tuple(payload.get("keywords"), name="keywords"),
            access=payload.get("access"),
            related_publications=_text_tuple(
                payload.get("related_publications"),
                name="related_publications",
            ),
            metadata=_metadata(raw_metadata, name="dataset share metadata"),
        )


@dataclass(frozen=True, slots=True)
class PoseModelProvenance:
    """Reproducibility provenance for an imported pose-prediction dataset."""

    tool: str
    tool_version: str | None = None
    model_name: str | None = None
    model_config: dict[str, Any] = field(default_factory=dict)
    training_set_reference: str | None = None
    checkpoint_id: str | None = None
    imported_from: str | None = None
    imported_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool", _required_text(self.tool, name="tool"))
        object.__setattr__(
            self,
            "tool_version",
            _optional_text(self.tool_version, name="tool_version"),
        )
        object.__setattr__(self, "model_name", _optional_text(self.model_name, name="model_name"))
        object.__setattr__(
            self,
            "model_config",
            _metadata(self.model_config, name="model_config"),
        )
        object.__setattr__(
            self,
            "training_set_reference",
            _optional_text(self.training_set_reference, name="training_set_reference"),
        )
        object.__setattr__(
            self,
            "checkpoint_id",
            _optional_text(self.checkpoint_id, name="checkpoint_id"),
        )
        object.__setattr__(
            self,
            "imported_from",
            _optional_text(self.imported_from, name="imported_from"),
        )
        object.__setattr__(
            self,
            "imported_at",
            _optional_text(self.imported_at, name="imported_at"),
        )
        object.__setattr__(
            self,
            "metadata",
            _metadata(self.metadata, name="pose provenance metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly pose provenance payload."""
        payload: dict[str, Any] = {"tool": self.tool}
        for key in (
            "tool_version",
            "model_name",
            "training_set_reference",
            "checkpoint_id",
            "imported_from",
            "imported_at",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.model_config:
            payload["model_config"] = dict(self.model_config)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PoseModelProvenance:
        """Hydrate pose provenance from a JSON-friendly payload."""
        if not isinstance(payload, Mapping):
            raise TypeError("pose provenance payload must be a mapping.")
        return cls(
            tool=str(payload.get("tool", "")),
            tool_version=payload.get("tool_version"),
            model_name=payload.get("model_name"),
            model_config=_metadata(payload.get("model_config"), name="model_config"),
            training_set_reference=payload.get("training_set_reference"),
            checkpoint_id=payload.get("checkpoint_id"),
            imported_from=payload.get("imported_from"),
            imported_at=payload.get("imported_at"),
            metadata=_metadata(payload.get("metadata"), name="pose provenance metadata"),
        )


__all__ = [
    "AcquisitionMetadata",
    "CameraMetadata",
    "DatasetShareMetadata",
    "PoseModelProvenance",
]
