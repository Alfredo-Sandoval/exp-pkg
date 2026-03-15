"""HDF5 helpers for canonical track identity storage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import h5py
import numpy as np

from posetta.core.annotations.instances import Track

TRACKS_SCHEMA_VERSION = "1.0.0"


def _normalize_track_name(track_id: int, name: str | None = None) -> str:
    cleaned = str(name or "").strip()
    return cleaned or f"track-{track_id}"


def _track_from_any(track_id: int, value: Track | str | None = None) -> Track:
    if isinstance(value, Track):
        return Track(
            spawned_on=int(value.id),
            name=_normalize_track_name(int(value.id), value.name),
        )
    return Track(
        spawned_on=int(track_id),
        name=_normalize_track_name(int(track_id), str(value or "")),
    )


def _prediction_instance_track_id(instance: Any) -> int | None:
    if hasattr(instance, "track_id"):
        raw_id = instance.track_id
        if raw_id is not None and int(raw_id) >= 0:
            return int(raw_id)
    track = getattr(instance, "track", None)
    if track is None:
        return None
    track_id = getattr(track, "id", None)
    if track_id is None:
        return None
    track_id_int = int(track_id)
    if track_id_int < 0:
        return None
    return track_id_int


def build_tracks_map(
    *,
    existing: Mapping[int, Track | str] | None = None,
    labels: Any | None = None,
    prediction_items: Sequence[Any] | None = None,
) -> dict[int, Track]:
    """Build a canonical track-id map from existing metadata plus new inputs."""

    tracks_by_id: dict[int, Track] = {}

    if existing is not None:
        for track_id, track_val in existing.items():
            track_id_int = int(track_id)
            if track_id_int < 0:
                continue
            tracks_by_id[track_id_int] = _track_from_any(track_id_int, track_val)

    labels_tracks = getattr(labels, "tracks", None)
    if isinstance(labels_tracks, Sequence):
        for track in labels_tracks:
            if not isinstance(track, Track):
                continue
            tracks_by_id[int(track.id)] = _track_from_any(int(track.id), track)

    if prediction_items is not None:
        for item in prediction_items:
            instances = getattr(item, "instances", None) or []
            for instance in instances:
                track_id = _prediction_instance_track_id(instance)
                if track_id is None or track_id in tracks_by_id:
                    continue
                tracks_by_id[track_id] = _track_from_any(track_id)

    return dict(sorted(tracks_by_id.items(), key=lambda item: item[0]))


def read_tracks_group(root: h5py.File | h5py.Group) -> dict[int, Track]:
    """Read `/tracks` into a canonical track-id mapping."""

    group = root.get("tracks")
    if not isinstance(group, h5py.Group):
        return {}

    track_ids_ds = group.get("track_id")
    names_ds = group.get("name")

    if isinstance(track_ids_ds, h5py.Dataset):
        track_ids = np.asarray(track_ids_ds[...], dtype=np.int32).ravel()
    else:
        track_ids = np.zeros((0,), dtype=np.int32)

    if isinstance(names_ds, h5py.Dataset):
        names = [str(item) for item in names_ds.asstr()[...].ravel()]
    else:
        names = []

    tracks_by_id: dict[int, Track] = {}
    for idx, track_id in enumerate(track_ids.tolist()):
        if int(track_id) < 0:
            continue
        name = names[idx] if idx < len(names) else ""
        tracks_by_id[int(track_id)] = Track(
            spawned_on=int(track_id),
            name=_normalize_track_name(int(track_id), name),
        )

    return tracks_by_id


def write_tracks_group(
    root: h5py.File | h5py.Group,
    *,
    existing: Mapping[int, Track | str] | None = None,
    labels: Any | None = None,
    prediction_items: Sequence[Any] | None = None,
) -> dict[int, Track]:
    """Write `/tracks` using a canonical track-id map."""

    tracks_by_id = build_tracks_map(
        existing=existing,
        labels=labels,
        prediction_items=prediction_items,
    )

    group = root.require_group("tracks")
    group.attrs["schema_version"] = TRACKS_SCHEMA_VERSION
    for dataset_name in ("track_id", "name"):
        if dataset_name in group:
            del group[dataset_name]

    track_ids = np.asarray(list(tracks_by_id.keys()), dtype=np.int32)
    names = np.asarray([track.name for track in tracks_by_id.values()], dtype=object)

    group.create_dataset("track_id", data=track_ids)
    group.create_dataset("name", data=names, dtype=h5py.string_dtype("utf-8"))
    return tracks_by_id


__all__ = [
    "TRACKS_SCHEMA_VERSION",
    "build_tracks_map",
    "read_tracks_group",
    "write_tracks_group",
]
