"""Name normalization routines for source-native pose and marker models."""

from __future__ import annotations

from collections.abc import Sequence


def strip_subject_prefix(name: str) -> str:
    """Return a label without its source subject namespace."""
    label = str(name).strip()
    if ":" in label:
        label = label.split(":", 1)[1]
    return label


def normalize_label_name(name: str) -> str:
    """Normalize exact labels for case-insensitive lookup."""
    return str(name).strip().lower()


def normalize_marker_name(name: str) -> str:
    """Normalize marker labels by stripping subject prefixes and case."""
    return strip_subject_prefix(name).lower()


def normalize_source_label(name: str) -> str:
    """Normalize source labels while preserving source namespace semantics."""
    return str(name).strip().lower()


def normalize_event_type(label: str) -> str:
    """Normalize event labels to stable lowercase slugs."""
    normalized = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    slug = "".join(ch for ch in normalized if ch.isalnum() or ch == "_").strip("_")
    return slug or "event"


def normalize_event_side(context: str) -> str | None:
    """Normalize a left/right event context, or return None when not sided."""
    normalized = str(context).strip().lower()
    if normalized in {"left", "right"}:
        return normalized
    return None


def lookup_unique_label_or_marker(labels: Sequence[str], name: str, *, kind: str) -> int:
    """Resolve one label by exact normalized label or canonical marker name."""
    normalized_label = normalize_label_name(name)
    exact_matches = [
        idx for idx, label in enumerate(labels) if normalize_label_name(label) == normalized_label
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise KeyError(f"{kind} {name!r} is ambiguous in {tuple(labels)}.")

    normalized_marker = normalize_marker_name(name)
    marker_matches = [
        idx
        for idx, label in enumerate(labels)
        if normalize_marker_name(label) == normalized_marker
    ]
    if len(marker_matches) == 1:
        return marker_matches[0]
    if len(marker_matches) > 1:
        matches = tuple(labels[idx] for idx in marker_matches)
        raise KeyError(f"{kind} {name!r} is ambiguous; use one of {matches}.")
    raise KeyError(f"{kind} {name!r} not found in {tuple(labels)}.")


__all__ = [
    "lookup_unique_label_or_marker",
    "normalize_event_side",
    "normalize_event_type",
    "normalize_label_name",
    "normalize_marker_name",
    "normalize_source_label",
    "strip_subject_prefix",
]
