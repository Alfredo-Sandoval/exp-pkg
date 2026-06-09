"""Case-insensitive DataFrame column lookup shared by table readers."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def column_by_name(frame: pd.DataFrame, name: str) -> str:
    names = {str(column).lower(): str(column) for column in frame.columns}
    key = str(name).lower()
    if key not in names:
        raise ValueError(
            f"Column {name!r} was not found. Available columns: {list(frame.columns)}."
        )
    return names[key]


def first_matching_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    names = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        match = names.get(candidate.lower())
        if match is not None:
            return match
    return None


def resolve_column(
    frame: pd.DataFrame, explicit: str | None, candidates: Sequence[str]
) -> str | None:
    if explicit is not None:
        return column_by_name(frame, explicit)
    return first_matching_column(frame, candidates)


__all__ = ["column_by_name", "first_matching_column", "resolve_column"]
