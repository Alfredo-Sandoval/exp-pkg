"""Public adapter entry points for external pose ecosystems."""

from __future__ import annotations

from posetta.adapters.dlc import convert_dlc_csv, convert_dlc_h5, convert_dlc_project
from posetta.adapters.sleap import convert_sleap_package
from posetta.io.converters.converter_helpers import ConversionResult

__all__ = [
    "ConversionResult",
    "convert_dlc_csv",
    "convert_dlc_h5",
    "convert_dlc_project",
    "convert_sleap_package",
]
