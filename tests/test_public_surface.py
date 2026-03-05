from __future__ import annotations

from posetta.adapters import convert_dlc_csv, convert_dlc_h5, convert_dlc_project, convert_sleap_package
from posetta.formats import read_siesta, summarize_project, update_labels_siesta, validate_project, write_siesta


def test_public_exports_are_callable() -> None:
    assert callable(read_siesta)
    assert callable(summarize_project)
    assert callable(update_labels_siesta)
    assert callable(validate_project)
    assert callable(write_siesta)
    assert callable(convert_dlc_csv)
    assert callable(convert_dlc_h5)
    assert callable(convert_dlc_project)
    assert callable(convert_sleap_package)
