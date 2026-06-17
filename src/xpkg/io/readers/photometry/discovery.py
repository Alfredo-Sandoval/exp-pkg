"""Photometry session-entry discovery across supported source formats."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from xpkg.io.readers.photometry.fiber import (
    _is_tdt_block_directory,
    is_doric_photometry_file,
    is_neurophotometrics_csv,
    is_rwd_ofrs_session,
    is_teleopto_h5,
)
from xpkg.io.readers.photometry.nwb import is_nwb_photometry_file
from xpkg.io.readers.photometry.pyphotometry import (
    is_pyphotometry_csv,
    is_pyphotometry_ppd_file,
)

_GENERIC_HDF5_SUFFIXES = frozenset({".h5", ".hdf5"})
_FILE_DETECTORS: tuple[Callable[[Path], bool], ...] = (
    is_teleopto_h5,
    is_doric_photometry_file,
    is_neurophotometrics_csv,
    is_nwb_photometry_file,
    is_pyphotometry_csv,
    is_pyphotometry_ppd_file,
)


def _visible(path: Path, *, include_hidden_dirs: bool) -> bool:
    return include_hidden_dirs or not path.name.startswith(".")


def _is_session_directory(path: Path) -> bool:
    return is_rwd_ofrs_session(path) or _is_tdt_block_directory(path)


def _is_session_file(path: Path, *, include_generic_hdf5: bool) -> bool:
    if include_generic_hdf5 and path.suffix.lower() in _GENERIC_HDF5_SUFFIXES:
        return True
    return any(detector(path) for detector in _FILE_DETECTORS)


def find_photometry_session_entries(
    path: str | Path,
    *,
    include_hidden_dirs: bool = False,
    include_generic_hdf5: bool = True,
) -> list[Path]:
    """Return loadable photometry session files and directories under ``path``."""

    root = Path(path)
    if root.is_file():
        if _is_session_file(root, include_generic_hdf5=include_generic_hdf5):
            return [root.resolve()]
        return []
    if not root.is_dir():
        return []

    entries: list[Path] = []

    def visit(directory: Path) -> None:
        if _is_session_directory(directory):
            entries.append(directory.resolve())
            return
        for child in sorted(directory.iterdir()):
            if child.is_dir():
                if _visible(child, include_hidden_dirs=include_hidden_dirs):
                    visit(child)
                continue
            if child.is_file() and _is_session_file(
                child,
                include_generic_hdf5=include_generic_hdf5,
            ):
                entries.append(child.resolve())

    visit(root)
    return sorted(set(entries), key=lambda entry: str(entry))


__all__ = ["find_photometry_session_entries"]
