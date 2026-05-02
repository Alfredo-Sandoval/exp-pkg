#!/usr/bin/env python3
"""Fetch public fiber-photometry files used by external reader tests.

The fetched files are intentionally stored under tests/vendor_data/, which is
ignored by git. The script records a local manifest with source URLs, sizes, and
checksums so fixture provenance stays inspectable without vendoring large data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

DEFAULT_ROOT = Path("tests/vendor_data/fiber_photometry")
PY_PHOTOMETRY_ZIP_URL = (
    "https://raw.githubusercontent.com/pyPhotometry/manuscript/master/data.zip"
)


@dataclass(frozen=True)
class DownloadSpec:
    name: str
    url: str
    relative_path: Path
    expected_size: int | None = None


class FixtureManifest(TypedDict):
    schema: str
    root: str
    records: list[dict[str, object]]


DOWNLOADS = (
    DownloadSpec(
        name="doric_official_example",
        url="https://raw.githubusercontent.com/doriclenses/readDoric/main/Console_Acq_0000.doric",
        relative_path=Path("doric/Console_Acq_0000.doric"),
        expected_size=8_272_048,
    ),
    DownloadSpec(
        name="pmat_photometry_csv",
        url=(
            "https://raw.githubusercontent.com/djamesbarker/pMAT/master/"
            "Sample%20Data/Example%20of%20CSV%20formatting/Ca2Data%20.csv"
        ),
        relative_path=Path("pmat_csv/Ca2Data .csv"),
        expected_size=17_816_733,
    ),
    DownloadSpec(
        name="pmat_events_csv",
        url=(
            "https://raw.githubusercontent.com/djamesbarker/pMAT/master/"
            "Sample%20Data/Example%20of%20CSV%20formatting/BehData%20.csv"
        ),
        relative_path=Path("pmat_csv/BehData .csv"),
        expected_size=768,
    ),
    DownloadSpec(
        name="pyfiber_doric_csv",
        url=(
            "https://gitlab.com/api/v4/projects/inserm-u1215%2Fpyfiber/repository/files/"
            "notebooks%2FFiber%2FData%20Fiber%2FAS21RSA7Rat1228032022_0.csv/raw?ref=main"
        ),
        relative_path=Path("doric_csv/AS21RSA7Rat1228032022_0.csv"),
    ),
)

PMAT_TDT_BLOCK_FILES = (
    ("99761-170207-161634_Photometry-161823.Tbk", 293_232),
    ("99761-170207-161634_Photometry-161823.Tdx", 660_056),
    ("99761-170207-161634_Photometry-161823.tev", 72_594_040),
    ("99761-170207-161634_Photometry-161823.tin", 19_050),
    ("99761-170207-161634_Photometry-161823.tnt", 22),
    ("99761-170207-161634_Photometry-161823.tsq", 2_315_840),
)

PY_PHOTOMETRY_MEMBERS = (
    (
        "dopamine data/P10V_16-2018-08-16-085115.ppd",
        Path("pyphotometry/dopamine_data/P10V_16-2018-08-16-085115.ppd"),
    ),
    (
        "dopamine data/P14-NAc-L-2018-11-29-143403.ppd",
        Path("pyphotometry/dopamine_data/P14-NAc-L-2018-11-29-143403.ppd"),
    ),
)

GUPPY_NPM_PATH = Path("stubbed_testing_data/npm/sampleData_NPM_1/bl72bl82_12feb2024_fp.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path, *, expected_size: int | None = None) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"download {path}")
        request = urllib.request.Request(url, headers={"User-Agent": "exp-pkg-fixture-fetcher"})
        with urllib.request.urlopen(request) as response, path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    size = path.stat().st_size
    if expected_size is not None and size != expected_size:
        raise RuntimeError(f"{path} has size {size}; expected {expected_size}.")
    return {
        "path": str(path),
        "url": url,
        "size": size,
        "sha256": sha256(path),
    }


def fetch_pyphotometry(root: Path) -> list[dict[str, object]]:
    archive = root / "_downloads/pyphotometry_manuscript_data.zip"
    records = [
        download(PY_PHOTOMETRY_ZIP_URL, archive, expected_size=18_484_500)
        | {"name": "pyphotometry_manuscript_zip"}
    ]
    with zipfile.ZipFile(archive) as zipped:
        for member, relative_path in PY_PHOTOMETRY_MEMBERS:
            target = root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                print(f"extract {target}")
                with zipped.open(member) as source, target.open("wb") as handle:
                    shutil.copyfileobj(source, handle)
            records.append(
                {
                    "name": "pyphotometry_ppd",
                    "path": str(target),
                    "source_archive": str(archive),
                    "archive_member": member,
                    "size": target.stat().st_size,
                    "sha256": sha256(target),
                }
            )
    return records


def fetch_pmat_tdt(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    base_url = (
        "https://raw.githubusercontent.com/djamesbarker/pMAT/master/"
        "Sample%20Data/99761-170207-161634/Photometry-161823"
    )
    target_dir = root / "pmat_tdt/Photometry-161823"
    for filename, size in PMAT_TDT_BLOCK_FILES:
        url = f"{base_url}/{filename}"
        record = download(url, target_dir / filename, expected_size=size)
        records.append(record | {"name": "pmat_tdt_block_file"})
    return records


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required command not found: {cmd[0]}") from exc


def fetch_guppy_npm(root: Path, *, ref: str = "main") -> list[dict[str, object]]:
    if shutil.which("git") is None or shutil.which("git-lfs") is None:
        print("skip GuPPy NPM sample: git and git-lfs are required", file=sys.stderr)
        return []
    target = root / "neurophotometrics/bl72bl82_12feb2024_fp.csv"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="xpkg-guppy-") as tmp:
            clone_path = Path(tmp) / "GuPPy"
            run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--filter=blob:none",
                    "--sparse",
                    "--branch",
                    ref,
                    "https://github.com/LernerLab/GuPPy.git",
                    str(clone_path),
                ]
            )
            run(["git", "sparse-checkout", "set", "--no-cone", str(GUPPY_NPM_PATH)], cwd=clone_path)
            run(["git", "lfs", "pull", "-I", str(GUPPY_NPM_PATH)], cwd=clone_path)
            shutil.copy2(clone_path / GUPPY_NPM_PATH, target)
    return [
        {
            "name": "guppy_neurophotometrics_csv",
            "path": str(target),
            "source_repository": "https://github.com/LernerLab/GuPPy",
            "source_path": str(GUPPY_NPM_PATH),
            "size": target.stat().st_size,
            "sha256": sha256(target),
        }
    ]


def fetch(root: Path, *, include_lfs: bool = True) -> FixtureManifest:
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    records.extend(fetch_pyphotometry(root))
    for spec in DOWNLOADS:
        records.append(
            download(spec.url, root / spec.relative_path, expected_size=spec.expected_size)
            | {"name": spec.name}
        )
    records.extend(fetch_pmat_tdt(root))
    if include_lfs:
        records.extend(fetch_guppy_npm(root))
    manifest: FixtureManifest = {
        "schema": "exp-pkg-fiber-photometry-fixtures-v1",
        "root": str(root),
        "records": records,
    }
    manifest_path = root / "manifest.json"
    manifest_json = json.dumps(manifest, indent=2, sort_keys=True)
    manifest_path.write_text(f"{manifest_json}\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--skip-lfs", action="store_true", help="skip GuPPy Git LFS sample fetch")
    args = parser.parse_args(argv)
    manifest = fetch(args.root, include_lfs=not args.skip_lfs)
    print(f"wrote {args.root / 'manifest.json'}")
    print(f"records: {len(manifest['records'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
