# Posetta

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Version: 0.1.0](https://img.shields.io/badge/version-0.1.0-green.svg)](pyproject.toml)

**A Rosetta Stone for pose data.**

The pose-estimation ecosystem is fragmented: DeepLabCut exports CSV and H5, SLEAP uses `.pkg.slp` packages, and every other tracker has its own format. Posetta bridges that gap with a canonical `Labels` object, a native HDF5 archive format (`.siesta` archives), and a lightweight labels JSON interchange path for GUI-friendly inspection workflows.

## What It Does

- Native `.siesta` archive IO (read, write, update, append, merge)
- Canonical labels JSON IO for fast interchange and GUI workflows
- Metrics table storage inside archives
- Skeleton loading from multiple formats
- DeepLabCut adapters (CSV, H5, whole-project)
- SLEAP adapter (`.pkg.slp` package import)

## Supported Formats

| Source | Format | Status |
|--------|--------|--------|
| DeepLabCut | CSV | ✅ Supported |
| DeepLabCut | H5 | ✅ Supported |
| DeepLabCut | Project | ✅ Supported |
| SLEAP | `.pkg.slp` | ✅ Supported |
| MMPose | — | 🔜 Planned |
| MediaPipe | — | 🔜 Planned |
| OpenPose | — | 🔜 Planned |
| Detectron2 | — | 🔜 Planned |

## Install

Not on PyPI yet. Clone and install locally:

```bash
git clone https://github.com/Alfredo-Sandoval/Posetta.git
cd Posetta
pip install -e .
```

For the documentation toolchain:

```bash
pip install -e '.[docs]'
```

## Documentation

Build and serve the docs locally with MkDocs:

```bash
make docs-build    # build the static site
make docs-serve    # live preview at localhost:8123
```

## Quick Start

Convert DeepLabCut tracking into a `.siesta` archive, then read it back:

```python
from posetta.adapters import convert_dlc_csv
from posetta.model import Labels

# Convert DeepLabCut tracking into a .siesta archive
convert_dlc_csv("tracking.csv", "video.mp4", "tracking.siesta")

# Read an archive back as the canonical Labels object
labels = Labels.load_file("tracking.siesta")
assert isinstance(labels, Labels)

# Write either native .siesta or fast JSON interchange
labels.save_file(labels, "copy.siesta")
labels.save_file(labels, "copy.json")
```

Load skeleton definitions from a config file:

```python
from posetta.model import load_skeleton

skeleton = load_skeleton("config.yaml")
print(skeleton.keypoint_names)
```

## CLI

After installation, Posetta provides a `posetta` command with the following subcommands:

**Convert DeepLabCut CSV:**
```bash
posetta convert dlc csv --csv tracking.csv --video video.mp4 --out tracking.siesta
```

**Convert DeepLabCut H5:**
```bash
posetta convert dlc h5 --h5 tracking.h5 --video video.mp4 --out tracking.siesta
```

**Convert an entire DeepLabCut project:**
```bash
posetta convert dlc project --project dlc_project --out exports
```

**Convert SLEAP labels:**
```bash
posetta convert sleap --slp labels.pkg.slp --out sleap_project --fps 30 --no-videos
```

## Contributing

Contributions are welcome! If you'd like to add an adapter for a new pose-estimation framework or improve existing functionality:

1. Open an issue describing the change you'd like to make.
2. Fork the repo and create a feature branch.
3. Make sure all tests pass with `pytest`.
4. Submit a pull request.

Please follow the existing code style (enforced by [Ruff](https://docs.astral.sh/ruff/) with the settings in `pyproject.toml`).

## License

This project is released under a **Proprietary License**. See the [LICENSE](LICENSE) file for full terms. © 2026 Alfredo and Joseph Sandoval. All rights reserved.
