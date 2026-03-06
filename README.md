# Posetta

A Rosetta Stone for pose data.

Posetta reads and writes `.siesta` bundles (HDF5-based pose archives) and
converts DLC and SLEAP tracking into that format.

## What It Does

- Native `.siesta` bundle IO (read, write, update, append, merge)
- Metrics table storage inside bundles
- Skeleton loading from multiple formats
- DeepLabCut adapters (CSV, H5, whole-project)
- SLEAP adapter (`.pkg.slp` package import)

The repo does not yet ship adapters for MMPose, MediaPipe, OpenPose,
Detectron2, or other ecosystems listed in older drafts.

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

```bash
make docs-build    # build the static site
make docs-serve    # live preview at localhost:8123
```

## Quick Start

```python
from posetta.adapters import convert_dlc_csv
from posetta.formats import read_siesta, write_siesta
from posetta.model import Labels

# Convert DeepLabCut tracking into a .siesta bundle
convert_dlc_csv("tracking.csv", "video.mp4", "tracking.siesta")

# Read a bundle back
payload = read_siesta("tracking.siesta", lazy=False)
labels = payload["labels"]
assert isinstance(labels, Labels)

# Write a new bundle
write_siesta("copy.siesta", labels)
```

Loading skeleton definitions:

```python
from posetta.model import load_skeleton

skeleton = load_skeleton("config.yaml")
print(skeleton.keypoint_names)
```

## CLI

After installation, Posetta provides a `posetta` command:

```bash
posetta convert dlc csv --csv tracking.csv --video video.mp4 --out tracking.siesta
posetta convert dlc h5 --h5 tracking.h5 --video video.mp4 --out tracking.siesta
posetta convert dlc project --project dlc_project --out exports
posetta convert sleap --slp labels.pkg.slp --out sleap_project --fps 30 --no-videos
```
