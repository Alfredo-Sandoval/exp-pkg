# Posetta

A Rosetta Stone for pose data.

Posetta is the standalone IO layer for the native **Siesta format** (`.siesta`).
It owns the bundle reader and writer, prediction append and merge helpers,
metrics table storage, skeleton loading, and adapter entry points for the file
formats we currently support.

## Current Surface

Today `Posetta` supports:

- Native `.siesta` bundle IO
- Prediction append and merge operations for `.siesta`
- Metrics table read and write helpers for `.siesta`
- DeepLabCut adapters
  - CSV tracking files
  - H5 tracking files
  - project-directory conversion
- SLEAP adapter
  - `.pkg.slp` package import

The repo does not yet ship adapters for MMPose, MediaPipe, OpenPose,
Detectron2, or other ecosystems listed in older drafts.

## Installation

For local development or internal use:

```bash
pip install /path/to/Posetta
```

For editable installs:

```bash
pip install -e /path/to/Posetta
```

## Public API

Primary library entry points:

Use `posetta.model` for in-memory pose objects. The `posetta.io.*` modules are
internal implementation detail and are not the install-facing contract.

- `posetta.formats`
  - `read_siesta`
  - `write_siesta`
  - `update_labels_siesta`
  - `append_predictions_siesta`
  - `merge_predictions_siesta`
  - `summarize_project`
  - `validate_project`
  - `PredictionAppendItem`
  - `SerializerPredictedInstance`
  - `MaxInstancesExceededError`
  - `LazyDatasetHandle`
  - `read_metrics_table`
  - `write_metrics_table`
- `posetta.adapters`
  - `ConversionResult`
  - `convert_dlc_csv`
  - `convert_dlc_h5`
  - `convert_dlc_project`
  - `convert_sleap_package`
- `posetta.model`
  - `Labels`
  - `SuggestionFrame`
  - `Skeleton`
  - `Keypoint`
  - `build_keypoint_skeleton`
  - `Track`
  - `LabeledFrame`
  - `Instance`
  - `PredictedInstance`
  - `Point`
  - `PredictedPoint`
  - `PointArray`
  - `PredictedPointArray`
  - `Video`
  - `load_skeleton`
  - `load_skeleton_dlc`
  - `load_skeleton_siesta_json`
  - `load_skeleton_sleap`
  - `load_skeleton_ultralytics`

## Quick Start

```python
from posetta.adapters import convert_dlc_csv
from posetta.formats import read_siesta, write_siesta
from posetta.model import Labels

# Convert DeepLabCut tracking into a native .siesta bundle
convert_dlc_csv("tracking.csv", "video.mp4", "tracking.siesta")

# Read a bundle back
payload = read_siesta("tracking.siesta", lazy=False)
labels = payload["labels"]
assert isinstance(labels, Labels)

# Write a new bundle
write_siesta("copy.siesta", labels)
```

Working directly with the canonical object model:

```python
from posetta.formats import read_siesta, write_siesta
from posetta.model import Labels

labels = Labels()
write_siesta("empty.siesta", labels)

loaded = read_siesta("empty.siesta", lazy=False)["labels"]
assert isinstance(loaded, Labels)
```

Loading external skeleton definitions through the public model surface:

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
