# Posetta

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Version: 0.1.0](https://img.shields.io/badge/version-0.1.0-green.svg)](pyproject.toml)

**Workspace-first IO for pose, segmentation, and related annotation data.**

The annotation ecosystem is fragmented: DeepLabCut exports CSV and H5, SLEAP uses
`.pkg.slp`, and every tool invents a different project shape. Posetta bridges
that gap with a canonical `Labels` object, adapter surfaces for multiple pose
ecosystems, and a locked v1 artifact contract built around editable workspace
folders plus portable `.poseproj` exports.

`.siesta` is now a legacy import/read compatibility format. It remains in the
codebase during the transition, but it is no longer the public native project
contract.

## Recommended Workspace API

`WorkspaceService` is the primary lifecycle entrypoint for workspace-based
projects. Use it when you want to create, open, validate, pack, or unpack a
project with a single object-oriented boundary:

```python
from posetta.services import WorkspaceService

workspace = WorkspaceService.create("./My Project", title="My Project")
layout = workspace.validate()
artifact = workspace.pack()
restored = WorkspaceService.unpack(artifact, "./Restored Project")
```

The older free functions in `posetta.formats` and `posetta.api` remain
available for compatibility and low-level workflows, but new code should prefer
`WorkspaceService`.

## What It Does

- Locked v1 project contract: workspace folder + `.posetta/` + `.poseproj`
- Legacy `.siesta` import/read compatibility during transition
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

## Public Artifact Contract

Posetta v1 defines exactly three artifact classes:

```text
My Project/
  PROJECT.json
  .posetta/
  Media/
  Exports/
    My Project.poseproj
```

- Editable project = workspace folder
- Authoritative mutable state = `.posetta/`
- Portable artifact = `.poseproj`
- `.siesta` = legacy import/read only

The locked spec lives in `docs/artifact_contract_v1.md`, with the matching
command surface in `docs/cli_command_spec_v1.md`.

## Current Compatibility Layer

The current implementation still exposes low-level `.siesta` helpers while the
workspace-first v1 workflow is being wired in. Those APIs remain useful for
fixtures, migration work, and legacy import/read paths.

Example:

```python
from posetta.adapters import convert_dlc_csv
from posetta.model import Labels

# Convert DeepLabCut tracking into a legacy .siesta compatibility archive
convert_dlc_csv("tracking.csv", "video.mp4", "tracking.siesta")

# Read a legacy archive back as the canonical Labels object
labels = Labels.load_file("tracking.siesta")
assert isinstance(labels, Labels)

# Write either legacy .siesta compatibility output or fast JSON interchange
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

The locked v1 public CLI is workspace-first:

```bash
posetta init "./My Project"
posetta import dlc csv --csv tracking.csv --video video.mp4 --out "./My Project"
posetta pack "./My Project"
posetta unpack "./My Project.poseproj" --out "./My Project"
posetta migrate "./My Project"
```

That command contract is documented in `docs/cli_command_spec_v1.md`.

The current implementation still provides legacy conversion-oriented commands
while the v1 CLI is being wired in:

**Legacy convert DeepLabCut CSV:**
```bash
posetta convert dlc csv --csv tracking.csv --video video.mp4 --out tracking.siesta
```

**Legacy convert DeepLabCut H5:**
```bash
posetta convert dlc h5 --h5 tracking.h5 --video video.mp4 --out tracking.siesta
```

**Legacy convert an entire DeepLabCut project:**
```bash
posetta convert dlc project --project dlc_project --out exports
```

**Legacy convert SLEAP labels:**
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
