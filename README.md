# exp-pkg

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Version: 0.1.0](https://img.shields.io/badge/version-0.1.0-green.svg)](pyproject.toml)

**Workspace-first toolkit for behavior-centered experiment data and portable project artifacts.**

Import from `xpkg` in Python and use `xpkg` for the CLI.

Many neurobehavior experiments need one project container for media, labels,
segmentation, event-aligned state, and durable exports. exp-pkg is built around
that workflow: editable workspaces, managed project state, and shareable
artifacts for experiment packaging and downstream analysis.

Today the implemented core is strongest for annotations, segmentation, media,
and workspace lifecycle management. The broader mission is behavior-centered
experiment packaging that keeps experiment state coherent across tools.

The old annotation ecosystem is fragmented: DeepLabCut exports CSV and H5,
SLEAP uses `.pkg.slp`, and every tool invents a different project shape.
exp-pkg bridges that gap with a canonical `Labels` object, adapter surfaces for
multiple pose ecosystems, and a locked v1 artifact contract built around
editable workspace folders plus portable `.expkg` exports.

`.siesta` is now a legacy import/read compatibility format. It remains in the
codebase during the transition, but it is no longer the public native project
contract.

If you want the blunt storage rationale, read
`docs/architecture/storage-direction.md`. The short answer is that `.siesta`
is still the only fully implemented round-trip archive engine behind workspace
saves, migration, and durable store commits.

## Mission

The intended stack is:

- editable workspace for a whole experiment session
- media, segmentation, labels, and experiment metadata in one project layout
- portable project exports for sharing, packaging, and downstream tools

The current codebase should be read as a workspace/project system for
behavior-centered experiments.

## Recommended Workspace API

`WorkspaceService` is the primary lifecycle entrypoint for workspace-based
projects. Use it when you want to create, open, validate, pack, or unpack a
project with a single object-oriented boundary:

```python
from xpkg.services import WorkspaceService

workspace = WorkspaceService.create("./My Project", title="My Project")
layout = workspace.validate()
artifact = workspace.pack()
restored = WorkspaceService.unpack(artifact, "./Restored Project")
```

The older free functions in `xpkg.formats` and `xpkg.api` remain
available for compatibility and low-level workflows, but new code should prefer
`WorkspaceService`.

## What It Does

- Locked v1 project contract: workspace folder + `.xpkg/` + `.expkg`
- Workspace lifecycle services for create/open/validate/pack/unpack flows
- Canonical pose/annotation containers (`Labels`, `Skeleton`, `Instance`, `Video`)
- Pose and segmentation storage plus media-aware project packaging
- Legacy `.siesta` import/read compatibility during transition
- Skeleton loading from multiple formats
- DeepLabCut adapters (CSV, H5, whole-project)
- SLEAP adapter (`.pkg.slp` package import)

## Current Scope vs Direction

Implemented today:

- pose tracks and labeled frames
- skeletons and keypoint semantics
- segmentation storage
- managed media roots and portable project exports
- workspace/project lifecycle operations

Mission direction:

- behavior-centered experiment workspaces
- pose-aligned auxiliary modalities
- cleaner packaging for analysis and GUI workflows built around experiment state

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
git clone https://github.com/Alfredo-Sandoval/exp-pkg.git
cd exp-pkg
make env
```

Fallback if you do not want the canonical setup target:

```bash
bash environment/setup.sh
```

For the documentation toolchain:

```bash
mamba run -n xpkg uv pip install -e '.[docs]'
```

## Documentation

Build and serve the docs locally with MkDocs:

```bash
make docs-build    # build the static site
make docs-serve    # live preview at localhost:8123
```

## Public Artifact Contract

exp-pkg v1 defines exactly three artifact classes:

```text
My Project/
  PROJECT.json
  .xpkg/
  Media/
  Exports/
    My Project.expkg
```

- Editable project = workspace folder
- Authoritative mutable state = `.xpkg/`
- Portable artifact = `.expkg`
- `.siesta` = legacy import/read only

The artifact model is workspace-first so experiment state, managed media, and
future aligned modalities have a clear home in one project layout.

The locked spec lives in `docs/artifact_contract_v1.md`, with the matching
command surface in `docs/cli_command_spec_v1.md`.

## Current Compatibility Layer

The current implementation still exposes low-level `.siesta` helpers while the
workspace-first v1 workflow is being wired in. Those APIs remain useful for
fixtures, migration work, and legacy import/read paths.

The longer write-up on why that is still true, and what has to change before
`.siesta` can shrink further, lives in
`docs/architecture/storage-direction.md`.

Example:

```python
from xpkg.adapters import convert_dlc_csv
from xpkg.model import Labels

# Convert DeepLabCut tracking into a native bundle
convert_dlc_csv("tracking.csv", "video.mp4", "tracking.sta")

# Read a native bundle back as the canonical Labels object
labels = Labels.load_file("tracking.sta")
assert isinstance(labels, Labels)

# Write either a native .sta bundle or fast JSON interchange
labels.save_file(labels, "copy.sta")
labels.save_file(labels, "copy.json")
```

Load skeleton definitions from a config file:

```python
from xpkg.model import load_skeleton

skeleton = load_skeleton("config.yaml")
print(skeleton.keypoint_names)
```

## CLI

The locked v1 public CLI is workspace-first:

```bash
xpkg init "./My Project"
xpkg import dlc csv --csv tracking.csv --video video.mp4 --out "./My Project"
xpkg pack "./My Project"
xpkg unpack "./My Project.expkg" --out "./My Project"
xpkg migrate "./My Project"
```

That command contract is documented in `docs/cli_command_spec_v1.md`.

The current implementation still provides legacy conversion-oriented commands
while the v1 CLI is being wired in:

That split is intentional: the public contract is already workspace/project
oriented, while the compatibility CLI still helps migrate older pipelines into
that model.

**Legacy convert DeepLabCut CSV:**
```bash
xpkg convert dlc csv --csv tracking.csv --video video.mp4 --out tracking.sta
```

**Legacy convert DeepLabCut H5:**
```bash
xpkg convert dlc h5 --h5 tracking.h5 --video video.mp4 --out tracking.sta
```

**Legacy convert an entire DeepLabCut project:**
```bash
xpkg convert dlc project --project dlc_project --out exports
```

**Legacy convert SLEAP labels:**
```bash
xpkg convert sleap --slp labels.pkg.slp --out sleap_project --fps 30 --no-videos
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
