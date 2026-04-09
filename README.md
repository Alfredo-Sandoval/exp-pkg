# exp-pkg

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Version: 0.1.0](https://img.shields.io/badge/version-0.1.0-green.svg)](pyproject.toml)

**Canonical IO and artifact layer for experiment data, managed workspaces, and portable project artifacts.**

Import from `xpkg` in Python and use `xpkg` for the CLI.

exp-pkg exists to give other repos one stable boundary for experiment-data IO.
It imports external formats, normalizes them into canonical `xpkg` objects,
stores them in a workspace-first project contract, and emits portable `.expkg`
artifacts.

This repo is not an analysis platform. It is the IO layer that analysis tools,
GUIs, and automation can build on when they need a coherent project/workspace
surface instead of a pile of ad hoc CSV, H5, JSON, and archive files.

The codebase grew out of older SLEAP / `.siesta`-shaped IO work, but the public
boundary is now generic: `Labels`, `Video`, `Skeleton`, adapter imports,
workspace lifecycle operations, and portable project artifacts.

`.siesta` now belongs to the edge of the system: migration, legacy aliases,
fixtures, and compatibility workflows. The explicit edge surface for that work
is `xpkg.compat`, with `.sta` as the canonical archive suffix and `.siesta` as
the older compatibility alias.

## Positioning

The intended stack is:

- external pose / annotation formats at the edge
- canonical in-memory objects in the middle
- editable workspace + private store + portable artifact at the boundary

The current codebase should be read as a generic IO and packaging layer for
experiment projects, not as an analysis framework.

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

If you are wiring another repo into xpkg, this is the place to start.

## What It Does

- Imports external pose / annotation formats into canonical xpkg objects
- Defines a stable project contract: workspace folder + `.xpkg/` + `.expkg`
- Manages workspace lifecycle: create, open, validate, pack, unpack
- Carries canonical containers such as `Labels`, `Skeleton`, `Instance`, and `Video`
- Handles media-aware packaging and workspace-relative project state
- Exposes migration and legacy compatibility surfaces where needed
- Ships DeepLabCut and SLEAP adapters today

## Current Scope vs Direction

Implemented today:

- canonical annotation and media data objects
- import adapters and readers for external formats
- workspace/store/artifact lifecycle operations
- media-aware packaging and portable exports
- legacy compatibility for `.siesta` migration and read/write

Mission direction:

- keep xpkg narrow as the stable IO and artifact boundary
- support more external ecosystems through adapters
- make downstream analysis and GUI repos depend on xpkg instead of inventing
  their own project formats
- continue shrinking `.siesta` toward an edge-only migration layer

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

When published, the distribution name will be `exp-pkg`. The Python import
name and CLI command remain `xpkg`.

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

The current implementation still exposes low-level `.sta` archive helpers and
older `.siesta` aliases, but they should be treated as edge compatibility
surfaces rather than the center of the product.

Use them for:

- migration from older `.siesta` archives
- fixtures and compatibility tests
- legacy read/write paths that have not been cut over yet

Use `xpkg.compat` when you need that edge layer. Avoid using it as the primary
integration boundary for new code. The longer write-up on why this layer still
exists, and what has to happen before it can shrink further, lives in
`docs/architecture/storage-direction.md`.

Example:

```python
from xpkg.compat import read_sta
from xpkg.adapters import convert_dlc_csv

# Convert DeepLabCut tracking into a canonical .sta bundle
convert_dlc_csv("tracking.csv", "video.mp4", "tracking.sta")

# Read the compatibility bundle back when you need direct archive access
payload = read_sta("tracking.sta", lazy=False)
labels = payload["labels"]
```

That example is intentionally compatibility-oriented. New integrations should
prefer workspace import + pack/unpack flows over direct legacy archive handling.

Load skeleton definitions from a config file:

```python
from xpkg.model import load_skeleton

skeleton = load_skeleton("config.yaml")
print(skeleton.keypoint_names)
```

## CLI

The current CLI is a hybrid of workspace-first project commands and transition
helpers for `.sta` archives and older `.siesta` aliases:

```bash
xpkg init "./My Project"
xpkg import dlc csv --csv tracking.csv --video video.mp4 --out "./My Project"
xpkg pack "./My Project"
xpkg unpack "./My Project.expkg" --out "./My Project"
xpkg validate "./My Project"
xpkg migrate "./legacy.sta" --out "./My Project"
```

The shipped command surface is documented in `docs/cli_command_spec_v1.md`.

The compatibility `convert` commands remain available during the transition for
pipelines that still need legacy `.sta` outputs at the edge of the system.

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
