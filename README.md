# exp-pkg

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

**Typed data I/O and portable project packaging for multimodal neuroscience experiments.**

Import from `xpkg` in Python and use `xpkg` for the CLI.

exp-pkg gives neuroscience repositories one stable boundary for experiment-data
IO. One project stores one typed `Experiment` containing subjects, protocols,
conditions, multiple recording sessions, sampled signals, pose outputs,
synchronized video, named event streams, behavior labels, calibration, and
sharing metadata. The signal contract includes generic time series,
photometry, EMG, and force-plate data.

It imports external formats, normalizes them into canonical `xpkg` objects,
stores them in a project-first contract, and emits portable `.expkg`
artifacts. The project import surface covers pose, calibration, generic
photometry CSV, event CSV, and behavior outputs from generic exports, BORIS,
B-SOiD, SimBA, and Keypoint-MoSeq. Paired synchronization CSV imports produce
evidence-backed timebase alignments.

This repo is not an analysis platform. It is the IO layer that analysis tools,
GUIs, and automation can build on when they need a coherent project surface
instead of a pile of ad hoc CSV, H5, JSON, and sidecars.

The public product contract is now intentionally narrow:

- editable project = project folder
- authoritative mutable state = private `.xpkg/`
- portable artifact = `.expkg`

`.expkg` v1 is a portable zip container with a root `EXPKG.json` manifest.
By default it includes managed media and records member paths, sizes, and
SHA-256 hashes. Pack commands can also emit package-media or manifest-only
media exports when users do not want to store every managed video byte inside
the `.expkg`.

## Positioning

The intended stack is:

- external neuroscience formats at the edge
- one canonical in-memory experiment aggregate in the middle
- shared timing, event, and signal contracts across modalities
- in-memory exchange helpers for arrays / tables / JSON payloads
- editable project + private store + portable artifact at the boundary

The current codebase should be read as a multimodal neuroscience IO and
packaging layer for experiment projects, not as an analysis framework.

## Recommended Project API

`ProjectService` is the primary service boundary for project-based
projects. Use it when you want one object that can create, open, import into,
validate, pack, or unpack a project:

```python
from xpkg.services import ProjectService

project = ProjectService.create("./My Project", title="My Project")
project.import_pose(
    "dlc-csv",
    path="tracking.csv",
    video="video.mp4",
    skeleton_name="subject",
)
layout = project.validate()
artifact = project.pack()
restored = ProjectService.unpack(artifact, "./Restored Project")
```

If you are wiring another repo into xpkg, this is the place to start.

Choose the public surface by job:

| Task | Preferred public entrypoint |
| --- | --- |
| Create, open, import into, validate, pack, or unpack a project | `xpkg.services.ProjectService` |
| Import foreign pose or calibration data into a project you already manage through the service | `project.import_pose(format, ...)` / `import_calibration(format, ...)` |
| Import generic photometry CSV into typed recording-session state | `project.import_signals("photometry-csv", path=...)` |
| Import events or behavior labels into typed recording-session state | `project.import_events(...)` / `project.import_behavior(...)` |
| Save or load typed recording-session state | `project.save_session(...)` / `project.load_session()` |
| Save or load the complete experiment aggregate | `project.save_experiment(...)` / `project.load_experiment()` |
| Register figures, tables, analyses, reports, or other output artifacts | `project.artifacts.*` from `xpkg.services.ProjectService` |
| Save figure outputs and their lineage manifests | `project.figures.*` from `xpkg.services.ProjectService` |
| Save or load frame-level segmentation masks | `project.segmentation.*` from `xpkg.services.ProjectService` |
| Save or window-read dense instance-mask outputs | `xpkg.segmentation.MaskTableReader` / `write_mask_table` |
| Populate GUI or project-picker rows | `xpkg project describe --json`, `ProjectService.describe()`, `xpkg inspect --json`, or `load_project_summary(...)` |

For downstream GUIs and catalog scans, keep list views shallow. Use descriptor,
layout, metadata, and current-state file stats for rows; hydrate labels,
predictions, and media only after a user selects a project. The full rule set
lives in `docs/performance.md`.

Artifacts use a generic registry under `.xpkg/artifacts/<kind>/`, with common
kind directories such as `figures`, `tables`, `analyses`, `reports`, and
`stats-reports`. Callers may also choose their own app namespace, such as
`.xpkg/neuro-analysis/figures/`, by passing `namespace="neuro-analysis"` to
`project.artifacts.register(...)` or `project.figures.save(...)`. `xpkg`
treats namespaces as caller-owned strings; it does not reserve or hard-code
downstream package names.

The shipped service and CLI project import surface currently covers:

- DeepLabCut CSV, H5, and project imports
- Lightning Pose prediction CSV (DLC-style MultiIndex)
- SLEAP analysis H5 and `.pkg.slp`
- MMPose top-down demo JSON (`--save-predictions`)
- MediaPipe pose-landmarks JSON
- Generic photometry CSV
- Generic event CSV
- Paired synchronization CSV
- Generic behavior-event CSV and JSON
- BORIS, B-SOiD, SimBA, and Keypoint-MoSeq CSV outputs

There are two tiers here. Project imports (pose, calibration, generic
photometry CSV, events, and behavior) go
through `ProjectService` and the CLI, write project state, and produce portable
`.expkg` artifacts. Other direct readers parse a file into typed in-memory
objects and stop there.

The direct reader surface includes typed, project-free, experimental readers
for:

- Generic photometry CSV and event CSV
- Generic behavior-event CSV and JSON
- BORIS, B-SOiD, SimBA, and Keypoint-MoSeq behavior CSV outputs
- pMAT-compatible photometry/event CSV
- pyPhotometry PPD and CSV+JSON
- RWD OFRS CSV session bundles
- Neurophotometrics/Bonsai CSV
- Doric `.doric` photometry containers
- Teleopto H5 exports
- TDT tank/block photometry streams through the optional `tdt` package
These readers remain useful for project-free conversion. The generic event and
behavior readers listed above are also reachable through `ProjectService` and
the CLI with managed source provenance.

## What It Does

- Imports external pose, calibration, and media-associated annotation formats into canonical xpkg objects
- Defines a stable project contract: project folder + private `.xpkg/` + `.expkg`
- Manages project lifecycle: create, open, validate, pack, unpack
- Validates `.expkg` manifests, archive member paths, sizes, and SHA-256 digests
- Carries canonical containers such as `Labels`, `Skeleton`, `Instance`, and `Video`
- Registers output artifacts with portable manifests for inputs, producer metadata, stats, checksums, and source data
- Provides figure convenience helpers on top of the generic artifact registry
- Saves and loads frame-level segmentation masks through `project.segmentation`
- Exposes a clean in-memory adapter layer through `xpkg.adapters`
- Handles media-aware packaging and project-relative project state

## Current Scope vs Direction

Implemented today:

- canonical annotation and media data objects
- canonical behavior-label objects for intervals, framewise motifs, and embeddings
- service/CLI project importers for pose, calibration, generic photometry CSV,
  generic events, behavior labels, and paired clock synchronization
- direct readers for signal, event, behavior, pose, and calibration files
- project/store/artifact lifecycle operations
- media-aware packaging and portable exports
- Parquet-backed `xpkg.rle.v1` mask tables for dense instance-mask outputs

Mission direction:

- keep xpkg narrow as the stable multimodal neuroscience IO and artifact boundary
- support more external ecosystems through project importers
- extend importer coverage across pose, video, signals, calibration, and
  acquisition systems without adding parallel ontologies
- make downstream analysis and GUI repos depend on xpkg instead of inventing
  their own project formats
- keep project storage centered on projects and portable `.expkg` exports

The canonical experiment state supports multiple sessions with acquisition,
signals, videos, pose, behavior, calibration, named event streams, timebases,
and provenance. It also supports direct behavior-to-subject links, bounded
subject-to-track assignments, typed event relationships, EMG, and force-plate
signals. Generic photometry CSV is the first complete signal importer for this
state.

## Supported Project Imports

| Source | Format | Status |
|--------|--------|--------|
| DeepLabCut | CSV | Supported |
| DeepLabCut | H5 | Supported |
| DeepLabCut | Project | Supported |
| Lightning Pose | Prediction CSV (DLC-style MultiIndex) | Supported |
| SLEAP | Analysis H5 | Supported |
| SLEAP | `.pkg.slp` | Supported |
| MMPose | Top-down demo JSON (`--save-predictions`) | Supported |
| MediaPipe | Pose landmarks JSON | Supported |
| Generic photometry | CSV | Supported as a session signal in experiment state |
| Generic events | CSV | Supported as a named session event stream |
| Generic behavior events | CSV / JSON | Supported as named session behavior links |
| BORIS | Event CSV | Supported as a named session behavior link |
| B-SOiD | Label or bout CSV | Supported as a named session behavior link |
| SimBA | Classifier CSV | Supported as a named session behavior link |
| Keypoint-MoSeq | Syllable CSV | Supported as a named session behavior link |
| Paired timebases | Synchronization CSV | Supported as an evidence-backed session alignment |

## Supported Direct Readers (experimental)

These readers parse files into typed in-memory objects through `xpkg.readers`.
Generic photometry, event CSV, generic behavior files, BORIS, B-SOiD, SimBA,
and Keypoint-MoSeq also have project importers. The remaining entries do not
yet have project actions or `.expkg` integration.

| Source | Format | Status |
|--------|--------|--------|
| Generic photometry | CSV | Direct reader and project importer |
| Generic events | CSV | Direct reader and project importer |
| Generic behavior events | CSV / JSON | Direct reader and project importer |
| BORIS | Tabular event CSV | Direct reader and project importer |
| B-SOiD | Label or bout CSV | Direct reader and project importer |
| SimBA | Framewise classifier CSV | Direct reader and project importer |
| Keypoint-MoSeq | Syllable CSV | Direct reader and project importer |
| pMAT | Photometry/event CSV | Direct reader (experimental) |
| pyPhotometry | PPD, CSV+JSON | Direct reader (experimental) |
| RWD OFRS | CSV session bundle | Direct reader (experimental) |
| Neurophotometrics/Bonsai | CSV | Direct reader (experimental) |
| Doric | `.doric` HDF5 photometry container | Direct reader (experimental) |
| Teleopto | H5 export | Direct reader (experimental) |
| TDT | Tank/block streams | Direct reader with optional `tdt` dependency (experimental) |
The fiber-photometry readers intentionally do not claim Inscopix `.isx` /
`.isxd`, Blackrock NEV/NSx, or Neuralynx Cheetah support. Those are imaging or
extracellular electrophysiology surfaces, not fiber-photometry IO.

## Install

Install the released package from PyPI:

```bash
uv pip install exp-pkg
```

The distribution name is `exp-pkg`. The Python import name and CLI command are
both `xpkg`.

For a source checkout or local development:

```bash
git clone https://github.com/Alfredo-Sandoval/exp-pkg.git
cd exp-pkg
make env
```

Fallback if you do not want the canonical setup target:

```bash
bash environment/setup.sh
```

If conda or mamba is unavailable on a machine that already has project
dependencies installed in an activated local virtualenv, a repo-local `.venv`,
or a non-base conda environment, the quality-gate wrapper can run inside that
active environment. This is only a runner fallback; `make env` remains the
canonical setup path.

The base install keeps heavyweight media/model stacks optional. Add extras when
you need richer media or deep-learning functionality:

```bash
uv pip install -e ".[media-rich]"  # PyAV / rich FFmpeg media handling
uv pip install -e ".[dl]"          # PyTorch + TorchCodec + TorchVision
uv pip install -e ".[inference]"   # ONNX Runtime
uv pip install -e ".[mlx]"         # MLX / Metal acceleration
uv pip install -e ".[nvidia]"      # PyTorch + TorchCodec for NVIDIA CUDA
uv pip install -e ".[vision]"      # Kornia + PyTorch
uv pip install -e ".[hardware-accel]"  # MLX + NVIDIA optional runtimes
uv pip install -e ".[media-dl]"    # Full optional media/deep-learning stack
```

Runtime code can inspect available stacks through `xpkg.media`:

```python
from xpkg.media import (
    available_hardware_accelerators,
    available_media_backends,
    require_hardware_acceleration,
    require_media_backend,
)
from xpkg.media.video import Video

print(available_media_backends())
print(available_hardware_accelerators())
require_media_backend("pyav")
require_hardware_acceleration("mlx")
require_hardware_acceleration("opencv-cuda")
video = Video.from_filename("session.mp4", backend="pyav")
```

On Linux NVIDIA hosts, use `nvpkg` as the provisioning layer for CUDA-enabled
media libraries, then let `xpkg.media` verify the result. `nvpkg` is an
external provisioning tool and is not installed by exp-pkg extras.

```bash
nvpkg system doctor
nvpkg package install ffmpeg
nvpkg package install opencv_cuda
nvpkg package install torchcodec_cuda
nvpkg package verify opencv_cuda --json
```

Then use the local quality gates:

```bash
make qa
make ci-local
```

Build and check the Python package from a source checkout:

```bash
make package-check
make build
uv pip install dist/exp_pkg-*.whl
```

Before a PyPI/TestPyPI cut, run the local release gate against real lab data:

```bash
make release-check REAL_DATA_ROOT=../xpkg-real-data
```

The local synthetic gates are the canonical public checks for this repository.
The local release gate additionally runs the opt-in real-data suite before a
package handoff or PyPI/TestPyPI cut.

Genuine vendor exports are tested separately and never join the default suite
because an ignored fixture directory exists. Run the explicit vendor lane with
all required corpora:

```bash
make test-vendor \
  FIBER_FIXTURE_ROOT=../xpkg-vendor-data/fiber-photometry \
  POSE_FIXTURE_ROOT=../xpkg-vendor-data/pose \
  BEHAVIOR_FIXTURE_ROOT=../xpkg-vendor-data/behavior
```

## Real Data Tests

The normal test suite uses deterministic synthetic fixtures. Production
readiness requires a private real-data corpus supplied through
`XPKG_REAL_DATA_ROOT` or `REAL_DATA_ROOT=...` when invoking `make`.

Create a manifest named `xpkg-real-data.json` at the corpus root, or point
`XPKG_REAL_DATA_MANIFEST` at a manifest file:

```json
{
  "schema_version": 1,
  "cases": [
    {
      "id": "dlc-session-001",
      "kind": "dlc",
      "tracking": "dlc/session_001/tracking.csv",
      "video": "dlc/session_001/video.mp4",
      "skeleton_name": "subject",
      "expect": {
        "state": "labels",
        "videos": 1,
        "skeletons": 1,
        "min_labeled_frames": 1
      }
    }
  ]
}
```

Supported real-data `kind` values are `dlc`, `lightning_pose`, `sleap`,
`mmpose`, and `mediapipe`. Use `kind: "dlc"` with either `tracking` plus `video` for a single
CSV/H5 tracking or labeled-data file, or `project` for a full DLC project folder; use
`kind: "lightning_pose"` with `tracking` plus `video` for a Lightning Pose
prediction CSV produced by `litpose predict`; use
`kind: "sleap"` with a `labels` file ending in `.slp`, `.pkg.slp`, `.h5`, or
`.hdf5`. SLEAP analysis H5 cases also need a matching `video`. Each case
imports into a fresh project, validates, packs to `.expkg`, unpacks, and
validates again unless `"skip_pack": true` is set.

## Documentation

Build and serve the docs locally with MkDocs:

```bash
make docs-build    # build the static site
make docs-serve    # live preview at localhost:8123
```

## Public Artifact Contract

The `.expkg` on-disk artifact format is the versioned durability contract. The
format is the zip container, the root `EXPKG.json` manifest, and the three-class
layout below: the editable project folder, the private `.xpkg/` store, and the
portable `.expkg` export. Embedded experiment and recording-session documents
are separate pre-1.0 schemas and reject unsupported versions after explicit
breaking bumps.

The Python API and CLI command surface are a separate concern. They are 0.x and
pre-1.0, and may still change before 1.0. The documents in
`docs/artifact_contract.md` and `docs/cli_command_spec_v1.md` describe the
intended v1 surface; the artifact format is frozen, the command and Python
surfaces are not yet.

The artifact format defines exactly three product-facing artifact classes:

```text
My Project/
  PROJECT.json
  .xpkg/
  Media/
  Exports/
    My Project.expkg
```

- Editable project = project folder
- Authoritative mutable state = `.xpkg/`
- Portable artifact = `.expkg`

The artifact model is project-first so experiment state, managed media, and
future aligned modalities have a clear home in one project layout.

The artifact-format specification lives in `docs/artifact_contract.md`. The
command surface, which is still pre-1.0 and may change, is in
`docs/cli_command_spec_v1.md`.

### Stability

- `.expkg` outer format (zip container, `EXPKG.json` manifest, and the project /
  `.xpkg/` / `.expkg` three-class layout) = versioned contract.
- Embedded experiment and recording-session documents = pre-1.0 versioned
  schemas. Breaking ontology changes bump the schema and reject old state.
- Python API and CLI command surface = 0.x, pre-1.0, may change before 1.0.

## CLI

The primary project-first CLI surface is:

```bash
xpkg project init "./My Project"
xpkg import pose dlc-csv --path tracking.csv --video video.mp4 --out "./My Project"
xpkg import pose lightning-pose-csv --path predictions.csv --video video.mp4 --out "./My Project"
xpkg import pose sleap-package --path labels.pkg.slp --out "./My Project"
xpkg import signals photometry-csv --path photometry.csv --out "./My Project" --session-id session-001
xpkg import events events-csv --path events.csv --out "./My Project" --session-id session-001 --event-stream-name task-events
xpkg import behavior boris-csv --path observations.csv --out "./My Project" --session-id session-001 --video-role behavior-camera --pose-name pose-2d
xpkg import synchronization synchronization-csv --path sync.csv --out "./My Project" --source-timebase camera --target-timebase daq --session-id session-001
xpkg inspect tracking.csv --json
xpkg project pack "./My Project"
xpkg project pack "./My Project" --media package
xpkg project unpack "./My Project.expkg" --out "./My Project"
xpkg project validate "./My Project"
xpkg project describe "./My Project" --json
xpkg artifacts list "./My Project" --kind figure
xpkg artifacts inspect "./My Project" session-summary-figure --kind figure
xpkg artifacts validate "./My Project" --kind figure
```

The same `xpkg import` command also ships source-specific project imports for
Anipose calibration, Lightning Pose CSV, SLEAP, MMPose JSON, and MediaPipe
JSON.
Every canonical command supports `--json` for machine-readable output, and
`xpkg describe --json` reports the current command contract for agents.
Use `xpkg inspect PATH --json` before import to identify likely formats,
importers, media metadata, and QC warnings without mutating a project.
Input files that are themselves JSON use `--input-json` so `--json` is reserved
for output mode.

For project pickers and startup catalogs, prefer
`xpkg project describe PATH --json` or `xpkg inspect PATH --json`. Save
`xpkg project validate PATH` for explicit validation, packing, publishing, or
CI.

`xpkg project pack` defaults to `--media full`, which stores all managed
`Media/` files in the `.expkg`. Use `--media package` to include package-sized
media such as image sequences while manifesting video containers without
storing them, or `--media manifest` to record managed media paths, sizes, and
hashes without storing media bytes.

## Development

If you want to add an importer for a new pose-estimation framework or improve
existing functionality:

1. Open an issue or local task describing the change.
2. Create a focused feature branch.
3. Run `make qa` for the fast gate.
4. Run `make release-check REAL_DATA_ROOT=../xpkg-real-data` before a
   package handoff or PyPI/TestPyPI cut.

Please follow the existing code style (enforced by [Ruff](https://docs.astral.sh/ruff/) with the settings in `pyproject.toml`).

## License

This project is licensed under the **BSD 3-Clause License**. See the [LICENSE](LICENSE) file for full terms.
