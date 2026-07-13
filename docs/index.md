---
hide:
  - toc
  - navigation
---

<div class="xpkg-hero" markdown>

# xpkg

<p class="tagline">
The IO and project boundary for multimodal neuroscience experiments.
Pose, synchronized video, segmentation masks, calibration metadata — one
stable contract, portable artifacts, no analysis platform attached.
</p>

<div class="terminal">
<span class="prompt">$</span> uv pip install exp-pkg<br>
<span class="prompt">$</span> xpkg project init "./My Experiment"<br>
<span class="prompt">$</span> xpkg import pose dlc-csv --path tracking.csv --video clip.mp4 --out "./My Experiment"
</div>

<div class="actions">
  <a href="getting-started/" class="primary">Get started →</a>
  <a href="api/services/" class="secondary">Service API</a>
  <a href="cli_command_spec_v1/" class="secondary">CLI spec</a>
</div>

</div>

<div class="xpkg-features" markdown>

<div class="feature" markdown>
<span class="label">Service-first</span>
### One object, full lifecycle
`ProjectService` creates, opens, imports into, validates, packs, and unpacks
projects. No surface to memorize beyond the dispatch methods.
</div>

<div class="feature" markdown>
<span class="label">Typed throughout</span>
### From bytes to dataclasses
External pose, calibration, and signal formats normalize into typed
`xpkg.model` objects through project importers or direct readers. No untyped
dicts crossing the IO boundary.
</div>

<div class="feature" markdown>
<span class="label">Portable</span>
### `.expkg` is the only export
A portable zip with `EXPKG.json` declaring members, sizes, and SHA-256
digests. Three media policies — full, package, manifest — for any size.
</div>

</div>

## Quickstart

```python
from xpkg.services import ProjectService

# Create a project — folder + private .xpkg/ store
project = ProjectService.create("./My Experiment", title="My Experiment")

# Import in one of two families: pose or calibration
project.import_pose("dlc-csv", path="tracking.csv", video="clip.mp4", skeleton_name="subject")

# Attach typed metadata (acquisition, dataset_share, datasheet, model_card)
project.metadata.update(
    acquisition={"acquisition_id": "session-001", "cameras": [{"camera_id": "cam-top", "frame_rate_hz": 120.0}]},
)

# Validate and pack into a portable .expkg
project.validate()
artifact = project.pack()
```

## What you import

<div class="xpkg-cards three" markdown>

<div class="card" markdown>
### Pose
DeepLabCut (CSV / H5 / project), Lightning Pose, SLEAP (analysis H5,
`.pkg.slp`), MMPose top-down JSON, MediaPipe pose-landmarks JSON.
</div>

<div class="card" markdown>
### Calibration
Anipose calibration TOML and OpenCV stereo calibration YAML for camera geometry
metadata.
</div>

<div class="card" markdown>
### Signals + events
Photometry (Doric, Neurophotometrics, pyPhotometry, RWD OFRS, Teleopto, TDT,
pMAT, generic CSV), behavior outputs, and named event streams, all through
`xpkg.readers`. Generic photometry CSV also has service and CLI project
integration through canonical experiment state.
</div>

</div>

## Pick by task

| You want to… | Reach for |
| --- | --- |
| Create / open / pack / unpack a project | `xpkg.services.ProjectService` |
| Import an external pose or calibration file | `project.import_pose / import_calibration` |
| Import generic photometry CSV | `project.import_signals("photometry-csv", ...)` |
| Attach durable typed metadata to a project | `project.metadata` / `project.metadata.update(...)` |
| Populate a GUI/project-picker row | `xpkg project describe --json` / `ProjectService.describe()` |
| Read a single file into typed objects | `xpkg.readers.read_*` |
| Build typed model objects in-memory | `xpkg.model` |
| Convert between arrays / tables / JSON payloads | `xpkg.adapters` |
| Inspect a path before importing | `xpkg.inspection.inspect_path` |
| Drive any of the above from a shell | `xpkg --help` (the CLI mirrors the Python surface) |

## The artifact contract

```mermaid
flowchart LR
    subgraph project["📁 Project folder (editable)"]
        direction TB
        descriptor["PROJECT.json<br/><span style='font-size:0.75em;opacity:0.7'>public descriptor</span>"]
        store[".xpkg/<br/><span style='font-size:0.75em;opacity:0.7'>private mutable state</span>"]
        media["Media/<br/><span style='font-size:0.75em;opacity:0.7'>managed video + image bytes</span>"]
        exports["Exports/<br/><span style='font-size:0.75em;opacity:0.7'>portable artifacts</span>"]
    end

    project -->|project.pack| expkg
    expkg["📦 My Experiment.expkg<br/><span style='font-size:0.75em;opacity:0.7'>EXPKG.json manifest + members</span>"]:::accent
    expkg -.->|ProjectService.unpack| project

    classDef accent fill:transparent,stroke-width:1.5px;
```

`.expkg` is the only portable export. xpkg never edits a `.expkg` file in
place — pack to publish, unpack to keep working.

[Read the artifact contract →](artifact_contract.md){ .md-button }
[Read the CLI spec →](cli_command_spec_v1.md){ .md-button }
[Read the performance guide →](performance.md){ .md-button }

## Lifecycle

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant FS as Project folder
    participant Svc as ProjectService
    participant Store as .xpkg/ store

    User->>Svc: ProjectService.create(path)
    Svc->>FS: write PROJECT.json + scaffold
    Svc-->>User: project

    User->>Svc: project.import_pose("dlc-csv", path=..., video=...)
    Svc->>Store: normalize + write state head
    Svc->>FS: copy media into Media/

    User->>Svc: project.save_acquisition(..., session_id=...)
    Svc->>Store: commit acquisition on recording session

    User->>Svc: project.validate()
    Svc-->>User: ProjectLayout (paths + descriptor + summary index)

    User->>Svc: project.pack(media="full")
    Svc->>FS: write Exports/My Experiment.expkg

    Note over User,FS: Later, somewhere else…
    User->>Svc: ProjectService.unpack(expkg, out=...)
    Svc->>FS: rebuild project at out
    Svc-->>User: restored project
```

## Where things live

<div class="xpkg-cards two" markdown>

<div class="card" markdown>
### Stable surfaces
- `xpkg.services` — `ProjectService` and accessors
- `xpkg.model` — typed dataclasses (pose, signals, metadata, reporting)
- `xpkg.readers` — single-file readers
- `xpkg.adapters` — in-memory exchange (numpy / pandas / JSON)
- `xpkg.media` — video and image IO with hardware accel detection
- `xpkg.inspection` — `inspect_path()` returning `InspectionReport`
</div>

<div class="card" markdown>
### Path-level seam
- `xpkg.project` — free functions (`init_project`, `pack_project`,
  `validate_project`, path helpers)

`ProjectService` is the public import path for pose, calibration, generic
photometry CSV, event CSV, and supported behavior outputs.
</div>

</div>
