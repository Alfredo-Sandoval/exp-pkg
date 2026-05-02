# exp-pkg Multimodal Neuroscience IO Plan

## Goal

`exp-pkg` should become a workspace-first IO toolkit for multimodal
neuroscience experiment sessions.

The public identity should be:

> sleap-io for the whole experiment.

That means practical, low-ceremony Python IO for pose, synchronized video,
fiber photometry and other signals, behavioral events, synchronization data,
metadata, workspaces, and portable artifacts.

This should not become "NWB but smaller" or an analysis platform. The package
should stay one layer lower: read lab files, normalize them into useful Python
objects, keep modalities aligned, and package sessions so downstream analysis
tools, GUIs, and agents can depend on one stable boundary.

## Reference Pattern

NeuralSet is the useful external reference, but not the thing to copy wholesale.

The design lesson is:

- represent recordings, stimuli, labels, events, and annotations against shared
  timelines
- keep modality-specific data objects
- expose a common time/event contract across modalities
- use extractors or exchange helpers to produce arrays, tables, tensors, or
  downstream views

For `xpkg`, the target stack is:

```text
raw lab files
  -> xpkg readers/importers
  -> xpkg session/workspace model
  -> validated portable .expkg artifact
  -> optional analysis, GUI, NWB export, or PyTorch dataset bridge
```

The core package should not require PyTorch, NWB, or any one analysis worldview.

## Product Contract

The public product contract remains intentionally simple:

- editable project = workspace folder
- authoritative mutable state = private `.xpkg/`
- portable artifact = `.expkg`
- direct Python readers for common file formats
- workspace imports for managed projects
- JSON-friendly CLI output for agents and automation

## Public Framing

Use this language consistently across package metadata, README, docs, and CLI
help:

> `exp-pkg` is a workspace-first IO toolkit for multimodal neuroscience
> experiment data and portable artifacts.

The package should name the real neuroscience modalities:

- pose estimation
- synchronized video
- fiber photometry and other sampled signals
- behavioral events
- synchronization pulses and time alignment
- motion capture, force, and EMG where supported
- metadata and project artifacts

Avoid framing that makes the repo sound like:

- a generic archive library
- a private GUI backend
- a pose-only package
- a replacement analysis platform
- a mandatory ontology or NWB-like schema commitment

## Architecture Direction

### 1. Add A Shared Time Spine

Add small, boring, strongly typed timing objects:

```text
src/xpkg/model/time.py
src/xpkg/model/events.py
src/xpkg/model/session.py
```

Candidate public objects:

```python
Timebase
Timeline
TimeRange
Event
EventTable
SyncEvent
RecordingSession
```

Every modality should eventually expose:

```python
recording.timeline
recording.start_time
recording.duration
recording.sample_rate  # for sampled signals
recording.frame_rate   # for frame-based media
recording.timestamps
```

The goal is alignment, not analysis. Downstream code should be able to ask:

- what happened at `t=32.4s`?
- which video frame matches this photometry sample?
- which pose frame overlaps this event?
- which trial contains this time range?
- did this modality start late, drop samples, or drift?

### 2. Add Signals And Photometry As First-Class Data

Add signal models before overfitting to any one vendor format:

```text
src/xpkg/model/signals.py
src/xpkg/io/readers/photometry_csv.py
```

Candidate public objects:

```python
SignalChannel
TimeSeries
PhotometryChannel
PhotometryRecording
```

Start with a flexible CSV reader:

```python
recording = xpkg.read_photometry_csv(
    "photometry.csv",
    time_column="time",
    signal_columns=["gcamp", "isosbestic"],
    sampling_rate=None,
)
```

Do not bake in a single rig. Later readers can support TDT, Doric,
Neurophotometrics, pyPhotometry, or lab-specific exports, but the core data
model should remain vendor-neutral.

### 3. Make Pose, Video, Photometry, And Events Alignable

The target user-facing shape should be session-oriented:

```python
session = xpkg.open_session("experiment.xpkg")

video = session.video["cam0"]
pose = session.pose["cam0"]
photometry = session.signals["fiber"]
events = session.events.query(kind="trial")
```

The important feature is not merely "we can read several file types." The
important feature is that those file types can live in one coherent experiment
session.

### 4. Add `xpkg inspect`

Add an agent-friendly inspection command before forcing users to import:

```bash
xpkg inspect session_dir --json
xpkg inspect photometry.csv --json
xpkg inspect video.mp4 --json
xpkg inspect predictions.slp --json
```

It should report:

- detected kind
- likely importer
- duration
- frame count or sample count
- channels or columns
- timestamp availability
- associated media
- missing files
- warnings
- machine-readable JSON

This becomes the first thing agents, scripts, and users run against unknown lab
files.

### 5. Extend Workspace Imports

After the models exist, add service imports:

```python
workspace.imports.photometry_csv(...)
workspace.imports.events_csv(...)
workspace.imports.sync_csv(...)
```

And CLI commands:

```bash
xpkg import photometry-csv photometry.csv --time time --signal gcamp --out project
xpkg import events-csv events.csv --out project
xpkg import sync-csv sync.csv --out project
```

Keep direct readers and workspace imports side by side:

- direct readers make the package pleasant like `sleap-io`
- workspace imports make it useful for durable projects and artifacts

### 6. Version The Session Manifest

Add or extend schemas for:

```text
timelines
recordings
videos
poses
signals
events
sync
artifacts
```

The manifest should be versioned, plain, and boring. Outside labs should be
able to depend on it without feeling like they are buying into a whole
framework.

### 7. Add Exchange Bridges Later

Keep the core IO package dependency-light. Add optional exchange helpers for
downstream use:

```python
xpkg.exchange.to_numpy(...)
xpkg.exchange.to_dataframe(...)
xpkg.exchange.to_torch_dataset(...)  # later, optional extra
```

PyTorch should be an optional extra, not a core dependency:

```bash
uv pip install "exp-pkg[torch]"
```

NWB should be an optional export target, not the required working format.

## Milestones

### 0.1 - Public Workspace And Pose/Video Release

Focus:

- keep current workspace, pose, video, Vicon, and artifact support stable
- finish public-release cleanup
- keep BSD-3-Clause metadata aligned
- publish package metadata, docs, and CLI surfaces that use the multimodal
  neuroscience IO framing

Deliverables:

- `pyproject.toml`, README, docs, and CLI description aligned
- `make qa` passing
- `make package-check docs-build` passing
- release notes clearly state current implemented support and future direction

### 0.2 - Timing, Event, Session, And Signal Model Alpha

Focus:

- add shared time spine
- add events
- add signal model
- expose a coherent session object

Deliverables:

- `xpkg.model.time`
- `xpkg.model.events`
- `xpkg.model.signals`
- `xpkg.model.session`
- unit tests for time ranges, timestamps, event queries, signal channels, and
  serialization
- docs page explaining the session/timeline contract

### 0.3 - Photometry And Sync Importers

Focus:

- make fiber photometry real in the package
- import event and sync tables
- support CSV-first workflows without vendor lock-in

Deliverables:

- `xpkg.read_photometry_csv(...)`
- `workspace.imports.photometry_csv(...)`
- `workspace.imports.events_csv(...)`
- `workspace.imports.sync_csv(...)`
- CLI commands for photometry, events, and sync imports
- test fixtures for realistic messy CSVs

### 0.4 - Inspection And Agent-Friendly Workflows

Focus:

- make unknown files easy to inspect
- make agent workflows reliable
- strengthen JSON output and error contracts

Deliverables:

- `xpkg inspect`
- JSON output for detected kind, metadata, warnings, and importer suggestion
- docs examples for direct readers, workspace imports, and inspect-first flows
- real-data gate expanded beyond pose/video

### 1.0 - Stable Multimodal Session Contract

Focus:

- freeze the public session/workspace contract enough for outside adoption
- provide enough readers that labs can use it without private knowledge
- keep the package simple, direct, and pleasant

Deliverables:

- stable session manifest version
- stable timing/event/signal API
- stable workspace import API
- documented migration path for pre-1.0 artifacts
- optional bridges for PyTorch and NWB exports if they are useful

## Real-Data Gate

The private real-data suite should grow from pose/video coverage into a
multimodal neuroscience gate:

```text
pose + video
pose + video + events
photometry + events
photometry + sync
pose + video + photometry + sync
Vicon / force / EMG where available
```

The release rule should be:

> Synthetic tests prove the contract. Private real-data tests prove the package
> survives real lab mess.

## Adoption Principles

Keep these constraints visible during implementation:

- direct readers first
- workspace imports second
- low ceremony
- simple Python objects
- no required ontology
- no mandatory NWB conversion
- no PyTorch dependency in core
- clear errors
- JSON CLI output for agents
- repo-relative paths only
- honest docs that separate supported features from planned direction

## Next Implementation Pass

Do this first:

1. Add `xpkg.model.time`.
2. Add `xpkg.model.events`.
3. Add `xpkg.model.signals`.
4. Add minimal serialization and validation tests.
5. Add docs for the timing/events/signals contract.
6. Add a short README example showing pose, video, photometry, and events as
   one aligned session concept.

Do not start with a vendor-specific photometry importer until the shared model
exists. The model should shape the readers, not the first available file.
