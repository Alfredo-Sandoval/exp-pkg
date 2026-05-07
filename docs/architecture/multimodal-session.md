# Multimodal Session Model

<div class="page-intro">
<p>
The multimodal session model is the bridge between today's pose/video project
support and the target neuroscience IO layer for pose, video, photometry,
events, and synchronization.
</p>
</div>

The guiding idea is simple: every timed modality should expose enough shared
time information to be aligned with the rest of the experiment, while still
keeping modality-specific data in typed objects.

## Layering

```text
raw lab files
  -> direct readers
  -> model objects
  -> project imports
  -> validated project
  -> portable .expkg artifact
```

The direct reader layer should stay pleasant and lightweight. The project
layer adds durable storage, validation, provenance, and packaging.

Luxem et al. 2023 makes the same architectural pressure practical: labs need
common formats, FAIR/share metadata, acquisition context, and accessible
interfaces before downstream behavioral video methods can be compared or reused
confidently. In this repository, that means the session model is an IO contract,
not an analysis framework.

## Current Public Objects

### Timing

`Timebase` names a time coordinate system. `Timeline` stores strictly
increasing timestamps. `TimeRange` represents a half-open interval in seconds.

```python
from xpkg.model import Timeline

timeline = Timeline.from_sample_rate(
    n_samples=3000,
    sample_rate_hz=100.0,
    start_s=0.0,
)

frame_or_sample = timeline.nearest_index(12.4)
```

### Events

`Event` and `SyncEvent` represent labeled intervals or pulses on a timeline.
`EventTable` stores sorted events and supports basic kind, label, time, and
overlap queries.

```python
from xpkg.model import Event, EventTable, TimeRange

events = EventTable.from_events(
    [
        Event(kind="trial", start_s=10.0, duration_s=2.5, label="A"),
        Event(kind="cue", start_s=10.4, duration_s=0.1, label="tone"),
    ]
)

trial_events = events.query(kind="trial")
events_during_window = events.query(overlaps=TimeRange(10.0, 11.0))
```

### Signals And Photometry

`SignalChannel` and `TimeSeries` represent source-neutral sampled signals.
`PhotometryChannel` and `PhotometryRecording` add photometry-specific naming
without tying the package to one vendor or rig.

```python
from xpkg.model import PhotometryRecording, TimeSeries

series = TimeSeries.from_samples(
    [[1.0, 0.5], [1.1, 0.48], [1.2, 0.47]],
    sample_rate_hz=20.0,
    channel_names=["gcamp", "isosbestic"],
    units=["dff", "dff"],
    name="fiber",
)

photometry = PhotometryRecording(
    series=series,
    signal_channel="gcamp",
    reference_channel="isosbestic",
)
```

### Recording Session

`RecordingSession` groups timed modalities without becoming an analysis object.
It is the target shape for downstream code that needs pose, video, signals, and
events in one place.

```python
from xpkg.model import Event, EventTable, RecordingSession

session = RecordingSession(session_id="session-001")
session = session.with_signal("fiber", photometry)
session = session.with_events(
    EventTable.from_events([Event(kind="trial", start_s=0.0, duration_s=1.0)])
)

print(session.modality_names)
print(session.time_range)
```

## Current Boundary

The session/time/events/signals classes are available today as model objects.
They are not yet a full project import stack.

Implemented now:

- typed timing primitives
- event and event-table primitives
- source-neutral signal channels and time series
- photometry recording wrapper
- session container
- direct readers for photometry CSV, event CSV, pMAT CSV, pyPhotometry PPD/CSV,
  RWD OFRS, Neurophotometrics CSV, Doric `.doric`, Teleopto H5, and optional
  TDT tank/block streams
- read-only `xpkg inspect PATH --json` summaries for projects, common files,
  pose exports, media, and `.expkg` artifacts
- acquisition and dataset-share metadata model primitives
- public exports from `xpkg.model` and `xpkg.api`
- focused tests for validation, queries, and time ranges

Explicitly not part of the fiber-photometry layer:

- Inscopix `.isx` / `.isxd`, which are miniscope/imaging formats
- Blackrock NEV/NSx, which are electrophysiology formats
- Neuralynx Cheetah collections, which are electrophysiology formats

Still ahead:

- `read_sync_csv(...)`
- `project.imports.photometry_csv(...)`
- `project.imports.events_csv(...)`
- `project.imports.sync_csv(...)`
- session manifest storage under the project contract
- richer `xpkg inspect --json` associated-media and sync checks
- acquisition metadata capture for cameras, frame rates, resolution,
  compression, dropped frames, hardware sync, and timing uncertainty
- import-time QC for timestamp monotonicity, frame/sample counts, associated
  media, missing columns, keypoint identity/confidence fields, and sync coverage
- FAIR/share metadata for source files, software producers, parameter summaries,
  checksums, units, coordinate systems, and experimental context
- behavior segmentation output imports from upstream tools or lab workflows

## Design Constraints

- Keep the core package IO-focused.
- Do not require NWB, PyTorch, or a downstream analysis stack.
- Keep direct readers usable without projects.
- Keep project imports useful for durable projects and artifacts.
- Preserve provenance and time alignment.
- Do not overfit the model to one photometry vendor, one camera stack, or one
  lab's file naming scheme.
- Treat behavior segmentation as imported interval or label data. `xpkg` may
  store outputs from tools such as SimBA, MARS, B-SOID, VAME, MoSeq, or local
  classifiers, but should not claim to train or run those algorithms.

## Session Contract Priorities

A useful session package should answer basic inspection questions without
requiring a user to run analysis code:

- What raw files and derived files are present?
- Which modality does each file represent?
- Which timeline does each file use?
- Which camera, acquisition, or synchronization assumptions affect alignment?
- Which software and parameters produced each derived output?
- Which QC warnings were observed during import?
- Which behavior labels, motifs, or intervals were imported, and what source
  pose/video/session do they reference?

These questions define the near-term manifest work. The manifest should make
session contents findable and reusable across tools, while leaving heavier
analysis libraries as optional consumers.

This gives `xpkg` a practical adoption path: direct neuroscience IO first,
portable project/session contracts second, optional analysis bridges later.
