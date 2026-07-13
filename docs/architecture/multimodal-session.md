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

### Synchronization

`SyncEvent` records one marker on one timeline. Alignment evidence uses
`TimebaseCorrespondence` instead because clock synchronization requires paired
observations of the same instant in source and target timebases.

`AlignmentModel` declares the mathematical transform (`offset` or `affine`).
`SynchronizationMethod` declares how the observations were obtained (`pulses`,
`timestamps`, or `manual`). `fit_timebase_alignment(...)` fits coefficients
from the pairs and derives residual error from the stored evidence.

```python
from xpkg.model import (
    AlignmentModel,
    SynchronizationMethod,
    Timebase,
    TimebaseCorrespondence,
    fit_timebase_alignment,
)

alignment = fit_timebase_alignment(
    name="camera-to-daq",
    source=Timebase(name="camera"),
    target=Timebase(name="daq"),
    model=AlignmentModel.AFFINE,
    method=SynchronizationMethod.PULSES,
    evidence=(
        TimebaseCorrespondence(0.0, 0.25, correspondence_id="pulse-1"),
        TimebaseCorrespondence(10.0, 10.35, correspondence_id="pulse-2"),
    ),
)
```

### Behavior Labels

`BehaviorLabels` is the ethogram-oriented layer above generic events. It can
hold interval bouts, framewise labels or motifs, and per-frame embedding vectors
from human annotation tools or upstream behavior-analysis packages. Time-indexed
intervals can be projected down to `EventTable` when downstream code only needs
the generic event surface.

```python
from xpkg.model import BehaviorInterval, BehaviorLabels

labels = BehaviorLabels(
    source_type="bsoid",
    intervals=[
        BehaviorInterval(label="rear", start_s=10.0, end_s=11.2, score=0.93)
    ],
)

events = labels.to_event_table()
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

`RecordingSession` groups acquisition, video, signal, pose, behavior,
calibration, alignment, and event objects. `SessionPose` links either `Labels`
or `PoseTrajectory` data to its videos, calibration, and model provenance. A
raw pose dictionary is not part of the contract.

```python
from xpkg.model import (
    Event,
    EventTable,
    RecordingSession,
    SessionSignal,
    add_session_signal,
    replace_session_events,
)

session = RecordingSession(session_id="session-001")
session = add_session_signal(session, SessionSignal("fiber", photometry))
session = replace_session_events(
    session,
    EventTable.from_events([Event(kind="trial", start_s=0.0, duration_s=1.0)])
)

print(session.modality_names)
print(session.time_range)
```

## Current Boundary

The session/time/events/signals classes are available as model objects and as
versioned project state. Photometry, events, behavior, and synchronization all
have complete paths through the service, CLI, durable store, shallow summary,
validation, and portable artifact layers.

Implemented now:

- typed timing primitives
- event and event-table primitives
- behavior-label primitives for intervals, framewise motifs, and embeddings
- source-neutral signal channels and time series
- photometry recording wrapper
- session container
- versioned `xpkg.recording-session` and `xpkg.experiment` JSON serialization
- governed `save_project_session` and `load_project_session` actions
- `project.import_signals("photometry-csv", ...)`
- `project.import_events("events-csv", ...)`
- `project.import_behavior(FORMAT, ...)` for generic, BORIS, B-SOiD, SimBA,
  and Keypoint-MoSeq behavior outputs
- `project.import_synchronization("synchronization-csv", ...)`
- `xpkg import signals photometry-csv ...`
- `xpkg import events events-csv ...`
- `xpkg import behavior FORMAT ...`
- `xpkg import synchronization synchronization-csv ...`
- direct readers for photometry CSV, event CSV, pMAT CSV, pyPhotometry PPD/CSV,
  RWD OFRS, Neurophotometrics CSV, Doric `.doric`, Teleopto H5, and optional
  TDT tank/block streams
- direct readers for generic behavior CSVs, BORIS CSVs, SimBA CSVs,
  Keypoint-MoSeq syllable CSVs, and behavior-event JSON sidecars
- read-only `xpkg inspect PATH --json` summaries for projects, common files,
  pose exports, media, and `.expkg` artifacts
- experiment-level subjects, protocols, conditions, session context, and
  dataset-sharing metadata
- session-level acquisition and pose provenance
- public exports from `xpkg.model` and `xpkg.readers`
- focused tests for validation, queries, and time ranges

Explicitly not part of the fiber-photometry layer:

- Inscopix `.isx` / `.isxd`, which are miniscope/imaging formats
- Blackrock NEV/NSx, which are electrophysiology formats
- Neuralynx Cheetah collections, which are electrophysiology formats

Still ahead:

- richer `xpkg inspect --json` associated-media and sync checks
- import coverage for acquisition evidence such as dropped frames, hardware
  sync, and timing uncertainty
- import-time QC for timestamp monotonicity, frame/sample counts, associated
  media, missing columns, keypoint identity/confidence fields, and sync coverage
- broader FAIR/share metadata for software producers, parameter summaries,
  coordinate systems, and experimental context
- additional behavior adapters when an upstream format carries semantics not
  represented by the current generic and source-specific readers

## Shallow Acquisition QC Evidence

Project inspect can warn about acquisition and sync quality only when the
generated project summary or a future session manifest records explicit
evidence. It must not demux videos, hydrate labels, load signal payloads, or
infer failure from missing optional metadata.

The shallow summary needs three evidence groups before inspect-time warnings
are meaningful:

- `media_timing`: one record per associated media item with media id/path,
  nominal FPS, observed FPS or timestamp-derived FPS when already known,
  frame count, duration, timebase id, timing source, and optional dropped-frame
  evidence such as count, indices, spans, or source-side warnings.
- `timed_streams`: one record per pose, behavior, event, photometry, or
  other timed stream with stream id, modality, sample/frame count, start/end
  time, sample rate when known, timebase id, source path, and whether the
  source explicitly declares that synchronization is required.
- `sync_evidence`: source-recorded links between timebases or streams, such as
  TTL pulses, frame clocks, trigger rows, hardware clock names, offset maps,
  uncertainty, source path, and provenance for the importer that observed the
  evidence.

Warnings should stay evidence-gated:

- Dropped-frame warnings require recorded dropped-frame evidence, not merely a
  video file or frame-count mismatch opportunity.
- FPS-drift warnings require both a nominal rate and an observed or
  timestamp-derived rate captured during import or summary generation.
- Missing-sync warnings require an explicit `sync_required` or equivalent
  multi-timebase declaration plus no usable `sync_evidence` for the declared
  relationship.
- Unknown timing, absent acquisition metadata, or older summaries without these
  fields should remain status information, not warnings.

## Design Constraints

- Keep the core package IO-focused.
- Do not require a downstream analysis stack.
- Keep direct readers usable without projects.
- Keep project imports useful for durable projects and artifacts.
- Preserve provenance and time alignment.
- Do not overfit the model to one photometry vendor, one camera stack, or one
  lab's file naming scheme.
- Treat behavior segmentation as imported interval or label data. `xpkg` may
  store outputs from tools such as SimBA, MARS, B-SOID, VAME, MoSeq, or local
  classifiers, but should not claim to train or run those algorithms.

## Rejected Synchronization Designs

- **Single-timestamp events as alignment evidence.** Rejected because one
  `SyncEvent` identifies an instant in only one clock. An alignment requires a
  first-class pair of source and target observations, so evidence is stored as
  `TimebaseCorrespondence` objects.
- **One enum for transform shape and observation method.** Rejected because
  `offset` and `affine` describe the mathematical mapping, while `pulses`,
  `timestamps`, and `manual` describe how the mapping was established. Mixing
  those concepts permits contradictory states and prevents precise queries.

## Session Contract Priorities

A useful session package should answer basic inspection questions without
requiring a user to run analysis code:

- What raw files and derived files are present?
- Which modality does each file represent?
- Which timeline does each file use?
- Which camera, acquisition, or synchronization assumptions affect alignment?
- Which software and parameters produced each derived output?
- Which QC warnings were observed during import?
- Which behavior labels, motifs, intervals, or embeddings were imported, and what source
  pose/video/session do they reference?

These questions define the near-term manifest work. The manifest should make
session contents findable and reusable across tools, while leaving heavier
analysis libraries as optional consumers.

This gives `xpkg` a practical adoption path: direct neuroscience IO first,
portable project/session contracts second, optional analysis bridges later.
