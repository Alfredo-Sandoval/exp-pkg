# Roadmap

<div class="page-intro">
<p>
This page separates the current package contract from the multimodal
neuroscience IO work that is still ahead.
</p>
</div>

`exp-pkg` is now a proper Python package baseline: it builds an sdist and wheel,
installs the `xpkg` console script, publishes typed package metadata, and ships
the public project schema in the wheel.

The product is not finished. The current package is strongest for project
lifecycle, portable artifacts, pose labels, video-associated imports, Vicon,
segmentation masks, and output artifact registration. The next work is to make
the multimodal session layer usable end to end for pose, video, photometry,
events, and synchronization.

## Current Baseline

Implemented and covered by the normal package gates:

- distribution name: `exp-pkg`
- Python import name: `xpkg`
- CLI command: `xpkg`
- project folder + private `.xpkg/` state + portable `.expkg` artifact
- `ProjectService` lifecycle API
- Typer CLI with JSON output for canonical commands
- read-only `xpkg inspect PATH --json` for projects, media, common tables,
  pose exports, and `.expkg` artifacts
- readers and project importers for Vicon, DeepLabCut, Lightning Pose, SLEAP,
  MMPose, and MediaPipe pose-landmark data
- segmentation-mask project helpers
- generic artifact and figure registries
- package metadata suitable for PyPI/TestPyPI checks
- inline typing marker through `xpkg/py.typed`
- public project schema included in the installed wheel

## Emerging Session Spine

The first multimodal model layer is public and intentionally small:

- `Timebase`
- `Timeline`
- `TimeRange`
- `Event`
- `SyncEvent`
- `EventTable`
- `SignalChannel`
- `PhotometryChannel`
- `TimeSeries`
- `PhotometryRecording`
- `RecordingSession`

These objects establish the direction: modality-specific containers with a
shared timing contract. They do not yet mean that photometry, events, and sync
CSV files are fully imported into projects.

## Next Implementation Priorities

Luxem et al. 2023 frames the useful priority order for this repository: make
behavior-video data easier to inspect, exchange, reproduce, and reuse before
adding more analysis surface. For `xpkg`, that translates into these near-term
product priorities:

- one common session/package contract for video, pose, events, photometry,
  synchronization, masks, and imported analysis outputs
- FAIR/share metadata that records enough source, tool, parameter, and
  experimental context for another lab to evaluate or reuse the package
- acquisition metadata for cameras, frame rates, resolution, compression,
  hardware sync, dropped-frame evidence, and timing uncertainty
- inspect-first CLI behavior so users and agents can identify files, likely
  importers, timestamps, missing media, and QC warnings before mutating a
  project
- import-time QC that checks timestamp monotonicity, frame/sample counts,
  associated media, keypoint confidence/identity fields, and sync coverage
- behavior segmentation outputs treated as imported data products, not
  algorithms that `xpkg` claims to train or run

Those priorities are intentionally about IO and contracts. `xpkg` should make
SimBA, MARS, B-SOID, VAME, MoSeq, or lab-specific outputs portable and
auditable when imported; it should not present itself as a behavior
segmentation package.

### 1. Direct Readers

Simple, low-ceremony CSV readers now exist before the project machinery:

```python
from xpkg import readers

readers.read_photometry_csv(...)
readers.read_events_csv(...)
readers.read_behavior_events_csv(...)
readers.read_behavior_events_json(...)
readers.read_pyphotometry_ppd(...)
readers.read_pyphotometry_csv(...)
readers.read_pmat_photometry_csv(...)
readers.read_pmat_events_csv(...)
readers.read_rwd_ofrs_session(...)
readers.read_neurophotometrics_csv(...)
readers.read_doric_photometry(...)
readers.read_teleopto_h5(...)
readers.read_tdt_photometry_block(...)
```

These return `PhotometryRecording`, `EventTable`, `BehaviorLabels`, or
session-level objects without requiring users to create a project first.
`read_sync_csv(...)` is still the next direct reader in this family.

The fiber-photometry reader set is scoped to fiber/session IO. Inscopix
miniscope files, Blackrock NEV/NSx, and Neuralynx Cheetah files are deliberately
excluded from this layer.

### 2. Project Imports

After direct readers exist, wire them into `ProjectService`:

```python
project.import_signals("photometry-csv", ...)
project.import_signals("events-csv", ...)
project.import_signals("sync-csv", ...)
```

These imports should store normalized data under the project contract and
preserve enough provenance to rebuild, inspect, and package the session.
They should also run lightweight QC at import time and keep warnings with the
project instead of hiding them in console output.

### 3. Inspect-First CLI

Expand the inspection command for users and agents:

```bash
xpkg inspect file_or_folder --json
```

The command now reports likely kinds, likely importers, selected media/table
metadata, project summaries, `.expkg` summaries, and basic pose-confidence QC.
Next it should grow associated-media matching, missing-file checks, acquisition
metadata, sync evidence, richer timestamp checks, and project-aware warnings.
It should remain read-only so users can decide whether a file is safe to import
before `xpkg` writes project state.

### 4. Session Manifest

Version a session-level manifest that can describe:

- acquisition metadata
- timelines
- videos
- pose recordings
- signal recordings
- photometry recordings
- events
- sync pulses and maps
- artifacts

The manifest should be plain enough for other labs and tools to depend on
without adopting an analysis framework. It should carry FAIR/share metadata
for source files, software producers, parameter summaries, checksums, units,
coordinate systems, and timing assumptions.

### 5. Behavioral Segmentation Imports

Support behavior segmentation results as imported outputs from upstream tools
or lab workflows. The first `BehaviorLabels` contract now preserves:

- behavior labels or motif identifiers
- start/end frames or start/end times
- source pose/video/session references
- producer metadata, parameters, and software version when available
- confidence scores or uncertainty fields when provided
- import-time QC warnings when segments do not align to available timelines

Package-specific importers for B-SOiD, A-SOiD, Keypoint-MoSeq, SimBA, VAME,
DeepEthogram, BORIS, and JAABA are still explicit follow-on work.

This is an IO priority, not an algorithmic one. `xpkg` should help compare and
share behavior annotations across tools by normalizing their outputs under the
session contract.

### 6. Adapters Bridges

Keep the core package dependency-light. Add optional bridges only after the core
IO contract is stable:

```python
xpkg.adapters.to_numpy(...)
xpkg.adapters.to_dataframe(...)
xpkg.adapters.to_torch_dataset(...)  # optional future extra
```

PyTorch and NWB should remain optional downstream bridges, not required working
formats.

## Release Gates

Before public package release:

```bash
make qa
make package-check
make docs-build
make release-check REAL_DATA_ROOT=/path/to/xpkg-real-data
```

`make release-check` is the gate that uses private representative data. The
normal synthetic suite proves the package contract; the private real-data suite
proves the package survives real lab files.

## Adoption Rule

Prefer this order for new modality support:

1. define or reuse the minimal model object
2. add a direct reader
3. add focused synthetic tests
4. add real-data cases privately
5. wire the reader into project imports
6. expose CLI JSON output
7. document the supported workflow honestly

That order keeps `xpkg` close to the `sleap-io` lesson: pleasant direct Python
IO first, durable projects second, optional analysis bridges later.
