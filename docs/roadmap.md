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

The product is not finished. The current package is strongest for workspace
lifecycle, portable artifacts, pose labels, video-associated imports, Vicon,
segmentation masks, and output artifact registration. The next work is to make
the multimodal session layer usable end to end for pose, video, photometry,
events, and synchronization.

## Current Baseline

Implemented and covered by the normal package gates:

- distribution name: `exp-pkg`
- Python import name: `xpkg`
- CLI command: `xpkg`
- workspace folder + private `.xpkg/` state + portable `.expkg` artifact
- `WorkspaceService` lifecycle API
- Typer CLI with JSON output for canonical commands
- readers and workspace importers for Vicon, DeepLabCut, Lightning Pose, SLEAP,
  MMPose, and MediaPipe pose-landmark data
- segmentation-mask workspace helpers
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
CSV files are fully imported into workspaces.

## Next Implementation Priorities

### 1. Direct Readers

Simple, low-ceremony CSV readers now exist before the workspace machinery:

```python
xpkg.read_photometry_csv(...)
xpkg.read_events_csv(...)
xpkg.read_pyphotometry_ppd(...)
```

These return `PhotometryRecording`, `EventTable`, or session-level objects
without requiring users to create a workspace first. `read_sync_csv(...)` is
still the next direct reader in this family.

### 2. Workspace Imports

After direct readers exist, wire them into `WorkspaceService`:

```python
workspace.imports.photometry_csv(...)
workspace.imports.events_csv(...)
workspace.imports.sync_csv(...)
```

These imports should store normalized data under the workspace contract and
preserve enough provenance to rebuild, inspect, and package the session.

### 3. Inspect-First CLI

Add an inspection command for users and agents:

```bash
xpkg inspect file_or_folder --json
```

The command should report the likely kind, likely importer, duration, frame or
sample count, channels or columns, timestamp availability, associated media,
missing files, and warnings.

### 4. Session Manifest

Version a session-level manifest that can describe:

- timelines
- videos
- pose recordings
- signal recordings
- photometry recordings
- events
- sync pulses and maps
- artifacts

The manifest should be plain enough for other labs and tools to depend on
without adopting an analysis framework.

### 5. Exchange Bridges

Keep the core package dependency-light. Add optional bridges only after the core
IO contract is stable:

```python
xpkg.exchange.to_numpy(...)
xpkg.exchange.to_dataframe(...)
xpkg.exchange.to_torch_dataset(...)  # optional future extra
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
5. wire the reader into workspace imports
6. expose CLI JSON output
7. document the supported workflow honestly

That order keeps `xpkg` close to the `sleap-io` lesson: pleasant direct Python
IO first, durable workspaces second, optional analysis bridges later.
