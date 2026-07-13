# Roadmap

<div class="page-intro">
<p>
This page separates the current package contract from the IO work that is still
ahead.
</p>
</div>

`exp-pkg` is a project-first IO package. It builds an sdist and wheel, installs
the `xpkg` console script, publishes typed package metadata, and ships the
public project schema in the wheel.
The wheel also ships generated schema-4 experiment, recording-session, and
ontology documents.

The current focus is intentionally narrow:

- project folder + private `.xpkg/` state + portable `.expkg` artifact
- `ProjectService` lifecycle API
- service and CLI project importers for pose, calibration, generic photometry,
  events, behavior, and paired timebase synchronization
- versioned `Experiment` project state for subjects, protocols, conditions,
  recording sessions, modalities, timing, and provenance
- read-only `xpkg inspect PATH --json` for projects, media, common tables,
  pose exports, and `.expkg` artifacts
- segmentation-mask project helpers
- generic artifact and figure registries
- package metadata suitable for PyPI/TestPyPI checks
- inline typing marker through `xpkg/py.typed`

## Near-Term Priorities

The next useful work is still about IO contracts, not analysis surface:

- keep project pickers shallow through descriptors and generated summaries
- make media handling more predictable for GUI consumers
- keep pose importers fast, typed, and provenance-aware
- add more source-specific synchronization adapters only when they provide
  paired clock observations or an equally explicit timing contract
- add more calibration and 3D pose boundary adapters
- improve inspect-first CLI behavior before mutating projects
- treat behavior segmentation outputs as imported data products, not algorithms
  that `xpkg` claims to train or run

## Direct Readers

Direct readers remain useful when another repo needs typed in-memory objects
without creating a project first. Today that lane covers photometry, behavior,
events, pose tracks, and calibration formats.

Project-bound imports are narrower by design. New direct readers should only
become `ProjectService` importers when they have a clear project-storage
contract, validation story, and portability behavior.

Generic photometry CSV, generic event CSV, and the supported behavior formats
meet that bar. They copy source files into managed media, record checksums and
relative paths, commit typed objects inside experiment state, and survive
`.expkg` pack/unpack.

## Non-Goals

- Do not turn `xpkg` into an analysis framework.
- Do not add project importers without a durable storage contract.
- Do not add heavyweight modality support just because a reader exists.
- Do not make downstream GUI repos parse private `.xpkg/` layout details.
