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

The current focus is intentionally narrow:

- project folder + private `.xpkg/` state + portable `.expkg` artifact
- `ProjectService` lifecycle API
- service and CLI project importers for pose, calibration, and generic
  photometry CSV
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
- add event and synchronization import actions to session state
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

Generic photometry CSV is the first signal importer to meet that bar. It
copies the source into managed media, records its checksum and relative path,
commits a typed session inside experiment state and survives `.expkg`
pack/unpack.

## Non-Goals

- Do not turn `xpkg` into an analysis framework.
- Do not add project importers without a durable storage contract.
- Do not add heavyweight modality support just because a reader exists.
- Do not make downstream GUI repos parse private `.xpkg/` layout details.
