# Local Quality Gates

`xpkg` is set up to run its quality gates locally on macOS and Linux. The
synthetic gate is the canonical public check; the real-data release gate remains
local/private.

## First-Time Setup

Use the canonical setup entrypoint:

```bash
make env
```

This creates or updates the `xpkg` conda or mamba environment, installs the
local development toolchain, and installs the package in editable mode.

## Main Commands

Run the standard quality gates individually:

```bash
make conflict-check
make export-stubs-check
make lint
make typecheck
make test
make test-vendor
make test-real
make docs-build
make package-check
```

Run the standard local quality gate:

```bash
make qa
```

## Documentation Toolchain

This repository remains on MkDocs 1.x. MkDocs 2 is not a compatible upgrade
for this site because it removes the plugin system and changes the theme and
configuration contracts used by Material for MkDocs and `mkdocstrings`.

The `docs-build` and `docs-serve` targets set `NO_MKDOCS_2_WARNING=1` to suppress
Material's advisory after this decision. Reconsider MkDocs 2 only when the
Material theme and API-documentation plugins have a supported migration path.
See the [Material for MkDocs compatibility assessment](https://squidfunk.github.io/mkdocs-material/blog/2026/02/18/mkdocs-2.0/).

Run the broader local quality pass:

```bash
make ci-local
```

Run the package gate before publishing or handing off a wheel:

```bash
make package-check
```

That gate builds the sdist and wheel, runs `twine check`, installs the wheel in
a temporary fresh virtual environment, verifies `xpkg describe --json` exposes
the expected machine contract, and checks that installed-wheel
`ProjectService.create()` writes a valid project descriptor and summary index.

Run the release gate before a package handoff or PyPI/TestPyPI cut:

```bash
make release-check REAL_DATA_ROOT=../xpkg-real-data
```

The installed wheel should include `xpkg/py.typed` and
`xpkg/schemas/project.schema.json`, so downstream users get both typing metadata
and the public project schema.

`ci-local` runs:

- lint
- typecheck
- tests
- packaging check
- strict docs build

`test-real` runs the opt-in real-data integration suite under
`tests/real_data`. It requires either `XPKG_REAL_DATA_ROOT` or the
`REAL_DATA_ROOT=...` make argument.

`test-vendor` runs the genuine vendor-export contract suite. It is excluded
from ordinary `pytest`, `make test`, and `make coverage` runs. Supply all three
fixture roots explicitly:

```bash
make test-vendor \
  FIBER_FIXTURE_ROOT=../xpkg-vendor-data/fiber-photometry \
  POSE_FIXTURE_ROOT=../xpkg-vendor-data/pose \
  BEHAVIOR_FIXTURE_ROOT=../xpkg-vendor-data/behavior
```

The target raises if a root or required fixture is absent. It never activates
because an ignored directory happens to exist in the checkout.

`release-check` runs:

- `qa`
- package check
- strict docs build
- real-data import/validate/pack/unpack tests

## Environment Behavior

The Makefile routes quality targets through `environment/run-in-env.sh`, which
executes them inside the configured `xpkg` environment. You do not need to
manually activate the environment before using the normal development targets.

If the environment is missing, the wrapper will tell you to run `make env`
first.

If conda or mamba is unavailable but a local virtualenv, repo-local `.venv`, or
non-base conda environment already has the project dependencies installed, the
wrapper runs the quality command in that environment. This is a fallback for
prepared developer machines and CI; `make env` remains the canonical setup
entrypoint.

`make ci-local` also runs `make performance-check`. The canonical performance
budgets are stored in `performance-budgets.json`, and CI enforces the same
versioned benchmark contract.

`xpkg.model` and `xpkg.pose` keep their lazy public exports in adjacent
`_exports.py` registries. Those registries are the source of truth for the
generated runtime facades and `__init__.pyi` typing facades. Run `make
export-stubs` after changing either registry. `make qa` rejects stale generated
files.

## Real Data Manifest

Put `xpkg-real-data.json` at the root of your private corpus, or set
`XPKG_REAL_DATA_MANIFEST` to a manifest file. Paths inside the manifest are
resolved relative to `XPKG_REAL_DATA_ROOT`; keep manifest entries relative for
portability.

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

Supported case kinds are:

- `dlc`
- `lightning_pose`
- `sleap`
- `mmpose`, `mediapipe`

Use `kind: "dlc"` with either `tracking` plus `video` for a single CSV/H5
tracking file, or `project` for a full DLC project
folder. Use `kind: "lightning_pose"` with `tracking` plus `video` for a
Lightning Pose prediction CSV produced by `litpose predict`. Use
`kind: "sleap"` with a `labels` file ending in `.slp`, `.pkg.slp`, `.h5`, or
`.hdf5`; SLEAP analysis H5 cases also need a matching `video`. Each
case imports into a fresh project, validates it, packs it to `.expkg`,
unpacks it, and validates the restored project. Set `"skip_pack": true` only
for a case that is deliberately too large for the pack/unpack release pass.

## Scope

This local quality surface is intended for:

- day-to-day development on macOS and Linux
- pre-handoff checks
- validating storage and project experiments cheaply

It is not a replacement for targeted regression tests, but it gives the repo a
consistent local quality bar.
