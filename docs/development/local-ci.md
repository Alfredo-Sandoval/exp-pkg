# Local Quality Gates

`xpkg` is set up to run its quality gates locally on macOS and Linux without
needing hosted CI.

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
make lint
make typecheck
make test
make test-real
make docs-build
make package-check
```

Run the standard local quality gate:

```bash
make qa
```

Run the broader local quality pass:

```bash
make ci-local
```

Run the package gate before publishing or handing off a wheel:

```bash
make package-check
```

That gate builds the sdist and wheel, runs `twine check`, installs the wheel in
a temporary fresh virtual environment, smoke-tests `xpkg --help`, and verifies
that `ProjectService` imports from the installed package.

Run the release gate before a package handoff or PyPI/TestPyPI cut:

```bash
make release-check REAL_DATA_ROOT=/path/to/xpkg-real-data
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

## Real Data Manifest

Put `xpkg-real-data.json` at the root of your private corpus, or set
`XPKG_REAL_DATA_MANIFEST` to a manifest file. Paths inside the manifest are
resolved relative to `XPKG_REAL_DATA_ROOT` unless they are absolute.

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
    },
    {
      "id": "vicon-trial-001",
      "kind": "vicon",
      "recording": "vicon/trial_001.c3d",
      "expect": {
        "state": "vicon"
      }
    }
  ]
}
```

Supported case kinds are:

- `vicon`
- `dlc`
- `lightning_pose`
- `sleap`
- `mmpose`, `mediapipe`

Use `kind: "vicon"` for both CSV and C3D recordings; the importer chooses the
reader from the file extension. Use `kind: "dlc"` with either `tracking` plus
`video` for a single CSV/H5 tracking file, or `project` for a full DLC project
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
