# Local CI

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
make docs-build
make package-check
```

Run the standard local quality gate:

```bash
make qa
```

Run the broader local CI pass:

```bash
make ci-local
```

`ci-local` runs:

- lint
- typecheck
- tests
- packaging check
- strict docs build

## Environment Behavior

The Makefile routes quality targets through `environment/run-in-env.sh`, which
executes them inside the configured `xpkg` environment. You do not need to
manually activate the environment before using the normal development targets.

If the environment is missing, the wrapper will tell you to run `make env`
first.

## Scope

This local CI surface is intended for:

- day-to-day development on macOS and Linux
- pre-commit or pre-push checks
- validating storage and workspace experiments cheaply

It is not a replacement for targeted regression tests, but it gives the repo a
consistent local quality bar.
