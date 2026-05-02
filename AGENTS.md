# AGENTS.md

These instructions apply to all agent edits in this repository.

## Scope and Precedence

If instructions conflict, follow this order:

1. Explicit user request
2. This `AGENTS.md`
3. Other repository docs

## Repository Identity

Repository identity is maintained directly in this file and package metadata.

- Name: `exp-pkg`
- Slug: `Alfredo-Sandoval/exp-pkg`
- Support matrix: `macos`, `linux`

## Instruction Routing

Use this `AGENTS.md`, explicit user requests, and the current task context as
the repository instruction source.

## Environment Policy

- Canonical setup entrypoint: `make env`
- Fallback setup entrypoint: `bash environment/setup.sh`
- Setup scripts:
  - `environment/setup.sh` (dispatcher)
  - `environment/macos/setup.sh`
  - `environment/linux/setup.sh`
- Do not add `environment/windows/` unless explicitly requested by the user.
- Use `mamba` (or `conda` fallback) with OS-local `environment.yml`.
- Install Python dependencies with environment-bound `uv pip`.

## Quality Gates

Use `Makefile` targets as the canonical command surface:

- `make lint` -> `ruff check .`
- `make typecheck` -> `ty check`
- `make test` -> `pytest`
- `make qa` -> lint + typecheck + test

## Licensing Policy

- Default license is BSD-3-Clause.
- Rights holders: `Alfredo and Joseph Sandoval`.
- Keep `LICENSE`, `pyproject.toml`, and README license text aligned.

## Portability Rules

- Use repo-relative paths only.
- Do not introduce absolute paths in code, docs, or setup scripts.
- Use `pathlib` for Python path handling.

## Git Safety

- Never discard user changes without explicit approval.
- Avoid destructive commands such as `git reset --hard`, `git clean -fd`, and `git checkout --`.
