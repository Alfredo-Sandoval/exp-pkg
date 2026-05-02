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

## GitNexus

- This repo is indexed by GitNexus as `exp-pkg`.
- Use GitNexus for graph-guided exploration, impact analysis, refactors, renames, and pre-commit scope checks.
- Before editing a function, class, or method, run:
  - `gitnexus_impact({target: "symbolName", direction: "upstream"})`
- Report the blast radius before editing code symbols.
- If impact is `HIGH` or `CRITICAL`, warn the user before proceeding.
- Use `gitnexus_query({query: "concept"})` for graph-guided exploration.
- Use `gitnexus_context({name: "symbolName"})` for callers, callees, and process membership.
- Before renaming symbols, run `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` and review the preview before applying it.
- Before committing, run `gitnexus_detect_changes()` to confirm the scope matches intent.
- If the index is stale, run `make gitnexus-analyze` or `npx gitnexus analyze --skip-agents-md`.
- Do not run plain `npx gitnexus analyze`; it may rewrite `AGENTS.md`, create unwanted agent files, or preserve stale repo names.
- Useful resources:
  - `gitnexus://repo/exp-pkg/context`
  - `gitnexus://repo/exp-pkg/clusters`
  - `gitnexus://repo/exp-pkg/processes`
  - `gitnexus://repo/exp-pkg/process/{name}`
