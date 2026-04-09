# AGENTS.md

These instructions apply to all agent edits in this repository.

## Scope and Precedence

If instructions conflict, follow this order:

1. Explicit user request
2. This `AGENTS.md`
3. Other repository docs

## Repository Identity

Canonical identity lives in `repo_profile.yaml`.

- `repo.name`: `exp-pkg`
- `repo.slug`: `Alfredo-Sandoval/exp-pkg`
- `repo.preset`: `base`
- `repo.support_matrix`: `macos`, `linux`

## Required Skills

For repository tasks, always load:

- `repo-setup`
- `agent-playbook`

Then use `repo_profile.yaml` `skills.routing` for intent-specific skills.

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

- Default license is proprietary (`All rights reserved`).
- Rights holders: `Alfredo and Joseph Sandoval`.
- Keep `LICENSE` aligned with `repo_profile.yaml`.

## Portability Rules

- Use repo-relative paths only.
- Do not introduce absolute paths in code, docs, or setup scripts.
- Use `pathlib` for Python path handling.

## Git Safety

- Never discard user changes without explicit approval.
- Avoid destructive commands such as `git reset --hard`, `git clean -fd`, and `git checkout --`.
