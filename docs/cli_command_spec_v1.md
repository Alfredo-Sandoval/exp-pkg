# xpkg v1 CLI Command Spec

This document defines the shipped CLI for the xpkg project and artifact
workflow. It targets the v1 artifact format, but the command surface itself is
pre-1.0 and may still change before 1.0. The package is at 0.x (alpha). The
frozen guarantee is the `.expkg` on-disk format described in
[artifact_contract_v1.md](artifact_contract_v1.md), not this command surface.

The current CLI is project-first for project creation, importing, packing,
unpacking, validation, and artifact inspection.

## Command Surface

The project-first command surface is:

```text
xpkg artifacts
xpkg completion
xpkg describe
xpkg inspect
xpkg import
xpkg project
```

## Shared Rules

- The primary editable unit is a project folder.
- `.expkg` is the only portable project artifact.
- xpkg never edits a `.expkg` file in place.
- Commands that create a project must produce a valid project containing
  `PROJECT.json` and `.xpkg/`.
- Project-internal paths are stored relative to the project root.
- Pack defaults to `--media full`.
- Pack must fail loudly if required media are external to `Media/`.
- Every canonical command supports `--json` for machine-readable output.
- In `--json` mode, success payloads are wrapped in the envelope
  `{"ok": true, "data": <command-specific JSON object>}` and written to stdout
  as one JSON object.
- In `--json` mode, progress text is suppressed and errors are written to
  stderr as the envelope
  `{"ok": false, "error": {"code": ..., "message": ..., "hint": ...}}`.
- Exit codes are `0` for success, `1` for usage or runtime errors, `2` reserved
  for future auth/config failures, and `3` for not found.
- `--json` is reserved for machine output. Commands that import JSON files use
  `--input-json` for those input paths.
- `xpkg describe --json` exposes the current command contract for agents.
- Shell completion is exposed through `xpkg completion bash`, `xpkg completion
  zsh`, and `xpkg completion fish`.
- `xpkg inspect` and `xpkg project describe` are the preferred lightweight
  surfaces for project pickers and startup catalog scans.
- Commands named `inspect`, `describe`, `list`, or `summary` must not hydrate
  full labels, predictions, recordings, or media unless their command-specific
  contract explicitly says so.
- Commands named `validate`, `pack`, `unpack`, `load`, or `import` may read full
  state because they are explicit lifecycle or data actions.
- Downstream apps should follow [Performance Guidance](performance.md): list
  with descriptor/layout metadata, hydrate only after user selection, and
  validate only on explicit validation, pack, publish, or CI actions.

## `xpkg inspect`

Inspect a file, folder, project, or `.expkg` artifact before import.

### Synopsis

```bash
xpkg inspect tracking.csv
xpkg inspect tracking.csv --json
xpkg inspect "./My Project" --json
```

### Required behavior

- Does not mutate projects or input files.
- Reports the inferred input kind and likely importer names when available.
- Reports lightweight media, table, project, or pose-QC summaries when the
  metadata can be read safely.
- For project folders, reports descriptor/current-state presence without
  materializing full labels, predictions, or media.
- Emits warnings for invalid present metadata, missing referenced media, unknown
  formats, failed QC reads, or unavailable media backends. Optional FAIR
  metadata slots are reported as absent rather than warned on by ordinary
  inspect.
- Uses `--confidence-threshold` / `--threshold` only for pose-confidence QC.

### Project JSON shape

In `--json` mode, `xpkg inspect PROJECT --json` uses the standard success
envelope:

```json
{
  "ok": true,
  "data": {
    "status": "inspected",
    "path": "/path/to/My Project",
    "name": "My Project",
    "suffix": "",
    "exists": true,
    "is_dir": true,
    "size_bytes": null,
    "kind": "xpkg_project",
    "description": "xpkg project",
    "likely_importers": [],
    "summary": {},
    "warnings": [],
    "warning_records": []
  }
}
```

For project directories, `data.summary` has these stable top-level keys:

- `project_id`: project descriptor id.
- `title`: project descriptor title.
- `state_kind`: shallow current-state kind, one of `empty`, `labels`,
  `present`, or `unreadable`.
- `has_current_state`: whether `.xpkg/state/current.json` exists.
- `state_bytes`: byte size of the current state, or `null`.
- `commit_id`: durable-store commit id when available, or `null`.
- `modalities`: shallow modality labels inferred from summary/index data.
- `summary_path`: canonical `.xpkg/indexes/project_summary.json` path.
- `metadata_slots`: slot status map described below.
- `media`: associated-media inventory described below.

`metadata_slots` is always an object with these keys, in this order:
`acquisition`, `dataset_share`, `datasheet`, `model_card`, and
`pose_provenance`. Each slot contains `path`, `present`, and `valid`. When a
slot is absent, `present` is `false` and `valid` is `null`. When present and
parseable, `valid` is `true`. When present but invalid, `valid` is `false` and
the slot also includes an `error` string.

`media` is a list copied from the generated shallow project summary. Inspect
adds `exists` for every media item without decoding media. Summary-recorded
items can include `index`, `kind`, `path`, `backend`, `video_id`, `label`,
`frame_count`, `fps`, `duration_s`, `timebase`, `height`, `width`, `channels`, `image_count`,
`label_frame_count`, `max_label_frame_index`, `prediction_frame_count`, and
`max_prediction_frame_index`. Image-sequence items also include
`current_image_count` when the sequence directory is resolvable. `fps`,
`duration_s`, and `timebase` come from the summary-recorded media object at
save/import time; project inspect does not demux video containers to infer
dropped frames or FPS drift.

`warnings` is always a list of strings. Ordinary project inspect does not warn
for absent optional metadata slots. It remains a backward-compatible human
message list.

`warning_records` is always a list of objects with `code`, `message`, `path`,
and `severity`. For project directories and packed `.expkg` metadata, this is
the stable machine-readable warning surface. Current codes include
`project_metadata_invalid`, `project_media_missing`,
`project_media_image_count_drift`, `project_media_label_frame_out_of_range`,
`project_media_prediction_frame_out_of_range`,
`project_media_inventory_unavailable`, `project_summary_warning`,
`packed_project_metadata_invalid`, and `packed_project_manifest_missing`.
Ordinary project inspect warns for invalid present metadata, missing recorded
media, image-sequence count drift, out-of-range recorded label/prediction frame
indices, unreadable summaries or artifact indexes, and older labels summaries
that do not contain associated-media inventory.

## `xpkg project`

Manage project-first project lifecycle operations.

### Commands

- `xpkg project describe`
- `xpkg project init`
- `xpkg project metadata set <slot>`
- `xpkg project metadata show <slot>`
- `xpkg project pack`
- `xpkg project unpack`
- `xpkg project validate`

## `xpkg project init`

Create a new empty project with the canonical public layout.

### Synopsis

```bash
xpkg project init "./My Project"
xpkg project init "./My Project" --title "My Project"
```

### Required behavior

- Creates the project root if needed.
- Creates `PROJECT.json`.
- Creates `.xpkg/`.
- Creates `Media/` and `Exports/` when bootstrapping a new project.
- Initializes `project_id`, timestamps, and default descriptor fields.
- Refuses to overwrite a non-empty target unless an explicit future overwrite
  flag is added.

### Output

- A valid project folder at the requested output path.
- `PROJECT.json` conforming to `schemas/project.schema.json`.

## `xpkg import`

Import supported external data into a project. The CLI mirrors the
``ProjectService`` Python dispatch surface and exposes three families
(``pose``, ``calibration``, ``motion``); each takes a kebab-case ``format``
positional argument that matches ``ProjectService.import_pose``,
``ProjectService.import_calibration``.

### Synopsis

```bash
xpkg import pose dlc-csv --path tracking.csv --video video.mp4 --out "./My Project"
xpkg import pose dlc-h5 --path tracking.h5 --video video.mp4 --out "./My Project"
xpkg import pose dlc-project --path dlc_project --out "./My Project"
xpkg import pose lightning-pose-csv --path predictions.csv --video video.mp4 --out "./My Project"
xpkg import pose sleap-h5 --path analysis.h5 --video video.mp4 --out "./My Project"
xpkg import pose sleap-package --path labels.pkg.slp --out "./My Project"
xpkg import pose mmpose-topdown-json --input-json results.json --video video.mp4 --out "./My Project"
xpkg import pose mediapipe-pose-landmarks-json --input-json pose_landmarks.json --video video.mp4 --out "./My Project"

xpkg import calibration anipose --path rig.toml --out "./My Project"
xpkg import calibration opencv-stereo-yaml --path stereo.yml --out "./My Project"

```

### Supported formats today

- `xpkg import pose dlc-csv`
- `xpkg import pose dlc-h5`
- `xpkg import pose dlc-project`
- `xpkg import pose lightning-pose-csv`
- `xpkg import pose mediapipe-pose-landmarks-json`
- `xpkg import pose mmpose-topdown-json`
- `xpkg import pose sleap-h5`
- `xpkg import pose sleap-package`
- `xpkg import calibration anipose`
- `xpkg import calibration opencv-stereo-yaml`

### Required behavior

- Imports into a project, never directly into a new opaque single-file native
  artifact.
- Creates the project if it does not already exist.
- Writes authoritative mutable state into `.xpkg/`.
- Populates `Media/` when the import produces managed media.
- Updates `PROJECT.json` metadata and timestamps.

### Non-goals

- Import is not an in-place update of packed `.expkg` artifacts.

## `xpkg artifacts`

Inspect and validate registered project output artifacts.

### Synopsis

```bash
xpkg artifacts list "./My Project"
xpkg artifacts list "./My Project" --kind figure
xpkg artifacts inspect "./My Project" session-summary-figure --kind figure
xpkg artifacts validate "./My Project" --kind figure
xpkg artifacts validate "./My Project" session-summary-figure --kind figure
xpkg artifacts rebuild-index "./My Project"
```

### Required behavior

- Reads artifact manifests from a project.
- Lists compact entries from `.xpkg/artifacts/index.json`, rebuilding the index
  if it is missing.
- Prints one full manifest for `inspect`.
- Validates referenced input, output, and stats files.
- Verifies recorded checksums and sizes when present.
- Supports optional artifact kind and caller-owned namespace filters.

### Non-goals

- `artifacts` does not render plots, compute statistics, or choose scientific
  models.
- `artifacts` does not make `.expkg` files mutable; use `xpkg project pack` after
  project changes.

## `xpkg project metadata`

Read or write the durable typed metadata slots stored under
`.xpkg/metadata/`. The CLI mirrors `ProjectService.metadata` on the Python
side: each slot is a kebab-case positional argument matching the typed
accessor.

### Supported slots

- `acquisition` — `AcquisitionMetadata` (acquisition.json)
- `dataset-share` — `DatasetShareMetadata` (dataset_share.json), FAIR sharing fields
- `datasheet` — `DatasetDatasheet` (datasheet.json), Gebru et al. 2021
- `model-card` — `ModelCard` (model_card.json), Mitchell et al. 2019

### Synopsis

```bash
xpkg project metadata set acquisition "./My Project" --from acquisition.json
xpkg project metadata set dataset-share "./My Project" --from share.json --json
xpkg project metadata show acquisition "./My Project"
xpkg project metadata show datasheet "./My Project" --json
```

### Required behavior

- `set <slot>` reads a JSON object from `--from FILE`, validates it through the
  slot's typed `from_dict`, and writes it under `.xpkg/metadata/<slot>.json`.
- `show <slot>` reads the slot's JSON if present and emits the typed payload;
  returns `status: "missing"` when the slot has not been written.
- Unknown slot names produce a `usage_error` with a helpful "choose from:"
  hint and exit code `1`.

## `xpkg project describe`

Describe the normalized project layout and descriptor.

### Synopsis

```bash
xpkg project describe "./My Project"
xpkg project describe "./My Project" --json
```

### Required behavior

- Resolves the owning project root from the supplied path.
- Returns the normalized managed paths for `PROJECT.json`, `.xpkg/`, `Media/`,
  `Exports/`, the current state cache, and
  `.xpkg/indexes/project_summary.json`.
- Emits the current `PROJECT.json` descriptor in JSON mode.
- Refreshes and emits the generated project summary index in JSON mode.
- Does not validate, load labels, load predictions, or parse media
  payloads.

## `xpkg project pack`

Create a portable `.expkg` artifact from a project.

### Synopsis

```bash
xpkg project pack "./My Project"
xpkg project pack "./My Project" --out "./release/My Project.expkg"
xpkg project pack "./My Project" --media package
xpkg project pack "./My Project" --media manifest
```

### Required behavior

- Reads a project folder as input.
- Validates the project before packing.
- Emits a `.expkg` file.
- Defaults output to `./My Project/Exports/My Project.expkg` when `--out` is
  omitted.
- Fails if required media are external to `Media/`.
- Supports `--media full`, `--media package`, and `--media manifest`.
- Defaults to `--media full`, storing every managed media file.
- `--media package` stores package-sized media such as image sequences while
  manifesting video containers without storing their bytes.
- `--media manifest` stores no media bytes and records managed media paths,
  sizes, and SHA-256 digests in `EXPKG.json`.
- Writes a root `EXPKG.json` manifest declaring member paths, member sizes,
  member SHA-256 digests, media inclusion status, and compression policy.
- Stores already-compressed media and common binary containers without
  additional zip compression.

## `xpkg project unpack`

Create a project from a `.expkg` artifact.

### Synopsis

```bash
xpkg project unpack "./My Project.expkg" --out "./My Project"
```

### Required behavior

- Accepts a `.expkg` file as input.
- Creates a project folder as output.
- Reconstructs `PROJECT.json` and `.xpkg/`.
- Restores included `Media/` files from the artifact.
- Creates an empty `Media/` root when media were manifested but not stored.
- Excludes temp files, locks, caches, and machine-local scratch state.
- Refuses to unpack into a conflicting non-empty directory unless an explicit
  future overwrite flag is added.

## `xpkg project validate`

Validate a project or packed `.expkg` artifact.

### Synopsis

```bash
xpkg project validate "./My Project"
xpkg project validate "./My Project.expkg"
```

### Required behavior

- Accepts a project folder or `.expkg` file.
- Fails loudly when the supplied path does not satisfy the corresponding
  contract.
- For `.expkg`, verifies `EXPKG.json`, member path safety, duplicate member
  absence, member sizes, member SHA-256 digests, and media/member consistency.
- Leaves the validated artifact unchanged.

## Open Behavior

GUI and shell tooling should treat the project folder as the primary open
target.

- Opening a project opens it directly.
- Opening a `.expkg` prompts for an unpack destination, then opens the
  resulting project.
- Packed artifacts remain immutable from the user’s perspective.

## Transition Guidance

- New project creation follows the project + `.expkg` contract.
- No public command should frame direct `.xpkg` conversion as the native or
  preferred project workflow.
