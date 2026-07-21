# xpkg Object and Action Registry

`xpkg` (distribution `exp-pkg`) is the core I/O layer for the pose/vision stack:
it owns project identity, media, pose/label, segmentation, calibration,
multimodal signals, provenance, and the portable `.expkg` artifact. Downstream
repos (for example Fiesta) consume these objects rather than defining their own.

This registry is the **root node** of the federated ontology: each repo owns its
object types, and consumers reference the objects defined here instead of
restating them. It is a map of contracts that already exist in code; keep it
aligned with the code and the existing design docs rather than duplicating them:

- [`identity_provenance.md`](../identity_provenance.md) — identity and provenance model.
- [`artifact_contract.md`](../artifact_contract.md) — frozen on-disk contract (v1).
- [`artifact-namespaces.md`](../artifact-namespaces.md) — namespace boundaries.
- [`api/model.md`](../api/model.md), [`api/project.md`](../api/project.md) — documented object/API surface.
- [`schemas/project.schema.json`](https://github.com/Alfredo-Sandoval/exp-pkg/blob/main/schemas/project.schema.json) — canonical `PROJECT.json` schema.

Ownership rule: an object's owner is the single module allowed to define its
durable shape. The public surface is the curated `__all__`/`_EXPORTS` of each
package (`xpkg.model`, `xpkg.project`, `xpkg.services`, `xpkg.adapters`); on-disk
layout constants and `project_*` path helpers are private — reach objects
through `ProjectService`.

## Status Labels

- `promoted`: safe as a shared contract. The on-disk **artifact contract is
  frozen at v1**; core project/pose/media objects are stable to depend on.
- `active`: implemented and public, but the Python/CLI surface is pre-1.0 and may
  still shift.
- `experimental`: private machinery or a target architecture that may change.
- `deprecated`: kept only to explain migration.
- `example`: synthetic/demo-only.

## Object Types

### Project and storage (root)

| Object | Status | Owner | Contract |
| --- | --- | --- | --- |
| `Project` | promoted | `xpkg.project` + `xpkg.services.ProjectService` | Root object. There is no `Project` class: a project is a `ProjectDescriptor` plus the on-disk private store `.xpkg/`, with `ProjectService` as the runtime handle. Layout: `PROJECT.json` (required), `.xpkg/` (private authoritative store), `Media/`, `Exports/`. |
| `ProjectDescriptor` | promoted | `xpkg.project.layout` | `PROJECT.json` payload. Ten required fields including `format="xpkg-project"`, `project_schema_version=1`, `layout_version=1`, `title`, `project_id` (UUID/ULID), `created_at`/`updated_at` (`Z`), `store_path=".xpkg"`, `media_root="Media"`, `exports_root="Exports"`. |
| `ProjectManifest` / `AssetEntry` | promoted | `xpkg.io.manifest` | Asset registry. `AssetEntry` = `id` (derived `sha1(project-relative path)[:8]`), `label`, `path`, `asset_type: AssetType`, `exists`, `modified_at`, `metadata`. `AssetType` = `VIDEO/MODEL/CHECKPOINT/SKELETON/PREDICTIONS/CONFIG/OTHER`. |
| `ArtifactManifest` / `ArtifactFile` / `ArtifactIndexEntry` | promoted | `xpkg.project.artifacts` | Portable artifact lineage record: `artifact_type`, `artifact_id`, `namespace`, `title`, `inputs`, `outputs`, `producer`, `stats`, per-file `ArtifactFile.sha256`. Rebuildable index at `.xpkg/artifacts/index.json`. |
| `ExportArtifact` (`.expkg`) | promoted | `xpkg.project` (`pack`/`unpack`) | Portable single-zip project with root `EXPKG.json` and per-file SHA-256 digests; media modes `full`/`package`/`manifest`. Lives under `Exports/`. Not an editable input. |

### Pose and labels

| Object | Status | Owner | Identity / join key |
| --- | --- | --- | --- |
| `Labels` | promoted | `xpkg.io.labels.model` | Container: `labeled_frames`, `videos`, `skeletons`, `tracks`, `suggestions`, `provenance`, `identity_provenance`, `session`. |
| `LabeledFrame` | promoted | `xpkg.pose.annotations.frames` | `(video, frame_idx)`. Holds instances, heatmaps, masks, ROIs. |
| `Instance` / `PredictedInstance` | promoted | `xpkg.pose.annotations.instances` | `track` + parent `frame`; `PredictedInstance` adds `score`; `from_predicted` links a corrected instance to its source prediction. |
| `Track` | promoted | `xpkg.pose.annotations.instances` | `id` (== `int(spawned_on)`) and `name` — the pose-track identity join key. |
| `Point` / `PredictedPoint` / `PointArray` | promoted | `xpkg.pose.annotations.points` | Positional by skeleton keypoint index; dtype `x,y,visible,complete,flags`. Flags: `KPFlag` (`OCCLUDED/NO_TRAIN/INTERP/LOCKED`). |
| `Skeleton` | promoted | `xpkg.pose.skeleton` | `name` + id-keyed `keypoints`; `links_ids`, `schema_version`, `triads`, `units`, `coordinate_system`, `calibration`. |
| `Keypoint` | promoted | `xpkg.pose.skeleton` | `name` (`__hash__ = hash(name)`) is identity; `id` is storage index; `mirror_partner`, `side`, `role`, `entity`. |
| `SuggestionFrame` | active | `xpkg.io.labels.model` | `(video, frame_idx)` + `group`, `score`. Frames proposed for labeling. |
| `PoseTrajectory` | active | `xpkg.pose.trajectory` | Analysis-facing adapter shape: `fps`, `frame_offset`, `skeleton_edges`, `source_kind`. |

### Media

| Object | Status | Owner | Identity / join key |
| --- | --- | --- | --- |
| `Video` | promoted | `xpkg.media` (`readers`/`video`) | **`sha256`** (file content hash) plus `id`/`filename` — the media join key across repos. Carries `width/height/frames/fps/channels`, `backend`. |
| `SingleImageVideo` / readers | active | `xpkg.media.readers` | Image-sequence and backend resources (OpenCV/PyAV/decord). |

### Segmentation

| Object | Status | Owner | Identity / join key |
| --- | --- | --- | --- |
| `SegmentationMask` | active | `xpkg.segmentation.model` | `mask_id`; RLE fields; `track` + `instance_ref` link to a pose `Instance`; `artifact_ref`. |
| `ROI` | active | `xpkg.segmentation.model` | `track` + `instance_ref`; box `x1,y1,x2,y2`. |
| `SegmentationPrompt` | active | `xpkg.segmentation.model` | `text`, `model_id`, `backend`. |

### Identity and provenance

| Object | Status | Owner | Identity / join key |
| --- | --- | --- | --- |
| `IdentityProvenanceRecord` | active | `xpkg.model.identity` | Companion keyed by `track_id`/`track_name` (not widened onto `Track`). Schema `xpkg.identity_provenance.v1`; `identity_source ∈ {mot,reid,manual,mixed,unknown}`. |
| `IdentityConfidenceSpan` / `IdentityEvent` / `IdentityProofreadingSpan` | active | `xpkg.model.identity` | Frame-scoped by `video_id` + `(start_frame,end_frame)`; swaps link `from_track_id`→`to_track_id`. |
| `PoseModelProvenance` | active | `xpkg.model.metadata` | Dataset-level: `tool`, `tool_version`, `model_name`, `checkpoint_id`, `training_set_reference`, `imported_from/at`. |

### Multimodal, metadata, and calibration

| Object | Status | Owner | Notes |
| --- | --- | --- | --- |
| `RecordingSession` | active | `xpkg.model.session` | `session_id` join key; binds `pose`, `videos`, `signals`, `events` on a shared `Timebase`. |
| `Timebase` / `Timeline` / `TimeRange` | active | `xpkg.model.time` | Shared time semantics across modalities. |
| `TimeSeries` / `SignalChannel` / `PhotometryChannel` / `PhotometryRecording` | active | `xpkg.model.signals` | Timed signal data. |
| `Event` / `EventTable` / `SyncEvent` | active | `xpkg.model.events` | Discrete and sync events. |
| `EMGSignalData` / `ForcePlateData` | active | `xpkg.model.emg`, `xpkg.model.force` | Biomechanics signals. |
| `BehaviorLabels` / `BehaviorInterval` / `BehaviorFrameLabel` / `BehaviorEmbedding` | active | `xpkg.model.behavior` | Behavior annotations (`BEHAVIOR_LABELS_SCHEMA_VERSION`). |
| `Calibration` / `Camera` (+ intrinsics/extrinsics/distortion/quality) | active | `xpkg.model.calibration` | `CALIBRATION_SCHEMA_VERSION = 1`; `world_frame`, `units`, `source` provenance. |
| `CameraMetadata` / `AcquisitionMetadata` / `DatasetShareMetadata` | active | `xpkg.model.metadata` | Acquisition and sharing/DOI/license metadata. |
| `ModelCard*` / `Dataset*` / `Datasheet*` | active | `xpkg.model.reporting` | Governance/reporting documents. |

## Action Types

The canonical action surface is `xpkg.services.ProjectService`
(`xpkg/services/project.py`); the `xpkg.project` function surface is the seam it
sits on. Reach actions through the service.

| Action group | Status | Owner | Output / effect |
| --- | --- | --- | --- |
| lifecycle: `create`, `ensure`, `open`, `init_project`, `ensure_project` | promoted | `ProjectService` / `xpkg.project` | New/opened `Project` (`.xpkg/` store + `PROJECT.json`). |
| `pack` / `unpack` (`pack_project`/`unpack_project`) | promoted | `ProjectService` / `xpkg.project` | `ExportArtifact` (`.expkg`) ↔ project folder. |
| `describe` / `validate` / `inspect` (`validate_project`, `validate_expkg`, `inspect_project`) | promoted | `ProjectService` / `xpkg.project` | `ProjectLayout` / `ProjectInspection`; fail-fast contract checks. |
| labels/state: `load_labels`, `save_labels`, `load_payload`, `load/save_state_metadata` | promoted | `ProjectService` / `xpkg.project` | Persisted `Labels`; durable-store commit; rebuildable `.xpkg/state/current.json`. |
| ingest: `import_pose`, `import_calibration` | active | `ProjectService` → `xpkg.io.converters.*` | Imports SLEAP, DLC, Lightning Pose, MediaPipe, MMPose, anipose, OpenCV stereo → project. |
| readers: `read_*` | active | `xpkg.readers` (`xpkg.io.readers`) | In-memory objects from lab files: pose, calibration, photometry (Doric/Neurophotometrics/pyPhotometry/NWB/TDT), events, behavior (BORIS/B-SOiD/SimBA/keypoint-MoSeq), skeletons. |
| exchange: `labels_to/from_json_payload`, `labels_numpy`, `labels_to_dataframe` | active | `xpkg.adapters` | Round-trip `Labels` ↔ JSON/numpy/DataFrame (not file formats). |
| artifacts: `artifacts().register/load/list/index/validate/delete/rebuild_index` | promoted | `ProjectService.artifacts` (`xpkg.project.artifacts`) | `ArtifactManifest` rows + rebuildable index. |
| segmentation: `segmentation().load_frames/load_masks/save_masks/clear_masks` | active | `ProjectService.segmentation` | Persisted `SegmentationMask`/`ROI`. |
| calibrations: `calibrations().save/load/list/import_anipose/import_opencv_stereo` | active | `ProjectService.calibrations` | Persisted `Calibration`. |
| metadata: `metadata().acquisition/dataset_share/pose_provenance/model_card/datasheet + update` | active | `ProjectService.metadata` | Persisted metadata objects. |

## Identity and Provenance (the cross-repo seam)

These are the stable identifiers the federation and downstream repos link on.
`xpkg` records where identities came from; it does not run ReID/MOT. See
[`identity_provenance.md`](../identity_provenance.md).

| Join key | Scope | Definition |
| --- | --- | --- |
| `Video.sha256` | media | SHA-256 of the source file; the primary media identity across repos. Portable `EXPKG.json` records per-file SHA-256 digests. |
| `Track.id` / `Track.name` | pose annotation | Track identity within labels (`id == int(spawned_on)`). |
| `Keypoint` `hash(name)` | skeleton | Keypoint identity by name; the skeleton/keypoint schema seam. |
| `project_id` | project | UUID/ULID in `PROJECT.json`; regex-validated. |
| `AssetEntry.id` | manifest | Deterministic `sha1(project-relative path)[:8]`. |
| `ArtifactFile.sha256` | artifact | Per-output content hash; links artifact outputs back to inputs. |
| `IdentityProvenanceRecord.track_id` | identity | Companion keyed to `Track`; frame spans use `video_id` + `(start,end)`. |
| `session_id` | multimodal | Binds pose/video/signals/events in a `RecordingSession`. |

## Link Types

- `project -> {video, labels, artifact, calibration, segmentation, metadata}`: the
  project store owns all of these.
- `video -> labeled_frame -> instance -> point`: the pose annotation graph.
- `skeleton -> instance`: instances are keyed to a skeleton; keypoint identity is
  `hash(name)`.
- `track -> instance` and `track -> identity_provenance_record`: track identity and
  its provenance companion.
- `instance -> segmentation_mask` / `instance -> roi`: via `track` + `instance_ref`.
- `session -> {pose, videos, signals, events}`: multimodal binding on a `Timebase`.
- `artifact -> {inputs, outputs}`: lineage via `ArtifactFile.sha256`.
- `video.sha256`: the media hash that lets any repo re-join to the same source.

## Boundaries

- `.xpkg/` is the **private authoritative store**. Consumers use the public
  `ProjectService`/`xpkg.project` API and never read `.xpkg/` internals; there is
  no `project → .xpkg` export command. On-disk layout constants and `project_*`
  path helpers are private.
- The on-disk **artifact contract is frozen at v1**; the Python/CLI surface is
  pre-1.0 and may change. Depend on object shapes and the v1 contract, not on
  private path helpers.
- Identity is content-addressed where it matters: `Video.sha256`,
  `ArtifactFile.sha256`, and portable `EXPKG.json` per-file digests.
- **Namespaces are caller-owned** ([`artifact-namespaces.md`](../artifact-namespaces.md)):
  xpkg reserves no downstream package names. Each consumer picks its own artifact
  namespace; xpkg does not hard-code Fiesta or any other repo.
- The durable-store durability layer and the media-ownership stack are marked
  experimental / target-architecture; do not depend on their internals.
- Licensing/security: see `LICENSE` and `SECURITY.md`.

## Federation (Root of the Cross-Repo Ontology)

`xpkg` is the center of gravity of the federated ontology: it owns the objects
that multiple repos consume. Consumers declare a seam that **links to this file**
rather than restating xpkg's object shapes.

- Known consumer: **Fiesta** (`sandovaljoseph/fiesta`) consumes `Project`,
  `Video`, `LabeledFrame`, `Skeleton`, `Labels`, `ProjectManifest`, and
  `ExportArtifact`, and adds its own model/training/inference objects. See
  `fiesta/docs/ontology/OBJECTS.md`.
- Because namespaces are caller-owned, xpkg does not enumerate its consumers as a
  contract; this list is informational. When an xpkg object shape changes, update
  this registry — the federation reads object identity from here.

## Versioning Constants

- `project_schema_version = 1`, `layout_version = 1` (`PROJECT.json`).
- `CALIBRATION_SCHEMA_VERSION = 1`.
- `IDENTITY_PROVENANCE_SCHEMA_VERSION = "xpkg.identity_provenance.v1"`.
- `BEHAVIOR_LABELS_SCHEMA_VERSION`, `ARTIFACT_SCHEMA_VERSION`,
  `FIGURE_ARTIFACT_SCHEMA_VERSION`, `Skeleton.schema_version`.

## Deprecated Names

- HDF5 (`.h5`) as a project artifact — removed from the v1 contract.
- `project → .xpkg` export — never exposed; `.xpkg/` is private-only.
