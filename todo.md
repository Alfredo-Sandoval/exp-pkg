# exp-pkg Todo

## Gap Analysis Direction

The Luxem 2023 gap analysis should function as a scope-control document for
behavior-video support. `exp-pkg` should absorb schemas, importers,
inspection, and QC. It should not become a training framework, behavior
classifier, ReID model, motif-discovery package, or analysis platform.

The strongest product line to protect:

> exp-pkg should normalize and package upstream behavioral-video outputs so
> downstream tools have one stable boundary to write into and read from.

## Next Milestone: Project-Level Inspect/QC

This is the cleanest next move. It honors the performance contract, uses the
contracts already present, and turns the existing pieces into a product surface.

Definition of done:

- `xpkg inspect PROJECT --json` reports project descriptor, shallow state
  summary, metadata slots, associated media, and QC warnings.
- The project inspect path remains shallow: it must not hydrate full labels,
  predictions, dense masks, Vicon recordings, or media payloads unless a future
  explicit command contract says so.
- Metadata slot presence is visible for acquisition, dataset-share, datasheet,
  model-card, and pose provenance.
- Missing datasheet/model-card/acquisition metadata is reported as absent slot
  status by ordinary inspect; completeness warnings belong in a future explicit
  release-readiness check.
- Common imported pose projects get basic media/frame/timing checks.
- Tests prove project inspect remains descriptor/summary-level and does not
  materialize full state payloads.

## Ordered Work

### 1. Project-Level Inspection Shape

Done:

- Public JSON shape for `xpkg inspect PROJECT --json` is documented in
  `docs/cli_command_spec_v1.md`.
- The runtime shape follows the existing CLI envelope:
  `{"ok": true, "data": <inspection-report>}`.
- Project inspection report keys are stable:
  `status`, `path`, `name`, `suffix`, `exists`, `is_dir`, `size_bytes`,
  `kind`, `description`, `likely_importers`, `summary`, `warnings`, and
  `warning_records`.
- Project `summary` keys are stable:
  `project_id`, `title`, `state_kind`, `has_current_state`, `state_bytes`,
  `commit_id`, `modalities`, `summary_path`, `metadata_slots`, and `media`.
- `metadata_slots` uses predictable slot names:
  `acquisition`, `dataset_share`, `datasheet`, `model_card`, and
  `pose_provenance`.
- `media` is the shallow summary-recorded associated-media inventory plus
  cheap existence checks and `current_image_count` for resolvable image
  sequences.
- `warnings` remains a backward-compatible list of strings, and
  `warning_records` is the stable machine-readable list with
  `code`/`message`/`path`/`severity`. Ordinary inspect does not warn for absent
  optional metadata slots; it does warn for invalid present metadata, missing
  media, media count/frame drift, unavailable media inventory, and unreadable
  shallow indexes.
- Regression tests lock the project JSON shape, metadata-slot shape, media
  shape, warnings behavior, and shallow/no-full-payload inspection behavior.

### 2. Metadata Slot Reporting

Done for project directories:

- Report whether these slots are absent, present, and parseable:
  acquisition, dataset-share, datasheet, model-card, and pose provenance.
- Include canonical metadata file paths in the JSON output.
- Validate datasheet/model-card parseability without treating optional fields
  as required.
- Add tests around absent, valid, and malformed metadata files.

Done for packed `.expkg` artifacts:

- Report the same metadata-slot presence and parseability by reading canonical
  archive members without unpacking.
- Add tests around absent, valid, and malformed packed metadata files.
- Ordinary inspect does not warn on absent optional FAIR slots; missing-slot
  completeness belongs in a future explicit release-readiness check.

### 3. Associated Media And Frame-Count Checks

Done for newly saved/imported project directories:

- Persist compact media inventory in the project summary at save/import time.
- Report associated media path, kind, backend, frame count, dimensions, and
  per-media label/prediction frame coverage through `xpkg inspect PROJECT`.
- Re-check whether recorded media paths still exist without decoding videos or
  hydrating labels.
- Warn when a recorded media path is missing or known label/prediction frame
  indices exceed the stored frame count.
- Warn when an older labels project summary does not have refreshed media
  inventory available.

Remaining follow-up:

- Keep deeper video timing/FPS work behind the acquisition-QC item unless it
  can use already-recorded metadata.

### 4. Timing And Acquisition QC

Done for newly generated labels media summaries:

- Persist summary-recorded `fps`, derived `duration_s`, and
  `timebase: "frame_index"` from the already-open media object at save/import
  time.
- Report those timing fields through `xpkg inspect PROJECT --json` via the
  existing shallow media inventory, without decoding media or hydrating project
  payloads.

Remaining follow-up:

- Report dropped-frame or FPS-drift evidence only when already recorded in a
  shallow descriptor/summary; do not demux media during project inspect.
- Add project-level warnings for missing timebase/sync metadata when the
  shallow summary can distinguish multi-timed-modality projects from ordinary
  pose/video projects.
- Track what is known versus unknown; do not invent certainty from absent
  metadata.

### 5. Behavior Label Importers

BehaviorLabels is the first source-neutral contract. The next adapter order
should favor simple, common formats before larger tool-specific surfaces:

Done:

- BORIS tabular event CSV exports for manual ethogram intervals.
- SimBA CSV outputs for supervised behavior labels.
- Keypoint-MoSeq syllable CSV outputs with provenance and row metadata.

Remaining:

- B-SOiD, A-SOiD, VAME, DeepEthogram, and JAABA after the first adapters prove
  the importer contract.

For each adapter:

- Preserve source metadata and package-specific row metadata.
- Preserve confidence or uncertainty fields when present.
- Produce import-time QC warnings when segments do not align to known timelines.
- Avoid presenting `xpkg` as the algorithm that trained or inferred the labels.

### 6. 3D Pose Design

Done for design docs:

- `docs/pose_3d_design.md` defines the coordinate-frame-first contract.
- Reuse existing `PoseTrajectory` for 2D/3D skeletal or named-marker
  trajectories; do not add a separate `PoseTrajectory3D` unless runtime
  storage diverges.
- Reserve a future `SpatialPointSeries` for non-skeletal point streams.
- Keep `Labels`, `Point`, and `PredictedPoint` image-space and 2D.
- Put units, coordinate frame, calibration provenance, reprojection error,
  visibility, confidence, and source-tool metadata at trajectory/project/quality
  sidecar scope rather than casual point fields.

Likely 3D importer sequence:

- Anipose 3D CSV.
- DANNCE.
- DeepFly3D.
- Additional tool outputs only after the core coordinate-frame contract is
  stable.

### 7. Identity Provenance

Tracks already exist, but multi-animal workflows need explicit identity
provenance.

Add fields or companion records for:

- ReID versus MOT identity source.
- Identity swap markers.
- Manual proofreading flags.
- Identity confidence intervals or spans.
- SLEAP/DLC multi-animal track ingestion that populates those fields.

This should remain data provenance, not a ReID model implementation.

### 8. Calibration Importers

After project-level QC and the 3D coordinate contract are clearer, add more
calibration importers:

- OpenCV stereo YAML.
- Kalibr `camchain.yaml`.
- MC-calib outputs.

Keep the calibration model tool-neutral and preserve source provenance.

## Documentation Notes

- Public docs now use `Behavior Video Roadmap` as the nav/page title instead
  of `Luxem Gap Analysis`.
- Keep Luxem et al. 2023 as the grounding source inside the page.
- Keep the explicit non-recommendations visible: no pose-estimation training,
  classifier training, unsupervised motif discovery, ReID models, or
  dataset/model zoos in core `xpkg`.

## Release Hygiene

- Release docs and `package-check` cover installed-wheel contract checks, not
  help-only or import-only checks.
- `make qa` includes the cheap conflict-marker quality check:
  `rg -n '(<){7}|(=){7}|(>){7}' .`

- Before a public release, the honest gate remains:

```bash
make release-check REAL_DATA_ROOT=../xpkg-real-data
```
