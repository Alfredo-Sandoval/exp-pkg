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
- Missing datasheet/model-card/acquisition metadata can be reported as warnings
  without making those schemas mandatory.
- Common imported pose projects get basic media/frame/timing checks.
- Tests prove project inspect remains descriptor/summary-level and does not
  materialize full state payloads.

## Ordered Work

### 1. Project-Level Inspection Shape

- Define the public JSON shape for `xpkg inspect PROJECT --json`.
- Keep the shape agent-friendly: stable top-level keys, explicit warning
  records, and predictable metadata-slot names.
- Reuse existing project summary/index data where possible.
- Add regression tests that would fail if inspect loads full labels,
  predictions, dense masks, Vicon recordings, or media payloads.

### 2. Metadata Slot Reporting

Done for project directories:

- Report whether these slots are absent, present, and parseable:
  acquisition, dataset-share, datasheet, model-card, and pose provenance.
- Include canonical metadata file paths in the JSON output.
- Validate datasheet/model-card parseability without treating optional fields
  as required.
- Add tests around absent, valid, and malformed metadata files.

Remaining follow-up:

- Decide whether packed `.expkg` inspect should expose the same metadata-slot
  status before unpacking.
- Decide which missing FAIR metadata slots should become warnings for release
  readiness versus ordinary project inspection.

### 3. Associated Media And Frame-Count Checks

Done for newly saved/imported project directories:

- Persist compact media inventory in the project summary at save/import time.
- Report associated media path, kind, backend, frame count, dimensions, and
  per-media label/prediction frame coverage through `xpkg inspect PROJECT`.
- Re-check whether recorded media paths still exist without decoding videos or
  hydrating labels.
- Warn when a recorded media path is missing or known label/prediction frame
  indices exceed the stored frame count.

Remaining follow-up:

- Decide whether old projects without refreshed summaries should get an
  explicit "media inventory unavailable" warning.
- Keep deeper video timing/FPS work behind the acquisition-QC item unless it
  can use already-recorded metadata.

### 4. Timing And Acquisition QC

- Report basic dropped-frame or FPS-drift evidence where already cheap.
- Add project-level warnings for missing timebase/sync metadata when a project
  has multiple timed modalities.
- Track what is known versus unknown; do not invent certainty from absent
  metadata.

### 5. Behavior Label Importers

BehaviorLabels is the first source-neutral contract. The next adapter order
should favor simple, common formats before larger tool-specific surfaces:

- BORIS exports for manual ethogram intervals.
- SimBA CSV outputs for supervised behavior labels.
- Keypoint-MoSeq syllables with careful provenance.
- B-SOiD, A-SOiD, VAME, DeepEthogram, and JAABA after the first adapters prove
  the importer contract.

For each adapter:

- Preserve source metadata and package-specific row metadata.
- Preserve confidence or uncertainty fields when present.
- Produce import-time QC warnings when segments do not align to known timelines.
- Avoid presenting `xpkg` as the algorithm that trained or inferred the labels.

### 6. 3D Pose Design

Do not casually add `z` to every 2D point. First decide how 3D coordinates
should encode coordinate frames and provenance.

Design questions:

- Do we need a separate `PoseTrajectory3D` or `SpatialPointSeries` type?
- How do we distinguish image coordinates, camera coordinates, world
  coordinates, lifted-model predictions, and marker-based mocap points?
- Where do units live?
- How does a 3D point reference calibration provenance?
- Where do reprojection error, visibility, confidence, and source-tool metadata
  live?

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

- Consider renaming the public nav label from `Luxem Gap Analysis` to
  `Behavior Video Roadmap` if the page is meant to be read as product
  direction rather than an internal audit.
- Keep Luxem et al. 2023 as the grounding source inside the page.
- Keep the explicit non-recommendations visible: no pose-estimation training,
  classifier training, unsupervised motif discovery, ReID models, or
  dataset/model zoos in core `xpkg`.

## Release Hygiene

- Keep release docs on installed-wheel contract checks, not help-only or
  import-only checks.
- Add a cheap conflict-marker check to the quality surface:

```bash
rg -n '(<){7}|(=){7}|(>){7}' .
```

- Before a public release, the honest gate remains:

```bash
make release-check REAL_DATA_ROOT=../xpkg-real-data
```
