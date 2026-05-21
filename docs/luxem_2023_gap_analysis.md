# Behavior Video Roadmap

Source paper: Luxem K, Sun JJ, Bradley SP, Krishnan K, Yttri E, Zimmermann J,
Pereira TD, Laubach M (2023). *Open-source tools for behavioral video
analysis: Setup, methods, and best practices.* eLife 12:e79305.
DOI: [10.7554/eLife.79305](https://doi.org/10.7554/eLife.79305).
Reference copies should not be committed to this repository; use the DOI link
above for the source paper.

## Scope reminder

`exp-pkg` is an IO / packaging layer — a stable boundary for experiment-data
import, normalization, validation, and portable artifact export. It is **not**
an analysis platform. Pose-training, multi-animal tracking algorithms,
behavioral motif discovery, and supervised behavior classifier training are
explicitly off-scope per `docs/roadmap.md`; they belong in DeepLabCut, SLEAP,
SimBA, VAME, etc.

The relevant question for Luxem alignment is therefore:

> Which items from the paper should land in xpkg as **schemas, importers, and
> inspect/QC**, so that downstream tools have a stable contract to write into
> and read from?

## What the paper covers

1. **Acquisition** — cameras, lenses, lighting, sync, codecs, storage.
2. **Calibration** — intrinsics/extrinsics, multi-camera, structure-from-motion.
3. **2D pose estimation** — DeepLabCut, SLEAP, DeepPoseKit, MARS, B-KinD.
4. **3D pose estimation** — Anipose, OpenMonkeyStudio, DeepFly3D, DANNCE,
   LiftPose3D.
5. **Multi-animal tracking & ReID** — part grouping, MOT vs. ReID, identity
   swaps.
6. **Behavior quantification** — supervised (SimBA, MARS, DeepEthogram,
   JAABA) and unsupervised (B-SOiD, VAME, MotionMapper, TREBA).
7. **Best practices** — FAIR data, datasheets (Gebru et al.), model cards
   (Mitchell et al.), reproducibility checklist, open data/code sharing.

## What exp-pkg already covers

- Multi-format **2D pose import**: DeepLabCut (CSV/H5/project), SLEAP
  (analysis H5, `.pkg.slp`), Lightning Pose, MMPose, MediaPipe, generic CSV/H5.
- A `Labels` / `LabeledFrame` / `Instance` / `PredictedInstance` / `Track`
  model with skeleton, keypoints, confidence, JSON serialization.
- **Calibration**: pinhole/fisheye/omnidir intrinsics, OpenCV / fisheye
  distortion models, multi-camera extrinsics, Anipose TOML round-trip.
- **Project metadata**: `AcquisitionMetadata` (cameras, arena, lighting, IR,
  software/hardware), `DatasetShareMetadata`, `PoseModelProvenance`,
  `DatasetDatasheet`, and `ModelCard`.
- **Media stack**: backend discovery (OpenCV/PyAV/FFmpeg/TorchCodec/ONNX,
  CUDA/MLX/Metal), frame readers/writers, `nvpkg` accelerator bridge.
- **Inspection CLI**: `xpkg inspect` autodetects DLC/SLEAP/MMPose/MediaPipe/
  Doric/`.expkg`/PROJECT.json and reports per-keypoint QC.
- **Frame-level segmentation masks**: COCO RLE, polygons, SAM bridge —
  *image* masks, not behavioral labels.
- **Portable artifact**: `.expkg` manifests, project lifecycle, services.

## Prioritized adds

### 1. Behavior labels / ethogram model — first contract now present

`segmentation/` today is pixel masks only. Luxem's entire §"Behavior
quantification" produces *temporal* labels: intervals (SimBA, MARS,
DeepEthogram), per-frame motif IDs (B-SOiD, VAME), or continuous embeddings
(TREBA, MotionMapper). `xpkg.model.BehaviorLabels` is now the first
source-neutral contract alongside `Labels`:

- intervals (start/end in frames or seconds, label, score, annotator/model)
- per-frame discrete motifs + confidence
- per-frame continuous embeddings (latent codes)
- source metadata and package-specific row metadata
- generic behavior CSV and behavior-event JSON readers
- BORIS tabular event CSV reader for manual ethogram intervals
- SimBA framewise classifier CSV reader for imported upstream outputs
- Keypoint-MoSeq syllable CSV reader for imported motif assignments

Remaining importer work:

- package-specific adapters for B-SOiD output, A-SOiD output, VAME motif
  export, DeepEthogram predictions, and JAABA exports
- classifier/model provenance mirroring `PoseModelProvenance`
- `inspect` autodetect + label-count / coverage summary

### 2. 3D pose support in `Labels`

Calibration plumbing exists; `Labels` is 2D-centric. Extend keypoints to
optional 3D coordinates + reprojection error + back-pointer to the calibration
used. Add Anipose 3D CSV, DANNCE, and DeepFly3D importers. Natural payoff of
the calibration work already shipped.

### 3. Multi-animal track + identity provenance

Tracks exist in the model, but Luxem stresses identity provenance: ReID vs.
MOT, swap markers, manual proofreading flags. Add identity-provenance fields
and a confidence-interval type on tracks, plus SLEAP/DLC multi-animal track
ingest that populates them.

### 4. FAIR: Datasheet + Model Card schemas — first contract now present

`DatasetDatasheet` and `ModelCard` are now formal typed metadata slots, with
project service helpers, CLI set/show commands, `.expkg` manifest round-trips,
and shallow inspect reporting for project-directory and packed-artifact metadata
slot presence and parseability. This covers the first FAIR schema step the
paper's "Best practices for developers" section motivates.

Remaining work:

- Project-level summaries should make these FAIR metadata slots visible without
  loading full labels, predictions, masks, or media payloads.
- A future release-readiness check can add practical completeness warnings for
  required release/share fields. Ordinary `xpkg inspect` reports absent optional
  FAIR slots as status, not warnings, so the schemas do not become a mandatory
  ontology.

### 5. Acquisition QC in `inspect`

File-level inspect already reports basic media/timing and pose-confidence
signals. Project-level acquisition QC is still shallow. Add inspect-time checks
the paper calls for explicitly:

- dropped-frame / FPS-drift evidence across associated videos
- sync coverage between video timebase and event/photometry streams
- per-keypoint confidence histograms + below-threshold spans across imported
  pose state
- camera-coverage check for multi-camera projects (≥ 1 camera sees each
  keypoint at all times)

Surface as warnings in `xpkg inspect --json` output.

### 6. More calibration importers

`anipose.py` is great. Add OpenCV stereo YAML, Kalibr (`camchain.yaml`), and
MC-calib (Rameau et al. 2022, cited in §"Data acquisition") so users coming
from other 3D stacks aren't locked out.

### 7. Cross-file / project-level inspection

`xpkg inspect` can identify project directories and report summary-recorded
media presence/frame coverage today, but it does not yet pair videos ↔ pose ↔
events ↔ calibration into one completeness report. Add a broader project-level
inspector that reports timebase alignment and missing sidecars. This is the
"common file format / interoperability" pitch from the paper's closing section.

## Explicit non-recommendations

Skip: pose-estimation training, classifier training, unsupervised motif
discovery, ReID models, dataset/model zoos. The roadmap already excludes
these and the paper's framing — that these belong in DLC/SLEAP/SimBA/VAME —
supports keeping them out.

## Suggested order

The first contracts for items 1, 3, and 4 are now present. The remaining order
is 5 → 2 → 6 → 7, with item 4's completeness checks folded into project-level
inspection. Item 5 gives the most user-visible coverage of Luxem's framework
with the least scope creep.
