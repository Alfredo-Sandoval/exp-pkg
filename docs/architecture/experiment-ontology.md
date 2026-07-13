# Experiment Ontology

## Aggregate boundary

An editable xpkg project is a storage and transaction boundary. Its canonical
scientific state is one `Experiment` aggregate. The project descriptor locates
the state; it is not the experiment ontology and it is not a recording session.

```text
Project folder
  -> Experiment
       -> Subject registry
       -> Protocol registry
       -> ExperimentalCondition registry
       -> DatasetShareMetadata
       -> ExperimentSessionLink[]
            -> RecordingSession
            -> SessionSubjectLink[]
            -> SessionProtocolLink[]
            -> SessionConditionLink[]
            -> BehaviorSubjectLink[]
            -> SubjectTrackAssignment[]
            -> EventRelationship[]
```

The aggregate enforces referential integrity with direct object links in
memory. Versioned JSON uses identifiers only at the serialization boundary and
resolves them back to objects during parsing.

## Core object types

### Experiment context

- `Experiment` owns reusable subjects, protocols, conditions, sessions, and
  dataset-sharing metadata.
- `Subject` is a biological participant that may appear in many sessions.
- `Protocol` identifies a versioned experimental or acquisition procedure.
- `ExperimentalCondition` represents a treatment, cohort, or environmental
  condition.
- `ExperimentSessionLink` supplies scientific context for one recording
  session.

### Session context

- `RecordingSession` is one bounded acquisition episode.
- `AcquisitionMetadata` belongs to the session because rigs and settings may
  change between episodes.
- `SessionVideo`, `SessionSignal`, `SessionPose`, `SessionBehavior`, and
  `SessionCalibration` are first-class modality relationships.
- `SessionEventStream` gives each `EventTable` a stable name and provenance.
- `Event` carries a stable identifier within its stream.
- `EventRelationship` records typed links such as trigger, response, outcome,
  order, and containment between identified events.
- `TimebaseAlignment` records evidence-backed mappings between clocks.
- `RecordingSession.timebases` is the complete named clock registry. Every
  modality and alignment endpoint must resolve to that registry.

### Camera and spatial context

- `CameraMetadata` identifies an acquisition camera.
- `SessionVideo.camera` links video bytes to that camera.
- `CalibrationCameraLink` connects an acquisition camera to its calibrated
  camera model.
- `SessionPose.calibration` links calibrated pose directly to session geometry.
- `PoseCoordinateFrame` declares coordinate system and units.

### Pose and identity context

- `Labels` represents editable and predicted 2D annotations.
- `PoseTrajectory` represents numeric 2D or 3D trajectories with explicit
  frame, track, keypoint, and coordinate axes.
- `PoseModelProvenance` belongs to one `SessionPose`, not the project.
- `BehaviorSubjectLink` attributes a named behavior stream to a participating
  subject without embedding subject identity in the labels payload.
- `SubjectTrackAssignment` assigns a biological subject to a technical pose
  track over an inclusive frame range. Nonoverlapping ranges allow identity
  changes after swaps or reassignment.
- `IdentityProvenanceRecord` supplies evidence for a track identity.

## Invariants

- Experiment, session, subject, protocol, condition, pose, calibration, camera,
  and track identities are unique within their owning scope.
- Every session relationship resolves to a registered experiment object.
- A condition cannot name a subject who does not participate in the session.
- A behavior-to-subject link must name a session behavior and a participating
  subject.
- A subject-to-track assignment must name a participating subject, existing
  pose, existing track, and frame range inside that pose product.
- Assignments on one technical track cannot overlap in time.
- An event relationship must resolve both streams and both event identifiers
  inside the same recording session.
- Every modality timebase and alignment endpoint must be registered by name in
  its recording session.
- A video or calibration camera link requires matching session acquisition
  metadata.
- A calibrated pose must name an existing session calibration.
- Two-dimensional numeric pose uses image pixels. Three-dimensional numeric
  pose uses a non-image coordinate frame.
- Persisted writes occur through experiment and session actions.

## Serialization

The canonical documents are:

- `xpkg.experiment`, schema version 4
- `xpkg.recording-session`, schema version 4
- `xpkg-packed-project`, artifact schema version 2

The generated machine-readable catalog is `schemas/ontology.json`. The two
document envelopes are `schemas/experiment.schema.json` and
`schemas/recording-session.schema.json`. `make ontology-schemas-check` fails
when committed schemas no longer match the model source.

Parsers construct typed objects at the file boundary. Interior project code
does not inspect raw experiment or session dictionaries.

## Rejected alternatives

- A project-wide session registry was rejected because it creates a second
  scientific ontology beside `Experiment`.
- One session per project was rejected because repeated and longitudinal
  experiments require multiple sessions.
- Project-scoped acquisition, calibration, and pose provenance were rejected
  because their cardinality is session or pose-product specific.
- Implicit camera, calibration, and subject relationships in metadata
  dictionaries were rejected because they cannot enforce referential
  integrity.
- A singular session event table was rejected because independent acquisition,
  stimulus, annotation, and derived-event streams need separate provenance and
  stable names.
- Subject identity inside `BehaviorLabels` was rejected because one behavior
  product can describe several actors and interactions. Identity is a
  first-class experiment link.
- Whole-track subject identity was rejected because tracking identities can
  swap within a recording. Assignments are bounded by frame range.
- Assay-specific session subclasses were rejected because an open field,
  operant task, social interaction, gait trial, and home-cage recording differ
  in protocols, participants, modalities, and event relationships rather than
  in their storage container type.
- Separate numeric types for 2D, 3D, and multi-animal pose were rejected because
  those are explicit axes and coordinate-frame properties of one concept.
- A general laboratory inventory or apparatus registry was deferred because
  current acquisition objects describe the recorded session without inventing
  an institution-wide LIMS. A reusable apparatus identity should be added only
  when a real cross-session requirement supplies its properties and links.
