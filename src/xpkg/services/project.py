"""Primary project-first service surface for xpkg project operations.

``ProjectService`` is the stable consumer-facing boundary for downstream
integrations that need to create, open, import into, validate, pack, or unpack
an xpkg project. Import methods select package-owned implementations by a
kebab-case ``format`` string.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from xpkg._core.json_utils import load_json_dict
from xpkg.project import (
    ProjectDescriptor,
    ProjectInspection,
    ProjectSummaryIndex,
    ensure_project,
    init_project,
    inspect_project,
    load_project_acquisition,
    load_project_alignment,
    load_project_behavior,
    load_project_dataset_share,
    load_project_datasheet,
    load_project_descriptor,
    load_project_events,
    load_project_experiment,
    load_project_metadata,
    load_project_metadata_field,
    load_project_model_card,
    load_project_session,
    pack_project,
    refresh_project_summary,
    save_project_acquisition,
    save_project_alignment,
    save_project_behavior,
    save_project_dataset_share,
    save_project_datasheet,
    save_project_events,
    save_project_experiment,
    save_project_labels,
    save_project_metadata,
    save_project_metadata_field,
    save_project_model_card,
    save_project_session,
    unpack_project,
    validate_project,
)
from xpkg.project.calibration import (
    import_anipose_calibration_project,
    import_opencv_stereo_calibration_project,
)
from xpkg.project.layout import (
    ARTIFACTS_DIRNAME,
    CURRENT_STATE_FILENAME,
    INDEXES_DIRNAME,
    PROJECT_DESCRIPTOR_FILENAME,
    PROJECT_SUMMARY_FILENAME,
    STORE_STATE_DIRNAME,
    resolve_project_root,
)
from xpkg.project.metadata import project_metadata_root
from xpkg.project.recording import (
    import_behavior_project,
    import_events_csv_project,
    import_photometry_csv_project,
    import_synchronization_csv_project,
    load_project_labels,
)
from xpkg.services.artifacts import ProjectArtifacts
from xpkg.services.calibrations import ProjectCalibrations
from xpkg.services.figures import ProjectFigures
from xpkg.services.segmentation import ProjectSegmentation

if TYPE_CHECKING:
    from xpkg.model import (
        AcquisitionMetadata,
        BehaviorLabels,
        DatasetDatasheet,
        DatasetShareMetadata,
        EventTable,
        Experiment,
        Labels,
        ModelCard,
        PoseModelProvenance,
        RecordingSession,
        TimebaseAlignment,
    )
    from xpkg.model.session import AlignmentModel, SynchronizationMethod
    from xpkg.model.time import Timebase


PoseFormat = Literal[
    "dlc-csv",
    "dlc-h5",
    "dlc-project",
    "lightning-pose-csv",
    "mediapipe-pose-landmarks-json",
    "mmpose-topdown-json",
    "sleap-h5",
    "sleap-package",
]
"""Supported `format` values for ``ProjectService.import_pose``."""


CalibrationFormat = Literal["anipose", "opencv-stereo-yaml"]
"""Supported `format` values for ``ProjectService.import_calibration``."""


SignalFormat = Literal["photometry-csv"]
"""Supported `format` values for ``ProjectService.import_signals``."""


EventFormat = Literal["events-csv"]
"""Supported `format` values for ``ProjectService.import_events``."""


BehaviorFormat = Literal[
    "behavior-csv",
    "behavior-json",
    "boris-csv",
    "bsoid-csv",
    "keypoint-moseq-csv",
    "simba-csv",
]
"""Supported `format` values for ``ProjectService.import_behavior``."""


SynchronizationFormat = Literal["synchronization-csv"]
"""Supported `format` values for ``ProjectService.import_synchronization``."""


@dataclass(frozen=True, slots=True)
class _PoseImporter:
    """Per-format dispatch entry for ``ProjectService.import_pose``."""

    fn_name: str
    requires_video: bool
    accepts_skeleton_name: bool = True
    default_skeleton_name: str | None = None
    accepts_likelihood_threshold: bool = True
    accepts_instance_index: bool = False
    accepts_fps: bool = False
    accepts_encode_videos: bool = False


_POSE_IMPORTERS: dict[str, _PoseImporter] = {
    "dlc-csv": _PoseImporter(
        fn_name="import_dlc_csv_project",
        requires_video=True,
        default_skeleton_name="imported",
    ),
    "dlc-h5": _PoseImporter(
        fn_name="import_dlc_h5_project",
        requires_video=True,
        default_skeleton_name="imported",
    ),
    "dlc-project": _PoseImporter(
        fn_name="import_dlc_project_directory",
        requires_video=False,
    ),
    "lightning-pose-csv": _PoseImporter(
        fn_name="import_lightning_pose_csv_project",
        requires_video=True,
        default_skeleton_name="imported",
    ),
    "mediapipe-pose-landmarks-json": _PoseImporter(
        fn_name="import_mediapipe_pose_landmarks_json_project",
        requires_video=True,
        default_skeleton_name="mediapipe_pose",
    ),
    "mmpose-topdown-json": _PoseImporter(
        fn_name="import_mmpose_topdown_json_project",
        requires_video=True,
        default_skeleton_name="imported",
        accepts_instance_index=True,
    ),
    "sleap-h5": _PoseImporter(
        fn_name="import_sleap_h5_project",
        requires_video=True,
        default_skeleton_name="imported",
    ),
    "sleap-package": _PoseImporter(
        fn_name="import_sleap_package_project",
        requires_video=False,
        accepts_skeleton_name=False,
        accepts_likelihood_threshold=False,
        accepts_fps=True,
        accepts_encode_videos=True,
    ),
}


_CALIBRATION_IMPORTERS: dict[str, Callable[..., Path]] = {
    "anipose": import_anipose_calibration_project,
    "opencv-stereo-yaml": import_opencv_stereo_calibration_project,
}


def _cached_summary_for_descriptor(
    summary_path: Path,
    descriptor: ProjectDescriptor,
) -> ProjectSummaryIndex | None:
    """Return the cached summary when it matches the current descriptor."""

    try:
        summary = ProjectSummaryIndex.from_dict(load_json_dict(summary_path))
    except (OSError, TypeError, ValueError):
        return None

    if summary.project_id != descriptor.project_id:
        return None
    if summary.title != descriptor.title:
        return None
    if summary.descriptor_updated_at != descriptor.updated_at:
        return None
    return summary


@dataclass(frozen=True, slots=True)
class ProjectMetadata:
    """Accessor for dataset and model documentation under ``.xpkg/metadata/``.

    Read each slot as an attribute (returns ``None`` when unset) and write one
    or more slots in a single call via :meth:`update`. The state-bound free-form
    metadata dict on the current project head is a separate concept and is
    accessed through ``ProjectService.load_state_metadata`` /
    ``save_state_metadata``.
    """

    project_root: Path

    @property
    def root(self) -> Path:
        """Return the ``.xpkg/metadata/`` directory for this project."""
        return project_metadata_root(self.project_root)

    @property
    def datasheet(self) -> DatasetDatasheet | None:
        return load_project_datasheet(self.project_root)

    @property
    def model_card(self) -> ModelCard | None:
        return load_project_model_card(self.project_root)

    def update(
        self,
        *,
        datasheet: DatasetDatasheet | Mapping[str, Any] | None = None,
        model_card: ModelCard | Mapping[str, Any] | None = None,
    ) -> dict[str, Path]:
        """Write one or more documentation records and return their paths."""
        written: dict[str, Path] = {}
        if datasheet is not None:
            written["datasheet"] = save_project_datasheet(self.project_root, datasheet)
        if model_card is not None:
            written["model_card"] = save_project_model_card(self.project_root, model_card)
        if not written:
            raise ValueError("ProjectMetadata.update requires at least one slot keyword.")
        return written


@dataclass(frozen=True, slots=True)
class ProjectLayout:
    """Normalized descriptor and managed paths returned by service lifecycle calls."""

    project_root: Path
    descriptor: ProjectDescriptor
    descriptor_path: Path
    store_root: Path
    artifacts_root: Path
    state_root: Path
    media_root: Path
    exports_root: Path
    current_state_path: Path
    summary_path: Path
    summary: ProjectSummaryIndex
    has_current_state: bool


@dataclass(slots=True)
class ProjectService:
    """Stable project object for create/open/import/validate/pack workflows."""

    project_root: Path

    @classmethod
    def create(
        cls,
        project: str | Path,
        *,
        title: str | None = None,
        project_id: str | None = None,
        force: bool = False,
    ) -> ProjectService:
        """Create a project with the canonical public layout and open it."""
        init_project(
            project,
            title=title,
            project_id=project_id,
            force=force,
        )
        return cls.open(project)

    @classmethod
    def ensure(
        cls,
        project: str | Path,
        *,
        title: str | None = None,
        project_id: str | None = None,
    ) -> ProjectService:
        """Open an xpkg project, or adopt an existing folder as one."""
        ensure_project(
            project,
            title=title,
            project_id=project_id,
        )
        return cls.open(project)

    @classmethod
    def open(cls, project: str | Path) -> ProjectService:
        """Open an existing project root or a path inside one."""
        root = resolve_project_root(project)
        if root is None:
            raise FileNotFoundError(f"Not an xpkg project: {project}")
        return cls(project_root=root)

    @classmethod
    def unpack(
        cls,
        artifact: str | Path,
        out: str | Path,
        *,
        force: bool = False,
        rename_title: str | None = None,
    ) -> ProjectService:
        """Unpack a portable `.expkg` artifact into a project and open it."""
        unpack_project(
            artifact,
            out,
            force=force,
            rename_title=rename_title,
        )
        return cls.open(out)

    def descriptor(self) -> ProjectDescriptor:
        """Load the current project descriptor."""
        return load_project_descriptor(self.project_root)

    @property
    def segmentation(self) -> ProjectSegmentation:
        """Return project-bound segmentation-mask storage commands."""
        return ProjectSegmentation(project_root=self.project_root)

    @property
    def artifacts(self) -> ProjectArtifacts:
        """Return project-bound generic artifact registry commands."""
        return ProjectArtifacts(project_root=self.project_root)

    @property
    def figures(self) -> ProjectFigures:
        """Return project-bound figure artifact registry commands."""
        return ProjectFigures(project_root=self.project_root)

    @property
    def calibrations(self) -> ProjectCalibrations:
        """Return project-bound camera-calibration storage commands."""
        return ProjectCalibrations(project_root=self.project_root)

    @property
    def metadata(self) -> ProjectMetadata:
        """Return the typed metadata accessor for this project's durable slots."""
        return ProjectMetadata(project_root=self.project_root)

    def import_pose(
        self,
        format: PoseFormat,
        *,
        path: str | Path,
        video: str | Path | None = None,
        skeleton_name: str | None = None,
        likelihood_threshold: float | None = None,
        instance_index: int | None = None,
        fps: int | None = None,
        encode_videos: bool | None = None,
        prediction_provenance: Mapping[str, Any] | None = None,
        provenance: PoseModelProvenance | Mapping[str, Any] | None = None,
        session_id: str | None = None,
        force: bool = False,
        progress_callback: Callable[[str], None] | None = None,
    ) -> Path:
        """Import a pose track from one of the supported formats into this project."""
        importer = _POSE_IMPORTERS.get(format)
        if importer is None:
            raise ValueError(f"Unknown pose format: {format!r}")
        if importer.requires_video and video is None:
            raise ValueError(f"import_pose(format={format!r}) requires `video=`.")

        for opt_name, value, accepted in (
            ("skeleton_name", skeleton_name, importer.accepts_skeleton_name),
            (
                "likelihood_threshold",
                likelihood_threshold,
                importer.accepts_likelihood_threshold,
            ),
            ("instance_index", instance_index, importer.accepts_instance_index),
            ("fps", fps, importer.accepts_fps),
            ("encode_videos", encode_videos, importer.accepts_encode_videos),
        ):
            if value is not None and not accepted:
                raise ValueError(f"import_pose(format={format!r}) does not accept `{opt_name}=`.")

        kwargs: dict[str, Any] = {
            "project": self.project_root,
            "prediction_provenance": prediction_provenance,
            "provenance": provenance,
            "session_id": session_id,
            "force": force,
            "progress_callback": progress_callback,
        }
        if importer.accepts_likelihood_threshold:
            kwargs["likelihood_threshold"] = (
                0.0 if likelihood_threshold is None else likelihood_threshold
            )
        if importer.accepts_skeleton_name:
            kwargs["skeleton_name"] = skeleton_name or importer.default_skeleton_name
        if importer.accepts_instance_index:
            kwargs["instance_index"] = 0 if instance_index is None else int(instance_index)
        if importer.accepts_fps:
            kwargs["fps"] = 30 if fps is None else int(fps)
        if importer.accepts_encode_videos:
            kwargs["encode_videos"] = encode_videos

        if importer.requires_video:
            import xpkg.project.store.imports as project_imports

            assert video is not None
            fn = getattr(project_imports, importer.fn_name)
            return fn(path, video, **kwargs)
        import xpkg.project.store.imports as project_imports

        fn = getattr(project_imports, importer.fn_name)
        return fn(path, **kwargs)

    def import_calibration(
        self,
        format: CalibrationFormat,
        *,
        path: str | Path,
        calibration_id: str | None = None,
        session_id: str | None = None,
        name: str | None = None,
        camera_names: tuple[str, str] | None = None,
        units: str = "unknown",
        captured_at: str | None = None,
        tool_version: str | None = None,
        force: bool = False,
    ) -> Path:
        """Import a camera calibration from one of the supported formats."""
        importer = _CALIBRATION_IMPORTERS.get(format)
        if importer is None:
            raise ValueError(f"Unknown calibration format: {format!r}")
        kwargs: dict[str, Any] = {
            "project": self.project_root,
            "calibration_id": calibration_id,
            "session_id": session_id,
            "name": name,
            "units": units,
            "captured_at": captured_at,
            "tool_version": tool_version,
            "force": force,
        }
        if camera_names is not None:
            kwargs["camera_names"] = camera_names
        return importer(path, **kwargs)

    def import_signals(
        self,
        format: SignalFormat,
        *,
        path: str | Path,
        session_id: str | None = None,
        signal_name: str = "photometry",
        force: bool = False,
    ) -> Path:
        """Import sampled signals into canonical recording-session state."""
        if format != "photometry-csv":
            raise ValueError(f"Unknown signal format: {format!r}")
        return import_photometry_csv_project(
            path,
            project=self.project_root,
            session_id=session_id,
            signal_name=signal_name,
            force=force,
        )

    def import_events(
        self,
        format: EventFormat,
        *,
        path: str | Path,
        session_id: str | None = None,
        force: bool = False,
    ) -> Path:
        """Import event records into canonical recording-session state."""
        if format != "events-csv":
            raise ValueError(f"Unknown event format: {format!r}")
        return import_events_csv_project(
            path,
            project=self.project_root,
            session_id=session_id,
            force=force,
        )

    def import_behavior(
        self,
        format: BehaviorFormat,
        *,
        path: str | Path,
        behavior_name: str = "behavior",
        session_id: str | None = None,
        video_role: str | None = None,
        force: bool = False,
    ) -> Path:
        """Import behavior labels into a named recording-session link."""
        return import_behavior_project(
            format,
            path,
            project=self.project_root,
            behavior_name=behavior_name,
            session_id=session_id,
            video_role=video_role,
            force=force,
        )

    def import_synchronization(
        self,
        format: SynchronizationFormat,
        *,
        path: str | Path,
        source_timebase: Timebase,
        target_timebase: Timebase,
        model: AlignmentModel,
        method: SynchronizationMethod,
        alignment_name: str | None = None,
        session_id: str | None = None,
        force: bool = False,
    ) -> Path:
        """Import paired clock observations into one timebase alignment."""
        if format != "synchronization-csv":
            raise ValueError(f"Unknown synchronization format: {format!r}")
        return import_synchronization_csv_project(
            path,
            project=self.project_root,
            source_timebase=source_timebase,
            target_timebase=target_timebase,
            model=model,
            method=method,
            alignment_name=alignment_name,
            session_id=session_id,
            force=force,
        )

    def describe(self) -> ProjectLayout:
        """Return the normalized managed paths for this project.

        The describe path prefers the generated shallow summary index when it
        matches the current descriptor. That keeps repeated project-picker and
        GUI row rendering calls to descriptor + summary JSON reads, while still
        regenerating the summary if it is missing, unreadable, or stale.
        """
        root = self.project_root
        descriptor = self.descriptor()
        store_root = root / descriptor.store_path
        state_root = store_root / STORE_STATE_DIRNAME
        summary_path = store_root / INDEXES_DIRNAME / PROJECT_SUMMARY_FILENAME
        summary = _cached_summary_for_descriptor(summary_path, descriptor)
        if summary is None:
            summary = refresh_project_summary(root)

        return ProjectLayout(
            project_root=root,
            descriptor=descriptor,
            descriptor_path=root / PROJECT_DESCRIPTOR_FILENAME,
            store_root=store_root,
            artifacts_root=store_root / ARTIFACTS_DIRNAME,
            state_root=state_root,
            media_root=root / descriptor.media_root,
            exports_root=root / descriptor.exports_root,
            current_state_path=state_root / CURRENT_STATE_FILENAME,
            summary_path=summary_path,
            summary=summary,
            has_current_state=summary.has_current_state,
        )

    def validate(self) -> ProjectLayout:
        """Validate the project and return its normalized layout."""
        validate_project(self.project_root)
        return self.describe()

    def inspect(self) -> ProjectInspection:
        """Inspect the current project using canonical package-owned summary APIs."""
        return inspect_project(self.project_root)

    def load_labels(self, *, session_id: str | None = None) -> Labels:
        """Load the canonical pose link from the project session."""
        return load_project_labels(self.project_root, session_id=session_id)

    def load_experiment(self) -> Experiment:
        """Load the canonical experiment aggregate."""
        return load_project_experiment(self.project_root)

    def load_session(self, *, session_id: str | None = None) -> RecordingSession:
        """Load a named session, or the sole session when unambiguous."""
        return load_project_session(self.project_root, session_id=session_id)

    def load_events(self, *, session_id: str | None = None) -> EventTable:
        """Load the canonical event table for one recording session."""
        return load_project_events(self.project_root, session_id=session_id)

    def save_events(
        self,
        events: EventTable,
        *,
        session_id: str | None = None,
    ) -> Path:
        """Commit a typed event table to one recording session."""
        return save_project_events(self.project_root, events, session_id=session_id)

    def load_behavior(
        self,
        *,
        behavior_name: str = "behavior",
        session_id: str | None = None,
    ) -> BehaviorLabels:
        """Load one named behavior-label link from a recording session."""
        return load_project_behavior(
            self.project_root,
            behavior_name=behavior_name,
            session_id=session_id,
        )

    def save_behavior(
        self,
        labels: BehaviorLabels,
        *,
        behavior_name: str = "behavior",
        session_id: str | None = None,
        video_role: str | None = None,
        replace_existing: bool = True,
    ) -> Path:
        """Add or replace one typed behavior-label link."""
        return save_project_behavior(
            self.project_root,
            labels,
            behavior_name=behavior_name,
            session_id=session_id,
            video_role=video_role,
            replace_existing=replace_existing,
        )

    def load_alignment(
        self,
        *,
        alignment_name: str,
        session_id: str | None = None,
    ) -> TimebaseAlignment:
        """Load one named timebase alignment from a recording session."""
        return load_project_alignment(
            self.project_root,
            alignment_name=alignment_name,
            session_id=session_id,
        )

    def save_alignment(
        self,
        alignment: TimebaseAlignment,
        *,
        session_id: str | None = None,
        replace_existing: bool = True,
    ) -> Path:
        """Add or replace one typed timebase alignment."""
        return save_project_alignment(
            self.project_root,
            alignment,
            session_id=session_id,
            replace_existing=replace_existing,
        )

    def load_acquisition(
        self, *, session_id: str | None = None
    ) -> AcquisitionMetadata | None:
        """Load acquisition context owned by one recording session."""
        return load_project_acquisition(self.project_root, session_id=session_id)

    def save_acquisition(
        self,
        acquisition: AcquisitionMetadata | Mapping[str, object],
        *,
        session_id: str | None = None,
    ) -> Path:
        """Commit acquisition context to one recording session."""
        return save_project_acquisition(
            self.project_root, acquisition, session_id=session_id
        )

    def load_dataset_share(self) -> DatasetShareMetadata | None:
        """Load dataset-sharing metadata owned by the experiment."""
        return load_project_dataset_share(self.project_root)

    def save_dataset_share(
        self, dataset_share: DatasetShareMetadata | Mapping[str, object]
    ) -> Path:
        """Commit dataset-sharing metadata to the experiment."""
        return save_project_dataset_share(self.project_root, dataset_share)

    def load_state_metadata(self) -> dict[str, Any] | None:
        """Load the current project state's free-form metadata dict."""
        return load_project_metadata(self.project_root)

    def load_state_metadata_field(self, field: str) -> dict[str, Any] | None:
        """Load one mapping-valued field from the current project state metadata."""
        return load_project_metadata_field(self.project_root, field)

    def save_state_metadata(
        self,
        metadata: dict[str, Any] | None,
        *,
        reason: str = "project.save.metadata",
    ) -> Path:
        """Commit free-form metadata onto the current project state head."""
        return save_project_metadata(
            self.project_root,
            metadata,
            reason=reason,
        )

    def save_state_metadata_field(
        self,
        field: str,
        value: Mapping[str, Any],
        *,
        reason: str = "project.save.metadata_field",
    ) -> Path:
        """Persist one mapping-valued metadata field onto the current project state head."""
        return save_project_metadata_field(
            self.project_root,
            field,
            value,
            reason=reason,
        )

    def save_labels(
        self,
        labels: Labels,
        *,
        metadata: dict[str, Any] | None = None,
        provenance: PoseModelProvenance | None = None,
        session_id: str | None = None,
    ) -> Path:
        """Replace the canonical pose link on the project session."""
        return save_project_labels(
            self.project_root,
            labels,
            metadata=metadata,
            provenance=provenance,
            session_id=session_id,
        )

    def save_experiment(
        self,
        experiment: Experiment,
        *,
        reason: str = "project.save.experiment",
    ) -> Path:
        """Commit the canonical experiment aggregate."""
        return save_project_experiment(self.project_root, experiment, reason=reason)

    def save_session(
        self,
        session: RecordingSession,
        *,
        reason: str = "project.save.session",
    ) -> Path:
        """Add or replace one recording session in the experiment."""
        return save_project_session(self.project_root, session, reason=reason)

    def pack(
        self,
        *,
        out: str | Path | None = None,
        media: str | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Pack the project into a portable `.expkg` artifact."""
        return pack_project(
            self.project_root,
            out=out,
            media=media,
            overwrite=overwrite,
        )


__all__ = [
    "ProjectService",
    "ProjectLayout",
    "ProjectInspection",
    "ProjectArtifacts",
    "ProjectCalibrations",
    "ProjectFigures",
    "ProjectMetadata",
    "ProjectSegmentation",
    "PoseFormat",
    "CalibrationFormat",
    "SignalFormat",
    "EventFormat",
    "BehaviorFormat",
    "SynchronizationFormat",
]
