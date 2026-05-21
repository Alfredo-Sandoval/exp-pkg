"""Primary project-first service surface for xpkg project operations.

``ProjectService`` is the stable consumer-facing boundary for downstream
integrations that need to create, open, import into, validate, pack, or unpack
an xpkg project. The ``import_pose`` / ``import_calibration`` /
``import_motion`` dispatch methods select package-owned importer
implementations by kebab-case ``format`` string.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from xpkg.project import (
    ProjectDescriptor,
    ProjectInspection,
    ProjectSummaryIndex,
    current_project_state_path,
    init_project,
    inspect_project,
    load_project_acquisition_metadata,
    load_project_dataset_share_metadata,
    load_project_datasheet,
    load_project_descriptor,
    load_project_metadata,
    load_project_metadata_field,
    load_project_model_card,
    load_project_payload,
    load_project_pose_provenance,
    load_project_vicon_recording,
    pack_project,
    project_artifacts_root,
    project_descriptor_path,
    project_exports_root,
    project_media_root,
    project_metadata_root,
    project_state_root,
    project_store_root,
    project_summary_path,
    refresh_project_summary,
    resolve_project_root,
    save_project_acquisition_metadata,
    save_project_dataset_share_metadata,
    save_project_datasheet,
    save_project_labels,
    save_project_metadata,
    save_project_metadata_field,
    save_project_model_card,
    save_project_pose_provenance,
    unpack_project,
    validate_project,
)
from xpkg.project.calibration import (
    import_anipose_calibration_project,
    import_opencv_stereo_calibration_project,
)
from xpkg.project.state import project_state_kind
from xpkg.project.store import ensure_current_project_state_cache
from xpkg.project.store.imports import (
    import_dlc_csv_project,
    import_dlc_h5_project,
    import_dlc_project_directory,
    import_lightning_pose_csv_project,
    import_mediapipe_pose_landmarks_json_project,
    import_mmpose_topdown_json_project,
    import_sleap_h5_project,
    import_sleap_package_project,
    import_vicon_c3d_project,
    import_vicon_csv_project,
    import_vicon_project,
)
from xpkg.services.artifacts import ProjectArtifacts
from xpkg.services.calibrations import ProjectCalibrations
from xpkg.services.figures import ProjectFigures
from xpkg.services.segmentation import ProjectSegmentation

if TYPE_CHECKING:
    from xpkg.model import (
        AcquisitionMetadata,
        DatasetDatasheet,
        DatasetShareMetadata,
        Labels,
        ModelCard,
        PoseModelProvenance,
        ViconRecording,
    )


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


MotionFormat = Literal["vicon", "vicon-csv", "vicon-c3d"]
"""Supported `format` values for ``ProjectService.import_motion``.

``"vicon"`` auto-detects CSV vs C3D from the input path; the dashed forms
force the matching reader.
"""


@dataclass(frozen=True, slots=True)
class _PoseImporter:
    """Per-format dispatch entry for ``ProjectService.import_pose``.

    Captures the per-format quirks (video requirement, skeleton-name default,
    extra kwargs, omitted common kwargs) so the dispatch method can stay a
    single thin pass over the registry.
    """

    fn: Callable[..., Path]
    requires_video: bool
    accepts_skeleton_name: bool = True
    default_skeleton_name: str | None = None
    accepts_likelihood_threshold: bool = True
    accepts_instance_index: bool = False
    accepts_fps: bool = False
    accepts_encode_videos: bool = False


_POSE_IMPORTERS: dict[str, _PoseImporter] = {
    "dlc-csv": _PoseImporter(
        fn=import_dlc_csv_project,
        requires_video=True,
        default_skeleton_name="imported",
    ),
    "dlc-h5": _PoseImporter(
        fn=import_dlc_h5_project,
        requires_video=True,
        default_skeleton_name="imported",
    ),
    "dlc-project": _PoseImporter(
        fn=import_dlc_project_directory,
        requires_video=False,
    ),
    "lightning-pose-csv": _PoseImporter(
        fn=import_lightning_pose_csv_project,
        requires_video=True,
        default_skeleton_name="imported",
    ),
    "mediapipe-pose-landmarks-json": _PoseImporter(
        fn=import_mediapipe_pose_landmarks_json_project,
        requires_video=True,
        default_skeleton_name="mediapipe_pose",
    ),
    "mmpose-topdown-json": _PoseImporter(
        fn=import_mmpose_topdown_json_project,
        requires_video=True,
        default_skeleton_name="imported",
        accepts_instance_index=True,
    ),
    "sleap-h5": _PoseImporter(
        fn=import_sleap_h5_project,
        requires_video=True,
        default_skeleton_name="imported",
    ),
    "sleap-package": _PoseImporter(
        fn=import_sleap_package_project,
        requires_video=False,
        accepts_skeleton_name=False,
        accepts_likelihood_threshold=False,
        accepts_fps=True,
        accepts_encode_videos=True,
    ),
}


_MOTION_IMPORTERS: dict[str, Callable[..., Path]] = {
    "vicon": import_vicon_project,
    "vicon-csv": import_vicon_csv_project,
    "vicon-c3d": import_vicon_c3d_project,
}


_CALIBRATION_IMPORTERS: dict[str, Callable[..., Path]] = {
    "anipose": import_anipose_calibration_project,
    "opencv-stereo-yaml": import_opencv_stereo_calibration_project,
}


@dataclass(frozen=True, slots=True)
class ProjectMetadata:
    """Accessor for the durable typed metadata slots under ``.xpkg/metadata/``.

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
    def acquisition(self) -> AcquisitionMetadata | None:
        return load_project_acquisition_metadata(self.project_root)

    @property
    def dataset_share(self) -> DatasetShareMetadata | None:
        return load_project_dataset_share_metadata(self.project_root)

    @property
    def pose_provenance(self) -> PoseModelProvenance | None:
        return load_project_pose_provenance(self.project_root)

    @property
    def datasheet(self) -> DatasetDatasheet | None:
        return load_project_datasheet(self.project_root)

    @property
    def model_card(self) -> ModelCard | None:
        return load_project_model_card(self.project_root)

    def update(
        self,
        *,
        acquisition: AcquisitionMetadata | Mapping[str, Any] | None = None,
        dataset_share: DatasetShareMetadata | Mapping[str, Any] | None = None,
        pose_provenance: PoseModelProvenance | Mapping[str, Any] | None = None,
        datasheet: DatasetDatasheet | Mapping[str, Any] | None = None,
        model_card: ModelCard | Mapping[str, Any] | None = None,
    ) -> dict[str, Path]:
        """Write one or more typed metadata slots and return the paths written.

        Slots left as ``None`` are not touched. Each provided slot accepts the
        canonical typed value or a JSON-shaped mapping that will be coerced
        through the slot's ``from_dict``. Writes happen in a fixed order; if a
        write raises, slots written before it remain on disk.
        """
        written: dict[str, Path] = {}
        if acquisition is not None:
            written["acquisition"] = save_project_acquisition_metadata(
                self.project_root, acquisition
            )
        if dataset_share is not None:
            written["dataset_share"] = save_project_dataset_share_metadata(
                self.project_root, dataset_share
            )
        if pose_provenance is not None:
            written["pose_provenance"] = save_project_pose_provenance(
                self.project_root, pose_provenance
            )
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
        likelihood_threshold: float = 0.0,
        instance_index: int = 0,
        fps: int = 30,
        encode_videos: bool | None = None,
        prediction_provenance: Mapping[str, Any] | None = None,
        provenance: PoseModelProvenance | Mapping[str, Any] | None = None,
        force: bool = False,
        progress_callback: Any | None = None,
    ) -> Path:
        """Import a pose track from one of the supported formats into this project.

        The ``video`` keyword is required for the per-clip formats
        (``dlc-csv``, ``dlc-h5``, ``lightning-pose-csv``,
        ``mediapipe-pose-landmarks-json``, ``mmpose-topdown-json``,
        ``sleap-h5``) and ignored for ``dlc-project`` and ``sleap-package``.
        ``instance_index`` only applies to ``mmpose-topdown-json``;
        ``fps`` and ``encode_videos`` only apply to ``sleap-package``.
        """
        importer = _POSE_IMPORTERS.get(format)
        if importer is None:
            raise ValueError(f"Unknown pose format: {format!r}")
        if importer.requires_video and video is None:
            raise ValueError(f"import_pose(format={format!r}) requires `video=`.")

        kwargs: dict[str, Any] = {
            "project": self.project_root,
            "prediction_provenance": prediction_provenance,
            "provenance": provenance,
            "force": force,
            "progress_callback": progress_callback,
        }
        if importer.accepts_likelihood_threshold:
            kwargs["likelihood_threshold"] = likelihood_threshold
        if importer.accepts_skeleton_name:
            kwargs["skeleton_name"] = skeleton_name or importer.default_skeleton_name
        if importer.accepts_instance_index:
            kwargs["instance_index"] = int(instance_index)
        if importer.accepts_fps:
            kwargs["fps"] = int(fps)
        if importer.accepts_encode_videos:
            kwargs["encode_videos"] = encode_videos

        if importer.requires_video:
            assert video is not None  # narrowed above
            return importer.fn(path, video, **kwargs)
        return importer.fn(path, **kwargs)

    def import_calibration(
        self,
        format: CalibrationFormat,
        *,
        path: str | Path,
        calibration_id: str | None = None,
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
            "name": name,
            "units": units,
            "captured_at": captured_at,
            "tool_version": tool_version,
            "force": force,
        }
        if camera_names is not None:
            kwargs["camera_names"] = camera_names
        return importer(
            path,
            **kwargs,
        )

    def import_motion(
        self,
        format: MotionFormat = "vicon",
        *,
        path: str | Path,
        force: bool = False,
        progress_callback: Any | None = None,
    ) -> Path:
        """Import a motion-capture recording from one of the supported formats.

        ``"vicon"`` auto-detects CSV vs C3D from the path; the dashed forms
        force the matching reader.
        """
        importer = _MOTION_IMPORTERS.get(format)
        if importer is None:
            raise ValueError(f"Unknown motion format: {format!r}")
        return importer(
            path,
            project=self.project_root,
            force=force,
            progress_callback=progress_callback,
        )

    def describe(self) -> ProjectLayout:
        """Return the normalized managed paths for this project."""
        descriptor = self.descriptor()
        state_path = current_project_state_path(self.project_root)
        summary = refresh_project_summary(self.project_root)
        return ProjectLayout(
            project_root=self.project_root,
            descriptor=descriptor,
            descriptor_path=project_descriptor_path(self.project_root),
            store_root=project_store_root(self.project_root),
            artifacts_root=project_artifacts_root(self.project_root),
            state_root=project_state_root(self.project_root),
            media_root=project_media_root(self.project_root),
            exports_root=project_exports_root(self.project_root),
            current_state_path=state_path,
            summary_path=project_summary_path(self.project_root),
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

    def load_labels(self) -> Labels:
        """Load the current project labels through the public project root."""
        from xpkg.model import Labels

        state_path = ensure_current_project_state_cache(self.project_root)
        if state_path is None:
            state_path = current_project_state_path(self.project_root)
        if state_path.exists() and state_path.suffix.lower() == ".json":
            if project_state_kind(state_path) == "vicon":
                raise ValueError(
                    "Project current state is a Vicon recording. "
                    "Use ProjectService.load_vicon_recording()."
                )
        return Labels.load_file(self.project_root.as_posix())

    def load_vicon_recording(self) -> ViconRecording:
        """Load the current project Vicon recording."""
        return load_project_vicon_recording(self.project_root)

    def load_state_metadata(self) -> dict[str, Any] | None:
        """Load the current project state's free-form metadata dict.

        This is distinct from :attr:`metadata` — that accessor reads typed
        durable slots under ``.xpkg/metadata/``; this method reads the
        free-form ``metadata`` field on the current project state head.
        """
        return load_project_metadata(self.project_root)

    def load_state_metadata_field(self, field: str) -> dict[str, Any] | None:
        """Load one mapping-valued field from the current project state metadata."""
        return load_project_metadata_field(self.project_root, field)

    def load_payload(self) -> dict[str, Any]:
        """Load the current project payload with project-relative media rebased."""
        return load_project_payload(self.project_root)

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
        journal: bool = True,
        regenerate_predictions: bool = False,
    ) -> Path:
        """Commit labels into the project-managed durable state."""
        return save_project_labels(
            self.project_root,
            labels,
            metadata=metadata,
            journal=journal,
            regenerate_predictions=regenerate_predictions,
        )

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
    "MotionFormat",
]
