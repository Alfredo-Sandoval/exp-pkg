"""Primary project-first service surface for xpkg project operations.

``ProjectService`` is the stable consumer-facing boundary for downstream
integrations that need to create, open, import into, validate, pack, or unpack
an xpkg project. ``ProjectImports`` mirrors the public
``xpkg.project.import_*_project(...)`` helpers on a project-bound object so
new code can stay on the same service path end to end.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xpkg.project import (
    ProjectDescriptor,
    ProjectInspection,
    current_project_state_path,
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
    init_project,
    inspect_project,
    load_project_descriptor,
    load_project_metadata,
    load_project_metadata_field,
    load_project_payload,
    load_project_vicon_recording,
    pack_project,
    project_artifacts_root,
    project_descriptor_path,
    project_exports_root,
    project_media_root,
    project_state_root,
    project_store_root,
    resolve_project_root,
    save_project_labels,
    save_project_metadata,
    save_project_metadata_field,
    unpack_project,
    validate_project,
)
from xpkg.project.state import project_state_kind
from xpkg.project.store import ensure_current_project_state_cache
from xpkg.services.artifacts import ProjectArtifacts
from xpkg.services.figures import ProjectFigures
from xpkg.services.segmentation import ProjectSegmentation

if TYPE_CHECKING:
    from xpkg.model import Labels, ViconRecording


@dataclass(frozen=True, slots=True)
class ProjectImports:
    """Project-bound mirror of the public ``xpkg.project`` import helpers."""

    project_root: Path

    def _import(self, importer: Callable[..., Path], /, *args: Any, **kwargs: Any) -> Path:
        return importer(*args, project=self.project_root, **kwargs)

    def vicon(
        self,
        recording_path: str | Path,
        *,
        force: bool = False,
        progress_callback: Any | None = None,
    ) -> Path:
        """Import a Vicon CSV or C3D recording into this project."""
        return self._import(
            import_vicon_project,
            recording_path,
            force=force,
            progress_callback=progress_callback,
        )

    def vicon_csv(
        self,
        csv_path: str | Path,
        *,
        force: bool = False,
        progress_callback: Any | None = None,
    ) -> Path:
        """Import a Vicon CSV recording into this project."""
        return self._import(
            import_vicon_csv_project,
            csv_path,
            force=force,
            progress_callback=progress_callback,
        )

    def vicon_c3d(
        self,
        c3d_path: str | Path,
        *,
        force: bool = False,
        progress_callback: Any | None = None,
    ) -> Path:
        """Import a Vicon C3D recording into this project."""
        return self._import(
            import_vicon_c3d_project,
            c3d_path,
            force=force,
            progress_callback=progress_callback,
        )

    def dlc_csv(
        self,
        csv_path: str | Path,
        video_path: str | Path,
        *,
        skeleton_name: str = "imported",
        likelihood_threshold: float = 0.0,
        force: bool = False,
        progress_callback: Any | None = None,
    ) -> Path:
        """Import a DeepLabCut CSV plus video into this project."""
        return self._import(
            import_dlc_csv_project,
            csv_path,
            video_path,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            force=force,
            progress_callback=progress_callback,
        )

    def dlc_h5(
        self,
        h5_path: str | Path,
        video_path: str | Path,
        *,
        skeleton_name: str = "imported",
        likelihood_threshold: float = 0.0,
        force: bool = False,
        progress_callback: Any | None = None,
    ) -> Path:
        """Import a DeepLabCut H5 export plus video into this project."""
        return self._import(
            import_dlc_h5_project,
            h5_path,
            video_path,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            force=force,
            progress_callback=progress_callback,
        )

    def dlc_project(
        self,
        project_dir: str | Path,
        *,
        skeleton_name: str | None = None,
        likelihood_threshold: float = 0.0,
        force: bool = False,
        progress_callback: Any | None = None,
    ) -> Path:
        """Import a supported DeepLabCut project into this project."""
        return self._import(
            import_dlc_project_directory,
            project_dir,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            force=force,
            progress_callback=progress_callback,
        )

    def lightning_pose_csv(
        self,
        csv_path: str | Path,
        video_path: str | Path,
        *,
        skeleton_name: str = "imported",
        likelihood_threshold: float = 0.0,
        force: bool = False,
        progress_callback: Any | None = None,
    ) -> Path:
        """Import a Lightning Pose prediction CSV plus video into this project."""
        return self._import(
            import_lightning_pose_csv_project,
            csv_path,
            video_path,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            force=force,
            progress_callback=progress_callback,
        )

    def sleap_h5(
        self,
        h5_path: str | Path,
        video_path: str | Path,
        *,
        skeleton_name: str = "imported",
        likelihood_threshold: float = 0.0,
        force: bool = False,
        progress_callback: Any | None = None,
    ) -> Path:
        """Import a SLEAP analysis H5 export plus video into this project."""
        return self._import(
            import_sleap_h5_project,
            h5_path,
            video_path,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            force=force,
            progress_callback=progress_callback,
        )

    def sleap_package(
        self,
        slp: str | Path,
        *,
        fps: int = 30,
        encode_videos: bool | None = None,
        force: bool = False,
        progress_callback: Any | None = None,
    ) -> Path:
        """Import a SLEAP package into this project."""
        return self._import(
            import_sleap_package_project,
            slp,
            fps=int(fps),
            encode_videos=encode_videos,
            force=force,
            progress_callback=progress_callback,
        )

    def mmpose_topdown_json(
        self,
        json_path: str | Path,
        video_path: str | Path,
        *,
        skeleton_name: str = "imported",
        instance_index: int = 0,
        likelihood_threshold: float = 0.0,
        force: bool = False,
        progress_callback: Any | None = None,
    ) -> Path:
        """Import an MMPose top-down JSON export plus video into this project."""
        return self._import(
            import_mmpose_topdown_json_project,
            json_path,
            video_path,
            skeleton_name=skeleton_name,
            instance_index=int(instance_index),
            likelihood_threshold=likelihood_threshold,
            force=force,
            progress_callback=progress_callback,
        )

    def mediapipe_pose_landmarks_json(
        self,
        json_path: str | Path,
        video_path: str | Path,
        *,
        skeleton_name: str = "mediapipe_pose",
        likelihood_threshold: float = 0.0,
        force: bool = False,
        progress_callback: Any | None = None,
    ) -> Path:
        """Import MediaPipe pose-landmarks JSON plus video into this project."""
        return self._import(
            import_mediapipe_pose_landmarks_json_project,
            json_path,
            video_path,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            force=force,
            progress_callback=progress_callback,
        )


@dataclass(frozen=True, slots=True)
class ProjectLayout:
    """Normalized project summary returned by ``describe()`` and ``validate()``."""

    project_root: Path
    descriptor: ProjectDescriptor
    descriptor_path: Path
    store_root: Path
    artifacts_root: Path
    state_root: Path
    media_root: Path
    exports_root: Path
    current_state_path: Path
    has_current_state: bool


@dataclass(slots=True)
class ProjectService:
    """Stable public service for project-first project lifecycle operations."""

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
    def imports(self) -> ProjectImports:
        """Return service-bound import helpers backed by the public format API."""
        return ProjectImports(project_root=self.project_root)

    @property
    def segmentation(self) -> ProjectSegmentation:
        """Return service-bound segmentation mask helpers."""
        return ProjectSegmentation(project_root=self.project_root)

    @property
    def artifacts(self) -> ProjectArtifacts:
        """Return service-bound generic artifact registry helpers."""
        return ProjectArtifacts(project_root=self.project_root)

    @property
    def figures(self) -> ProjectFigures:
        """Return service-bound figure artifact helpers."""
        return ProjectFigures(project_root=self.project_root)

    def describe(self) -> ProjectLayout:
        """Return the normalized managed paths for this project."""
        descriptor = self.descriptor()
        state_path = current_project_state_path(self.project_root)
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
            has_current_state=state_path.exists(),
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

    def load_metadata(self) -> dict[str, Any] | None:
        """Load the current project metadata payload."""
        return load_project_metadata(self.project_root)

    def load_metadata_field(self, field: str) -> dict[str, Any] | None:
        """Load one mapping-valued metadata field from the current project head."""
        return load_project_metadata_field(self.project_root, field)

    def load_payload(self) -> dict[str, Any]:
        """Load the current project payload with project-relative media rebased."""
        return load_project_payload(self.project_root)

    def save_metadata(
        self,
        metadata: dict[str, Any] | None,
        *,
        reason: str = "project.save.metadata",
    ) -> Path:
        """Commit metadata onto the current project head."""
        return save_project_metadata(
            self.project_root,
            metadata,
            reason=reason,
        )

    def save_metadata_field(
        self,
        field: str,
        value: Mapping[str, Any],
        *,
        reason: str = "project.save.metadata_field",
    ) -> Path:
        """Persist one mapping-valued metadata field onto the current project head."""
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
    "ProjectImports",
    "ProjectLayout",
    "ProjectInspection",
    "ProjectArtifacts",
    "ProjectFigures",
    "ProjectSegmentation",
]
