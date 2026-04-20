"""Primary workspace-first service surface for xpkg project operations.

``WorkspaceService`` is the stable consumer-facing boundary for downstream
integrations that need to create, open, import into, validate, pack, or unpack
an xpkg workspace. ``WorkspaceImports`` mirrors the public
``xpkg.formats.import_*_workspace(...)`` helpers on a workspace-bound object so
new code can stay on the same service path end to end.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from xpkg.formats.project import (
    ProjectDescriptor,
    current_project_state_path,
    import_detectron2_coco_workspace,
    import_dlc_csv_workspace,
    import_dlc_h5_workspace,
    import_dlc_project_workspace,
    import_mediapipe_pose_landmarks_json_workspace,
    import_mmpose_topdown_json_workspace,
    import_openpose_json_workspace,
    import_sleap_h5_workspace,
    import_sleap_package_workspace,
    init_project,
    load_project_descriptor,
    pack_project,
    project_descriptor_path,
    resolve_workspace_root,
    save_workspace_labels,
    unpack_project,
    validate_workspace,
    workspace_exports_root,
    workspace_media_root,
    workspace_state_root,
    workspace_store_root,
)

if TYPE_CHECKING:
    from xpkg.model import Labels

PackMode = Literal["portable", "snapshot"]


@dataclass(frozen=True, slots=True)
class WorkspaceImports:
    """Workspace-bound mirror of the public ``xpkg.formats`` import helpers."""

    workspace_root: Path

    def _import(self, importer: Callable[..., Path], /, *args: Any, **kwargs: Any) -> Path:
        return importer(*args, workspace=self.workspace_root, **kwargs)

    def dlc_csv(
        self,
        csv_path: str | Path,
        video_path: str | Path,
        *,
        skeleton_name: str = "imported",
        likelihood_threshold: float = 0.0,
        default_pack_mode: PackMode = "portable",
        force: bool = False,
        progress_callback: Any | None = None,
    ) -> Path:
        """Import a DeepLabCut CSV plus video into this workspace."""
        return self._import(
            import_dlc_csv_workspace,
            csv_path,
            video_path,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            default_pack_mode=default_pack_mode,
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
        default_pack_mode: PackMode = "portable",
        force: bool = False,
        progress_callback: Any | None = None,
    ) -> Path:
        """Import a DeepLabCut H5 export plus video into this workspace."""
        return self._import(
            import_dlc_h5_workspace,
            h5_path,
            video_path,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            default_pack_mode=default_pack_mode,
            force=force,
            progress_callback=progress_callback,
        )

    def dlc_project(
        self,
        project_dir: str | Path,
        *,
        skeleton_name: str | None = None,
        likelihood_threshold: float = 0.0,
        default_pack_mode: PackMode = "portable",
        force: bool = False,
        progress_callback: Any | None = None,
    ) -> Path:
        """Import a supported DeepLabCut project into this workspace."""
        return self._import(
            import_dlc_project_workspace,
            project_dir,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            default_pack_mode=default_pack_mode,
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
        default_pack_mode: PackMode = "portable",
        force: bool = False,
        progress_callback: Any | None = None,
    ) -> Path:
        """Import a SLEAP analysis H5 export plus video into this workspace."""
        return self._import(
            import_sleap_h5_workspace,
            h5_path,
            video_path,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            default_pack_mode=default_pack_mode,
            force=force,
            progress_callback=progress_callback,
        )

    def sleap_package(
        self,
        slp: str | Path,
        *,
        fps: int = 30,
        encode_videos: bool | None = None,
        default_pack_mode: PackMode = "portable",
        force: bool = False,
        progress_callback: Any | None = None,
    ) -> Path:
        """Import a SLEAP package into this workspace."""
        return self._import(
            import_sleap_package_workspace,
            slp,
            fps=int(fps),
            encode_videos=encode_videos,
            default_pack_mode=default_pack_mode,
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
        default_pack_mode: PackMode = "portable",
        force: bool = False,
        progress_callback: Any | None = None,
    ) -> Path:
        """Import an MMPose top-down JSON export plus video into this workspace."""
        return self._import(
            import_mmpose_topdown_json_workspace,
            json_path,
            video_path,
            skeleton_name=skeleton_name,
            instance_index=int(instance_index),
            likelihood_threshold=likelihood_threshold,
            default_pack_mode=default_pack_mode,
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
        default_pack_mode: PackMode = "portable",
        force: bool = False,
        progress_callback: Any | None = None,
    ) -> Path:
        """Import MediaPipe pose-landmarks JSON plus video into this workspace."""
        return self._import(
            import_mediapipe_pose_landmarks_json_workspace,
            json_path,
            video_path,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            default_pack_mode=default_pack_mode,
            force=force,
            progress_callback=progress_callback,
        )

    def openpose_json(
        self,
        json_dir: str | Path,
        video_path: str | Path,
        *,
        skeleton_name: str = "imported",
        likelihood_threshold: float = 0.0,
        default_pack_mode: PackMode = "portable",
        force: bool = False,
        progress_callback: Any | None = None,
    ) -> Path:
        """Import an OpenPose JSON directory plus video into this workspace."""
        return self._import(
            import_openpose_json_workspace,
            json_dir,
            video_path,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            default_pack_mode=default_pack_mode,
            force=force,
            progress_callback=progress_callback,
        )

    def detectron2_coco(
        self,
        predictions_path: str | Path,
        dataset_json_path: str | Path,
        image_root: str | Path,
        *,
        category_id: int | None = None,
        skeleton_name: str | None = None,
        likelihood_threshold: float = 0.0,
        default_pack_mode: PackMode = "portable",
        force: bool = False,
        progress_callback: Any | None = None,
    ) -> Path:
        """Import Detectron2 COCO keypoint results into this workspace."""
        return self._import(
            import_detectron2_coco_workspace,
            predictions_path,
            dataset_json_path,
            image_root,
            category_id=category_id,
            skeleton_name=skeleton_name,
            likelihood_threshold=likelihood_threshold,
            default_pack_mode=default_pack_mode,
            force=force,
            progress_callback=progress_callback,
        )

@dataclass(frozen=True, slots=True)
class WorkspaceLayout:
    """Normalized workspace summary returned by ``describe()`` and ``validate()``."""

    workspace_root: Path
    descriptor: ProjectDescriptor
    descriptor_path: Path
    store_root: Path
    state_root: Path
    media_root: Path
    exports_root: Path
    current_state_path: Path
    has_current_state: bool


@dataclass(slots=True)
class WorkspaceService:
    """Stable public service for workspace-first project lifecycle operations."""

    workspace_root: Path

    @classmethod
    def create(
        cls,
        workspace: str | Path,
        *,
        title: str | None = None,
        project_id: str | None = None,
        default_pack_mode: PackMode = "portable",
        force: bool = False,
    ) -> WorkspaceService:
        """Create a workspace with the canonical public layout and open it."""
        init_project(
            workspace,
            title=title,
            project_id=project_id,
            default_pack_mode=default_pack_mode,
            force=force,
        )
        return cls.open(workspace)

    @classmethod
    def open(cls, workspace: str | Path) -> WorkspaceService:
        """Open an existing workspace root or a path inside one."""
        root = resolve_workspace_root(workspace)
        if root is None:
            raise FileNotFoundError(f"Not an xpkg workspace: {workspace}")
        return cls(workspace_root=root)

    @classmethod
    def unpack(
        cls,
        artifact: str | Path,
        out: str | Path,
        *,
        force: bool = False,
        rename_title: str | None = None,
    ) -> WorkspaceService:
        """Unpack a portable `.expkg` artifact into a workspace and open it."""
        unpack_project(
            artifact,
            out,
            force=force,
            rename_title=rename_title,
        )
        return cls.open(out)

    def descriptor(self) -> ProjectDescriptor:
        """Load the current workspace descriptor."""
        return load_project_descriptor(self.workspace_root)

    @property
    def imports(self) -> WorkspaceImports:
        """Return service-bound import helpers backed by the public format API."""
        return WorkspaceImports(workspace_root=self.workspace_root)

    def describe(self) -> WorkspaceLayout:
        """Return the normalized managed paths for this workspace."""
        descriptor = self.descriptor()
        state_path = current_project_state_path(self.workspace_root)
        return WorkspaceLayout(
            workspace_root=self.workspace_root,
            descriptor=descriptor,
            descriptor_path=project_descriptor_path(self.workspace_root),
            store_root=workspace_store_root(self.workspace_root),
            state_root=workspace_state_root(self.workspace_root),
            media_root=workspace_media_root(self.workspace_root),
            exports_root=workspace_exports_root(self.workspace_root),
            current_state_path=state_path,
            has_current_state=state_path.exists(),
        )

    def validate(self) -> WorkspaceLayout:
        """Validate the workspace and return its normalized layout."""
        validate_workspace(self.workspace_root)
        return self.describe()

    def load_labels(self) -> Labels:
        """Load the current workspace labels through the public workspace root."""
        from xpkg.model import Labels

        return Labels.load_file(self.workspace_root.as_posix())

    def save_labels(
        self,
        labels: Labels,
        *,
        metadata: dict[str, Any] | None = None,
        journal: bool = True,
        regenerate_predictions: bool = False,
    ) -> Path:
        """Commit labels into the workspace-managed durable state."""
        return save_workspace_labels(
            self.workspace_root,
            labels,
            metadata=metadata,
            journal=journal,
            regenerate_predictions=regenerate_predictions,
        )

    def pack(
        self,
        *,
        out: str | Path | None = None,
        mode: PackMode | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Pack the workspace into a portable `.expkg` artifact."""
        return pack_project(
            self.workspace_root,
            out=out,
            mode=mode,
            overwrite=overwrite,
        )


__all__ = ["WorkspaceService", "WorkspaceImports", "WorkspaceLayout"]
