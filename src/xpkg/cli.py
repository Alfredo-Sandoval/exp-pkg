"""xpkg command-line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from xpkg.core.json_utils import dump_json
from xpkg.formats import (
    import_detectron2_coco_workspace,
    import_dlc_csv_workspace,
    import_dlc_h5_workspace,
    import_dlc_project_workspace,
    import_mediapipe_pose_landmarks_json_workspace,
    import_mmpose_topdown_json_workspace,
    import_openpose_json_workspace,
    import_sleap_h5_workspace,
    import_sleap_package_workspace,
    import_vicon_c3d_workspace,
    import_vicon_csv_workspace,
    import_vicon_workspace,
    init_project,
    list_workspace_artifact_index,
    load_workspace_artifact,
    migrate_legacy_archive,
    pack_project,
    rebuild_workspace_artifact_index,
    unpack_project,
    validate_artifact,
    validate_workspace_artifact,
    validate_workspace_artifacts,
)
from xpkg.io.archive_format.shared import CANONICAL_ARCHIVE_SUFFIX
from xpkg.version import __version__

CliCommand = Callable[[argparse.Namespace], int]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {parsed}")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {parsed}")
    return parsed


def _likelihood_threshold(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError(f"expected a likelihood threshold in [0, 1], got {parsed}")
    return parsed


def _emit_progress(message: str) -> None:
    sys.stdout.write(message + "\n")


def _write_path(path: Path) -> None:
    sys.stdout.write(f"{path}\n")


def _add_workspace_tracking_parser(
    parser: argparse.ArgumentParser,
    *,
    data_flag: str,
    data_help: str,
    skeleton_default: str,
    skeleton_help: str,
    command: CliCommand,
) -> None:
    parser.add_argument(data_flag, required=True, help=data_help)
    parser.add_argument("--video", required=True, help="Path to the matching video file.")
    parser.add_argument("--out", required=True, help="Output workspace directory.")
    parser.add_argument(
        "--skeleton-name",
        default=skeleton_default,
        help=skeleton_help,
    )
    parser.add_argument(
        "--threshold",
        type=_likelihood_threshold,
        default=0.0,
        help="Likelihood threshold for including keypoints (0 to 1).",
    )
    parser.set_defaults(func=command)


def _add_import_parser(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    imported = parent.add_parser(
        "import",
        help="Import external tracking data into a workspace-first exp-pkg project.",
    )
    import_subparsers = imported.add_subparsers(dest="import_source", required=True)

    vicon = import_subparsers.add_parser(
        "vicon",
        help="Import a Vicon CSV or C3D recording into a workspace.",
    )
    vicon_group = vicon.add_mutually_exclusive_group(required=True)
    vicon_group.add_argument(
        "--recording",
        help="Path to a Vicon recording (.csv or .c3d).",
    )
    vicon_group.add_argument(
        "--csv",
        help="Path to a Vicon Nexus CSV trajectory export.",
    )
    vicon_group.add_argument(
        "--c3d",
        help="Path to a Vicon C3D recording.",
    )
    vicon.add_argument("--out", required=True, help="Output workspace directory.")
    vicon.set_defaults(func=_cmd_import_vicon)

    dlc = import_subparsers.add_parser("dlc", help="Import DeepLabCut tracking into a workspace.")
    dlc_subparsers = dlc.add_subparsers(dest="dlc_source", required=True)

    csv_parser = dlc_subparsers.add_parser("csv", help="Import a DLC CSV tracking file.")
    _add_workspace_tracking_parser(
        csv_parser,
        data_flag="--csv",
        data_help="Path to a DLC CSV tracking file.",
        skeleton_default="imported",
        skeleton_help="Skeleton name to store in the imported workspace.",
        command=_cmd_import_dlc_csv,
    )

    h5_parser = dlc_subparsers.add_parser("h5", help="Import a DLC H5 tracking file.")
    _add_workspace_tracking_parser(
        h5_parser,
        data_flag="--h5",
        data_help="Path to a DLC H5 tracking file.",
        skeleton_default="imported",
        skeleton_help="Skeleton name to store in the imported workspace.",
        command=_cmd_import_dlc_h5,
    )

    project_parser = dlc_subparsers.add_parser(
        "project",
        help="Import an entire DLC project into one workspace.",
    )
    project_parser.add_argument("--project", required=True, help="Path to the DLC project root.")
    project_parser.add_argument("--out", required=True, help="Output workspace directory.")
    project_parser.add_argument(
        "--threshold",
        type=_likelihood_threshold,
        default=0.0,
        help="Likelihood threshold for including keypoints (0 to 1).",
    )
    project_parser.set_defaults(func=_cmd_import_dlc_project)

    sleap = import_subparsers.add_parser("sleap", help="Import SLEAP package or analysis H5 data.")
    source_group = sleap.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--slp", help="Path to the input .pkg.slp archive.")
    source_group.add_argument("--h5", help="Path to the SLEAP analysis H5 export.")
    sleap.add_argument("--video", help="Path to the matching video file when using --h5.")
    sleap.add_argument("--out", required=True, help="Output workspace directory.")
    sleap.add_argument(
        "--skeleton-name",
        default="imported",
        help="Skeleton name to store when importing --h5 tracking data.",
    )
    sleap.add_argument(
        "--threshold",
        type=_likelihood_threshold,
        default=0.0,
        help="Likelihood threshold for including keypoints when using --h5 (0 to 1).",
    )
    sleap.add_argument(
        "--fps",
        type=_positive_int,
        default=30,
        help="Frame rate to use when encoding videos.",
    )
    encode_group = sleap.add_mutually_exclusive_group()
    encode_group.add_argument(
        "--videos",
        dest="encode_videos",
        action="store_true",
        help="Encode MP4 videos into the internal store.",
    )
    encode_group.add_argument(
        "--no-videos",
        dest="encode_videos",
        action="store_false",
        help="Keep extracted frame sequences instead of encoding MP4 videos.",
    )
    sleap.set_defaults(func=_cmd_import_sleap, encode_videos=True)

    mmpose = import_subparsers.add_parser(
        "mmpose",
        help="Import MMPose top-down demo JSON predictions into a workspace.",
    )
    _add_workspace_tracking_parser(
        mmpose,
        data_flag="--json",
        data_help="Path to an MMPose --save-predictions JSON export.",
        skeleton_default="imported",
        skeleton_help="Skeleton name to store in the imported workspace.",
        command=_cmd_import_mmpose,
    )
    mmpose.add_argument(
        "--instance-index",
        type=_nonnegative_int,
        default=0,
        help="Per-frame instance slot to import from the MMPose predictions.",
    )

    mediapipe = import_subparsers.add_parser(
        "mediapipe",
        help="Import MediaPipe pose-landmarks JSON into a workspace.",
    )
    _add_workspace_tracking_parser(
        mediapipe,
        data_flag="--json",
        data_help="Path to MediaPipe pose-landmarks JSON.",
        skeleton_default="mediapipe_pose",
        skeleton_help="Skeleton name to store in the imported workspace.",
        command=_cmd_import_mediapipe,
    )

    openpose = import_subparsers.add_parser(
        "openpose",
        help="Import OpenPose BODY_25 JSON directories into a workspace.",
    )
    _add_workspace_tracking_parser(
        openpose,
        data_flag="--json",
        data_help="Path to an OpenPose --write_json directory.",
        skeleton_default="imported",
        skeleton_help="Skeleton name to store in the imported workspace.",
        command=_cmd_import_openpose,
    )

    detectron2 = import_subparsers.add_parser(
        "detectron2",
        help="Import Detectron2 COCO keypoint predictions into a workspace.",
    )
    detectron2.add_argument(
        "--predictions",
        required=True,
        help="Path to Detectron2 COCOEvaluator coco_instances_results.json.",
    )
    detectron2.add_argument(
        "--dataset-json",
        required=True,
        help="Path to the registered COCO dataset JSON.",
    )
    detectron2.add_argument(
        "--image-root",
        required=True,
        help="Image root paired with the COCO dataset JSON.",
    )
    detectron2.add_argument("--out", required=True, help="Output workspace directory.")
    detectron2.add_argument(
        "--category-id",
        type=int,
        help="COCO keypoint category id to import when multiple keypoint categories exist.",
    )
    detectron2.add_argument(
        "--skeleton-name",
        help="Optional skeleton name override for the imported keypoints.",
    )
    detectron2.add_argument(
        "--threshold",
        type=_likelihood_threshold,
        default=0.0,
        help="Likelihood threshold for including keypoints (0 to 1).",
    )
    detectron2.set_defaults(func=_cmd_import_detectron2)


def _add_workspace_parsers(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    init = parent.add_parser("init", help="Create a new empty exp-pkg workspace.")
    init.add_argument("workspace", help="Workspace directory to create.")
    init.add_argument("--title", help="Optional project title override.")
    init.add_argument("--id", dest="project_id", help="Optional project identifier override.")
    init.add_argument(
        "--pack-mode",
        choices=("portable", "snapshot"),
        default="portable",
        help="Default pack mode recorded in PROJECT.json.",
    )
    init.add_argument(
        "--force",
        action="store_true",
        help="Allow initialization into an existing empty directory.",
    )
    init.set_defaults(func=_cmd_init)

    migrate = parent.add_parser(
        "migrate",
        help=(
            f"Migrate a canonical {CANONICAL_ARCHIVE_SUFFIX} archive into a "
            "workspace-first project."
        ),
    )
    migrate.add_argument(
        "legacy_archive",
        help=f"Path to a canonical {CANONICAL_ARCHIVE_SUFFIX} archive.",
    )
    migrate.add_argument("--out", required=True, help="Output workspace directory.")
    migrate.add_argument("--title", help="Optional project title override.")
    migrate.add_argument(
        "--force",
        action="store_true",
        help="Allow initialization into an existing empty directory.",
    )
    migrate.set_defaults(func=_cmd_migrate)

    pack = parent.add_parser("pack", help="Pack a workspace into a .expkg artifact.")
    pack.add_argument("workspace", help="Workspace directory to pack.")
    pack.add_argument("--out", help="Explicit output .expkg path.")
    pack.add_argument(
        "--mode",
        choices=("portable", "snapshot"),
        help="Pack mode. Defaults to the workspace default.",
    )
    pack.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output artifact.",
    )
    pack.set_defaults(func=_cmd_pack)

    unpack = parent.add_parser("unpack", help="Unpack a .expkg artifact into a workspace.")
    unpack.add_argument("artifact", help="Path to the .expkg artifact.")
    unpack.add_argument("--out", required=True, help="Destination workspace directory.")
    unpack.add_argument(
        "--force",
        action="store_true",
        help="Allow unpacking into an existing empty directory.",
    )
    unpack.add_argument("--rename", help="Optional new project title during unpack.")
    unpack.set_defaults(func=_cmd_unpack)

    validate = parent.add_parser(
        "validate",
        help="Validate a workspace or packed .expkg artifact.",
    )
    validate.add_argument("path", help="Path to validate.")
    validate.set_defaults(func=_cmd_validate)


def _add_artifacts_parser(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    artifacts = parent.add_parser(
        "artifacts",
        help="Inspect and validate workspace artifact manifests.",
    )
    artifact_subparsers = artifacts.add_subparsers(dest="artifact_command", required=True)

    list_parser = artifact_subparsers.add_parser("list", help="List registered artifacts.")
    list_parser.add_argument("workspace", help="Workspace directory to inspect.")
    list_parser.add_argument("--kind", help="Optional artifact kind filter, such as figure.")
    list_parser.add_argument("--namespace", help="Optional caller-owned namespace filter.")
    list_parser.set_defaults(func=_cmd_artifacts_list)

    inspect = artifact_subparsers.add_parser("inspect", help="Print one artifact manifest.")
    inspect.add_argument("workspace", help="Workspace directory to inspect.")
    inspect.add_argument("artifact_id", help="Artifact id to inspect.")
    inspect.add_argument("--kind", help="Optional artifact kind, such as figure.")
    inspect.add_argument("--namespace", help="Optional caller-owned namespace.")
    inspect.set_defaults(func=_cmd_artifacts_inspect)

    validate = artifact_subparsers.add_parser(
        "validate",
        help="Validate one artifact or every matching artifact.",
    )
    validate.add_argument("workspace", help="Workspace directory to validate.")
    validate.add_argument("artifact_id", nargs="?", help="Optional artifact id to validate.")
    validate.add_argument("--kind", help="Optional artifact kind, such as figure.")
    validate.add_argument("--namespace", help="Optional caller-owned namespace filter.")
    validate.set_defaults(func=_cmd_artifacts_validate)

    rebuild = artifact_subparsers.add_parser(
        "rebuild-index",
        help="Rebuild the workspace artifact index from manifests.",
    )
    rebuild.add_argument("workspace", help="Workspace directory to index.")
    rebuild.set_defaults(func=_cmd_artifacts_rebuild_index)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xpkg", description="xpkg CLI tools")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_workspace_parsers(subparsers)
    _add_artifacts_parser(subparsers)
    _add_import_parser(subparsers)
    return parser


def _cmd_init(args: argparse.Namespace) -> int:
    init_project(
        args.workspace,
        title=args.title,
        project_id=args.project_id,
        default_pack_mode=args.pack_mode,
        force=args.force,
    )
    sys.stdout.write(f"Initialized workspace {Path(args.workspace)}\n")
    return 0


def _cmd_migrate(args: argparse.Namespace) -> int:
    path = migrate_legacy_archive(
        args.legacy_archive,
        args.out,
        title=args.title,
        force=args.force,
    )
    sys.stdout.write(f"Migrated {args.legacy_archive} -> {args.out}\n")
    _write_path(path)
    return 0


def _cmd_pack(args: argparse.Namespace) -> int:
    path = pack_project(
        args.workspace,
        out=args.out,
        mode=args.mode,
        overwrite=args.overwrite,
    )
    sys.stdout.write(f"Packed {args.workspace}\n")
    _write_path(path)
    return 0


def _cmd_unpack(args: argparse.Namespace) -> int:
    path = unpack_project(
        args.artifact,
        args.out,
        force=args.force,
        rename_title=args.rename,
    )
    sys.stdout.write(f"Unpacked {args.artifact}\n")
    _write_path(path)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    validate_artifact(args.path)
    sys.stdout.write(f"Valid {args.path}\n")
    return 0


def _cmd_artifacts_list(args: argparse.Namespace) -> int:
    entries = list_workspace_artifact_index(
        args.workspace,
        artifact_type=args.kind,
        namespace=args.namespace,
    )
    for entry in entries:
        namespace = entry.namespace or "-"
        sys.stdout.write(
            f"{entry.artifact_type}\t{namespace}\t{entry.artifact_id}\t{entry.manifest_path}\n"
        )
    if not entries:
        sys.stdout.write("No artifacts\n")
    return 0


def _cmd_artifacts_inspect(args: argparse.Namespace) -> int:
    artifact = load_workspace_artifact(
        args.workspace,
        args.artifact_id,
        artifact_type=args.kind,
        namespace=args.namespace,
    )
    sys.stdout.write(dump_json(artifact.to_dict(), indent=2, sort_keys=False) + "\n")
    return 0


def _cmd_artifacts_validate(args: argparse.Namespace) -> int:
    if args.artifact_id:
        artifact = validate_workspace_artifact(
            args.workspace,
            args.artifact_id,
            artifact_type=args.kind,
            namespace=args.namespace,
        )
        sys.stdout.write(f"Valid artifact {artifact.artifact_id}\n")
        return 0
    artifacts = validate_workspace_artifacts(
        args.workspace,
        artifact_type=args.kind,
        namespace=args.namespace,
    )
    sys.stdout.write(f"Valid artifacts {len(artifacts)}\n")
    return 0


def _cmd_artifacts_rebuild_index(args: argparse.Namespace) -> int:
    entries = rebuild_workspace_artifact_index(args.workspace)
    sys.stdout.write(f"Indexed artifacts {len(entries)}\n")
    return 0


def _cmd_import_dlc_csv(args: argparse.Namespace) -> int:
    path = import_dlc_csv_workspace(
        args.csv,
        args.video,
        args.out,
        skeleton_name=args.skeleton_name,
        likelihood_threshold=args.threshold,
        progress_callback=_emit_progress,
    )
    sys.stdout.write(f"Imported DLC CSV into {args.out}\n")
    _write_path(path)
    return 0


def _cmd_import_vicon(args: argparse.Namespace) -> int:
    if args.csv:
        path = import_vicon_csv_workspace(
            args.csv,
            args.out,
            progress_callback=_emit_progress,
        )
        sys.stdout.write(f"Imported Vicon CSV into {args.out}\n")
        _write_path(path)
        return 0

    if args.c3d:
        path = import_vicon_c3d_workspace(
            args.c3d,
            args.out,
            progress_callback=_emit_progress,
        )
        sys.stdout.write(f"Imported Vicon C3D into {args.out}\n")
        _write_path(path)
        return 0

    path = import_vicon_workspace(
        args.recording,
        args.out,
        progress_callback=_emit_progress,
    )
    sys.stdout.write(f"Imported Vicon recording into {args.out}\n")
    _write_path(path)
    return 0


def _cmd_import_dlc_h5(args: argparse.Namespace) -> int:
    path = import_dlc_h5_workspace(
        args.h5,
        args.video,
        args.out,
        skeleton_name=args.skeleton_name,
        likelihood_threshold=args.threshold,
        progress_callback=_emit_progress,
    )
    sys.stdout.write(f"Imported DLC H5 into {args.out}\n")
    _write_path(path)
    return 0


def _cmd_import_dlc_project(args: argparse.Namespace) -> int:
    path = import_dlc_project_workspace(
        args.project,
        args.out,
        likelihood_threshold=args.threshold,
        progress_callback=_emit_progress,
    )
    sys.stdout.write(f"Imported DLC project into {args.out}\n")
    _write_path(path)
    return 0


def _cmd_import_sleap(args: argparse.Namespace) -> int:
    if args.h5:
        if not args.video:
            raise ValueError("--video is required when importing SLEAP --h5 inputs")
        path = import_sleap_h5_workspace(
            args.h5,
            args.video,
            args.out,
            skeleton_name=args.skeleton_name,
            likelihood_threshold=args.threshold,
            progress_callback=_emit_progress,
        )
        sys.stdout.write(f"Imported SLEAP H5 into {args.out}\n")
        _write_path(path)
        return 0

    path = import_sleap_package_workspace(
        args.slp,
        args.out,
        fps=args.fps,
        encode_videos=args.encode_videos,
        progress_callback=_emit_progress,
    )
    sys.stdout.write(f"Imported SLEAP package into {args.out}\n")
    _write_path(path)
    return 0


def _cmd_import_mmpose(args: argparse.Namespace) -> int:
    path = import_mmpose_topdown_json_workspace(
        args.json,
        args.video,
        args.out,
        skeleton_name=args.skeleton_name,
        instance_index=args.instance_index,
        likelihood_threshold=args.threshold,
        progress_callback=_emit_progress,
    )
    sys.stdout.write(f"Imported MMPose JSON into {args.out}\n")
    _write_path(path)
    return 0


def _cmd_import_mediapipe(args: argparse.Namespace) -> int:
    path = import_mediapipe_pose_landmarks_json_workspace(
        args.json,
        args.video,
        args.out,
        skeleton_name=args.skeleton_name,
        likelihood_threshold=args.threshold,
        progress_callback=_emit_progress,
    )
    sys.stdout.write(f"Imported MediaPipe JSON into {args.out}\n")
    _write_path(path)
    return 0


def _cmd_import_openpose(args: argparse.Namespace) -> int:
    path = import_openpose_json_workspace(
        args.json,
        args.video,
        args.out,
        skeleton_name=args.skeleton_name,
        likelihood_threshold=args.threshold,
        progress_callback=_emit_progress,
    )
    sys.stdout.write(f"Imported OpenPose JSON into {args.out}\n")
    _write_path(path)
    return 0


def _cmd_import_detectron2(args: argparse.Namespace) -> int:
    path = import_detectron2_coco_workspace(
        args.predictions,
        args.dataset_json,
        args.image_root,
        args.out,
        category_id=args.category_id,
        skeleton_name=args.skeleton_name,
        likelihood_threshold=args.threshold,
        progress_callback=_emit_progress,
    )
    sys.stdout.write(f"Imported Detectron2 COCO into {args.out}\n")
    _write_path(path)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


__all__ = ["main"]
