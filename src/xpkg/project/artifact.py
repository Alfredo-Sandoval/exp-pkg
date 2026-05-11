"""Pack, unpack, and validate portable ``.expkg`` project artifacts."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from xpkg.project.layout import (
    EXPKG_SUFFIX,
    MEDIA_DIRNAME,
    PROJECT_DESCRIPTOR_FILENAME,
    STORE_DIRNAME,
    ProjectDescriptor,
    _candidate_project_root,
    _now_utc_iso,
    default_expkg_path,
    load_project_descriptor,
    project_artifacts_root,
    project_descriptor_path,
    project_media_root,
    project_store_root,
    resolve_project_root,
    write_project_descriptor,
)

from .._core.hashing import sha256_file
from .._core.json_utils import dump_json, parse_json_dict
from .._core.path_registry import resolve_path

EXPKG_MANIFEST_FILENAME = "EXPKG.json"
EXPKG_FORMAT = "xpkg-packed-project"
EXPKG_SCHEMA_VERSION = 1
PackMediaMode = Literal["full", "package", "manifest"]
PACK_MEDIA_MODES: tuple[str, ...] = ("full", "package", "manifest")

_COMPRESSED_MEDIA_SUFFIXES = {
    ".avi",
    ".bz2",
    ".doric",
    ".gz",
    ".h264",
    ".h265",
    ".h5",
    ".hdf5",
    ".jpeg",
    ".jpg",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".png",
    ".slp",
    ".webm",
    ".zip",
}

_VIDEO_MEDIA_SUFFIXES = {
    ".avi",
    ".h264",
    ".h265",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}


def _iter_project_files(
    project_root: Path,
    *,
    include_media: bool = True,
) -> list[Path]:
    files: list[Path] = []
    descriptor_path = project_descriptor_path(project_root)
    if descriptor_path.is_file():
        files.append(descriptor_path)
    root_dirs = [project_store_root(project_root)]
    if include_media:
        root_dirs.append(project_media_root(project_root))
    for root_dir in root_dirs:
        if not root_dir.exists():
            continue
        for candidate in sorted(root_dir.rglob("*")):
            if candidate.is_file():
                if candidate.is_symlink():
                    raise ValueError(
                        f"Packed project artifacts cannot include symlinks: {candidate}"
                    )
                files.append(candidate)
    return files


def _iter_project_media_files(project_root: Path) -> list[Path]:
    media_root = project_media_root(project_root)
    if not media_root.exists():
        return []
    files: list[Path] = []
    for candidate in sorted(media_root.rglob("*")):
        if candidate.is_file():
            if candidate.is_symlink():
                raise ValueError(f"Packed project media cannot include symlinks: {candidate}")
            files.append(candidate)
    return files


def _relative_member_path(path: Path, project_root: Path) -> str:
    return path.relative_to(project_root).as_posix()


def _member_role(member_path: str) -> str:
    if member_path == PROJECT_DESCRIPTOR_FILENAME:
        return "project_descriptor"
    if member_path.startswith(f"{MEDIA_DIRNAME}/"):
        return "media"
    if member_path.startswith(f"{STORE_DIRNAME}/"):
        return "store"
    return "project"


def _compression_name_for_member(member_path: str) -> str:
    suffix = PurePosixPath(member_path).suffix.lower()
    return "stored" if suffix in _COMPRESSED_MEDIA_SUFFIXES else "deflate"


def _zip_compression_for_member(member_path: str) -> int:
    return (
        zipfile.ZIP_STORED
        if _compression_name_for_member(member_path) == "stored"
        else zipfile.ZIP_DEFLATED
    )


def _file_manifest_entry(path: Path, *, project_root: Path) -> dict[str, Any]:
    member_path = _relative_member_path(path, project_root)
    return {
        "path": member_path,
        "role": _member_role(member_path),
        "size": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "compression": _compression_name_for_member(member_path),
    }


def _media_manifest_entry(
    path: Path,
    *,
    project_root: Path,
    included: bool,
) -> dict[str, Any]:
    entry = _file_manifest_entry(path, project_root=project_root)
    entry["included"] = bool(included)
    if not included:
        entry["compression"] = "none"
    return entry


def _expkg_manifest_payload(
    *,
    descriptor: ProjectDescriptor,
    media_mode: PackMediaMode,
    member_entries: list[dict[str, Any]],
    media_entries: list[dict[str, Any]],
    acquisition: dict[str, Any] | None = None,
    dataset_share: dict[str, Any] | None = None,
    pose_provenance: dict[str, Any] | None = None,
    datasheet: dict[str, Any] | None = None,
    model_card: dict[str, Any] | None = None,
) -> dict[str, Any]:
    included_media_entries = [entry for entry in media_entries if bool(entry.get("included"))]
    external_media_entries = [entry for entry in media_entries if not bool(entry.get("included"))]
    payload = {
        "format": EXPKG_FORMAT,
        "artifact_schema_version": EXPKG_SCHEMA_VERSION,
        "container": "zip",
        "created_at": _now_utc_iso(),
        "project": {
            "project_id": descriptor.project_id,
            "title": descriptor.title,
            "project_schema_version": descriptor.project_schema_version,
            "layout_version": descriptor.layout_version,
            "store_path": descriptor.store_path,
            "media_root": descriptor.media_root,
            "exports_root": descriptor.exports_root,
        },
        "media": {
            "mode": media_mode,
            "included_files": len(included_media_entries),
            "included_bytes": sum(int(entry["size"]) for entry in included_media_entries),
            "external_files": len(external_media_entries),
            "external_bytes": sum(int(entry["size"]) for entry in external_media_entries),
            "files": media_entries,
        },
        "members": member_entries,
    }
    if dataset_share is not None:
        payload["dataset_share"] = dataset_share
    if acquisition is not None:
        payload["acquisition"] = acquisition
    if pose_provenance is not None:
        payload["pose_provenance"] = pose_provenance
    if datasheet is not None:
        payload["datasheet"] = datasheet
    if model_card is not None:
        payload["model_card"] = model_card
    return payload


def _project_manifest_metadata(
    project_root: Path,
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    from xpkg.project.metadata import (
        load_project_acquisition_metadata,
        load_project_dataset_share_metadata,
        load_project_datasheet,
        load_project_model_card,
        load_project_pose_provenance,
    )

    acquisition = load_project_acquisition_metadata(project_root)
    dataset_share = load_project_dataset_share_metadata(project_root)
    pose_provenance = load_project_pose_provenance(project_root)
    datasheet = load_project_datasheet(project_root)
    model_card = load_project_model_card(project_root)
    return (
        None if acquisition is None else acquisition.to_dict(),
        None if dataset_share is None else dataset_share.to_dict(),
        None if pose_provenance is None else pose_provenance.to_dict(),
        None if datasheet is None else datasheet.to_dict(),
        None if model_card is None else model_card.to_dict(),
    )


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _validate_vicon_bundle_paths(project_root: Path, recording: Any) -> None:
    store_root = project_store_root(project_root).resolve()
    managed_paths = {
        "recording": recording.path,
        "xcp": recording.xcp_path,
        "vsk": recording.vsk_path,
    }
    for label, raw_path in managed_paths.items():
        if raw_path is None:
            continue
        resolved = resolve_path(raw_path)
        if not resolved.is_file():
            raise FileNotFoundError(f"Project Vicon {label} file missing: {resolved}")
        if not _is_within(resolved, store_root):
            raise ValueError(
                "Project Vicon bundle files must live under the project store. "
                f"Found unmanaged {label} path: {resolved}"
            )


def _project_media_violations(project_root: Path) -> list[str]:
    from xpkg.model import Labels
    from xpkg.project.state import project_state_kind
    from xpkg.project.store import current_project_state_path

    state_path = current_project_state_path(project_root)
    if not state_path.exists():
        return []
    if state_path.suffix.lower() == ".json" and project_state_kind(state_path) == "vicon":
        return []
    labels = Labels.load_file(project_root.as_posix())
    media_root = project_media_root(project_root).resolve()
    violations: list[str] = []

    for video in labels.videos:
        filename = getattr(video, "filename", None)
        if filename:
            resolved = resolve_path(filename)
            if not _is_within(resolved, media_root):
                violations.append(str(filename))
        for frame_path in getattr(video, "image_filenames", []) or []:
            resolved = resolve_path(frame_path)
            if not _is_within(resolved, media_root):
                violations.append(str(frame_path))

    return list(dict.fromkeys(violations))


def _normalize_pack_media_mode(media: str | None) -> PackMediaMode:
    selected = "full" if media is None else str(media).strip().lower()
    if selected == "full":
        return "full"
    if selected == "package":
        return "package"
    if selected == "manifest":
        return "manifest"
    allowed = ", ".join(PACK_MEDIA_MODES)
    raise ValueError(f"Unsupported media mode {media!r}; expected one of: {allowed}")


def _include_media_in_pack(path: Path, *, media_mode: PackMediaMode) -> bool:
    if media_mode == "full":
        return True
    if media_mode == "manifest":
        return False
    return path.suffix.lower() not in _VIDEO_MEDIA_SUFFIXES


def pack_project(
    project: str | Path,
    *,
    out: str | Path | None = None,
    media: str | None = None,
    overwrite: bool = False,
) -> Path:
    root = resolve_project_root(project)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg project: {project}")
    media_mode = _normalize_pack_media_mode(media)
    validate_project(root)

    descriptor = load_project_descriptor(root)
    violations = _project_media_violations(root)
    if violations:
        joined = ", ".join(violations[:5])
        if len(violations) > 5:
            joined += ", ..."
        raise ValueError(
            "Portable pack requires all referenced media to live under Media/. "
            f"Found external or unmanaged references: {joined}"
        )

    out_path = resolve_path(out) if out is not None else default_expkg_path(root)
    if out_path.suffix.lower() != EXPKG_SUFFIX:
        raise ValueError(f"Packed project artifacts must use {EXPKG_SUFFIX}: {out_path}")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output artifact already exists: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    project_media_files = _iter_project_media_files(root)
    included_media_paths = {
        source_path
        for source_path in project_media_files
        if _include_media_in_pack(source_path, media_mode=media_mode)
    }
    members = [
        *_iter_project_files(root, include_media=False),
        *[path for path in project_media_files if path in included_media_paths],
    ]
    member_entries = [
        _file_manifest_entry(source_path, project_root=root)
        for source_path in members
    ]
    media_entries = [
        _media_manifest_entry(
            source_path,
            project_root=root,
            included=source_path in included_media_paths,
        )
        for source_path in project_media_files
    ]
    (
        acquisition,
        dataset_share,
        pose_provenance,
        datasheet,
        model_card,
    ) = _project_manifest_metadata(root)
    manifest = _expkg_manifest_payload(
        descriptor=descriptor,
        media_mode=media_mode,
        member_entries=member_entries,
        media_entries=media_entries,
        pose_provenance=pose_provenance,
        acquisition=acquisition,
        dataset_share=dataset_share,
        datasheet=datasheet,
        model_card=model_card,
    )
    with tempfile.NamedTemporaryFile(
        prefix=f".{out_path.stem}_",
        suffix=".tmp",
        dir=str(out_path.parent),
        delete=False,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        with zipfile.ZipFile(tmp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                EXPKG_MANIFEST_FILENAME,
                dump_json(manifest, indent=2, sort_keys=False) + "\n",
                compress_type=zipfile.ZIP_DEFLATED,
            )
            for source_path in members:
                relative_name = source_path.relative_to(root).as_posix()
                archive.write(
                    source_path,
                    arcname=relative_name,
                    compress_type=_zip_compression_for_member(relative_name),
                )
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return out_path


def _validated_zip_member(name: str) -> PurePosixPath:
    member = PurePosixPath(name)
    if member.as_posix() in {"", "."}:
        raise ValueError("Packed artifact contains an empty member path")
    if member.is_absolute():
        raise ValueError(f"Packed artifact contains an absolute path: {name!r}")
    if ".." in member.parts:
        raise ValueError(f"Packed artifact contains a parent traversal path: {name!r}")
    return member


def _zip_info_is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0o170000
    return mode == 0o120000


def _validated_zip_infos(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        member = _validated_zip_member(info.filename)
        if info.is_dir():
            raise ValueError(f"Packed artifact contains a directory entry: {info.filename!r}")
        if _zip_info_is_symlink(info):
            raise ValueError(f"Packed artifact contains a symlink entry: {info.filename!r}")
        name = member.as_posix()
        if name in infos:
            raise ValueError(f"Packed artifact contains a duplicate member: {name!r}")
        infos[name] = info
    return infos


def _load_expkg_manifest(archive: zipfile.ZipFile) -> dict[str, Any]:
    try:
        raw = archive.read(EXPKG_MANIFEST_FILENAME).decode("utf-8")
    except KeyError as exc:
        raise FileNotFoundError(
            f"Packed project artifact is missing {EXPKG_MANIFEST_FILENAME}"
        ) from exc
    try:
        return parse_json_dict(raw)
    except TypeError as exc:
        raise TypeError(f"Packed {EXPKG_MANIFEST_FILENAME} must contain a JSON object") from exc


def _zip_member_sha256(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    digest = hashlib.sha256()
    with archive.open(info, mode="r") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _actual_zip_compression(info: zipfile.ZipInfo) -> str:
    if info.compress_type == zipfile.ZIP_STORED:
        return "stored"
    if info.compress_type == zipfile.ZIP_DEFLATED:
        return "deflate"
    return f"zip:{info.compress_type}"


def _member_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    raw_entries = manifest.get("members")
    if not isinstance(raw_entries, list):
        raise TypeError(f"Packed {EXPKG_MANIFEST_FILENAME} members must be a list")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise TypeError(f"Packed {EXPKG_MANIFEST_FILENAME} member entries must be objects")
        path = raw_entry.get("path")
        if not isinstance(path, str):
            raise TypeError("Packed member entry is missing a string path")
        _validated_zip_member(path)
        if path in seen:
            raise ValueError(f"Packed manifest contains a duplicate member: {path!r}")
        seen.add(path)
        entries.append(raw_entry)
    return entries


def _bool_manifest_field(entry: dict[str, Any], field: str, *, path: str) -> bool:
    value = entry.get(field)
    if not isinstance(value, bool):
        raise TypeError(f"Packed manifest entry {path!r} has non-boolean {field}")
    return value


def _media_entries(manifest: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    media = manifest.get("media")
    if not isinstance(media, dict):
        raise TypeError(f"Packed {EXPKG_MANIFEST_FILENAME} media field must be an object")
    mode = media.get("mode")
    if mode not in PACK_MEDIA_MODES:
        allowed = ", ".join(PACK_MEDIA_MODES)
        raise ValueError(f"Packed media.mode must be one of: {allowed}")
    raw_entries = media.get("files")
    if not isinstance(raw_entries, list):
        raise TypeError(f"Packed {EXPKG_MANIFEST_FILENAME} media.files must be a list")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise TypeError("Packed media entries must be objects")
        path = raw_entry.get("path")
        if not isinstance(path, str):
            raise TypeError("Packed media entry is missing a string path")
        _validated_zip_member(path)
        if _member_role(path) != "media":
            raise ValueError(f"Packed media entry is outside {MEDIA_DIRNAME}/: {path!r}")
        if path in seen:
            raise ValueError(f"Packed media manifest contains a duplicate path: {path!r}")
        _int_manifest_field(raw_entry, "size", path=path)
        _str_manifest_field(raw_entry, "sha256", path=path)
        compression = _str_manifest_field(raw_entry, "compression", path=path)
        included = _bool_manifest_field(raw_entry, "included", path=path)
        if included:
            expected_compression = _compression_name_for_member(path)
            if compression != expected_compression:
                raise ValueError(
                    f"Packed media compression policy mismatch for {path!r}: "
                    f"manifest={compression}, expected={expected_compression}"
                )
            if mode == "manifest":
                raise ValueError("Manifest media mode cannot include media members")
            if mode == "package" and PurePosixPath(path).suffix.lower() in _VIDEO_MEDIA_SUFFIXES:
                raise ValueError(
                    f"Package media mode cannot include video container media: {path!r}"
                )
        elif compression != "none":
            raise ValueError(
                f"Packed external media compression must be 'none' for {path!r}"
            )
        if mode == "full" and not included:
            raise ValueError("Full media mode cannot declare external media")
        seen.add(path)
        entries.append(raw_entry)
    return media, entries


def _validate_expkg_metadata(manifest: dict[str, Any]) -> None:
    from xpkg.model import (
        AcquisitionMetadata,
        DatasetDatasheet,
        DatasetShareMetadata,
        ModelCard,
        PoseModelProvenance,
    )

    dataset_share = manifest.get("dataset_share")
    if dataset_share is not None:
        if not isinstance(dataset_share, dict):
            raise TypeError(f"Packed {EXPKG_MANIFEST_FILENAME} dataset_share must be an object")
        DatasetShareMetadata.from_dict(dataset_share)
    acquisition = manifest.get("acquisition")
    if acquisition is not None:
        if not isinstance(acquisition, dict):
            raise TypeError(f"Packed {EXPKG_MANIFEST_FILENAME} acquisition must be an object")
        AcquisitionMetadata.from_dict(acquisition)
    pose_provenance = manifest.get("pose_provenance")
    if pose_provenance is not None:
        if not isinstance(pose_provenance, dict):
            raise TypeError(f"Packed {EXPKG_MANIFEST_FILENAME} pose_provenance must be an object")
        PoseModelProvenance.from_dict(pose_provenance)
    datasheet = manifest.get("datasheet")
    if datasheet is not None:
        if not isinstance(datasheet, dict):
            raise TypeError(f"Packed {EXPKG_MANIFEST_FILENAME} datasheet must be an object")
        DatasetDatasheet.from_dict(datasheet)
    model_card = manifest.get("model_card")
    if model_card is not None:
        if not isinstance(model_card, dict):
            raise TypeError(f"Packed {EXPKG_MANIFEST_FILENAME} model_card must be an object")
        ModelCard.from_dict(model_card)


def _int_manifest_field(entry: dict[str, Any], field: str, *, path: str) -> int:
    value = entry.get(field)
    if not isinstance(value, int):
        raise TypeError(f"Packed manifest entry {path!r} has non-integer {field}")
    return value


def _str_manifest_field(entry: dict[str, Any], field: str, *, path: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str):
        raise TypeError(f"Packed manifest entry {path!r} has non-string {field}")
    return value


def _validate_member_payloads(
    archive: zipfile.ZipFile,
    infos: dict[str, zipfile.ZipInfo],
    entries: list[dict[str, Any]],
) -> None:
    for entry in entries:
        path = _str_manifest_field(entry, "path", path="<unknown>")
        info = infos[path]
        expected_size = _int_manifest_field(entry, "size", path=path)
        if info.file_size != expected_size:
            raise ValueError(
                f"Packed member size mismatch for {path!r}: "
                f"manifest={expected_size}, archive={info.file_size}"
            )
        expected_sha256 = _str_manifest_field(entry, "sha256", path=path)
        actual_sha256 = _zip_member_sha256(archive, info)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"Packed member checksum mismatch for {path!r}")
        expected_compression = _compression_name_for_member(path)
        manifest_compression = _str_manifest_field(entry, "compression", path=path)
        actual_compression = _actual_zip_compression(info)
        if manifest_compression != expected_compression:
            raise ValueError(
                f"Packed member compression policy mismatch for {path!r}: "
                f"manifest={manifest_compression}, expected={expected_compression}"
            )
        if actual_compression != manifest_compression:
            raise ValueError(
                f"Packed member compression mismatch for {path!r}: "
                f"manifest={manifest_compression}, archive={actual_compression}"
            )


def _validate_expkg_artifact(artifact_path: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(artifact_path, mode="r") as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"Packed artifact has a corrupt zip member: {bad_member!r}")
            infos = _validated_zip_infos(archive)
            manifest = _load_expkg_manifest(archive)
            if manifest.get("format") != EXPKG_FORMAT:
                raise ValueError(f"Packed artifact format must be {EXPKG_FORMAT!r}")
            if manifest.get("artifact_schema_version") != EXPKG_SCHEMA_VERSION:
                raise ValueError(
                    "Packed artifact schema version must be "
                    f"{EXPKG_SCHEMA_VERSION}, got {manifest.get('artifact_schema_version')!r}"
                )
            if manifest.get("container") != "zip":
                raise ValueError("Packed artifact container must be 'zip'")
            _validate_expkg_metadata(manifest)

            entries = _member_entries(manifest)
            member_paths = {str(entry["path"]) for entry in entries}
            if PROJECT_DESCRIPTOR_FILENAME not in member_paths:
                raise FileNotFoundError("Packed project artifact is missing PROJECT.json")
            expected_names = {EXPKG_MANIFEST_FILENAME, *member_paths}
            actual_names = set(infos)
            if actual_names != expected_names:
                missing = sorted(expected_names.difference(actual_names))
                extra = sorted(actual_names.difference(expected_names))
                parts: list[str] = []
                if missing:
                    parts.append(f"missing={missing}")
                if extra:
                    parts.append(f"extra={extra}")
                raise ValueError(
                    "Packed artifact members do not match manifest: " + ", ".join(parts)
                )

            media, media_files = _media_entries(manifest)
            included_media_files = [
                entry
                for entry in media_files
                if _bool_manifest_field(entry, "included", path=str(entry.get("path", "")))
            ]
            external_media_files = [
                entry
                for entry in media_files
                if not _bool_manifest_field(entry, "included", path=str(entry.get("path", "")))
            ]
            if media.get("included_files") != len(included_media_files):
                raise ValueError("Packed media included_files count does not match media.files")
            included_media_bytes = sum(
                _int_manifest_field(entry, "size", path=str(entry["path"]))
                for entry in included_media_files
            )
            if media.get("included_bytes") != included_media_bytes:
                raise ValueError("Packed media included_bytes does not match media.files")
            if media.get("external_files") != len(external_media_files):
                raise ValueError("Packed media external_files count does not match media.files")
            external_media_bytes = sum(
                _int_manifest_field(entry, "size", path=str(entry["path"]))
                for entry in external_media_files
            )
            if media.get("external_bytes") != external_media_bytes:
                raise ValueError("Packed media external_bytes does not match media.files")
            included_media_paths = {
                _str_manifest_field(entry, "path", path="<unknown>") for entry in media_files
                if _bool_manifest_field(entry, "included", path=str(entry.get("path", "")))
            }
            archived_media = {
                path for path in member_paths if path.startswith(f"{MEDIA_DIRNAME}/")
            }
            if archived_media != included_media_paths:
                raise ValueError("Packed archived media does not match media manifest")

            entry_by_path = {str(entry["path"]): entry for entry in entries}
            for media_entry in media_files:
                media_path = _str_manifest_field(media_entry, "path", path="<unknown>")
                included = _bool_manifest_field(media_entry, "included", path=media_path)
                member_entry = entry_by_path.get(media_path)
                if not included:
                    if member_entry is not None:
                        raise ValueError(
                            "Packed external media entry is present in artifact "
                            f"members: {media_path!r}"
                        )
                    continue
                if member_entry is None:
                    raise ValueError(
                        f"Packed media entry is absent from artifact members: {media_path!r}"
                    )
                if media_entry.get("sha256") != member_entry.get("sha256"):
                    raise ValueError(
                        f"Packed media checksum mismatch between manifests: {media_path!r}"
                    )
                if media_entry.get("size") != member_entry.get("size"):
                    raise ValueError(
                        f"Packed media size mismatch between manifests: {media_path!r}"
                    )

            _validate_member_payloads(archive, infos, entries)

            descriptor_raw = archive.read(PROJECT_DESCRIPTOR_FILENAME).decode("utf-8")
            try:
                descriptor_data = parse_json_dict(descriptor_raw)
            except TypeError as exc:
                raise TypeError("Packed PROJECT.json must contain a JSON object") from exc
            ProjectDescriptor.from_dict(descriptor_data)
            return manifest
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Packed artifact is not a valid zip container: {artifact_path}") from exc


def unpack_project(
    artifact: str | Path,
    out: str | Path,
    *,
    force: bool = False,
    rename_title: str | None = None,
) -> Path:
    artifact_path = resolve_path(artifact)
    if artifact_path.suffix.lower() != EXPKG_SUFFIX:
        raise ValueError(f"Packed project artifacts must use {EXPKG_SUFFIX}: {artifact_path}")
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Packed project artifact not found: {artifact_path}")
    manifest = _validate_expkg_artifact(artifact_path)

    out_root = _candidate_project_root(out)
    if out_root.exists() and not out_root.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {out_root}")
    if out_root.exists():
        entries = list(out_root.iterdir())
        if entries and not force:
            raise FileExistsError(f"Output directory is not empty: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    member_paths = {str(entry["path"]) for entry in _member_entries(manifest)}
    with zipfile.ZipFile(artifact_path, mode="r") as archive:
        for info in archive.infolist():
            if info.filename == EXPKG_MANIFEST_FILENAME:
                continue
            if info.filename not in member_paths:
                continue
            member = _validated_zip_member(info.filename)
            target = out_root.joinpath(*member.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, mode="r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)

    descriptor = load_project_descriptor(out_root)
    (out_root / descriptor.store_path).mkdir(parents=True, exist_ok=True)
    (out_root / descriptor.media_root).mkdir(parents=True, exist_ok=True)
    (out_root / descriptor.exports_root).mkdir(parents=True, exist_ok=True)
    if rename_title is not None and rename_title.strip():
        descriptor.title = rename_title.strip()
        descriptor.updated_at = _now_utc_iso()
        write_project_descriptor(out_root, descriptor)
    if _manifest_has_external_media(manifest):
        _validate_project_layout(out_root)
    else:
        validate_project(out_root)
    return out_root


def _manifest_has_external_media(manifest: dict[str, Any]) -> bool:
    _, media_files = _media_entries(manifest)
    return any(
        not _bool_manifest_field(entry, "included", path=str(entry.get("path", "")))
        for entry in media_files
    )


def _validate_project_layout(project: str | Path) -> tuple[Path, ProjectDescriptor]:
    root = resolve_project_root(project)
    if root is None:
        raise FileNotFoundError(f"Not an xpkg project: {project}")
    descriptor = load_project_descriptor(root)
    descriptor.validate()

    store_root = root / descriptor.store_path
    if not store_root.is_dir():
        raise FileNotFoundError(f"Project store directory missing: {store_root}")

    media_root = root / descriptor.media_root
    exports_root = root / descriptor.exports_root
    if media_root.exists() and not media_root.is_dir():
        raise ValueError(f"Project media root is not a directory: {media_root}")
    if exports_root.exists() and not exports_root.is_dir():
        raise ValueError(f"Project exports root is not a directory: {exports_root}")
    artifacts_root = project_artifacts_root(root)
    if artifacts_root.exists() and not artifacts_root.is_dir():
        raise ValueError(f"Project artifacts root is not a directory: {artifacts_root}")
    return root, descriptor


def validate_project(project: str | Path) -> None:
    root, _descriptor = _validate_project_layout(project)

    from xpkg.model import Labels
    from xpkg.project.artifacts import validate_project_artifacts
    from xpkg.project.state import project_state_kind
    from xpkg.project.store import (
        current_project_state_path,
        ensure_current_project_state_cache,
        load_project_vicon_recording,
    )

    validate_project_artifacts(root)

    state_path = ensure_current_project_state_cache(root)
    state_path = state_path if state_path is not None else current_project_state_path(root)
    if not state_path.exists():
        return
    if state_path.suffix.lower() == ".json":
        if project_state_kind(state_path) == "vicon":
            recording = load_project_vicon_recording(root)
            _validate_vicon_bundle_paths(root, recording)
            return
    labels = Labels.load_file(root.as_posix())
    labels.validate()


def validate_expkg(artifact: str | Path) -> None:
    artifact_path = resolve_path(artifact)
    if artifact_path.suffix.lower() != EXPKG_SUFFIX:
        raise ValueError(f"Packed project artifacts must use {EXPKG_SUFFIX}: {artifact_path}")
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Packed project artifact not found: {artifact_path}")
    _validate_expkg_artifact(artifact_path)


def validate_artifact(path: str | Path) -> None:
    resolved = resolve_path(path)
    if resolved.is_dir() or resolved.name == PROJECT_DESCRIPTOR_FILENAME:
        validate_project(resolved)
        return
    if resolved.suffix.lower() == EXPKG_SUFFIX:
        validate_expkg(resolved)
        return
    raise ValueError(f"Unsupported artifact path: {resolved}")


__all__ = [
    "EXPKG_MANIFEST_FILENAME",
    "pack_project",
    "unpack_project",
    "validate_artifact",
    "validate_expkg",
    "validate_project",
]
