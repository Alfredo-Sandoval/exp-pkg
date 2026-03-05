"""Filesystem/network side-effectful path operations."""

from __future__ import annotations

import re
from pathlib import Path

from posetta.core.logging_utils import get_logger
from posetta.core.path_registry import WeightSpec, resolve_optional_weight_path, resolve_weight_path

logger = get_logger(__name__)


def resolve_weight_with_download(
    spec: WeightSpec,
    model_key: str,
    *,
    override: str | Path | None = None,
    download_hint: str | None = None,
) -> Path:
    """Resolve a weight file, downloading it when missing."""
    if override is not None:
        return resolve_weight_path(spec, override=override)

    path = resolve_optional_weight_path(spec)
    if path is not None:
        return path

    from posetta.core.download import download_model

    if not download_model(model_key):
        hint = f" {download_hint}" if download_hint else ""
        raise FileNotFoundError(
            f"{spec.name} not found. Download failed for model '{model_key}'.{hint}"
        )

    return resolve_weight_path(spec)


def ensure_dir(path: str | Path) -> Path:
    """Ensure the directory exists (like mkdir -p) and return the resolved Path."""
    target = path if path else "."
    directory = Path(target)
    directory.mkdir(parents=True, exist_ok=True)
    return directory.resolve()


def _read_best_epoch(run_path: Path) -> int | None:
    from posetta.core.json_utils import load_json_dict

    summary_file = run_path / "artifacts" / "train_summary.json"
    if not summary_file.exists():
        summary_file = run_path / "train_summary.json"
    if not summary_file.exists():
        return None

    data = load_json_dict(summary_file)
    best_epoch = data.get("best_epoch")
    if best_epoch is None:
        logger.debug("No best_epoch found in %s", summary_file)
        return None

    return int(best_epoch)


def _validate_best_checkpoint(
    checkpoint: Path,
    run_dir: Path,
    all_checkpoints: list[Path],
) -> tuple[bool, str | None]:
    """Validate that the selected checkpoint is actually the best."""
    summary_exists = (run_dir / "train_summary.json").exists() or (
        run_dir / "artifacts" / "train_summary.json"
    ).exists()

    if not summary_exists and len(all_checkpoints) > 1:
        return (
            False,
            "Multiple checkpoints found but no train_summary.json; "
            "selected checkpoint based on modification time",
        )

    best_epoch = _read_best_epoch(run_dir)
    if best_epoch is not None:
        match = re.match(r"^(\d+)-", checkpoint.name)
        if match and int(match.group(1)) != best_epoch:
            return (
                False,
                f"Checkpoint epoch {match.group(1)} doesn't match best_epoch {best_epoch} "
                "from train_summary.json",
            )

    return (True, None)


def find_best_checkpoint(run_dir: str | Path) -> Path | None:
    """Return best checkpoint from run_dir."""
    run_path = Path(run_dir)
    ckpt_dir = run_path / "checkpoints"
    if not ckpt_dir.is_dir():
        return None

    selected_checkpoint: Path | None = None
    best_named: list[Path] = []
    checkpoint_exts = (".ckpt", ".pth", ".pt", ".safetensors")

    for ext in checkpoint_exts:
        best_named_file = ckpt_dir / f"best{ext}"
        if best_named_file.is_file():
            selected_checkpoint = best_named_file
            break

    if selected_checkpoint is None:
        best_metric_candidates: list[tuple[Path, float]] = []
        for ext in checkpoint_exts:
            ext_regex = re.escape(ext)
            pattern = rf"^best-(?P<val>-?\d+(?:\.\d+)?){ext_regex}$"
            for ckpt in ckpt_dir.glob(f"best-*{ext}"):
                if not ckpt.is_file():
                    continue
                match = re.match(pattern, ckpt.name)
                if match is None:
                    continue
                best_metric_candidates.append((ckpt, float(match.group("val"))))
        if best_metric_candidates:
            selected_checkpoint = min(best_metric_candidates, key=lambda item: item[1])[0]

    if selected_checkpoint is None:
        for ext in checkpoint_exts:
            best_named.extend(
                [ckpt for ckpt in ckpt_dir.glob(f"*-best{ext}") if ckpt.is_file()]
            )
        if best_named:
            best_epoch = _read_best_epoch(run_path)
            if best_epoch is not None:
                for ckpt in best_named:
                    ext_regex = re.escape(ckpt.suffix.lower())
                    match = re.match(rf"^(\d+)-\d+-best{ext_regex}$", ckpt.name)
                    if match and int(match.group(1)) == best_epoch:
                        selected_checkpoint = ckpt
                        break
            if selected_checkpoint is None:
                best_named_metrics: list[tuple[Path, float]] = []
                for ckpt in best_named:
                    ext_regex = re.escape(ckpt.suffix.lower())
                    match = re.match(
                        r"^(?P<epoch>\d+)-(?P<step>\d+)-(?P<val>-?\d+(?:\.\d+)?)"
                        rf"-best{ext_regex}$",
                        ckpt.name,
                    )
                    if match is None:
                        continue
                    best_named_metrics.append((ckpt, float(match.group("val"))))
                if best_named_metrics:
                    selected_checkpoint = min(best_named_metrics, key=lambda item: item[1])[0]
                else:
                    logger.warning(
                        "No epoch match found for best checkpoint in %s; "
                        "selecting by modification time",
                        run_path,
                    )
                    selected_checkpoint = max(
                        best_named,
                        key=lambda item: (item.stat().st_mtime, item.name.lower()),
                    )

    if selected_checkpoint is None:
        for ext in checkpoint_exts:
            last_named_file = ckpt_dir / f"last{ext}"
            if last_named_file.is_file():
                selected_checkpoint = last_named_file
                break

    if selected_checkpoint:
        is_valid, warning_msg = _validate_best_checkpoint(
            selected_checkpoint,
            run_path,
            all_checkpoints=best_named,
        )
        if not is_valid and warning_msg:
            logger.warning(
                "Best checkpoint selection for %s may be incorrect: %s",
                run_path,
                warning_msg,
            )

    return selected_checkpoint


def resolve_model_checkpoint(source: str | Path) -> Path:
    """Resolve a model checkpoint from a file or directory."""
    path = Path(source)
    if path.is_file():
        if path.suffix.lower() in {".ckpt", ".pth", ".pt", ".safetensors", ".siesta"}:
            return path
        raise ValueError(f"Unsupported checkpoint extension: {path.suffix}")

    best_checkpoint = find_best_checkpoint(path)
    if best_checkpoint:
        return best_checkpoint

    bundle = path / "model.siesta"
    if bundle.is_file():
        return bundle

    pose_bundle = path / "models" / "pose" / "model.siesta"
    if pose_bundle.is_file():
        return pose_bundle

    raise FileNotFoundError(f"No valid checkpoint or model.siesta found in {path}")


__all__ = [
    "ensure_dir",
    "find_best_checkpoint",
    "resolve_model_checkpoint",
    "resolve_weight_with_download",
]
