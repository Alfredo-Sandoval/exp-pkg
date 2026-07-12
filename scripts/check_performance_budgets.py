"""Run deterministic cold-start and shallow-project performance gates."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from xpkg.inspection import inspect_path
from xpkg.services import ProjectService

REPO_ROOT = Path(__file__).resolve().parents[1]
BUDGET_PATH = REPO_ROOT / "performance-budgets.json"


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """One declared performance budget."""

    budget_ms: float
    samples: int


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """One measured benchmark verdict."""

    name: str
    measured_ms: float
    budget_ms: float
    samples: int

    @property
    def passed(self) -> bool:
        return self.measured_ms <= self.budget_ms

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "measured_ms": round(self.measured_ms, 3),
            "budget_ms": self.budget_ms,
            "samples": self.samples,
            "passed": self.passed,
        }


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number, got {value!r}.")
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than zero, got {result}.")
    return result


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}.")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value}.")
    return value


def _load_specs() -> dict[str, BenchmarkSpec]:
    payload: object = json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("performance-budgets.json must contain a JSON object.")
    if set(payload) != {"schema_version", "benchmarks"}:
        raise ValueError("performance-budgets.json must contain schema_version and benchmarks.")
    if payload["schema_version"] != 1:
        raise ValueError(f"Unsupported performance budget schema: {payload['schema_version']!r}.")
    raw_specs = payload["benchmarks"]
    if not isinstance(raw_specs, Mapping):
        raise TypeError("performance benchmark definitions must be a JSON object.")

    specs: dict[str, BenchmarkSpec] = {}
    for raw_name, raw_spec in raw_specs.items():
        if not isinstance(raw_name, str) or not isinstance(raw_spec, Mapping):
            raise TypeError("performance benchmark names and definitions must be objects.")
        if set(raw_spec) != {"budget_ms", "samples"}:
            raise ValueError(f"Benchmark {raw_name!r} must define budget_ms and samples.")
        specs[raw_name] = BenchmarkSpec(
            budget_ms=_positive_float(raw_spec["budget_ms"], name=f"{raw_name} budget_ms"),
            samples=_positive_int(raw_spec["samples"], name=f"{raw_name} samples"),
        )
    return specs


def _median_runtime_ms(operation: Callable[[], object], samples: int) -> float:
    durations: list[float] = []
    for _ in range(samples):
        started = time.perf_counter()
        operation()
        durations.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(durations)


def _package_cold_import(samples: int) -> float:
    return _median_runtime_ms(
        lambda: subprocess.run([sys.executable, "-c", "import xpkg"], check=True),
        samples,
    )


def _cli_help_cold_start(samples: int) -> float:
    return _median_runtime_ms(
        lambda: subprocess.run(
            [sys.executable, "-m", "xpkg", "--help"],
            check=True,
            stdout=subprocess.DEVNULL,
        ),
        samples,
    )


def _project_create(samples: int) -> float:
    with tempfile.TemporaryDirectory() as raw_root:
        root = Path(raw_root)
        index = 0

        def create_project() -> object:
            nonlocal index
            index += 1
            title = f"Benchmark Project {index}"
            return ProjectService.create(root / title, title=title)

        return _median_runtime_ms(create_project, samples)


def _with_project(operation: Callable[[ProjectService], object], samples: int) -> float:
    with tempfile.TemporaryDirectory() as raw_root:
        service = ProjectService.create(Path(raw_root) / "Benchmark", title="Benchmark")
        operation(service)
        return _median_runtime_ms(lambda: operation(service), samples)


def _project_describe(samples: int) -> float:
    return _with_project(lambda service: service.describe(), samples)


def _project_inspect_path(samples: int) -> float:
    return _with_project(lambda service: inspect_path(service.project_root), samples)


_BENCHMARKS: dict[str, Callable[[int], float]] = {
    "package_cold_import": _package_cold_import,
    "cli_help_cold_start": _cli_help_cold_start,
    "project_create": _project_create,
    "project_describe": _project_describe,
    "project_inspect_path": _project_inspect_path,
}


def _run_benchmarks(specs: Mapping[str, BenchmarkSpec]) -> list[BenchmarkResult]:
    if set(specs) != set(_BENCHMARKS):
        missing = sorted(set(_BENCHMARKS).difference(specs))
        unknown = sorted(set(specs).difference(_BENCHMARKS))
        raise ValueError(
            f"Performance benchmark registry mismatch; missing={missing}, unknown={unknown}."
        )
    return [
        BenchmarkResult(
            name=name,
            measured_ms=_BENCHMARKS[name](spec.samples),
            budget_ms=spec.budget_ms,
            samples=spec.samples,
        )
        for name, spec in specs.items()
    ]


def main() -> int:
    """Run every declared benchmark and return failure on a budget regression."""
    results = _run_benchmarks(_load_specs())
    payload = {
        "schema_version": 1,
        "benchmarks": [result.to_dict() for result in results],
        "passed": all(result.passed for result in results),
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
