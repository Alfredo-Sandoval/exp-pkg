from __future__ import annotations

import json

import pytest
import typer

from xpkg.cli.shared import run_command


def _run_failing_command(exc: Exception, capsys: pytest.CaptureFixture[str]) -> tuple[int, dict]:
    def action() -> dict[str, object]:
        raise exc

    with pytest.raises(typer.Exit) as exc_info:
        run_command(json_output=True, action=action, human_output=lambda _payload: None)
    captured = capsys.readouterr()
    assert captured.out == ""
    return int(exc_info.value.exit_code or 0), json.loads(captured.err)


def test_run_command_wraps_success_payload_in_envelope(capsys: pytest.CaptureFixture[str]) -> None:
    run_command(
        json_output=True,
        action=lambda: {"value": 7},
        human_output=lambda _payload: None,
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"ok": True, "data": {"value": 7}}


def test_run_command_maps_missing_file_to_not_found(capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = _run_failing_command(FileNotFoundError("missing.json"), capsys)

    assert code == 3
    assert payload["ok"] is False
    assert payload["error"]["code"] == "not_found"


def test_run_command_maps_value_error_to_invalid_input(capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = _run_failing_command(ValueError("bad threshold"), capsys)

    assert code == 1
    assert payload["error"]["code"] == "invalid_input"
    assert payload["error"]["message"] == "bad threshold"


def test_run_command_maps_operational_failure_to_runtime_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, payload = _run_failing_command(RuntimeError("store lock contention"), capsys)

    assert code == 1
    assert payload["error"]["code"] == "runtime_error"


@pytest.mark.parametrize("exc", [KeyError("frames"), TypeError("bad arg"), AttributeError("oops")])
def test_run_command_maps_bug_shaped_errors_to_internal_error(
    exc: Exception,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, payload = _run_failing_command(exc, capsys)

    assert code == 1
    assert payload["error"]["code"] == "internal_error"
    assert payload["error"]["message"].startswith(exc.__class__.__name__ + ":")
    assert "bug in xpkg" in payload["error"]["hint"]
