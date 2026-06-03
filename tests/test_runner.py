"""Tests for the subprocess wrapper."""

from __future__ import annotations

import subprocess
import sys

import pytest

from kagglepipe import runner


def test_run_invokes_python_with_utf8_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """All kaggle CLI invocations must go through `python -X utf8 -m kaggle`."""
    captured: list[list[str]] = []

    def fake_subprocess_run(cmd, **kwargs):
        captured.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    runner.run(["kernels", "list"])
    assert captured, "subprocess.run was not called"
    cmd = captured[0]
    assert cmd[0] == sys.executable
    assert cmd[1:3] == ["-X", "utf8"]
    assert cmd[3] == "-m"
    assert cmd[4] == "kaggle"
    assert cmd[5:] == ["kernels", "list"]


def test_run_passes_utf8_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs: dict = {}

    def fake_subprocess_run(cmd, **kwargs):
        captured_kwargs.update(kwargs)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    runner.run(["datasets", "list"])
    assert captured_kwargs.get("encoding") == "utf-8"
    assert captured_kwargs.get("text") is True
    assert captured_kwargs.get("capture_output") is True


def test_run_check_raises_kaggle_error_on_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_subprocess_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=1, stdout="oops", stderr="bad"
        )

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    with pytest.raises(runner.KaggleError) as excinfo:
        runner.run(["kernels", "list"], check=True)
    assert excinfo.value.returncode == 1
    assert "bad" in str(excinfo.value)


def test_run_no_check_returns_result(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_subprocess_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=2, stdout="x", stderr="y"
        )

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    result = runner.run(["kernels", "list"])
    assert result.returncode == 2
    assert result.stdout == "x"
    assert result.stderr == "y"


def test_run_json_parses_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_subprocess_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout='{"foo": "bar"}', stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    out = runner.run_json(["datasets", "list"])
    assert out == {"foo": "bar"}


def test_run_json_raises_on_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_subprocess_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="not json", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    with pytest.raises(runner.KaggleError):
        runner.run_json(["datasets", "list"])
