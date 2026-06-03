"""Tests for the `kernels` command group."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from kagglepipe.commands import kernels as kernels_cmd


def test_list_kernels_csv(monkeypatch: pytest.MonkeyPatch, fake_creds: Path,
                         capsys: pytest.CaptureFixture[str]) -> None:
    csv = "ref,status,title\ntestuser/a,complete,A\ntestuser/b,running,B\n"
    monkeypatch.setattr(
        "kagglepipe.runner.run",
        lambda *a, **kw: subprocess.CompletedProcess(
            args=[], returncode=0, stdout=csv, stderr=""
        ),
    )
    rc = kernels_cmd.list_kernels(csv_output=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "ref,status,title" in out
    assert "testuser/a" in out


def test_list_kernels_json(monkeypatch: pytest.MonkeyPatch, fake_creds: Path,
                            capsys: pytest.CaptureFixture[str]) -> None:
    import json
    csv = "ref,status,title\ntestuser/a,complete,A\n"
    monkeypatch.setattr(
        "kagglepipe.runner.run",
        lambda *a, **kw: subprocess.CompletedProcess(
            args=[], returncode=0, stdout=csv, stderr=""
        ),
    )
    rc = kernels_cmd.list_kernels(json_output=True)
    parsed = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert parsed[0]["ref"] == "testuser/a"


def test_kernel_status_returns_zero_on_complete(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "kagglepipe.runner.run",
        lambda *a, **kw: subprocess.CompletedProcess(
            args=[], returncode=0, stdout='"complete"', stderr=""
        ),
    )
    rc = kernels_cmd.kernel_status("u/k")
    assert rc == 0
    assert capsys.readouterr().out.strip() == "complete"


def test_kernel_output_invokes_kernels_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr("kagglepipe.runner.run", fake_run)
    target = tmp_path / "out"
    rc = kernels_cmd.kernel_output("u/some_kernel", path=target)
    assert rc == 0
    assert calls[0][:3] == ["kernels", "output", "u/some_kernel"]
    assert calls[0][3] == "-p"
    assert Path(calls[0][4]) == target


def test_kernel_logs_url(capsys: pytest.CaptureFixture[str]) -> None:
    rc = kernels_cmd.kernel_logs("user/slug")
    assert rc == 0
    assert "https://www.kaggle.com/user/slug/logs" in capsys.readouterr().out


def test_kernel_stop_invokes_kernels_stop(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr("kagglepipe.runner.run", fake_run)
    rc = kernels_cmd.kernel_stop("u/slug")
    assert rc == 0
    assert calls[0] == ["kernels", "stop", "-k", "u/slug"]
