"""Tests for the `status` command."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from kagglepipe.commands import status as status_cmd
from kagglepipe.config import Config, FeatureSection


def _cfg() -> Config:
    return Config(
        feature=FeatureSection(
            kernel_title_prefix="myproj",
        ),
    )


def test_status_filters_by_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_creds: Path,
    capsys: pytest.CaptureFixture[str]
) -> None:
    csv = (
        "ref,status,lastRunTime\n"
        "testuser/myproj-a,complete,2026-06-01\n"
        "testuser/otherproj-x,complete,2026-06-02\n"
        "testuser/myproj-b,running,2026-06-03\n"
    )
    monkeypatch.setattr(
        "kagglepipe.runner.run",
        lambda *a, **kw: subprocess.CompletedProcess(
            args=[], returncode=0, stdout=csv, stderr=""
        ),
    )
    rc = status_cmd.status(_cfg())
    out = capsys.readouterr().out
    assert rc == 0
    assert "myproj-a" in out
    assert "myproj-b" in out
    assert "otherproj-x" not in out


def test_status_all_includes_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_creds: Path,
    capsys: pytest.CaptureFixture[str]
) -> None:
    csv = (
        "ref,status,lastRunTime\n"
        "testuser/otherproj-x,complete,2026-06-02\n"
    )
    monkeypatch.setattr(
        "kagglepipe.runner.run",
        lambda *a, **kw: subprocess.CompletedProcess(
            args=[], returncode=0, stdout=csv, stderr=""
        ),
    )
    rc = status_cmd.status(_cfg(), all_kernels=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "otherproj-x" in out


def test_status_no_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_creds: Path,
    capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "kagglepipe.runner.run",
        lambda *a, **kw: subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ref,status,lastRunTime\n", stderr=""
        ),
    )
    rc = status_cmd.status(_cfg())
    assert rc == 0
    assert "No kernels matched" in capsys.readouterr().out
