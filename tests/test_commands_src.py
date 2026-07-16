"""Tests for the `src upload` command."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from kagglepipe.commands import src as src_cmd
from kagglepipe.config import Config, SourceSection


def _cfg() -> Config:
    return Config(
        source=SourceSection(
            include=["scripts"],
            exclude_dirs=[],
            exclude_exts=[],
            src_dataset_slug="{username}/proj-src",
        ),
    )


def test_upload_first_time_uses_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_creds: Path
) -> None:
    cfg = _cfg()
    root = tmp_path / "project"
    root.mkdir()
    (root / "scripts").mkdir()
    (root / "scripts" / "run.py").write_text("print('x')")

    csv_empty = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="ref,title\n", stderr=""
    )
    create_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="created", stderr=""
    )
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[:2] == ["datasets", "list"]:
            return csv_empty
        if args[:2] == ["datasets", "create"]:
            return create_result
        raise AssertionError(args)

    monkeypatch.setattr("kagglepipe.runner.run", fake_run)
    rc = src_cmd.upload(cfg, src_root=root)
    assert rc == 0
    # Should call datasets create (not version) since dataset doesn't exist
    assert any(args[:2] == ["datasets", "create"] for args in calls)
    assert not any(args[:2] == ["datasets", "version"] for args in calls)


def test_upload_existing_uses_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_creds: Path
) -> None:
    cfg = _cfg()
    root = tmp_path / "project"
    root.mkdir()
    (root / "scripts").mkdir()
    (root / "scripts" / "run.py").write_text("print('x')")

    csv_existing = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout='ref,title\ntestuser/proj-src,title\n',
        stderr="",
    )
    version_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="versioned", stderr=""
    )
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[:2] == ["datasets", "list"]:
            return csv_existing
        if args[:2] == ["datasets", "version"]:
            return version_result
        raise AssertionError(args)

    monkeypatch.setattr("kagglepipe.runner.run", fake_run)
    rc = src_cmd.upload(cfg, src_root=root)
    assert rc == 0
    assert any(args[:2] == ["datasets", "version"] for args in calls)


def test_upload_explicit_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_creds: Path
) -> None:
    cfg = _cfg()
    root = tmp_path / "project"
    root.mkdir()
    (root / "scripts").mkdir()
    (root / "scripts" / "run.py").write_text("print('x')")

    version_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="", stderr=""
    )
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[:2] == ["datasets", "version"]:
            return version_result
        raise AssertionError(args)

    monkeypatch.setattr("kagglepipe.runner.run", fake_run)
    rc = src_cmd.upload(cfg, src_root=root, version=5)
    assert rc == 0
    # The upload message should reference version 5.
    for call in calls:
        if call[:2] == ["datasets", "version"]:
            assert "v5" in " ".join(call)
            break


def test_upload_empty_tarball_returns_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_creds: Path
) -> None:
    cfg = _cfg()
    root = tmp_path / "empty"
    root.mkdir()
    rc = src_cmd.upload(cfg, src_root=root)
    assert rc == 1
