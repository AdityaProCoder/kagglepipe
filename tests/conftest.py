"""Shared pytest fixtures."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def fake_creds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Write a fake kaggle.json and point KAGGLE_USERNAME/KEY at it."""
    creds = tmp_path / "kaggle.json"
    creds.write_text('{"username": "testuser", "key": "testkey123"}\n')
    monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
    monkeypatch.setenv("KAGGLE_KEY", "testkey123")
    # Also point the default-path loader at the fake file by monkeypatching
    # the home directory via Path.home.
    monkeypatch.setattr(
        "pathlib.Path.home", classmethod(lambda cls: tmp_path)
    )
    return creds


@pytest.fixture
def fake_run(monkeypatch: pytest.MonkeyPatch):
    """Patch `kagglepipe.runner.run` so tests never hit Kaggle."""

    def _install(
        *,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
        record: list[list[str]] | None = None,
        side_effect=None,
    ):
        calls: list[list[str]] = record if record is not None else []

        def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
            calls.append(args)
            if side_effect is not None:
                return side_effect(args, **kwargs)
            return subprocess.CompletedProcess(
                args=args, returncode=returncode, stdout=stdout, stderr=stderr
            )

        monkeypatch.setattr("kagglepipe.runner.run", fake_run)
        return calls

    return _install
