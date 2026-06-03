"""Tests for the `config` command group."""

from __future__ import annotations

from pathlib import Path

import pytest

from kagglepipe.commands import config_cmd


def test_init_scaffolds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    rc = config_cmd.init(project_name="myproj")
    assert rc == 0
    target = tmp_path / "kaggle.toml"
    assert target.exists()
    text = target.read_text()
    assert "[project]" in text
    assert "myproj" in text


def test_init_refuses_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "kaggle.toml").write_text("existing")
    rc = config_cmd.init()
    assert rc == 1


def test_init_force_overwrites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "kaggle.toml").write_text("old")
    rc = config_cmd.init(force=True)
    assert rc == 0
    assert "[project]" in (tmp_path / "kaggle.toml").read_text()


def test_show_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                   capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    import json
    rc = config_cmd.show(json_output=True)
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert "project" in parsed
    assert "feature" in parsed


def test_path_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    rc = config_cmd.path()
    assert rc == 1
