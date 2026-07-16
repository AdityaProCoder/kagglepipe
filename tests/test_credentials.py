"""Tests for credentials discovery and writing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kagglepipe import credentials


def test_load_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "alice")
    monkeypatch.setenv("KAGGLE_KEY", "secret")
    creds = credentials.load()
    assert creds.username == "alice"
    assert creds.key == "secret"


def test_load_from_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "kaggle.json"
    cfg.write_text(json.dumps({"username": "bob", "key": "xyz"}))
    creds = credentials.load(cfg)
    assert creds.username == "bob"
    assert creds.key == "xyz"


def test_load_missing_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path))
    with pytest.raises(credentials.CredentialsError, match="not found"):
        credentials.load()


def test_load_invalid_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "kaggle.json"
    cfg.write_text("{not json")
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path))
    with pytest.raises(credentials.CredentialsError, match="Invalid JSON"):
        credentials.load(cfg)


def test_load_unreadable_file_raises_credentials_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "kaggle.json"
    cfg.write_text(json.dumps({"username": "bob", "key": "xyz"}))

    def denied(*_args, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(credentials.CredentialsError, match="Could not read"):
        credentials.load(cfg)


def test_load_missing_field(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "kaggle.json"
    cfg.write_text(json.dumps({"username": "bob"}))
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path))
    with pytest.raises(credentials.CredentialsError, match="missing required field"):
        credentials.load(cfg)


def test_write_creates_dir_and_writes_json(tmp_path: Path) -> None:
    target = tmp_path / "subdir" / "kaggle.json"
    written = credentials.write("user", "key", target)
    assert written == target
    data = json.loads(target.read_text())
    assert data == {"username": "user", "key": "key"}


def test_masked_does_not_leak_key() -> None:
    c = credentials.Credentials("alice", "supersecretkey")
    masked = c.masked()
    assert "alice" in masked
    assert "supersecretkey" not in masked
