"""Tests for auth commands (whoami, login)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from kagglepipe.cli import _build_parser, main
from kagglepipe.commands import auth as auth_cmd


def test_legacy_auth_commands_remain_available() -> None:
    """The grouped auth UX must not break existing scripts."""
    parser = _build_parser()
    assert parser.parse_args(["whoami"]).cmd == "whoami"
    assert parser.parse_args(["login", "--username", "alice", "--key", "secret"]).cmd == "login"
    grouped = parser.parse_args(["auth", "whoami"])
    assert grouped.cmd == "auth"
    assert grouped.auth_cmd == "whoami"


def test_login_with_custom_path_uses_that_path_for_project_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A custom credentials path must not be discarded after login."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path / "home"))
    target = tmp_path / "credentials.json"

    assert main(["auth", "login", "--username", "alice", "--key", "secret", "--path", str(target)]) == 0
    assert (tmp_path / "kaggle.toml").exists()


def test_whoami_prints_username(
    monkeypatch: pytest.MonkeyPatch, fake_creds: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "kagglepipe.runner.run",
        lambda *a, **kw: subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        ),
    )
    rc = auth_cmd.whoami()
    assert rc == 0
    assert capsys.readouterr().out.strip() == "testuser"


def test_whoami_json_output(
    monkeypatch: pytest.MonkeyPatch, fake_creds: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "kagglepipe.runner.run",
        lambda *a, **kw: subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        ),
    )
    rc = auth_cmd.whoami(json_output=True)
    assert rc == 0
    out = capsys.readouterr().out
    assert '"username": "testuser"' in out
    assert '"auth_ok": true' in out


def test_whoami_auth_check_fails(
    monkeypatch: pytest.MonkeyPatch, fake_creds: Path,
) -> None:
    monkeypatch.setattr(
        "kagglepipe.runner.run",
        lambda *a, **kw: subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="auth failed"
        ),
    )
    rc = auth_cmd.whoami()
    assert rc == 1


def test_whoami_no_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path))
    rc = auth_cmd.whoami()
    assert rc == 1


def test_login_writes_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "kaggle.json"
    rc = auth_cmd.login(username="alice", key="secret", target=target)
    assert rc == 0
    import json
    data = json.loads(target.read_text())
    assert data == {"username": "alice", "key": "secret"}
