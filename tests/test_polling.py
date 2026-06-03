"""Tests for kernel polling."""

from __future__ import annotations

import subprocess

import pytest

from kagglepipe import polling


def _completed(*, returncode: int = 0, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_poll_returns_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_run(args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            return _completed(stdout='"running"')
        return _completed(stdout='"complete"')

    monkeypatch.setattr("kagglepipe.runner.run", fake_run)
    fake_time = [0, 5, 10, 15]
    sleeps: list[float] = []

    state = polling.poll_kernel_status(
        "u/k",
        timeout_sec=600,
        poll_interval_sec=30,
        time_fn=lambda: fake_time.pop(0),
        sleep_fn=sleeps.append,
    )
    assert state == "complete"
    assert sleeps  # we did sleep between polls


def test_poll_returns_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "kagglepipe.runner.run",
        lambda *a, **kw: _completed(stdout='"running"'),
    )
    # First call: 0 (sets deadline to 600). Subsequent: 1000 (past deadline).
    times = iter([0, 1000, 2000, 3000])
    state = polling.poll_kernel_status(
        "u/k",
        timeout_sec=600,
        poll_interval_sec=30,
        time_fn=lambda: next(times),
        sleep_fn=lambda _: None,
    )
    assert state == "timeout"


def test_poll_returns_error_on_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "kagglepipe.runner.run",
        lambda *a, **kw: _completed(returncode=1, stderr="not found"),
    )
    state = polling.poll_kernel_status(
        "u/k",
        timeout_sec=600,
        poll_interval_sec=30,
        time_fn=lambda: 0,
        sleep_fn=lambda _: None,
    )
    assert state == "error"


def test_poll_returns_error_on_error_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "kagglepipe.runner.run",
        lambda *a, **kw: _completed(stdout='"Kernel error"'),
    )
    state = polling.poll_kernel_status(
        "u/k",
        timeout_sec=600,
        poll_interval_sec=30,
        time_fn=lambda: 0,
        sleep_fn=lambda _: None,
    )
    assert state == "error"
