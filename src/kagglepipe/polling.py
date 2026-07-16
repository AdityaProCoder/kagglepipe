"""Kernel status polling."""

from __future__ import annotations

import time

from kagglepipe import runner


def poll_kernel_status(
    kernel_slug: str,
    *,
    timeout_sec: int,
    poll_interval_sec: int = 30,
    sleep_fn: callable | None = None,
    time_fn: callable | None = None,
) -> str:
    """Poll `kaggle kernels status` until terminal state or timeout.

    Returns one of: "complete", "error", "timeout".

    The `sleep_fn` and `time_fn` parameters exist for unit tests; production
    callers should leave them as None.
    """
    _sleep = sleep_fn or time.sleep
    _time = time_fn or time.time
    deadline = _time() + timeout_sec
    while _time() < deadline:
        result = runner.run(["kernels", "status", "-k", kernel_slug])
        if result.returncode != 0:
            return "error"
        stdout = result.stdout.strip().strip('"').lower()
        if "complete" in stdout:
            return "complete"
        if "error" in stdout or "fail" in stdout:
            return "error"
        _sleep(poll_interval_sec)
    return "timeout"
