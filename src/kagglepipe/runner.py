"""Subprocess wrapper around the official `kaggle` CLI.

The upstream CLI emits Unicode (box-drawing, checkmarks) that crashes Windows
consoles using `cp1252`. Centralizing the call here lets every command get
UTF-8 safety for free and makes subprocess behavior testable by patching
`_run`.
"""

from __future__ import annotations

import os
import subprocess
import sys

# Force UTF-8 in this process and any children we spawn. Done at import so
# every entry point (cli.py, __main__.py) gets it for free.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass  # Python < 3.7 or non-reconfigurable stream


class KaggleError(RuntimeError):
    """Raised when a `kaggle` CLI invocation fails."""

    def __init__(self, args: list[str], returncode: int, stdout: str, stderr: str) -> None:
        self.args = args
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        snippet = (stderr or stdout or "").strip().splitlines()
        detail = snippet[-1] if snippet else "(no output)"
        super().__init__(f"kaggle {' '.join(args)} failed (rc={returncode}): {detail}")


def run(
    args: list[str],
    *,
    check: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    """Run `python -X utf8 -m kaggle <args>` and return the CompletedProcess.

    Args:
        args: Args after `kaggle` (e.g., ["kernels", "list"]).
        check: If True, raise KaggleError on non-zero returncode.
        timeout: Optional subprocess timeout in seconds.
    """
    cmd = [sys.executable, "-X", "utf8", "-m", "kaggle", *args]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
    )
    if check and result.returncode != 0:
        raise KaggleError(cmd, result.returncode, result.stdout, result.stderr)
    return result


def run_json(args: list[str], **kwargs) -> object:
    """Run a kaggle command that emits JSON and parse the result.

    Raises KaggleError on non-zero returncode or invalid JSON.
    """
    import json

    result = run([*args, "--json"], **kwargs)
    if result.returncode != 0:
        raise KaggleError([*args, "--json"], result.returncode, result.stdout, result.stderr)
    try:
        return json.loads(result.stdout) if result.stdout.strip() else None
    except json.JSONDecodeError as exc:
        raise KaggleError(
            [*args, "--json"], result.returncode, result.stdout, result.stderr
        ) from exc
