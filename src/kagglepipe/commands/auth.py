"""whoami / login commands."""

from __future__ import annotations

import getpass
import json
import sys
from pathlib import Path

from kagglepipe import credentials, runner


def whoami(*, json_output: bool = False) -> int:
    """Print the current Kaggle username (verified against the API)."""
    try:
        creds = credentials.load()
    except credentials.CredentialsError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    # The upstream CLI's `whoami` is fragile; we trust the local credentials
    # file and only ping `kernels list` to verify auth works.
    result = runner.run(["kernels", "list", "--page-size", "1"])
    if result.returncode != 0:
        print(f"Auth check failed: {result.stderr.strip()}", file=sys.stderr)
        return 1
    if json_output:
        print(json.dumps({"username": creds.username, "auth_ok": True}))
    else:
        print(creds.username)
    return 0


def login(
    *,
    username: str | None = None,
    key: str | None = None,
    path: Path | None = None,
    target: Path | None = None,
) -> int:
    """Bootstrap ~/.kaggle/kaggle.json. Reads from stdin / prompts if needed."""
    if not username:
        username = input("Kaggle username: ").strip()
    if not username:
        print("Username required.", file=sys.stderr)
        return 1
    if not key:
        if not sys.stdin.isatty():
            print(
                "No key supplied and stdin is not a TTY. "
                "Pass --key or set KAGGLE_KEY.",
                file=sys.stderr,
            )
            return 1
        key = getpass.getpass("Kaggle API key: ").strip()
    if not key:
        print("Key required.", file=sys.stderr)
        return 1
    target = target or credentials.default_path()
    written = credentials.write(username, key, target)
    print(f"Wrote credentials to {written}")
    return 0
