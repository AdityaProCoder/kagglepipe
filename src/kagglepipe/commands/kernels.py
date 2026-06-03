"""kernels — list, status, output, logs, stop, push."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from kagglepipe import credentials, kaggle_api, runner
from kagglepipe.config import Config


def list_kernels(
    *,
    user: str | None = None,
    search: str | None = None,
    page_size: int = 20,
    csv_output: bool = False,
    json_output: bool = False,
) -> int:
    """List kernels visible to the current user (or a specified user)."""
    creds = credentials.load()
    rows = kaggle_api.kernels_list(
        user=user or creds.username, search=search, page_size=page_size
    )
    if json_output:
        print(json.dumps(rows, indent=2))
        return 0
    if csv_output:
        if rows:
            keys = list(rows[0].keys())
            print(",".join(keys))
            for r in rows:
                print(",".join(r.get(c, "") for c in keys))
        return 0
    if not rows:
        print("No kernels.")
        return 0
    for r in rows:
        ref = r.get("ref", "?")
        title = r.get("title", "")
        state = r.get("status", "?")
        print(f"{ref:<60}  {state:<10}  {title}")
    return 0


def kernel_status(slug: str) -> int:
    """Print the status of one kernel."""
    state = kaggle_api.kernel_status(slug)
    print(state)
    return 0 if state in {"complete", "running", "queued"} else 1


def kernel_output(slug: str, *, path: Path | None = None) -> int:
    """Download a kernel's output to a directory."""
    dest = (path or Path.cwd() / f"{slug.replace('/', '_')}_output").resolve()
    kaggle_api.download_kernel_output(slug, dest)
    print(dest)
    return 0


def kernel_logs(slug: str) -> int:
    """Print the logs URL for a kernel."""
    print(kaggle_api.kernels_logs_url(slug))
    return 0


def kernel_stop(slug: str) -> int:
    """Stop a running kernel."""
    kaggle_api.kernels_stop(slug)
    print(f"Stop requested for {slug}")
    return 0


def kernel_push(path: Path) -> int:
    """Push a directory containing a kernel-metadata.json."""
    result = runner.run(["kernels", "push", "-p", str(path)])
    print(result.stdout, file=sys.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode
