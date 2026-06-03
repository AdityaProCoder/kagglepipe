"""High-level wrappers around the kaggle CLI for kagglepipe commands.

Each function returns parsed Python data (dicts / lists / strings) instead of
raw CLI output, so commands can format and `--json` serialize easily.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from kagglepipe import runner


def datasets_list(*, search: str | None = None, user: str | None = None) -> list[dict[str, str]]:
    """List datasets. `search` and `user` are optional kaggle CLI filters."""
    args = ["datasets", "list"]
    if search:
        args += ["--search", search]
    if user:
        args += ["--user", user]
    result = runner.run([*args, "--csv"])
    return _parse_csv(result.stdout)


def datasets_versions(slug: str) -> list[dict[str, str]]:
    """List versions of a dataset. Note: the kaggle CLI does not expose this
    directly, so we list and filter by slug. The version count is inferred
    from the dataset page; the canonical source is the dataset page itself."""
    result = runner.run(["datasets", "list", "--search", slug, "--csv"])
    rows = _parse_csv(result.stdout)
    return [r for r in rows if r.get("ref") == slug]


def dataset_exists(slug: str) -> bool:
    """True if a dataset with this slug is visible to the current user.

    Note: `kaggle datasets list --search <slug>` does not match against the
    full slug reliably; we list the current user's datasets and filter.
    """
    try:
        username = slug.split("/", 1)[0]
    except IndexError:
        return False
    return any(r.get("ref") == slug for r in datasets_list(user=username))


def get_next_version(slug: str) -> int:
    """Return the next version number for a dataset (1 if absent, 2 if present).

    The kaggle CLI doesn't expose "list versions of a dataset" cleanly, so we
    use `datasets list --search` as a proxy.
    """
    return 2 if dataset_exists(slug) else 1


def kernels_list(
    *, user: str | None = None, search: str | None = None, page_size: int = 20
) -> list[dict[str, str]]:
    """List kernels visible to the current user."""
    args = ["kernels", "list", "--page-size", str(page_size)]
    if user:
        args += ["--user", user]
    if search:
        args += ["--search", search]
    result = runner.run([*args, "--csv"])
    return _parse_csv(result.stdout)


def kernel_status(slug: str) -> str:
    """Return the raw status string for a kernel (e.g. "complete", "running")."""
    result = runner.run(["kernels", "status", "-k", slug])
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, ["kaggle", "kernels", "status", "-k", slug]
        )
    return result.stdout.strip().strip('"').lower()


def kernels_stop(slug: str) -> None:
    """Cancel a running kernel."""
    runner.run(["kernels", "stop", "-k", slug], check=True)


def kernels_logs_url(slug: str) -> str:
    """Return the canonical logs URL for a kernel (no API call)."""
    return f"https://www.kaggle.com/{slug}/logs"


def download_kernel_output(slug: str, dest: Path) -> Path:
    """Run `kaggle kernels output` into a tmp dir and return that path.

    The caller decides what to do with the contents.
    """
    dest.mkdir(parents=True, exist_ok=True)
    runner.run(["kernels", "output", slug, "-p", str(dest)], check=True)
    return dest


def find_artifact(artifact_dir: Path, glob_pattern: str) -> Path:
    """Search `artifact_dir` for the first file matching `glob_pattern`.

    Searches recursively under `artifact_dir/<branch>/` first, then
    recursively anywhere in `artifact_dir` as a fallback.
    """
    candidates = list(artifact_dir.rglob(Path(glob_pattern).name))
    if not candidates:
        raise FileNotFoundError(
            f"No file matching {glob_pattern!r} in {artifact_dir}. "
            f"Check kernel logs at https://www.kaggle.com/<user>/<slug>/logs"
        )
    return candidates[0]


def copy_artifact(src: Path, dest: Path) -> Path:
    """Copy `src` to `dest`, creating parent directories as needed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def competitions_list() -> list[dict[str, str]]:
    """List active competitions."""
    result = runner.run(["competitions", "list", "--csv"])
    return _parse_csv(result.stdout)


def competitions_leaderboard(slug: str, *, top: int | None = None) -> list[dict[str, str]]:
    """Show the leaderboard for a competition."""
    args = ["competitions", "leaderboard", "-c", slug, "--csv"]
    result = runner.run(args)
    rows = _parse_csv(result.stdout)
    if top is not None:
        rows = rows[:top]
    return rows


def competitions_submit(slug: str, file_path: Path, message: str) -> None:
    """Submit a file to a competition."""
    runner.run(
        [
            "competitions",
            "submit",
            "-c",
            slug,
            "-f",
            str(file_path),
            "-m",
            message,
        ],
        check=True,
    )


# --- helpers --------------------------------------------------------------


def _parse_csv(stdout: str) -> list[dict[str, str]]:
    """Parse kaggle CLI's --csv output into list of dicts. Empty -> []."""
    if not stdout or not stdout.strip():
        return []
    reader = csv.DictReader(io.StringIO(stdout))
    return [dict(row) for row in reader]
