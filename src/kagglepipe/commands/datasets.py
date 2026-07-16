"""datasets — list, versions, get, create, version."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from kagglepipe import kaggle_api, runner


def list_datasets(
    *,
    user: str | None = None,
    search: str | None = None,
    csv_output: bool = False,
    json_output: bool = False,
) -> int:
    """List datasets visible to the current user."""
    rows = kaggle_api.datasets_list(user=user, search=search)
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
        print("No datasets.")
        return 0
    for r in rows:
        ref = r.get("ref", "?")
        title = r.get("title", "")
        size = r.get("totalBytes", "?")
        print(f"{ref:<60}  {title:<40}  {size}")
    return 0


def dataset_versions(slug: str) -> int:
    """List versions of a dataset (best-effort: upstream CLI has no per-dataset
    version listing, so we report whether the dataset exists)."""
    if kaggle_api.dataset_exists(slug):
        print(f"{slug} exists (use the web UI to see individual versions).")
        return 0
    print(f"{slug} not found.", file=sys.stderr)
    return 1


def dataset_get(slug: str, path: Path) -> int:
    """Download a dataset to a path."""
    path.mkdir(parents=True, exist_ok=True)
    result = runner.run(["datasets", "download", slug, "-p", str(path), "--unzip"])
    print(result.stdout, file=sys.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode


def dataset_create(path: Path, *, public: bool = False) -> int:
    """Create a new dataset from a directory containing dataset-metadata.json."""
    args = ["datasets", "create", "-p", str(path)]
    if public:
        args.append("--public")
    result = runner.run(args)
    print(result.stdout, file=sys.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode


def dataset_version(path: Path, *, message: str, dir_mode: str = "zip") -> int:
    """Create a new version of an existing dataset."""
    result = runner.run(
        [
            "datasets",
            "version",
            "-p",
            str(path),
            "-m",
            message,
            "-r",
            dir_mode,
        ]
    )
    print(result.stdout, file=sys.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode
