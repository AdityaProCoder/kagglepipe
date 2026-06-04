"""Provenance capture (P11.5 / P13).

Small helpers that read git state and the local RunStore to assemble
the "what produced this score?" record. Used by `submissions` and
written into every run manifest.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from kagglepipe.cache import CacheStore
from kagglepipe.state import RunStore


def git_commit() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if r.returncode != 0:
            return None
        return r.stdout.strip() or None
    except (FileNotFoundError, OSError):
        return None


def git_dirty() -> bool | None:
    """True if the working tree has uncommitted changes. None if no git."""
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if r.returncode != 0:
            return None
        return bool(r.stdout.strip())
    except (FileNotFoundError, OSError):
        return None


def artifact_hashes_from_cache(branches: list[str]) -> dict[str, str]:
    """Return branch -> cache inputs hash for any branch with a cache entry."""
    out: dict[str, str] = {}
    store = CacheStore()
    for b in branches:
        e = store.get(b)
        if e is not None:
            out[b] = e.inputs_hash
    return out


def feature_branches_from_runs(branches: list[str] | None = None) -> list[str]:
    """Return the list of branches that have at least one RunRecord.

    If `branches` is given, only return those (preserving order, dropping
    branches that have no runs).
    """
    store = RunStore()
    seen: set[str] = set()
    for r in store.all():
        if branches is not None and r.branch not in branches:
            continue
        if r.branch not in seen:
            seen.add(r.branch)
            if branches is not None:
                # Preserve the requested order.
                pass
    if branches is not None:
        return [b for b in branches if b in seen]
    return list(seen)


def build_provenance(
    *,
    experiment_id: str | None = None,
    feature_branches: list[str] | None = None,
    dataset_versions: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Assemble the standard provenance dict for a submission / run."""
    branches = feature_branches or feature_branches_from_runs()
    return {
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "experiment_id": experiment_id,
        "feature_branches": branches,
        "dataset_versions": dataset_versions or {},
        "artifact_hashes": artifact_hashes_from_cache(branches),
    }


def hash_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()
