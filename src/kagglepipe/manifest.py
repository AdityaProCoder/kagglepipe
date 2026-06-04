"""Run manifest writer (P13).

Every `kagglepipe feature run` (or `feature all`) writes a manifest JSON
to `.kagglepipe/manifests/<branch>-<timestamp>.json` containing all the
provenance needed to reproduce the run. The RunRecord in RunStore is
updated with the manifest path so a single run can be located by branch
or by timestamp.

Manifest schema:

    {
      "schema": "kagglepipe.manifest.v1",
      "branch": "user_features",
      "state": "complete",
      "kernel_slug": "user/myproj-user_features",
      "started_at": 1717400000.0,
      "finished_at": 1717401000.0,
      "git_commit": "a7d9c13...",
      "git_dirty": false,
      "gpu": "t4 x2",
      "src_slug": "user/myproj-src",
      "src_version": 3,
      "dataset_versions": {"src": 3, "data": 1},
      "notebook_hash": "sha256:...",
      "config_hash": "sha256:...",
      "artifact_path": "features_kaggle/user_features.parquet",
      "artifact_hash": "sha256:...",
      "kaggle_url": "https://www.kaggle.com/user/myproj-user_features"
    }
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from kagglepipe.provenance import git_commit, git_dirty, hash_file
from kagglepipe.state import RunRecord, state_dir


MANIFESTS_DIRNAME = "manifests"
MANIFEST_SCHEMA = "kagglepipe.manifest.v1"


def manifests_dir(root: Path | None = None) -> Path:
    d = state_dir(root) / MANIFESTS_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_manifest(record: RunRecord) -> Path:
    """Write a manifest JSON for the given run record and update the record.

    Returns the path to the manifest file.
    """
    payload: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        **asdict(record),
    }
    # Augment with live data that may not have been set when the record
    # was created (e.g., git state at finish time).
    if record.git_commit is None:
        payload["git_commit"] = git_commit()
    if record.git_dirty is None:
        payload["git_dirty"] = git_dirty()
    if record.artifact_path and os.path.exists(record.artifact_path):
        h = hash_file(Path(record.artifact_path))
        if h:
            payload["artifact_hash"] = h
    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime(record.started_at))
    fname = f"{record.branch}-{ts}.json"
    path = manifests_dir() / fname
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    record.manifest_path = str(path)
    return path


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
