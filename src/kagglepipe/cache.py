"""Artifact caching (P5).

A per-branch cache records the SHA-256 of all inputs that determine the
output artifact:
  * the source dataset version (P5 source-of-truth)
  * the rendered notebook content (captures branch, mounts, command, GPU)
  * the kernel-metadata.json content
  * the relevant subset of the project config
  * the configured output_glob

If the hash hasn't changed since the last successful run, the existing
artifact is reused and the branch is skipped.

Cache files live at `.kagglepipe/cache/<branch>.json`.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kagglepipe.config import Config
from kagglepipe.state import state_dir


CACHE_DIRNAME = "cache"


def cache_dir(root: Path | None = None) -> Path:
    base = state_dir(root)
    d = base / CACHE_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class CacheEntry:
    branch: str
    inputs_hash: str
    artifact_path: str
    src_version: int
    kernel_slug: str
    created_at: float
    notebook_hash: str  # the SHA of the notebook JSON that produced this artifact

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CacheEntry":
        return cls(**d)


def compute_inputs_hash(
    *,
    branch: str,
    notebook: dict[str, Any],
    kernel_metadata: dict[str, Any],
    src_version: int,
    output_glob: str,
    config_hash: str,
) -> str:
    """Return SHA-256 over all inputs that determine the output artifact.

    Stable across runs: keys are sorted and dumps are sorted.
    """
    blob = {
        "branch": branch,
        "notebook": notebook,
        "kernel_metadata": kernel_metadata,
        "src_version": src_version,
        "output_glob": output_glob,
        "config_hash": config_hash,
    }
    encoded = json.dumps(blob, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def config_hash_for_branch(cfg: Config, branch: str) -> str:
    """Hash the parts of kaggle.toml that affect this branch's output."""
    relevant = {
        "feature.notebook_command": cfg.feature.notebook_command,
        "feature.notebook_template": cfg.feature.notebook_template,
        "feature.data_mount": cfg.feature.data_mount,
        "feature.src_mount": cfg.feature.src_mount,
        "feature.out_dir": cfg.feature.out_dir,
        "feature.output_glob": cfg.feature.output_glob,
        "kernels.is_private": cfg.kernels.is_private,
        "kernels.enable_internet": cfg.kernels.enable_internet,
        "kernels.language": cfg.kernels.language,
        "kernels.kernel_type": cfg.kernels.kernel_type,
    }
    encoded = json.dumps(relevant, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class CacheStore:
    """Thread-safe per-branch cache."""

    def __init__(self, root: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._dir = cache_dir(root)
        self._entries: dict[str, CacheEntry] = {}
        self._load()

    def _load(self) -> None:
        for path in self._dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                entry = CacheEntry.from_dict(data)
                self._entries[entry.branch] = entry
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

    def _persist(self, entry: CacheEntry) -> None:
        path = self._dir / f"{entry.branch}.json"
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(entry.to_dict(), indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def get(self, branch: str) -> CacheEntry | None:
        with self._lock:
            return self._entries.get(branch)

    def put(self, entry: CacheEntry) -> None:
        with self._lock:
            self._entries[entry.branch] = entry
            self._persist(entry)

    def all(self) -> list[CacheEntry]:
        with self._lock:
            return list(self._entries.values())

    def clear(self, branch: str | None = None) -> int:
        """Clear cache for a single branch (or all). Returns the count cleared."""
        with self._lock:
            if branch is None:
                paths = list(self._dir.glob("*.json"))
                count = len(paths)
                for p in paths:
                    p.unlink(missing_ok=True)
                self._entries.clear()
                return count
            entry = self._entries.pop(branch, None)
            if entry is None:
                return 0
            (self._dir / f"{branch}.json").unlink(missing_ok=True)
            return 1


# --- CLI wrappers --------------------------------------------------------


def cmd_cache_status(*, json_output: bool = False) -> int:
    store = CacheStore()
    entries = sorted(store.all(), key=lambda e: e.branch)
    if json_output:
        import json
        print(json.dumps([e.to_dict() for e in entries], indent=2))
        return 0
    if not entries:
        print("Cache is empty. Enable with `feature.cache = 1` in kaggle.toml.")
        return 0
    print(f"{'BRANCH':<30} {'HASH':<16} {'ARTIFACT'}")
    for e in entries:
        print(f"{e.branch:<30} {e.inputs_hash[:12]:<16} {e.artifact_path}")
    return 0


def cmd_cache_clear(branch: str | None = None) -> int:
    store = CacheStore()
    n = store.clear(branch)
    if branch:
        print(f"Cleared cache for {branch!r} ({n} entry).")
    else:
        print(f"Cleared {n} cache entries.")
    return 0
