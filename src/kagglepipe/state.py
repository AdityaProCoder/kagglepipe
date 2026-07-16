"""Local state store for kagglepipe.

Persists run history, submissions, experiments, features, and lineage as
JSON files under `.kagglepipe/`. P1-P8 all read/write through this layer.

Layout:
    .kagglepipe/
        runs.json          -- per-branch run history (P2, P5, P6)
        submissions.json   -- competition submissions (P3)
        experiments.json   -- experiment tracking (P6)
        features.json      -- feature registry (P7)
        lineage.json       -- upstream/downstream graph (P8)
        cache/<branch>.json -- per-branch input hash + cached artifact path (P5)
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STATE_DIRNAME = ".kagglepipe"


def state_dir(root: Path | None = None) -> Path:
    """Return the kagglepipe state directory for the given (or cwd) project."""
    base = (root or Path.cwd()).resolve()
    target = base / STATE_DIRNAME
    target.mkdir(parents=True, exist_ok=True)
    return target


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically: tmp file + replace, so concurrent writers don't corrupt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    """Read JSON, returning None if the file doesn't exist or is malformed."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# --- runs (P2) ------------------------------------------------------------


@dataclass
class RunRecord:
    """One attempt at running a feature branch (P13 strong manifest).

    Captures every input that determined the output artifact, so any
    historical run is fully reproducible from this record alone.
    """

    branch: str
    kernel_slug: str
    state: str  # "queued" | "running" | "complete" | "error" | "timeout"
    artifact_path: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
    config_hash: str | None = None  # P5
    skipped: bool = False  # P5: hit cache, did not actually run
    # P13: extra provenance
    git_commit: str | None = None
    git_dirty: bool | None = None
    gpu: str | None = None
    src_slug: str | None = None
    src_version: int | None = None
    dataset_versions: dict[str, int] = field(default_factory=dict)
    notebook_hash: str | None = None
    manifest_path: str | None = None  # P13: where the on-disk manifest was written

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> RunRecord:
        # Backward-compat: ignore unknown fields from older records.
        known = {f.name for f in __import__("dataclasses").fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class RunStore:
    """Thread-safe append/read store for RunRecord objects.

    Backed by `.kagglepipe/runs.json`. Writes are serialized via a lock
    and atomic file replacement so concurrent threads (P1 parallel) can't
    corrupt the JSON.
    """

    def __init__(self, root: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._root = state_dir(root) if root else state_dir()
        self._path = self._root / "runs.json"
        self._records: list[RunRecord] = []
        self._load()

    def _load(self) -> None:
        data = _read_json(self._path)
        if isinstance(data, list):
            self._records = [RunRecord.from_dict(r) for r in data if isinstance(r, dict)]

    def _persist(self) -> None:
        _atomic_write_json(self._path, [r.to_dict() for r in self._records])

    def add(self, record: RunRecord) -> None:
        with self._lock:
            self._records.append(record)
            self._persist()

    def update(self, branch: str, kernel_slug: str, **changes: Any) -> None:
        """Update the most recent record matching branch+kernel_slug."""
        with self._lock:
            for r in reversed(self._records):
                if r.branch == branch and r.kernel_slug == kernel_slug:
                    for k, v in changes.items():
                        setattr(r, k, v)
                    break
            self._persist()

    def latest_for_branch(self, branch: str) -> RunRecord | None:
        with self._lock:
            for r in reversed(self._records):
                if r.branch == branch:
                    return r
            return None

    def all_for_branch(self, branch: str) -> list[RunRecord]:
        with self._lock:
            return [r for r in self._records if r.branch == branch]

    def is_branch_successful(self, branch: str) -> bool:
        """True if the most recent run for this branch completed and produced an artifact."""
        latest = self.latest_for_branch(branch)
        if latest is None:
            return False
        return latest.state == "complete" and latest.artifact_path is not None

    def failed_or_incomplete(self, branches: list[str]) -> list[str]:
        """Return subset of branches whose latest run did not complete cleanly."""
        return [b for b in branches if not self.is_branch_successful(b)]

    def all(self) -> list[RunRecord]:
        with self._lock:
            return list(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records = []
            self._persist()


# --- submissions (P3) -----------------------------------------------------


@dataclass
class SubmissionRecord:
    competition: str
    file_path: str
    message: str
    submitted_at: float = field(default_factory=time.time)
    submission_id: str | None = None  # filled in after upload if Kaggle returns one
    score: float | None = None
    rank: int | None = None  # P11: position on the leaderboard when known
    status: str = "submitted"  # "submitted" | "scored" | "failed"
    # P11.5: submission provenance — answers "what code produced this score?"
    experiment_id: str | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None
    feature_branches: list[str] = field(default_factory=list)
    dataset_versions: dict[str, int] = field(default_factory=dict)
    artifact_hashes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SubmissionRecord:
        # Backward-compat: older records may not have the P11.5 fields.
        known = {f.name for f in __import__("dataclasses").fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class SubmissionStore:
    def __init__(self, root: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._root = state_dir(root) if root else state_dir()
        self._path = self._root / "submissions.json"
        self._records: list[SubmissionRecord] = []
        self._load()

    def _load(self) -> None:
        data = _read_json(self._path)
        if isinstance(data, list):
            self._records = [SubmissionRecord.from_dict(r) for r in data if isinstance(r, dict)]

    def _persist(self) -> None:
        _atomic_write_json(self._path, [r.to_dict() for r in self._records])

    def add(self, record: SubmissionRecord) -> None:
        with self._lock:
            self._records.append(record)
            self._persist()

    def all(self) -> list[SubmissionRecord]:
        with self._lock:
            return list(self._records)

    def latest(self, competition: str | None = None) -> SubmissionRecord | None:
        with self._lock:
            filtered = (
                [r for r in self._records if r.competition == competition]
                if competition
                else self._records
            )
            return filtered[-1] if filtered else None


# --- experiments (P6) -----------------------------------------------------


@dataclass
class ExperimentRecord:
    id: str
    created_at: float = field(default_factory=time.time)
    git_commit: str | None = None
    dataset_versions: dict[str, int] = field(default_factory=dict)
    feature_branches: list[str] = field(default_factory=list)
    submission_id: str | None = None
    score: float | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentRecord:
        return cls(**d)


class ExperimentStore:
    def __init__(self, root: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._root = state_dir(root) if root else state_dir()
        self._path = self._root / "experiments.json"
        self._records: list[ExperimentRecord] = []
        self._load()

    def _load(self) -> None:
        data = _read_json(self._path)
        if isinstance(data, list):
            self._records = [ExperimentRecord.from_dict(r) for r in data if isinstance(r, dict)]

    def _persist(self) -> None:
        _atomic_write_json(self._path, [r.to_dict() for r in self._records])

    def add(self, record: ExperimentRecord) -> None:
        with self._lock:
            self._records.append(record)
            self._persist()

    def all(self) -> list[ExperimentRecord]:
        with self._lock:
            return list(self._records)

    def get(self, exp_id: str) -> ExperimentRecord | None:
        with self._lock:
            for r in self._records:
                if r.id == exp_id:
                    return r
            return None


# --- features (P7) --------------------------------------------------------


@dataclass
class FeatureRecord:
    name: str
    dataset_slug: str
    version: int
    artifact_path: str
    created_at: float = field(default_factory=time.time)
    branch: str | None = None
    parents: list[str] = field(default_factory=list)  # upstream features (P8)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> FeatureRecord:
        return cls(**d)


class FeatureStore:
    def __init__(self, root: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._root = state_dir(root) if root else state_dir()
        self._path = self._root / "features.json"
        self._records: list[FeatureRecord] = []
        self._load()

    def _load(self) -> None:
        data = _read_json(self._path)
        if isinstance(data, list):
            self._records = [FeatureRecord.from_dict(r) for r in data if isinstance(r, dict)]

    def _persist(self) -> None:
        _atomic_write_json(self._path, [r.to_dict() for r in self._records])

    def add(self, record: FeatureRecord) -> None:
        with self._lock:
            self._records.append(record)
            self._persist()

    def all(self) -> list[FeatureRecord]:
        with self._lock:
            return list(self._records)

    def latest(self, name: str) -> FeatureRecord | None:
        with self._lock:
            for r in reversed(self._records):
                if r.name == name:
                    return r
            return None

    def get(self, name: str) -> list[FeatureRecord]:
        with self._lock:
            return [r for r in self._records if r.name == name]
