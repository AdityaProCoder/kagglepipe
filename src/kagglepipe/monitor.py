"""Read-only state collection for the `kagglepipe monitor` dashboard.

This module is the single source of truth for what the monitor displays.
It reads from existing kagglepipe stores (no new storage) and returns
a `MonitorSnapshot` data class. The Textual UI in
`kagglepipe/commands/monitor.py` consumes this snapshot — there is no
UI logic here.

The split keeps the collection side fast and testable, and lets the
view be a thin renderer.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kagglepipe import credentials
from kagglepipe.cache import CacheStore
from kagglepipe.config import load as load_config
from kagglepipe.manifest import manifests_dir
from kagglepipe.state import (
    ExperimentStore,
    FeatureStore,
    RunStore,
    SubmissionStore,
)


# ---- view-model types ----------------------------------------------------


@dataclass(frozen=True)
class JobView:
    """One feature branch reduced to a row in the Active Jobs panel."""

    branch: str
    state: str  # "complete" | "running" | "queued" | "error" | "timeout" | "skipped"
    elapsed_seconds: float
    gpu: str | None
    cache: str  # "HIT" | "MISS" | "N/A"
    kernel_slug: str

    @property
    def is_terminal(self) -> bool:
        return self.state in {"complete", "error", "timeout", "skipped"}


@dataclass(frozen=True)
class ArtifactView:
    """One feature artifact rendered in the Latest Artifacts panel."""

    name: str
    size_bytes: int
    timestamp: float

    @property
    def size_human(self) -> str:
        return _humanize_bytes(self.size_bytes)

    @property
    def timestamp_human(self) -> str:
        return _humanize_time(self.timestamp)


@dataclass(frozen=True)
class SubmissionView:
    competition: str
    score: float | None
    rank: int | None
    submission_id: str | None
    timestamp: float

    @property
    def timestamp_human(self) -> str:
        return _humanize_time(self.timestamp)


@dataclass(frozen=True)
class BestSubmissionView:
    score: float | None
    rank: int | None
    git_commit: str | None
    experiment_id: str | None
    competition: str
    timestamp: float

    @property
    def timestamp_human(self) -> str:
        return _humanize_time(self.timestamp)


# ---- the snapshot itself -------------------------------------------------


@dataclass
class MonitorSnapshot:
    """Pure data view of the kagglepipe project state. Built once per tick."""

    project_name: str
    user: str
    collected_at: float

    jobs: list[JobView] = field(default_factory=list)
    artifacts: list[ArtifactView] = field(default_factory=list)
    latest_submission: SubmissionView | None = None
    best_submission: BestSubmissionView | None = None

    # Pipeline overview counters
    total_branches: int = 0
    completed: int = 0
    running: int = 0
    failed: int = 0
    queued: int = 0

    # Experiment summary
    total_experiments: int = 0
    total_manifests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    registered_features: int = 0

    @property
    def percent_complete(self) -> float:
        if self.total_branches == 0:
            return 0.0
        return (self.completed / self.total_branches) * 100.0

    @property
    def collected_at_human(self) -> str:
        return datetime.fromtimestamp(self.collected_at, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )


# ---- collectors ----------------------------------------------------------


def _humanize_bytes(n: int) -> str:
    """Render a byte count as a compact human string (e.g. '4.2 MB')."""
    if n < 0:
        return "—"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            if unit == "B":
                return f"{n} B"
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _humanize_time(ts: float) -> str:
    """Render a timestamp as a compact relative time (e.g. '2h ago')."""
    if ts <= 0:
        return "—"
    delta = time.time() - ts
    if delta < 0:
        return "just now"
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    if delta < 86400 * 30:
        return f"{int(delta // 86400)}d ago"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _safe_load_state(loader, *args: Any, default: Any) -> Any:
    """Read a store, returning default on any error so the monitor never crashes.

    State files may be missing, corrupted, or partially written during
    a parallel run. The monitor should never propagate these failures.
    """
    try:
        return loader(*args)
    except Exception:
        return default


def _collect_jobs(cfg, run_store: RunStore, cache_store: CacheStore) -> tuple[list[JobView], dict[str, int]]:
    """Build JobViews for every configured branch.

    Returns the per-branch views and counters used by the Pipeline
    Overview panel.
    """
    jobs: list[JobView] = []
    counters = {"total": 0, "completed": 0, "running": 0, "failed": 0, "queued": 0}

    branches = list(cfg.feature.branches or cfg.feature.heavy_branches or [])
    if not branches:
        return jobs, counters

    counters["total"] = len(branches)
    cache_hits = 0
    cache_misses = 0

    for branch in branches:
        latest = run_store.latest_for_branch(branch)
        cache_entry = cache_store.get(branch)
        if cache_entry is not None and cfg.feature.cache:
            cache_status = "HIT"
            cache_hits += 1
        else:
            cache_status = "MISS" if cfg.feature.cache else "N/A"
            cache_misses += 1

        if latest is None:
            jobs.append(
                JobView(
                    branch=branch,
                    state="queued",
                    elapsed_seconds=0.0,
                    gpu=None,
                    cache=cache_status,
                    kernel_slug="",
                )
            )
            counters["queued"] += 1
            continue

        if latest.state == "running":
            started = latest.started_at
            elapsed = max(0.0, time.time() - started) if started else 0.0
        elif latest.finished_at and latest.started_at:
            elapsed = max(0.0, latest.finished_at - latest.started_at)
        else:
            elapsed = 0.0

        jobs.append(
            JobView(
                branch=branch,
                state=latest.state,
                elapsed_seconds=elapsed,
                gpu=latest.gpu,
                cache=cache_status,
                kernel_slug=latest.kernel_slug,
            )
        )

        if latest.state == "complete":
            counters["completed"] += 1
        elif latest.state in {"error", "timeout"}:
            counters["failed"] += 1
        elif latest.state == "running":
            counters["running"] += 1
        else:
            counters["queued"] += 1

    return jobs, counters


def _collect_artifacts(cfg, run_store: RunStore) -> list[ArtifactView]:
    """One row per branch that produced an artifact, sorted newest first."""
    views: list[ArtifactView] = []
    for branch in list(cfg.feature.branches or []):
        latest = run_store.latest_for_branch(branch)
        if latest is None or latest.artifact_path is None:
            continue
        path = Path(latest.artifact_path)
        try:
            size = path.stat().st_size if path.exists() else 0
        except OSError:
            size = 0
        ts = latest.finished_at or latest.started_at
        views.append(
            ArtifactView(
                name=path.name,
                size_bytes=size,
                timestamp=ts,
            )
        )
    views.sort(key=lambda a: a.timestamp, reverse=True)
    return views


def _collect_latest_submission(store: SubmissionStore) -> SubmissionView | None:
    rec = store.latest()
    if rec is None:
        return None
    return SubmissionView(
        competition=rec.competition,
        score=rec.score,
        rank=rec.rank,
        submission_id=rec.submission_id or "—",
        timestamp=rec.submitted_at,
    )


def _collect_best_submission(store: SubmissionStore) -> BestSubmissionView | None:
    recs = [r for r in store.all() if r.score is not None]
    if not recs:
        return None
    best = max(recs, key=lambda r: r.score)  # type: ignore[arg-type,return-value]
    return BestSubmissionView(
        score=best.score,
        rank=best.rank,
        git_commit=best.git_commit,
        experiment_id=best.experiment_id,
        competition=best.competition,
        timestamp=best.submitted_at,
    )


def collect_snapshot(project_root: Path | None = None) -> MonitorSnapshot:
    """Read all stores, build a snapshot. Never raises.

    Empty/corrupt state is handled gracefully so the dashboard always
    renders, even on a freshly-initialized project.
    """
    root = project_root or Path.cwd()
    config_path = root / "kaggle.toml"
    cfg = _safe_load_state(load_config, config_path, default=None)
    if cfg is None:
        cfg = type("EmptyCfg", (), {})()  # type: ignore[assignment]

    user = "—"
    try:
        creds = credentials.load()
        user = creds.username
    except Exception:
        pass

    project_name = getattr(getattr(cfg, "project", None), "name", None) or root.name

    run_store = _safe_load_state(RunStore, root, default=RunStore(root))
    cache_store = _safe_load_state(CacheStore, root, default=CacheStore(root))
    submission_store = _safe_load_state(SubmissionStore, root, default=SubmissionStore(root))
    experiment_store = _safe_load_state(ExperimentStore, root, default=ExperimentStore(root))
    feature_store = _safe_load_state(FeatureStore, root, default=FeatureStore(root))

    jobs, counters = _collect_jobs(cfg, run_store, cache_store)
    artifacts = _collect_artifacts(cfg, run_store)

    # Manifests = count of files in .kagglepipe/manifests/
    try:
        manifest_count = sum(1 for _ in manifests_dir(root).glob("*.json"))
    except Exception:
        manifest_count = 0

    # Cache hits/misses from jobs
    cache_hits = sum(1 for j in jobs if j.cache == "HIT")
    cache_misses = sum(1 for j in jobs if j.cache == "MISS")

    try:
        registered_features = len(feature_store.all())
    except Exception:
        registered_features = 0

    try:
        total_experiments = len(experiment_store.all())
    except Exception:
        total_experiments = 0

    return MonitorSnapshot(
        project_name=project_name,
        user=user,
        collected_at=time.time(),
        jobs=jobs,
        artifacts=artifacts,
        latest_submission=_collect_latest_submission(submission_store),
        best_submission=_collect_best_submission(submission_store),
        total_branches=counters["total"],
        completed=counters["completed"],
        running=counters["running"],
        failed=counters["failed"],
        queued=counters["queued"],
        total_experiments=total_experiments,
        total_manifests=manifest_count,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        registered_features=registered_features,
    )


def render_static(snapshot: MonitorSnapshot) -> str:
    """Render the snapshot as a static Rich string.

    Useful for non-interactive contexts (CI logs, `cat`-style
    inspection) and for the `--once` flag.
    """
    from rich.console import Console
    from io import StringIO

    from kagglepipe.commands.monitor import build_layout

    layout = build_layout(snapshot)
    buf = StringIO()
    console = Console(file=buf, width=120, force_terminal=True, color_system="truecolor")
    console.print(layout)
    return buf.getvalue()
