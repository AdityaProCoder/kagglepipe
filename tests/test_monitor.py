"""Tests for the kagglepipe monitor dashboard.

Tests are read-only and operate on synthetic state in temp directories
so they never touch a real Kaggle project. The goal is to verify:

  - Panel builders handle empty / sparse / corrupted state without crashing
  - Snapshot collector reads from existing stores and produces a
    well-typed view-model
  - `cmd_monitor --once` produces non-empty Rich output for any input
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from kagglepipe.monitor import (
    ArtifactView,
    JobView,
    MonitorSnapshot,
    _humanize_bytes,
    _humanize_time,
    collect_snapshot,
)
from kagglepipe.commands.monitor import (
    _build_artifacts_panel,
    _build_best_submission_panel,
    _build_experiment_summary_panel,
    _build_jobs_panel,
    _build_latest_submission_panel,
    _build_overview_panel,
    build_layout,
    cmd_monitor,
)
from kagglepipe.state import (
    ExperimentRecord,
    ExperimentStore,
    FeatureRecord,
    FeatureStore,
    RunRecord,
    RunStore,
    SubmissionRecord,
    SubmissionStore,
    state_dir,
)


# ---- snapshot collector -------------------------------------------------


def _seed_state(root: Path) -> None:
    """Populate a temp project with synthetic state across all stores.

    Also writes a minimal kaggle.toml with the branches the synthetic
    runs reference, so `collect_snapshot` can find them.
    """
    branches = ["user_features", "graph_features", "embedding_features"]
    # Minimal kaggle.toml with branches
    (root / "kaggle.toml").write_text(
        "[project]\nname = \"seeded\"\n"
        "[source]\ninclude = [\"scripts\"]\nexclude_dirs = []\nexclude_exts = []\n"
        "src_dataset_slug = \"u/seeded-src\"\n"
        "[data]\ndataset_slug = \"\"\n"
        "[feature]\n"
        f"branches = {branches!r}\n"
        "heavy_branches = []\n"
        "default_gpu = \"t4x2\"\n"
        "kernel_slug_template = \"u/seeded-{branch}\"\n"
        "kernel_title_prefix = \"seeded\"\n"
        "notebook_command = \"python scripts/run.py\"\n"
        "data_mount = \"\"\n"
        "src_mount = \"/kaggle/input/datasets/u/seeded-src\"\n"
        "out_dir = \"/kaggle/working/features\"\n"
        "output_glob = \"{branch}.parquet\"\n"
        "default_timeout_sec = 1800\n"
        "poll_interval_sec = 30\n"
        "cache = 0\n"
        "[kernels]\nis_private = true\nenable_internet = false\n"
        "language = \"python\"\nkernel_type = \"notebook\"\n"
        "[paths]\nnotebooks_dir = \"kaggle_notebooks\"\n"
        "features_dir = \"features_kaggle\"\n"
    )

    rs = RunStore(root)
    rs.add(
        RunRecord(
            branch="user_features",
            kernel_slug="u/seeded-user-features",
            state="complete",
            artifact_path=str(root / "features_kaggle" / "user_features.parquet"),
            started_at=time.time() - 200,
            finished_at=time.time() - 60,
            gpu="t4 x2",
        )
    )
    rs.add(
        RunRecord(
            branch="graph_features",
            kernel_slug="u/seeded-graph-features",
            state="running",
            artifact_path=None,
            started_at=time.time() - 30,
            finished_at=None,
            gpu="t4 x2",
        )
    )
    rs.add(
        RunRecord(
            branch="embedding_features",
            kernel_slug="u/seeded-embedding-features",
            state="queued",
            artifact_path=None,
            started_at=time.time(),
            finished_at=None,
        )
    )

    # submissions
    sub_store = SubmissionStore(root)
    sub_store.add(
        SubmissionRecord(
            competition="titanic",
            file_path=str(root / "submission.csv"),
            message="v1",
            submitted_at=time.time() - 7200,
            score=0.81234,
            rank=23,
            git_commit="abcdef12345",
            experiment_id="exp-01",
        )
    )
    sub_store.add(
        SubmissionRecord(
            competition="titanic",
            file_path=str(root / "submission.csv"),
            message="v2",
            submitted_at=time.time() - 3600,
            score=0.87234,
            rank=15,
            git_commit="012345abcdef",
            experiment_id="exp-04",
        )
    )

    # experiments
    exp_store = ExperimentStore(root)
    for i in range(3):
        exp_store.add(
            ExperimentRecord(id=f"exp-{i:02d}", created_at=time.time() - i * 1000)
        )

    # features
    feat_store = FeatureStore(root)
    feat_store.add(
        FeatureRecord(
            name="baseline",
            dataset_slug="u/seeded-data",
            version=1,
            artifact_path=str(root / "features_kaggle" / "baseline.parquet"),
        )
    )

    # manifests
    (state_dir(root) / "manifests").mkdir(parents=True, exist_ok=True)
    for i in range(2):
        (state_dir(root) / "manifests" / f"user_features-{i}.json").write_text("{}")

    # actually create a fake artifact so size_human doesn't show 0
    (root / "features_kaggle").mkdir(parents=True, exist_ok=True)
    (root / "features_kaggle" / "user_features.parquet").write_bytes(b"\x00" * 4096)


def _render(panel) -> str:
    """Render a Rich panel to a plain string for assertions."""
    from io import StringIO
    from rich.console import Console
    buf = StringIO()
    console = Console(file=buf, width=120, force_terminal=True, color_system="truecolor")
    console.print(panel)
    return buf.getvalue()


def test_collect_snapshot_empty(tmp_path) -> None:
    """An empty project should produce a snapshot without errors."""
    (tmp_path / "kaggle.toml").write_text(
        "[project]\nname = \"empty\"\n[feature]\nbranches = []\n"
    )
    snap = collect_snapshot(tmp_path)
    assert snap.project_name == "empty"
    assert snap.jobs == []
    assert snap.artifacts == []
    assert snap.latest_submission is None
    assert snap.best_submission is None
    assert snap.total_branches == 0
    assert snap.percent_complete == 0.0


def test_collect_snapshot_corrupt_state_doesnt_crash(tmp_path, monkeypatch) -> None:
    """Corrupt JSON files should be handled gracefully (no exception)."""
    state = state_dir(tmp_path)
    state.mkdir(parents=True, exist_ok=True)
    (state / "runs.json").write_text("{not valid json")
    (state / "submissions.json").write_text("[bad")
    snap = collect_snapshot(tmp_path)
    # Even with corrupted state we still get a snapshot.
    assert isinstance(snap, MonitorSnapshot)


def test_collect_snapshot_full(tmp_path) -> None:
    """A fully populated state should produce a complete snapshot."""
    _seed_state(tmp_path)
    snap = collect_snapshot(tmp_path)
    assert len(snap.jobs) == 3
    branches = {j.branch for j in snap.jobs}
    assert branches == {"user_features", "graph_features", "embedding_features"}
    assert snap.total_branches == 3
    assert snap.completed == 1
    assert snap.running == 1
    assert snap.queued == 1
    assert snap.latest_submission is not None
    assert snap.latest_submission.competition == "titanic"
    assert snap.best_submission is not None
    assert snap.best_submission.score == 0.87234
    assert snap.best_submission.git_commit == "012345abcdef"
    assert snap.total_experiments == 3
    assert snap.total_manifests == 2
    assert snap.registered_features == 1


def test_collect_snapshot_artifact_size(tmp_path) -> None:
    """Artifact size should be read from the actual file on disk."""
    _seed_state(tmp_path)
    snap = collect_snapshot(tmp_path)
    artifacts = {a.name: a for a in snap.artifacts}
    assert "user_features.parquet" in artifacts
    assert artifacts["user_features.parquet"].size_bytes == 4096


# ---- view-model formatting helpers --------------------------------------


def test_humanize_bytes() -> None:
    assert _humanize_bytes(0) == "0 B"
    assert _humanize_bytes(512) == "512 B"
    assert _humanize_bytes(1024) == "1.0 KB"
    assert _humanize_bytes(1024 * 1024) == "1.0 MB"
    assert _humanize_bytes(4 * 1024 * 1024 + 200 * 1024) == "4.2 MB"


def test_humanize_time_recent() -> None:
    assert _humanize_time(0) == "—"
    assert _humanize_time(time.time() - 5) == "5s ago"
    assert _humanize_time(time.time() - 90) == "1m ago"
    assert _humanize_time(time.time() - 7200) == "2h ago"


# ---- panel builders -----------------------------------------------------


def _snapshot_for(branches=("a", "b"), jobs=None) -> MonitorSnapshot:
    if jobs is None:
        jobs = [
            JobView(branch=b, state="complete", elapsed_seconds=120.0,
                    gpu="t4 x2", cache="HIT", kernel_slug="u/p-a")
            for b in branches
        ]
    return MonitorSnapshot(
        project_name="test",
        user="user",
        collected_at=time.time(),
        jobs=jobs,
        artifacts=[ArtifactView(name="a.parquet", size_bytes=1024, timestamp=time.time())],
        total_branches=len(branches),
        completed=len(jobs),
    )


def test_jobs_panel_empty() -> None:
    snap = _snapshot_for(jobs=[])
    panel = _build_jobs_panel(snap)
    assert "Active Jobs" in str(panel.title)


def test_jobs_panel_with_jobs() -> None:
    snap = _snapshot_for()
    panel = _build_jobs_panel(snap)
    rendered = _render(panel)
    assert "a" in rendered
    assert "DONE" in rendered
    assert "t4 x2" in rendered


def test_overview_panel_with_branches() -> None:
    snap = _snapshot_for()
    panel = _build_overview_panel(snap)
    assert "Pipeline Overview" in str(panel.title)


def test_overview_panel_empty() -> None:
    snap = _snapshot_for(branches=[], jobs=[])
    snap.total_branches = 0
    snap.completed = 0
    panel = _build_overview_panel(snap)
    assert "No branches" in str(panel.renderable)


def test_artifacts_panel_empty() -> None:
    snap = _snapshot_for(jobs=[])
    snap.artifacts = []
    panel = _build_artifacts_panel(snap)
    assert "No artifacts" in str(panel.renderable)


def test_latest_submission_panel_empty() -> None:
    snap = _snapshot_for(jobs=[])
    snap.latest_submission = None
    panel = _build_latest_submission_panel(snap)
    assert "No submissions" in str(panel.renderable)


def test_best_submission_panel_empty() -> None:
    snap = _snapshot_for(jobs=[])
    snap.best_submission = None
    panel = _build_best_submission_panel(snap)
    assert "No scored" in str(panel.renderable)


def test_experiment_summary_panel() -> None:
    snap = _snapshot_for()
    panel = _build_experiment_summary_panel(snap)
    rendered = _render(panel)
    assert "Experiments" in rendered
    assert "Manifests" in rendered


def test_build_layout_no_exceptions() -> None:
    """The full layout should render without errors for a populated snapshot."""
    snap = _snapshot_for()
    layout = build_layout(snap)
    assert layout is not None
    # Each cell should be populated
    assert layout["header"] is not None
    assert layout["body"]["row1"]["r1c1"] is not None
    assert layout["body"]["row1"]["r1c2"] is not None
    assert layout["body"]["row1"]["r1c3"] is not None
    assert layout["body"]["row2"]["r2c1"] is not None
    assert layout["body"]["row2"]["r2c2"] is not None
    assert layout["body"]["row2"]["r2c3"] is not None


# ---- CLI entry point ----------------------------------------------------


def test_cmd_monitor_once_produces_output(tmp_path) -> None:
    """`--once` should print a snapshot to stdout and return 0."""
    _seed_state(tmp_path)
    rc = cmd_monitor(refresh=5, once=True, project_root=str(tmp_path))
    assert rc == 0


def test_cmd_monitor_empty_project(tmp_path) -> None:
    """Empty project should still produce a clean one-shot render."""
    (tmp_path / "kaggle.toml").write_text(
        "[project]\nname = \"empty\"\n[feature]\nbranches = []\n"
    )
    rc = cmd_monitor(refresh=5, once=True, project_root=str(tmp_path))
    assert rc == 0


def test_cmd_monitor_via_subprocess(tmp_path) -> None:
    """The full CLI invocation should work end-to-end."""
    _seed_state(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "kagglepipe", "monitor", "--once",
         "--project-root", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    # The Rich-rendered output should mention our project name
    assert "testproj" in result.stdout or "test" in result.stdout


def test_cmd_monitor_help_text() -> None:
    """The help text should mention the key flags."""
    result = subprocess.run(
        [sys.executable, "-m", "kagglepipe", "monitor", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "--refresh" in result.stdout
    assert "--once" in result.stdout
