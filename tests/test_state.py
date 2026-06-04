"""Tests for the shared state layer (state.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def test_state_dir_creates_directory(tmp_path: Path) -> None:
    d = state_dir(tmp_path)
    assert d.exists()
    assert d.name == ".kagglepipe"


def test_run_store_add_and_latest(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    rec = RunRecord(branch="a", kernel_slug="u/a", state="complete")
    store.add(rec)
    latest = store.latest_for_branch("a")
    assert latest is not None
    assert latest.state == "complete"


def test_run_store_is_branch_successful(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    assert store.is_branch_successful("a") is False
    store.add(RunRecord(branch="a", kernel_slug="u/a", state="complete", artifact_path="/tmp/a.parquet"))
    assert store.is_branch_successful("a") is True
    store.add(RunRecord(branch="a", kernel_slug="u/a", state="error", error="boom"))
    assert store.is_branch_successful("a") is False  # latest is error


def test_run_store_failed_or_incomplete(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.add(RunRecord(branch="a", kernel_slug="u/a", state="complete", artifact_path="/x"))
    store.add(RunRecord(branch="b", kernel_slug="u/b", state="error"))
    store.add(RunRecord(branch="c", kernel_slug="u/c", state="timeout"))
    # Branches with no record at all are also "incomplete" (not started).
    out = store.failed_or_incomplete(["a", "b", "c", "d"])
    assert out == ["b", "c", "d"]


def test_run_store_persists_across_instances(tmp_path: Path) -> None:
    s1 = RunStore(tmp_path)
    s1.add(RunRecord(branch="a", kernel_slug="u/a", state="complete"))
    s2 = RunStore(tmp_path)
    assert s2.latest_for_branch("a") is not None
    assert s2.latest_for_branch("a").state == "complete"


def test_run_store_update(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.add(RunRecord(branch="a", kernel_slug="u/a", state="running"))
    store.update("a", "u/a", state="complete", artifact_path="/x.parquet")
    latest = store.latest_for_branch("a")
    assert latest.state == "complete"
    assert latest.artifact_path == "/x.parquet"


def test_submission_store_add_and_latest(tmp_path: Path) -> None:
    store = SubmissionStore(tmp_path)
    store.add(SubmissionRecord(competition="c1", file_path="/a.csv", message="v1"))
    store.add(SubmissionRecord(competition="c2", file_path="/b.csv", message="v2"))
    assert store.latest().competition == "c2"
    assert store.latest("c1").file_path == "/a.csv"


def test_experiment_store_add_and_get(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path)
    rec = ExperimentRecord(id="e1", feature_branches=["a", "b"], score=0.95)
    store.add(rec)
    assert store.get("e1") is not None
    assert store.get("e1").score == 0.95


def test_feature_store_latest(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path)
    store.add(FeatureRecord(name="user", dataset_slug="u/user", version=1, artifact_path="/a", branch="user"))
    store.add(FeatureRecord(name="user", dataset_slug="u/user", version=2, artifact_path="/b", branch="user"))
    latest = store.latest("user")
    assert latest.version == 2
    assert latest.artifact_path == "/b"
    all_user = store.get("user")
    assert len(all_user) == 2
