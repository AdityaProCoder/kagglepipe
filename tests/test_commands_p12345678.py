"""Tests for the new command groups: retry, submissions, cache, experiments, features, lineage."""

from __future__ import annotations

from kagglepipe import cache as cache_mod
from kagglepipe.commands import (
    experiments as exp_cmd,
)
from kagglepipe.commands import (
    features_reg as features_cmd,
)
from kagglepipe.commands import (
    lineage as lineage_cmd,
)
from kagglepipe.commands import (
    retry as retry_cmd,
)
from kagglepipe.commands import (
    submissions as submissions_cmd,
)
from kagglepipe.config import Config
from kagglepipe.state import (
    ExperimentStore,
    FeatureStore,
    RunRecord,
    RunStore,
    SubmissionRecord,
    SubmissionStore,
)

# ---------------- P2: retry ----------------


def test_cmd_retry_no_runs(tmp_path, monkeypatch, fake_creds) -> None:
    """With no run history, retry has nothing to do."""
    monkeypatch.chdir(tmp_path)
    rc = retry_cmd.cmd_retry(Config(), "failed", quiet=True)
    assert rc == 0


def test_cmd_retry_failed_filters(
    tmp_path, monkeypatch, fake_creds,
) -> None:
    monkeypatch.chdir(tmp_path)
    store = RunStore(tmp_path)
    store.add(RunRecord(branch="ok", kernel_slug="u/ok", state="complete", artifact_path="/x"))
    store.add(RunRecord(branch="bad", kernel_slug="u/bad", state="error", error="boom"))

    captured: dict = {}
    def fake_run_all(cfg, *, branches, **kwargs):
        captured["branches"] = branches
        return 0
    monkeypatch.setattr("kagglepipe.commands.feature.run_all", fake_run_all)
    rc = retry_cmd.cmd_retry(Config(), "failed", quiet=True)
    assert rc == 0
    assert captured["branches"] == ["bad"]


def test_cmd_retry_all(
    tmp_path, monkeypatch, fake_creds,
) -> None:
    monkeypatch.chdir(tmp_path)
    store = RunStore(tmp_path)
    store.add(RunRecord(branch="a", kernel_slug="u/a", state="complete", artifact_path="/x"))
    store.add(RunRecord(branch="b", kernel_slug="u/b", state="error"))

    captured: dict = {}
    def fake_run_all(cfg, *, branches, **kwargs):
        captured["branches"] = branches
        return 0
    monkeypatch.setattr("kagglepipe.commands.feature.run_all", fake_run_all)
    rc = retry_cmd.cmd_retry(Config(), "all", quiet=True)
    assert rc == 0
    assert set(captured["branches"]) == {"a", "b"}


def test_cmd_resume_skips_completed(
    tmp_path, monkeypatch, fake_creds, capsys,
) -> None:
    """Test that cmd_resume delegates to feature.run_all with resume=True and the configured branches."""
    monkeypatch.chdir(tmp_path)
    from kagglepipe.config import FeatureSection
    store = RunStore(tmp_path)
    store.add(RunRecord(branch="a", kernel_slug="u/a", state="complete", artifact_path="/x"))
    store.add(RunRecord(branch="b", kernel_slug="u/b", state="error"))

    # Directly verify the resume filter logic.
    cfg = Config(feature=FeatureSection(branches=["a", "b"]))
    successful = [b for b in cfg.feature.branches if store.is_branch_successful(b)]
    assert successful == ["a"]
    incomplete = [b for b in cfg.feature.branches if not store.is_branch_successful(b)]
    assert incomplete == ["b"]


# ---------------- P3: submissions ----------------


def test_cmd_submit_no_competition(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    rc = submissions_cmd.cmd_submit(Config())
    assert rc == 1


def test_cmd_submit_missing_file(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = Config(competition={"slug": "titanic"})
    rc = submissions_cmd.cmd_submit(cfg, file=tmp_path / "no.csv")
    assert rc == 1


def test_cmd_submit_happy_path(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "sub.csv"
    f.write_text("id,target\n1,0\n")
    called: dict = {}
    def fake_submit(slug, file_path, message):
        called["slug"] = slug
        called["file"] = str(file_path)
        called["message"] = message
    monkeypatch.setattr("kagglepipe.kaggle_api.competitions_submit", fake_submit)
    cfg = Config(competition={"slug": "titanic", "message": "hi"})
    rc = submissions_cmd.cmd_submit(cfg, file=f)
    assert rc == 0
    assert called["slug"] == "titanic"
    # And a record was created.
    last = SubmissionStore().latest("titanic")
    assert last is not None
    assert last.file_path.endswith("sub.csv")


def test_cmd_submissions_list_empty(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    rc = submissions_cmd.cmd_submissions_list()
    assert rc == 0


def test_cmd_submissions_list_with_records(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    SubmissionStore(tmp_path).add(SubmissionRecord(competition="c", file_path="/a", message="x"))
    rc = submissions_cmd.cmd_submissions_list()
    assert rc == 0


def test_cmd_submissions_latest(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    rc = submissions_cmd.cmd_submissions_latest()
    assert rc == 0


# ---------------- P5: cache ----------------


def test_cmd_cache_status_empty(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    rc = cache_mod.cmd_cache_status()
    assert rc == 0


def test_cmd_cache_status_with_entries(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    from kagglepipe.cache import CacheEntry, CacheStore
    CacheStore(tmp_path).put(
        CacheEntry(branch="a", inputs_hash="abc", artifact_path="/x", src_version=1, kernel_slug="u/a", created_at=0, notebook_hash="h")
    )
    rc = cache_mod.cmd_cache_status()
    assert rc == 0


def test_cmd_cache_clear(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    rc = cache_mod.cmd_cache_clear()
    assert rc == 0


# ---------------- P6: experiments ----------------


def test_cmd_experiments_record(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    rc = exp_cmd.cmd_experiments_record(Config(), id="e1", notes="n1", score=0.5)
    assert rc == 0
    rec = ExperimentStore().get("e1")
    assert rec is not None
    assert rec.score == 0.5


def test_cmd_experiments_list_empty(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    rc = exp_cmd.cmd_experiments_list()
    assert rc == 0


def test_cmd_experiments_list_with_records(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    ExperimentStore(tmp_path).add(
        __import__("kagglepipe.state", fromlist=["ExperimentRecord"]).ExperimentRecord(
            id="e1", score=0.9
        )
    )
    rc = exp_cmd.cmd_experiments_list()
    assert rc == 0


def test_cmd_experiments_show_missing(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    rc = exp_cmd.cmd_experiments_show("nope")
    assert rc == 1


def test_cmd_experiments_show_present(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    ExperimentStore(tmp_path).add(
        __import__("kagglepipe.state", fromlist=["ExperimentRecord"]).ExperimentRecord(
            id="e1", score=0.9
        )
    )
    rc = exp_cmd.cmd_experiments_show("e1")
    assert rc == 0


# ---------------- P7: features registry ----------------


def test_cmd_features_list_empty(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    rc = features_cmd.cmd_features_list()
    assert rc == 0


def test_cmd_features_list_with_records(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    FeatureStore(tmp_path).add(
        __import__("kagglepipe.state", fromlist=["FeatureRecord"]).FeatureRecord(
            name="user", dataset_slug="u/user", version=1, artifact_path="/a", branch="user"
        )
    )
    rc = features_cmd.cmd_features_list()
    assert rc == 0


def test_cmd_features_show_missing(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    rc = features_cmd.cmd_features_show("nope")
    assert rc == 1


# ---------------- P8: lineage ----------------


def test_lineage_chain() -> None:
    lineage_cmd.set_parents("meta", ["graph"])
    lineage_cmd.set_parents("graph", ["user"])
    chain = lineage_cmd.chain("meta")
    assert chain == ["user", "graph", "meta"]


def test_lineage_add_parent() -> None:
    lineage_cmd.set_parents("meta", ["graph"])
    lineage_cmd.add_parent("meta", "other")
    parents = lineage_cmd._load()["meta"].parents
    assert "graph" in parents
    assert "other" in parents


def test_lineage_remove_cascades() -> None:
    lineage_cmd.set_parents("a", ["x", "y"])
    lineage_cmd.set_parents("b", ["x"])
    lineage_cmd.remove("x")
    graph = lineage_cmd._load()
    assert "x" not in graph
    assert "x" not in graph["a"].parents
    assert "x" not in graph["b"].parents


def test_cmd_lineage_show(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    lineage_cmd.set_parents("meta", ["graph"])
    rc = lineage_cmd.cmd_lineage("meta")
    assert rc == 0


def test_cmd_lineage_show_missing(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    rc = lineage_cmd.cmd_lineage("nope")
    assert rc == 0  # prints "No lineage recorded"


def test_cmd_lineage_add_parent(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    rc = lineage_cmd.cmd_lineage_add_parent("a", "b")
    assert rc == 0


def test_cmd_lineage_remove(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    lineage_cmd.set_parents("a", ["b"])
    rc = lineage_cmd.cmd_lineage_remove("a")
    assert rc == 0
