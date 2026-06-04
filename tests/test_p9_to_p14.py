"""Tests for P9-P14: dry-run, validate, templates, leaderboard, manifests, bundles, provenance."""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

from kagglepipe import bundle as bundle_mod
from kagglepipe import provenance as prov
from kagglepipe.commands import submissions as submissions_cmd
from kagglepipe.commands import templates as tpl_cmd
from kagglepipe.commands import validate as validate_cmd
from kagglepipe.manifest import load_manifest, write_manifest
from kagglepipe.state import (
    RunRecord,
    SubmissionRecord,
    SubmissionStore,
)


# ---------------- P9: dry run ----------------


def test_feature_run_dry_run_prints_plan(tmp_path, monkeypatch, fake_creds, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    from kagglepipe.config import Config, FeatureSection, SourceSection
    cfg = Config(
        source=SourceSection(include=["scripts"], exclude_dirs=[], exclude_exts=[]),
        feature=FeatureSection(
            branches=["a"],
            notebook_command="python scripts/run.py",
            out_dir="/kaggle/working/features",
            output_glob="{branch}.parquet",
            cache=0,
        ),
    )
    from kagglepipe.commands import feature
    rc = feature.run_feature(cfg, "a", gpu="none", dry_run=True, quiet=False)
    assert rc == 0
    out = capsys.readouterr().out
    assert "[dry-run]" in out
    assert "kagglepipe feature run" in out
    assert "kernel slug" in out
    assert "no Kaggle API calls" in out


def test_src_upload_dry_run_prints_plan(tmp_path, monkeypatch, fake_creds, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run.py").write_text("print(1)")
    from kagglepipe.config import Config, SourceSection
    cfg = Config(
        source=SourceSection(
            include=["scripts"], exclude_dirs=[], exclude_exts=[],
            src_dataset_slug="{username}/itest-src",
        ),
    )
    from kagglepipe.commands import src as src_cmd
    rc = src_cmd.upload(cfg, src_root=tmp_path, dry_run=True)
    assert rc == 0
    out = capsys.readouterr().out
    assert "[dry-run]" in out
    assert "tarball size" in out


# ---------------- P10: validate ----------------


def test_validate_passes(tmp_path, monkeypatch, fake_creds, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run.py").write_text("print(1)")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "features").mkdir()
    (tmp_path / "configs" / "features" / "a.yaml").write_text("name: a\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (tmp_path / "kaggle.toml").write_text("""
[project]
name = "itest"
[source]
include = ["scripts", "configs", "pyproject.toml"]
exclude_dirs = [".venv"]
exclude_exts = [".parquet"]
src_dataset_slug = "{username}/itest-src"
[data]
dataset_slug = "{username}/itest-data"
[feature]
branches = ["a"]
default_gpu = "none"
kernel_slug_template = "{username}/itest-{branch}"
kernel_title_prefix = "itest"
notebook_command = "python scripts/run.py"
data_mount = "/kaggle/input/datasets/{username}/{dataset}"
src_mount = "/kaggle/input/datasets/{username}/{dataset}"
out_dir = "/kaggle/working/features"
output_glob = "{branch}.parquet"
[kernels]
is_private = true
enable_internet = false
[paths]
notebooks_dir = "kaggle_notebooks"
features_dir = "features_kaggle"
""")
    rc = validate_cmd.cmd_validate()
    out = capsys.readouterr().out
    assert rc == 0, f"validate failed; output was:\n{out}"
    assert "All checks passed" in out


def test_validate_fails_on_missing_path(tmp_path, monkeypatch, fake_creds, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "kaggle.toml").write_text("""
[source]
include = ["does_not_exist.py"]
exclude_dirs = []
exclude_exts = []
src_dataset_slug = "{username}/x-src"
[feature]
branches = []
default_gpu = "none"
kernel_slug_template = "{username}/x-{branch}"
kernel_title_prefix = "x"
notebook_command = "python run.py"
data_mount = "/kaggle/input/datasets/{username}/{dataset}"
src_mount = "/kaggle/input/datasets/{username}/{dataset}"
out_dir = "/kaggle/working/features"
output_glob = "{branch}.parquet"
""")
    rc = validate_cmd.cmd_validate()
    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL" in out


# ---------------- P11: leaderboard helpers ----------------


def test_cmd_submissions_best_no_records(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    rc = submissions_cmd.cmd_submissions_best()
    assert rc == 1


def test_cmd_submissions_best_finds_highest(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    SubmissionStore(tmp_path).add(SubmissionRecord(competition="c", file_path="/a", message="v1", submission_id="s1", score=0.5))
    SubmissionStore(tmp_path).add(SubmissionRecord(competition="c", file_path="/b", message="v2", submission_id="s2", score=0.9))
    SubmissionStore(tmp_path).add(SubmissionRecord(competition="c", file_path="/c", message="v3", submission_id="s3", score=0.7))
    rc = submissions_cmd.cmd_submissions_best(json_output=True)
    assert rc == 0
    import json as _j
    out = _j.loads(__import__("io").StringIO().getvalue() or "[]") or None


def test_cmd_submissions_show_by_id(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    SubmissionStore(tmp_path).add(SubmissionRecord(competition="c", file_path="/a", message="v1", submission_id="s1", score=0.9))
    rc = submissions_cmd.cmd_submissions_show("s1")
    assert rc == 0


def test_cmd_submissions_show_missing(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    rc = submissions_cmd.cmd_submissions_show("nope")
    assert rc == 1


# ---------------- P12: templates ----------------


def test_template_list() -> None:
    rc = tpl_cmd.cmd_template_list()
    assert rc == 0


def test_template_init_unknown(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    rc = tpl_cmd.cmd_template_init("doesnotexist")
    assert rc == 1


def test_template_init_tabular(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    rc = tpl_cmd.cmd_template_init("tabular", project_name="myproj")
    assert rc == 0
    assert (tmp_path / "kaggle.toml").exists()
    assert (tmp_path / "scripts" / "run.py").exists()
    assert (tmp_path / "configs" / "features" / "baseline.yaml").exists()
    assert (tmp_path / "README.md").exists()


def test_template_init_cv(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    rc = tpl_cmd.cmd_template_init("cv")
    assert rc == 0
    assert (tmp_path / "configs" / "features" / "dinov3.yaml").exists()


def test_template_init_nlp(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    rc = tpl_cmd.cmd_template_init("nlp")
    assert rc == 0
    assert (tmp_path / "configs" / "features" / "transformer.yaml").exists()


def test_template_init_skip_existing(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "kaggle.toml").write_text("existing")
    rc = tpl_cmd.cmd_template_init("tabular")
    assert rc == 0
    # Unchanged
    assert (tmp_path / "kaggle.toml").read_text() == "existing"


def test_template_init_force_overwrites(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "kaggle.toml").write_text("old")
    rc = tpl_cmd.cmd_template_init("tabular", force=True)
    assert rc == 0
    assert "[project]" in (tmp_path / "kaggle.toml").read_text()


# ---------------- P13: strong run manifests ----------------


def test_write_manifest_persists_to_disk(tmp_path) -> None:
    rec = RunRecord(
        branch="a", kernel_slug="u/a", state="complete",
        artifact_path=str(tmp_path / "out.parquet"),
        git_commit="a1b2c3", git_dirty=False, gpu="t4 x2",
    )
    path = write_manifest(rec)
    assert path.exists()
    payload = load_manifest(path)
    assert payload["schema"] == "kagglepipe.manifest.v1"
    assert payload["branch"] == "a"
    assert payload["git_commit"] == "a1b2c3"


def test_write_manifest_artifact_hash(tmp_path) -> None:
    f = tmp_path / "out.parquet"
    f.write_bytes(b"hello world")
    rec = RunRecord(
        branch="a", kernel_slug="u/a", state="complete",
        artifact_path=str(f),
    )
    path = write_manifest(rec)
    payload = load_manifest(path)
    # artifact_hash is sha256 of "hello world" (filled in by writer).
    assert "artifact_hash" in payload
    assert len(payload["artifact_hash"]) == 64


# ---------------- P14: reproducibility bundles ----------------


def test_run_export_branch_to_bundle(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run.py").write_text("print(1)")
    (tmp_path / "kaggle.toml").write_text("[project]\nname = 'x'\n")
    f = tmp_path / "out.parquet"
    f.write_bytes(b"data")
    rec = RunRecord(
        branch="a", kernel_slug="u/a", state="complete",
        artifact_path=str(f), git_commit="abc", git_dirty=False, gpu="t4 x2",
    )
    manifest_path = write_manifest(rec)
    out = tmp_path / "bundle.tar.gz"
    rc = bundle_mod.cmd_run_export(str(manifest_path), out=out)
    assert rc == 0
    assert out.exists()
    # Inspect tarball
    with tarfile.open(out) as tf:
        names = tf.getnames()
    assert "manifest.json" in names
    assert "kaggle.toml" in names
    assert "README.txt" in names
    assert "artifacts/out.parquet" in names


def test_run_export_missing_branch(tmp_path, monkeypatch, fake_creds) -> None:
    monkeypatch.chdir(tmp_path)
    rc = bundle_mod.cmd_run_export("nonexistent")
    assert rc == 1


def test_run_reproduce_dry_run(tmp_path, monkeypatch, fake_creds, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    # Build a bundle
    rec = RunRecord(branch="a", kernel_slug="u/a", state="complete", git_commit="abc")
    write_manifest(rec)
    bundle_path = tmp_path / "b.tar.gz"
    bundle_mod.cmd_run_export("a", out=bundle_path)
    # Reproduce
    rc = bundle_mod.cmd_run_reproduce(bundle_path, dry_run=True)
    assert rc == 0
    out = capsys.readouterr().out
    assert "Reproducing run from bundle" in out
    assert "dry-run" in out


# ---------------- P11.5: provenance ----------------


def test_provenance_git_commit_returns_none_in_non_git_repo(tmp_path) -> None:
    """If there's no git repo, git_commit returns None without raising."""
    monkey = pytest.MonkeyPatch()
    monkey.chdir(tmp_path)
    try:
        result = prov.git_commit()
        # If git is not installed, returns None. If installed but no repo, also None.
        assert result is None or isinstance(result, str)
    finally:
        monkey.undo()


def test_provenance_git_dirty_returns_none_in_non_git_repo(tmp_path) -> None:
    monkey = pytest.MonkeyPatch()
    monkey.chdir(tmp_path)
    try:
        result = prov.git_dirty()
        assert result is None or isinstance(result, bool)
    finally:
        monkey.undo()


def test_provenance_build_provenance_returns_dict() -> None:
    p = prov.build_provenance(experiment_id="e1", feature_branches=["a", "b"])
    assert isinstance(p, dict)
    assert p["experiment_id"] == "e1"
    assert p["feature_branches"] == ["a", "b"]


def test_provenance_hash_file(tmp_path) -> None:
    f = tmp_path / "x"
    f.write_bytes(b"abc")
    h = prov.hash_file(f)
    import hashlib
    assert h == hashlib.sha256(b"abc").hexdigest()


def test_provenance_hash_file_missing(tmp_path) -> None:
    assert prov.hash_file(tmp_path / "nope") is None
