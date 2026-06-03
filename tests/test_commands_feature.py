"""Tests for the `feature` command."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from kagglepipe.commands import feature
from kagglepipe.config import Config, FeatureSection, KernelsSection, SourceSection


def _cfg(tmp_path: Path) -> Config:
    return Config(
        source=SourceSection(
            include=["scripts"],
            exclude_dirs=[],
            exclude_exts=[],
            src_dataset_slug="{username}/proj-src",
        ),
        feature=FeatureSection(
            branches=["a", "b"],
            heavy_branches=["a", "b"],
            default_gpu="t4x2",
            kernel_slug_template="{username}/proj-{branch}",
            kernel_title_prefix="proj",
            notebook_command="python run.py --out {out_dir}",
            data_mount="/kaggle/input/proj-data",
            src_mount="/kaggle/input/proj-src",
            out_dir="/kaggle/working/features",
            output_glob="{branch}.parquet",
            default_timeout_sec=600,
            poll_interval_sec=10,
        ),
        kernels=KernelsSection(is_private=True, enable_internet=True),
    )


def test_validate_branch_accepts_known() -> None:
    cfg = _cfg(Path("/tmp"))
    assert feature.validate_branch(cfg, "a") == "a"


def test_validate_branch_rejects_unknown_when_whitelist_set() -> None:
    cfg = _cfg(Path("/tmp"))
    with pytest.raises(ValueError, match="not in the whitelist"):
        feature.validate_branch(cfg, "z")


def test_validate_branch_passes_when_whitelist_empty() -> None:
    cfg = _cfg(Path("/tmp"))
    cfg.feature.branches = []
    assert feature.validate_branch(cfg, "anything") == "anything"


def test_gpu_instance_map() -> None:
    assert feature.GPU_INSTANCE_MAP["p100"] == "p100"
    assert feature.GPU_INSTANCE_MAP["t4x2"] == "t4 x2"
    assert feature.GPU_INSTANCE_MAP["none"] is None


def test_run_feature_writes_notebook_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_creds: Path
) -> None:
    cfg = _cfg(tmp_path)
    cwd = tmp_path / "project"
    cwd.mkdir()
    (cwd / "scripts").mkdir()
    (cwd / "scripts" / "run.py").write_text("print('hi')")

    # Stub the kaggle API: dataset exists -> version 2, then push, then status complete
    csv = subprocess.CompletedProcess(
        args=[], returncode=0, stdout='ref,title\nproj-src,title\n', stderr=""
    )
    push = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="pushed", stderr=""
    )
    status = subprocess.CompletedProcess(
        args=[], returncode=0, stdout='"complete"', stderr=""
    )
    output = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="downloaded", stderr=""
    )
    call_log: list[list[str]] = []

    def fake_run(args, **kwargs):
        call_log.append(args)
        if args[:2] == ["datasets", "list"]:
            return csv
        if args[:2] == ["kernels", "push"]:
            return push
        if args[:2] == ["kernels", "status"]:
            return status
        if args[:2] == ["kernels", "output"]:
            # Write a fake artifact to the dest path the runner was given.
            dest = Path(args[args.index("-p") + 1])
            (dest / "features").mkdir(parents=True, exist_ok=True)
            (dest / "features" / "a.parquet").write_bytes(b"PAR1")
            return output
        raise AssertionError(f"unexpected call: {args}")

    monkeypatch.setattr("kagglepipe.runner.run", fake_run)
    monkeypatch.chdir(cwd)

    rc = feature.run_feature(cfg, "a")
    assert rc == 0
    # Notebook was written
    nb_path = cwd / "kaggle_notebooks" / "extract_a.ipynb"
    assert nb_path.exists()
    nb = json.loads(nb_path.read_text())
    assert "testuser/proj-src" in nb["metadata"]["dataset_sources"]
    # Kernel metadata
    assert (cwd / "kaggle_notebooks" / "kernel-metadata.json").exists()
    kmd = json.loads((cwd / "kaggle_notebooks" / "kernel-metadata.json").read_text())
    assert kmd["id"] == "testuser/proj-a"
    assert kmd["title"] == "proj-a"
    # Artifact was copied
    assert (cwd / "features_kaggle" / "a.parquet").exists()
    assert (cwd / "features_kaggle" / "a.parquet").read_bytes() == b"PAR1"


def test_run_feature_returns_1_on_kernel_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_creds: Path
) -> None:
    cfg = _cfg(tmp_path)
    cwd = tmp_path / "project"
    cwd.mkdir()
    (cwd / "scripts").mkdir()
    monkeypatch.chdir(cwd)

    csv = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="ref,title\n", stderr=""
    )
    push = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    status = subprocess.CompletedProcess(
        args=[], returncode=0, stdout='"error: something failed"', stderr=""
    )

    def fake_run(args, **kwargs):
        if args[:2] == ["datasets", "list"]:
            return csv
        if args[:2] == ["kernels", "push"]:
            return push
        if args[:2] == ["kernels", "status"]:
            return status
        raise AssertionError(args)

    monkeypatch.setattr("kagglepipe.runner.run", fake_run)
    rc = feature.run_feature(cfg, "a")
    assert rc == 1


def test_run_all_uses_heavy_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_creds: Path
) -> None:
    cfg = _cfg(tmp_path)
    cwd = tmp_path / "project"
    cwd.mkdir()
    (cwd / "scripts").mkdir()
    monkeypatch.chdir(cwd)

    csv = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="ref,title\n", stderr=""
    )
    push = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    status = subprocess.CompletedProcess(
        args=[], returncode=0, stdout='"complete"', stderr=""
    )
    output = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="", stderr=""
    )

    def fake_run(args, **kwargs):
        if args[:2] == ["datasets", "list"]:
            return csv
        if args[:2] == ["kernels", "push"]:
            return push
        if args[:2] == ["kernels", "status"]:
            return status
        if args[:2] == ["kernels", "output"]:
            dest = Path(args[args.index("-p") + 1])
            (dest / "features").mkdir(parents=True, exist_ok=True)
            # args[2] is the kernel slug like "testuser/proj-a"; pick the last
            # component so the artifact name matches the output_glob.
            slug = args[2]
            branch = slug.rsplit("-", 1)[-1]
            (dest / "features" / f"{branch}.parquet").write_bytes(b"X")
            return output
        raise AssertionError(args)

    monkeypatch.setattr("kagglepipe.runner.run", fake_run)
    rc = feature.run_all(cfg)
    assert rc == 0
    # Both branches produced a parquet.
    assert (cwd / "features_kaggle" / "a.parquet").exists()
    assert (cwd / "features_kaggle" / "b.parquet").exists()
