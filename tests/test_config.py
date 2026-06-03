"""Tests for kaggle.toml loading and env overrides."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kagglepipe import config as cfg_mod


def test_load_returns_defaults_when_file_missing(tmp_path: Path) -> None:
    cfg = cfg_mod.load(tmp_path / "missing.toml")
    assert cfg.feature.default_gpu == "none"
    assert cfg.source.include == ["src", "configs", "scripts", "pyproject.toml", "README.md"]
    assert cfg.config_path is None


def test_load_reads_toml(tmp_path: Path) -> None:
    p = tmp_path / "kaggle.toml"
    p.write_text(
        """
[project]
name = "myproj"

[source]
src_dataset_slug = "{username}/myproj-src"
exclude_dirs = [".venv", "data", "node_modules"]

[feature]
branches = ["a", "b"]
heavy_branches = ["a"]
default_gpu = "t4x2"
kernel_slug_template = "{username}/mp-{branch}"
kernel_title_prefix = "mp"
notebook_command = "python run.py --out {out_dir}"
"""
    )
    cfg = cfg_mod.load(p)
    assert cfg.project.name == "myproj"
    assert cfg.source.src_dataset_slug == "{username}/myproj-src"
    assert cfg.source.exclude_dirs == [".venv", "data", "node_modules"]
    assert cfg.feature.branches == ["a", "b"]
    assert cfg.feature.heavy_branches == ["a"]
    assert cfg.feature.default_gpu == "t4x2"
    assert cfg.feature.kernel_slug_template == "{username}/mp-{branch}"
    assert cfg.feature.kernel_title_prefix == "mp"
    assert cfg.feature.notebook_command == "python run.py --out {out_dir}"
    assert cfg.config_path == p


def test_load_invalid_toml_raises(tmp_path: Path) -> None:
    p = tmp_path / "kaggle.toml"
    p.write_text("this is not = = toml")
    with pytest.raises(ValueError, match="Invalid TOML"):
        cfg_mod.load(p)


def test_env_overrides_int(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    p = tmp_path / "kaggle.toml"
    p.write_text("[feature]\ndefault_timeout_sec = 60\n")
    monkeypatch.setenv("KAGGLEPIPE_FEATURE__DEFAULT_TIMEOUT_SEC", "9999")
    cfg = cfg_mod.load(p)
    assert cfg.feature.default_timeout_sec == 9999


def test_env_overrides_string(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    p = tmp_path / "kaggle.toml"
    p.write_text('[feature]\ndefault_gpu = "t4x2"\n')
    monkeypatch.setenv("KAGGLEPIPE_FEATURE__DEFAULT_GPU", "p100")
    cfg = cfg_mod.load(p)
    assert cfg.feature.default_gpu == "p100"


def test_env_overrides_list(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    p = tmp_path / "kaggle.toml"
    p.write_text('[feature]\nbranches = ["a", "b"]\n')
    monkeypatch.setenv("KAGGLEPIPE_FEATURE__BRANCHES", "x,y,z")
    cfg = cfg_mod.load(p)
    assert cfg.feature.branches == ["x", "y", "z"]


def test_env_overrides_bool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    p = tmp_path / "kaggle.toml"
    p.write_text("[kernels]\nis_private = true\n")
    monkeypatch.setenv("KAGGLEPIPE_KERNELS__IS_PRIVATE", "false")
    cfg = cfg_mod.load(p)
    assert cfg.kernels.is_private is False


def test_env_overrides_unknown_section_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p = tmp_path / "kaggle.toml"
    p.write_text("")
    monkeypatch.setenv("KAGGLEPIPE_BOGUS__FOO", "x")
    cfg = cfg_mod.load(p)
    # Just ensure no exception.
    assert cfg.feature.default_gpu == "none"


def test_to_dict_returns_serializable(tmp_path: Path) -> None:
    cfg = cfg_mod.load(tmp_path / "missing.toml")
    d = cfg_mod.to_dict(cfg)
    assert d["project"]["name"] == "kagglepipe"
    assert "feature" in d


def test_scaffold_creates_file(tmp_path: Path) -> None:
    target = tmp_path / "kaggle.toml"
    written = cfg_mod.scaffold(target, project_name="myproj")
    assert written == target
    text = target.read_text()
    assert "[project]" in text
    assert "[feature]" in text
    assert "myproj" in text


def test_scaffold_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "kaggle.toml"
    target.write_text("existing")
    with pytest.raises(FileExistsError):
        cfg_mod.scaffold(target, project_name="x")
