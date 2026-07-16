"""Tests for the cache module (P5)."""

from __future__ import annotations

from pathlib import Path

from kagglepipe.cache import (
    CacheEntry,
    CacheStore,
    compute_inputs_hash,
    config_hash_for_branch,
)
from kagglepipe.config import Config, FeatureSection, KernelsSection, SourceSection


def _cfg() -> Config:
    return Config(
        source=SourceSection(include=["src"], exclude_dirs=[], exclude_exts=[]),
        feature=FeatureSection(
            notebook_command="python run.py",
            notebook_template="kagglepipe.templates.notebook_default",
            out_dir="/kaggle/working/features",
            output_glob="{branch}.parquet",
        ),
        kernels=KernelsSection(is_private=True, enable_internet=False),
    )


def test_compute_inputs_hash_stable() -> None:
    nb = {"cells": [{"cell_type": "code", "source": ["print(1)"]}]}
    kmd = {"id": "u/k", "language": "python"}
    h1 = compute_inputs_hash(
        branch="a", notebook=nb, kernel_metadata=kmd, src_version=1,
        output_glob="{branch}.parquet", config_hash="abc",
    )
    h2 = compute_inputs_hash(
        branch="a", notebook=nb, kernel_metadata=kmd, src_version=1,
        output_glob="{branch}.parquet", config_hash="abc",
    )
    assert h1 == h2


def test_compute_inputs_hash_changes_with_src_version() -> None:
    nb = {}
    kmd = {}
    h1 = compute_inputs_hash(branch="a", notebook=nb, kernel_metadata=kmd, src_version=1, output_glob="x", config_hash="x")
    h2 = compute_inputs_hash(branch="a", notebook=nb, kernel_metadata=kmd, src_version=2, output_glob="x", config_hash="x")
    assert h1 != h2


def test_compute_inputs_hash_changes_with_branch() -> None:
    nb = {}
    kmd = {}
    h1 = compute_inputs_hash(branch="a", notebook=nb, kernel_metadata=kmd, src_version=1, output_glob="x", config_hash="x")
    h2 = compute_inputs_hash(branch="b", notebook=nb, kernel_metadata=kmd, src_version=1, output_glob="x", config_hash="x")
    assert h1 != h2


def test_config_hash_for_branch_stable() -> None:
    cfg = _cfg()
    h1 = config_hash_for_branch(cfg, "a")
    h2 = config_hash_for_branch(cfg, "a")
    assert h1 == h2


def test_config_hash_for_branch_changes_when_command_changes() -> None:
    cfg1 = _cfg()
    cfg2 = _cfg()
    cfg2.feature.notebook_command = "python different.py"
    h1 = config_hash_for_branch(cfg1, "a")
    h2 = config_hash_for_branch(cfg2, "a")
    assert h1 != h2


def test_cache_store_put_and_get(tmp_path: Path) -> None:
    store = CacheStore(tmp_path)
    entry = CacheEntry(
        branch="a", inputs_hash="abc", artifact_path="/x.parquet",
        src_version=1, kernel_slug="u/a", created_at=0, notebook_hash="d",
    )
    store.put(entry)
    fetched = store.get("a")
    assert fetched is not None
    assert fetched.inputs_hash == "abc"


def test_cache_store_clear_all_and_one(tmp_path: Path) -> None:
    store = CacheStore(tmp_path)
    store.put(CacheEntry(branch="a", inputs_hash="x", artifact_path="/a", src_version=1, kernel_slug="u/a", created_at=0, notebook_hash="h"))
    store.put(CacheEntry(branch="b", inputs_hash="x", artifact_path="/b", src_version=1, kernel_slug="u/b", created_at=0, notebook_hash="h"))
    assert len(store.all()) == 2
    n = store.clear("a")
    assert n == 1
    assert len(store.all()) == 1
    n = store.clear()
    assert n == 1
    assert len(store.all()) == 0


def test_cache_store_persists(tmp_path: Path) -> None:
    s1 = CacheStore(tmp_path)
    s1.put(CacheEntry(branch="a", inputs_hash="x", artifact_path="/a", src_version=1, kernel_slug="u/a", created_at=0, notebook_hash="h"))
    s2 = CacheStore(tmp_path)
    assert s2.get("a") is not None
