"""Tests for the source tarball builder."""

from __future__ import annotations

import tarfile
from pathlib import Path

from kagglepipe import tarball


def test_build_tarball_includes_listed_entries(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mypkg").mkdir()
    (tmp_path / "src" / "mypkg" / "__init__.py").write_text("")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "base.yaml").write_text("seed: 42")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run.py").write_text("print('x')")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")

    out = tmp_path / "src.tar.gz"
    tarball.build_tarball(
        src_root=tmp_path,
        dest=out,
        include=["src", "configs", "scripts", "pyproject.toml"],
        exclude_dirs=[],
        exclude_exts=[],
    )
    assert out.exists()
    with tarfile.open(out, "r:gz") as tf:
        names = tf.getnames()
    assert "src/mypkg/__init__.py" in names
    assert "configs/base.yaml" in names
    assert "scripts/run.py" in names
    assert "pyproject.toml" in names


def test_build_tarball_excludes_dirs(tmp_path: Path) -> None:
    for d in [".venv", "data", "models", ".git"]:
        (tmp_path / d).mkdir()
        (tmp_path / d / "skip.py").write_text("")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "keep.py").write_text("")

    out = tmp_path / "src.tar.gz"
    tarball.build_tarball(
        src_root=tmp_path,
        dest=out,
        include=["src", ".venv", "data", "models", ".git"],
        exclude_dirs=[".venv", "data", "models", ".git"],
        exclude_exts=[],
    )
    with tarfile.open(out, "r:gz") as tf:
        names = tf.getnames()
    assert "src/keep.py" in names
    assert not any(n.startswith(".venv/") for n in names)
    assert not any(n.startswith("data/") for n in names)
    assert not any(n.startswith("models/") for n in names)
    assert not any(n.startswith(".git/") for n in names)


def test_build_tarball_excludes_exts(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "keep.py").write_text("ok")
    (tmp_path / "src" / "big.parquet").write_bytes(b"x")
    (tmp_path / "src" / "model.pt").write_bytes(b"x")

    out = tmp_path / "src.tar.gz"
    tarball.build_tarball(
        src_root=tmp_path,
        dest=out,
        include=["src"],
        exclude_dirs=[],
        exclude_exts=[".parquet", ".pt"],
    )
    with tarfile.open(out, "r:gz") as tf:
        names = tf.getnames()
    assert "src/keep.py" in names
    assert "src/big.parquet" not in names
    assert "src/model.pt" not in names


def test_build_tarball_drops_pycache(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "keep.py").write_text("ok")
    (tmp_path / "src" / "__pycache__").mkdir()
    (tmp_path / "src" / "__pycache__" / "junk.pyc").write_bytes(b"x")

    out = tmp_path / "src.tar.gz"
    tarball.build_tarball(
        src_root=tmp_path, dest=out, include=["src"], exclude_dirs=[], exclude_exts=[]
    )
    with tarfile.open(out, "r:gz") as tf:
        names = tf.getnames()
    assert "src/keep.py" in names
    assert not any("__pycache__" in n for n in names)


def test_build_tarball_missing_entry_skipped(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("")

    out = tmp_path / "src.tar.gz"
    tarball.build_tarball(
        src_root=tmp_path,
        dest=out,
        include=["src", "nonexistent_dir"],
        exclude_dirs=[],
        exclude_exts=[],
    )
    with tarfile.open(out, "r:gz") as tf:
        names = tf.getnames()
    assert names == ["src", "src/a.py"]
