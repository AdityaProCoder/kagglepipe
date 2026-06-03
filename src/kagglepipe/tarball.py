"""Build a source tarball for upload to a Kaggle Dataset."""

from __future__ import annotations

import tarfile
from pathlib import Path


def build_tarball(
    src_root: Path,
    dest: Path,
    *,
    include: list[str],
    exclude_dirs: list[str],
    exclude_exts: list[str],
) -> Path:
    """Package the requested entries from `src_root` into a .tar.gz at `dest`.

    Args:
        src_root: Directory whose top-level entries will be packaged.
        dest: Output path (e.g., /tmp/src.tar.gz).
        include: Top-level entry names to include (e.g., ["src", "configs"]).
        exclude_dirs: Top-level directory names to skip entirely.
        exclude_exts: File extensions to skip (e.g., [".parquet", ".pt"]).

    Returns the `dest` path.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    src_root = src_root.resolve()
    exclude_dirs_set = set(exclude_dirs)
    exclude_exts_set = set(exclude_exts)

    def _filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = tarinfo.name.split("/")
        top = parts[0] if parts else ""
        if top in exclude_dirs_set:
            return None
        if Path(tarinfo.name).suffix in exclude_exts_set:
            return None
        # Blocklist of cache directory components anywhere in the path.
        for banned in ("__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"):
            if banned in parts:
                return None
        return tarinfo

    with tarfile.open(dest, "w:gz") as tf:
        for entry in sorted(include):
            full = src_root / entry
            if not full.exists():
                continue
            tf.add(str(full), arcname=entry, filter=_filter)
    return dest
