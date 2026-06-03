"""src upload — package and upload the source as a Kaggle Dataset."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from kagglepipe import credentials
from kagglepipe import kaggle_api
from kagglepipe import notebook as nb_mod
from kagglepipe import runner, tarball
from kagglepipe.config import Config
from kagglepipe.slug import resolve_template


def upload(
    cfg: Config,
    *,
    src_root: Path | None = None,
    version: int | None = None,
    slug: str | None = None,
    quiet: bool = False,
) -> int:
    """Build a tarball of the configured `source.include` and upload it."""
    creds = credentials.load()
    root = (src_root or Path.cwd()).resolve()
    actual_slug = slug or resolve_template(cfg.source.src_dataset_slug, username=creds.username)
    if version is None:
        version = kaggle_api.get_next_version(actual_slug)
    if not quiet:
        print(f"Packaging {root} -> {actual_slug} v{version}")
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        tarball_path = tmp / "src.tar.gz"
        tarball.build_tarball(
            root,
            tarball_path,
            include=cfg.source.include,
            exclude_dirs=cfg.source.exclude_dirs,
            exclude_exts=cfg.source.exclude_exts,
        )
        if not tarball_path.exists() or tarball_path.stat().st_size == 0:
            print("Tarball is empty; nothing to upload. Check `source.include` in kaggle.toml.",
                  file=sys.stderr)
            return 1
        if not quiet:
            print(f"Built tarball: {tarball_path} ({tarball_path.stat().st_size} bytes)")
        # dataset-metadata.json
        (tmp / "dataset-metadata.json").write_text(
            json.dumps(nb_mod.write_dataset_metadata(slug=actual_slug), indent=2),
            encoding="utf-8",
        )
        if version == 1:
            cmd = ["datasets", "create", "-p", str(tmp), "-r", "tar"]
        else:
            cmd = [
                "datasets",
                "version",
                "-p",
                str(tmp),
                "-m",
                f"kagglepipe src v{version} (auto)",
                "-r",
                "tar",
            ]
        result = runner.run(cmd)
        # The upstream kaggle CLI returns rc=0 even when a `create` call
        # fails because the dataset title is already in use. Detect that
        # and surface as a real failure.
        combined = (result.stdout or "") + (result.stderr or "")
        if result.returncode != 0 or "Dataset creation error" in combined:
            print(result.stdout, file=sys.stdout)
            print(result.stderr, file=sys.stderr)
            return 1 if result.returncode == 0 else result.returncode
    if not quiet:
        print(f"Uploaded: {actual_slug} v{version}")
    return 0
