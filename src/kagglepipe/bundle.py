"""Reproducibility bundles (P14).

`kagglepipe run export <branch|manifest-path>` writes a portable tarball
containing the run manifest, the kaggle.toml in effect, and a copy of
any cached artifacts. A teammate (or future-you) can `kagglepipe run
reproduce <bundle.tar.gz>` on a different machine and get an exact
re-run plan printed — or actually re-execute it.

Bundle layout (inside the tar.gz):

    manifest.json          the strong run manifest
    kaggle.toml            the config that produced this run
    README.txt             human-readable summary
    artifacts/             copies of produced artifacts (if available)
"""

from __future__ import annotations

import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

from kagglepipe.manifest import load_manifest


def find_latest_manifest(branch: str) -> Path | None:
    """Return the most recent manifest for the given branch, or None."""
    from kagglepipe.manifest import manifests_dir
    candidates = sorted(manifests_dir().glob(f"{branch}-*.json"), reverse=True)
    return candidates[0] if candidates else None


def cmd_run_export(
    target: str,
    *,
    out: Path | None = None,
    include_artifacts: bool = True,
) -> int:
    """Export a run as a portable tarball.

    `target` is either a branch name (picks the latest manifest) or a
    direct path to a manifest.json file.
    """
    if os.path.exists(target):
        manifest_path = Path(target)
    else:
        manifest_path = find_latest_manifest(target)
    if manifest_path is None or not manifest_path.exists():
        print(f"No manifest found for {target!r}.", file=sys.stderr)
        return 1
    manifest = load_manifest(manifest_path)
    bundle = out or Path(f"kagglepipe-bundle-{manifest['branch']}.tar.gz")
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        # 1. copy the manifest
        shutil.copy2(manifest_path, tmp / "manifest.json")
        # 2. copy kaggle.toml from the project if available
        cfg = Path.cwd() / "kaggle.toml"
        if cfg.exists():
            shutil.copy2(cfg, tmp / "kaggle.toml")
        # 3. copy any artifact referenced by the manifest
        if include_artifacts:
            ap = manifest.get("artifact_path")
            if ap:
                src = Path(ap)
                if src.exists():
                    art_dir = tmp / "artifacts"
                    art_dir.mkdir(parents=True, exist_ok=True)
                    dst = art_dir / src.name
                    shutil.copy2(src, dst)
        # 4. write a human-readable README
        readme = (
            f"KagglePipe run bundle\n"
            f"======================\n"
            f"Branch:    {manifest.get('branch')}\n"
            f"State:     {manifest.get('state')}\n"
            f"Kernel:    {manifest.get('kernel_slug')}\n"
            f"GPU:       {manifest.get('gpu')}\n"
            f"Git:       {(manifest.get('git_commit') or '')[:12]}\n"
            f"Started:   {manifest.get('started_at')}\n"
            f"Finished:  {manifest.get('finished_at')}\n"
            f"Artifact:  {manifest.get('artifact_path')}\n"
        )
        (tmp / "README.txt").write_text(readme, encoding="utf-8")
        # 5. tar everything
        with tarfile.open(bundle, "w:gz") as tf:
            for p in tmp.rglob("*"):
                tf.add(p, arcname=str(p.relative_to(tmp)))
    print(f"Exported bundle: {bundle} ({bundle.stat().st_size} bytes)")
    return 0


def cmd_run_reproduce(
    bundle_path: Path,
    *,
    dry_run: bool = True,
) -> int:
    """Reproduce a run from a bundle.

    In dry-run mode (default), prints the plan: which dataset versions
    were used, what command would be invoked, what would be downloaded.
    With dry_run=False, calls feature.run_feature to actually re-run.
    """
    if not bundle_path.exists():
        print(f"Bundle not found: {bundle_path}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        with tarfile.open(bundle_path, "r:gz") as tf:
            tf.extractall(tmp)
        manifest_path = tmp / "manifest.json"
        if not manifest_path.exists():
            print("Bundle has no manifest.json", file=sys.stderr)
            return 1
        manifest = load_manifest(manifest_path)
        print("Reproducing run from bundle:")
        print(f"  branch      : {manifest.get('branch')}")
        print(f"  state       : {manifest.get('state')}")
        print(f"  kernel_slug : {manifest.get('kernel_slug')}")
        print(f"  gpu         : {manifest.get('gpu')}")
        print(f"  git_commit  : {(manifest.get('git_commit') or '')[:12]}")
        print(f"  src_version : {manifest.get('src_version')}")
        print(f"  artifact    : {manifest.get('artifact_path')}")
        cfg = tmp / "kaggle.toml"
        if cfg.exists():
            print(f"  config      : {cfg.name} (bundled)")
        if dry_run:
            print("(dry-run: not actually re-executing. Pass --no-dry-run to run.)")
            return 0
        # Real re-run: hand off to feature.run_feature.
        # Use the bundled kaggle.toml as the source of truth.
        from kagglepipe.commands import feature
        from kagglepipe.config import load as load_cfg
        cfg_obj = load_cfg(cfg)
        gpu_token = (manifest.get("gpu") or "none").lower()
        if "t4 x2" in gpu_token or "t4x2" in gpu_token:
            gpu = "t4x2"
        elif "p100" in gpu_token:
            gpu = "p100"
        else:
            gpu = "none"
        return feature.run_feature(
            cfg_obj,
            manifest["branch"],
            gpu=gpu,
            dry_run=False,
        )
