"""feature run / feature all — render, push, poll, download."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

from kagglepipe import credentials, kaggle_api, notebook as nb_mod, runner, slug
from kagglepipe.config import Config
from kagglepipe.polling import poll_kernel_status
from kagglepipe.slug import normalize_slug, resolve_template


# Map CLI token -> Kaggle kernel metadata value. Kaggle expects "t4 x2".
GPU_INSTANCE_MAP: dict[str, str] = {"p100": "p100", "t4x2": "t4 x2", "none": None}


def validate_branch(cfg: Config, branch: str) -> str:
    """Return branch if in the configured whitelist; raise ValueError otherwise."""
    if cfg.feature.branches and branch not in cfg.feature.branches:
        raise ValueError(
            f"branch {branch!r} is not in the whitelist; "
            f"allowed: {sorted(cfg.feature.branches)}"
        )
    return branch


def run_feature(
    cfg: Config,
    branch: str,
    *,
    gpu: str = "t4x2",
    timeout_sec: int | None = None,
    src_version: int | None = None,
    src_dataset: str | None = None,
    data_dataset: str | None = None,
    features_dir: Path | None = None,
    notebooks_dir: Path | None = None,
    no_download: bool = False,
    quiet: bool = False,
) -> int:
    """Render a notebook, push it as a kernel, poll, download the output."""
    branch = validate_branch(cfg, branch)
    creds = credentials.load()
    src_slug = src_dataset or resolve_template(
        cfg.source.src_dataset_slug, username=creds.username
    )
    data_slug = data_dataset or (
        resolve_template(cfg.data.dataset_slug, username=creds.username)
        if cfg.data.dataset_slug
        else ""
    )
    if src_version is None:
        src_version = kaggle_api.get_next_version(src_slug)
    kernel_slug = resolve_template(
        cfg.feature.kernel_slug_template,
        username=creds.username,
        branch=normalize_slug(branch),
    )
    gpu_value = GPU_INSTANCE_MAP.get(gpu, gpu)
    nb = nb_mod.render(
        cfg.feature.notebook_template,
        branch=branch,
        src_dataset_slug=src_slug,
        src_version=src_version,
        src_mount=slug.resolve_template(
            cfg.feature.src_mount, username=creds.username, dataset=src_slug.split("/", 1)[-1]
        ),
        data_dataset_slug=data_slug,
        data_mount=(
            slug.resolve_template(
                cfg.feature.data_mount, username=creds.username, dataset=data_slug.split("/", 1)[-1]
            )
            if data_slug
            else ""
        ),
        out_dir=cfg.feature.out_dir,
        notebook_command=cfg.feature.notebook_command,
        gpu=gpu_value,
    )
    nb_dir = (notebooks_dir or Path.cwd() / cfg.paths.notebooks_dir).resolve()
    nb_dir.mkdir(parents=True, exist_ok=True)
    nb_path = nb_dir / f"extract_{normalize_slug(branch)}.ipynb"
    nb_path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    if not quiet:
        print(f"Wrote notebook: {nb_path}")

    kernel_md = nb_mod.write_kernel_metadata(
        kernel_slug=kernel_slug,
        title=f"{cfg.feature.kernel_title_prefix}-{normalize_slug(branch)}",
        code_file=nb_path.name,
        dataset_sources=[s for s in (src_slug, data_slug) if s],
        enable_internet=cfg.kernels.enable_internet,
        is_private=cfg.kernels.is_private,
        language=cfg.kernels.language,
        kernel_type=cfg.kernels.kernel_type,
        gpu=gpu_value,
    )
    (nb_dir / "kernel-metadata.json").write_text(
        json.dumps(kernel_md, indent=2), encoding="utf-8"
    )

    result = runner.run(["kernels", "push", "-p", str(nb_dir)])
    if result.returncode != 0:
        print(result.stdout, file=sys.stdout)
        print(result.stderr, file=sys.stderr)
        return result.returncode
    if not quiet:
        print(f"Pushed kernel: {kernel_slug}")

    state = poll_kernel_status(
        kernel_slug,
        timeout_sec=timeout_sec or cfg.feature.default_timeout_sec,
        poll_interval_sec=cfg.feature.poll_interval_sec,
    )
    if not quiet:
        print(f"Kernel state: {state}")
    if state != "complete":
        print(f"Inspect logs: {kaggle_api.kernels_logs_url(kernel_slug)}", file=sys.stderr)
        return 1

    if no_download:
        return 0

    out_target_dir = (features_dir or Path.cwd() / cfg.paths.features_dir).resolve()
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        kaggle_api.download_kernel_output(kernel_slug, tmp)
        glob_pattern = cfg.feature.output_glob.format(branch=branch)
        try:
            src_artifact = kaggle_api.find_artifact(tmp, glob_pattern)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        out_target_dir.mkdir(parents=True, exist_ok=True)
        dest = out_target_dir / f"{normalize_slug(branch)}{src_artifact.suffix}"
        kaggle_api.copy_artifact(src_artifact, dest)
        if not quiet:
            print(f"Downloaded: {dest}")
    return 0


def run_all(
    cfg: Config,
    *,
    branches: Sequence[str] | None = None,
    gpu: str = "t4x2",
    timeout_sec: int | None = None,
    data_dataset: str | None = None,
    features_dir: Path | None = None,
    notebooks_dir: Path | None = None,
    quiet: bool = False,
) -> int:
    """Run the configured `heavy_branches` (or `branches`) sequentially."""
    seq = list(branches) if branches else list(cfg.feature.heavy_branches or cfg.feature.branches)
    if not seq:
        print("No branches configured. Set `feature.heavy_branches` in kaggle.toml.",
              file=sys.stderr)
        return 1
    for b in seq:
        validate_branch(cfg, b)
    failures: list[str] = []
    started = time.time()
    for b in seq:
        if not quiet:
            print(f"\n=== {b} ===")
        rc = run_feature(
            cfg,
            b,
            gpu=gpu,
            timeout_sec=timeout_sec,
            data_dataset=data_dataset,
            features_dir=features_dir,
            notebooks_dir=notebooks_dir,
            quiet=quiet,
        )
        if rc != 0:
            failures.append(b)
    elapsed = time.time() - started
    if not quiet:
        print(f"\n=== Summary ({elapsed:.0f}s) ===")
        print(f"Total: {len(seq)}, OK: {len(seq) - len(failures)}, Failed: {len(failures)}")
        for b in failures:
            print(f"  FAILED: {b}")
    return 0 if not failures else 1
