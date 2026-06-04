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
from kagglepipe.state import RunRecord, RunStore, state_dir
from kagglepipe.manifest import write_manifest
from kagglepipe.provenance import build_provenance, git_commit, git_dirty, hash_file


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


def resolve_notebook_command(
    command: str,
    *,
    username: str,
    branch: str,
    out_dir: str = "/kaggle/working/features",
    src_mount: str = "",
    data_mount: str = "",
) -> str:
    """Resolve the notebook command string for the current run.

    We support both the current `{branch}` placeholder and the legacy
    `{{branch}}` form emitted by older scaffolded projects.
    """
    return _resolve_notebook_command(
        command,
        username=username,
        branch=branch,
        out_dir=out_dir,
        src_mount=src_mount,
        data_mount=data_mount,
    )


class _SafeFormatDict(dict[str, str]):
    """Preserve unknown placeholders instead of raising KeyError."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _resolve_notebook_command(
    command: str,
    *,
    username: str,
    branch: str,
    out_dir: str,
    src_mount: str,
    data_mount: str,
) -> str:
    command = resolve_template(
        command.replace("{{branch}}", "{branch}"),
        username=username,
        branch=branch,
    )
    return command.format_map(
        _SafeFormatDict(
            username=username,
            branch=branch,
            out_dir=out_dir,
            src_mount=src_mount,
            data_mount=data_mount,
        )
    )


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
    dry_run: bool = False,
) -> int:
    """Render a notebook, push it as a kernel, poll, download the output.

    dry_run=True (P9) prints the plan (datasets, notebook path, kernel slug,
    GPU, cache status, expected artifact) without actually invoking the
    kaggle CLI for any state-changing operation.
    """
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
        if dry_run:
            src_version = "?"
        else:
            src_version = kaggle_api.get_next_version(src_slug)
    # resolve_template handles {username} and {branch}. normalize_slug is only
    # for the branch segment so the template placeholder {branch} is preserved.
    kernel_slug_raw = cfg.feature.kernel_slug_template.replace(
        "{branch}", normalize_slug(branch)
    )
    kernel_slug = resolve_template(
        kernel_slug_raw,
        username=creds.username,
    )
    gpu_value = GPU_INSTANCE_MAP.get(gpu, gpu)
    nb = nb_mod.render(
        cfg.feature.notebook_template,
        branch=branch,
        src_dataset_slug=src_slug,
        src_version=src_version if isinstance(src_version, int) else 1,
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
        # Resolve {username} (and $name if present) in notebook_command
        # so the rendered notebook contains the actual resolved paths.
        notebook_command=resolve_notebook_command(
            cfg.feature.notebook_command,
            username=creds.username,
            branch=branch,
            out_dir=cfg.feature.out_dir,
            src_mount=slug.resolve_template(
                cfg.feature.src_mount,
                username=creds.username,
                dataset=src_slug.split("/", 1)[-1],
            ),
            data_mount=(
                slug.resolve_template(
                    cfg.feature.data_mount,
                    username=creds.username,
                    dataset=data_slug.split("/", 1)[-1],
                )
                if data_slug
                else ""
            ),
        ),
        gpu=gpu_value,
    )

    # P5 cache check (P9 dry-run also reports this).
    from kagglepipe.cache import CacheStore
    from kagglepipe.cache import config_hash_for_branch as _cfg_hash_for_branch
    cache_store = CacheStore()
    cached = cache_store.get(branch)
    cache_status = "MISS"
    if cached is not None and cfg.feature.cache:
        cache_status = "HIT (would skip)"

    if dry_run:
        _print_dry_run_plan(
            branch=branch,
            src_slug=src_slug,
            src_version=src_version,
            data_slug=data_slug,
            kernel_slug=kernel_slug,
            gpu_value=gpu_value,
            nb=nb,
            cfg=cfg,
            cache_status=cache_status,
            cached=cached,
            features_dir=features_dir or (Path.cwd() / cfg.paths.features_dir),
            notebooks_dir=notebooks_dir or (Path.cwd() / cfg.paths.notebooks_dir),
        )
        return 0

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

    # P13: capture provenance now (in case the run fails later, we still
    # have a partial manifest).
    from kagglepipe.provenance import build_provenance, git_commit, git_dirty, hash_file
    provenance = build_provenance()
    notebook_hash = hash_file(nb_path)

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
    artifact_path: str | None = None
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
        artifact_path = str(dest)
        if not quiet:
            print(f"Downloaded: {dest}")

    # P13: write the strong run manifest.
    rec = RunRecord(
        branch=branch,
        kernel_slug=kernel_slug,
        state="complete",
        artifact_path=artifact_path,
        finished_at=time.time(),
        config_hash=_cfg_hash_for_branch(cfg, branch),
        git_commit=git_commit(),
        git_dirty=git_dirty(),
        gpu=gpu_value,
        src_slug=src_slug,
        src_version=src_version if isinstance(src_version, int) else None,
        dataset_versions=provenance.get("dataset_versions", {}),
        notebook_hash=notebook_hash,
    )
    write_manifest(rec)
    return 0


def _ext_from_glob(s: str) -> str:
    """Best-effort extension extraction from a glob pattern."""
    import re
    m = re.search(r"\.([A-Za-z0-9]+)$", s)
    return f".{m.group(1)}" if m else ""


def _print_dry_run_plan(
    *,
    branch: str,
    src_slug: str,
    src_version,
    data_slug: str,
    kernel_slug: str,
    gpu_value: str | None,
    nb: dict,
    cfg: Config,
    cache_status: str,
    cached,
    features_dir: Path,
    notebooks_dir: Path,
) -> None:
    """Pretty-print the dry-run plan for `feature run`."""
    print("[dry-run] kagglepipe feature run")
    print(f"  branch          : {branch}")
    print(f"  source dataset  : {src_slug} v{src_version}")
    if data_slug:
        print(f"  data dataset    : {data_slug}")
    else:
        print("  data dataset    : (none)")
    print(f"  kernel slug     : {kernel_slug}")
    print(f"  gpu             : {gpu_value or 'none'}")
    print(f"  notebook path   : {notebooks_dir / f'extract_{branch}.ipynb'}")
    print(f"  features dir    : {features_dir}")
    expected = features_dir / f"{branch}{_ext_from_glob(cfg.feature.output_glob)}"
    print(f"  expected output : {expected}")
    print(f"  cache           : {cache_status}")
    if cached is not None:
        print(f"  cached artifact : {cached.artifact_path}")
    print(f"  notebook cells  : {len(nb.get('cells', []))}")
    sources = nb.get("metadata", {}).get("dataset_sources", [])
    print(f"  dataset_sources : {sources}")
    print("  (no Kaggle API calls will be made)")


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
    parallel: int = 1,
    resume: bool = False,
) -> int:
    """Run the configured `heavy_branches` (or `branches`).

    parallel=1 (default) preserves the original sequential behavior.
    parallel>=2 submits up to N kernels concurrently, polls them all from a
    single poller thread, and downloads artifacts as they complete.

    resume=True (P2): skip branches whose latest run in RunStore is complete
    with an artifact.
    """
    seq = list(branches) if branches else list(cfg.feature.heavy_branches or cfg.feature.branches)
    if not seq:
        print("No branches configured. Set `feature.heavy_branches` in kaggle.toml.",
              file=sys.stderr)
        return 1
    for b in seq:
        validate_branch(cfg, b)

    # P2: skip already-successful branches if --resume.
    if resume:
        from kagglepipe.state import RunStore
        store = RunStore()
        skipped = [b for b in seq if store.is_branch_successful(b)]
        for b in skipped:
            if not quiet:
                print(f"[skip] {b} (already complete with artifact)")
        seq = [b for b in seq if b not in skipped]
        if not seq:
            if not quiet:
                print("Nothing to do; all branches already complete.")
            return 0

    # Sequential fast path: no parallelism requested, no resume, no cache, no
    # dependency-graph requested. Match the original implementation exactly
    # so existing tests and the freuid pipeline keep working.
    if parallel <= 1 and not resume:
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

    # Parallel path (P1).
    from kagglepipe.parallel import ParallelRunner
    started = time.time()
    runner_obj = ParallelRunner(
        cfg,
        seq,
        gpu=gpu,
        workers=parallel,
        timeout_sec=timeout_sec,
        data_dataset=data_dataset,
        features_dir=features_dir,
        quiet=quiet,
    )
    if not quiet:
        print(f"Running {len(seq)} branches with {parallel} workers")
    results = runner_obj.run()
    elapsed = time.time() - started
    failures = [b for b, r in results.items() if r["status"] != "complete"]
    if not quiet:
        print(f"\n=== Summary ({elapsed:.0f}s) ===")
        print(f"Total: {len(seq)}, OK: {len(seq) - len(failures)}, Failed: {len(failures)}")
        for b in failures:
            err = results[b].get("error") or "unknown"
            print(f"  FAILED: {b} ({err})")
        # Clear the progress line
        sys.stdout.write("\r" + " " * 120 + "\r")
        sys.stdout.flush()
    return 0 if not failures else 1
