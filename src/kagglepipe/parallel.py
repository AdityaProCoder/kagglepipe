"""Parallel feature runner.

Used by `kagglepipe feature all --parallel N` (P1). Submits up to N kernels
concurrently, polls all active kernels in a single thread, and downloads
artifacts as each completes.

Design:
- One ThreadPoolExecutor of size N for *submission* (render + push is I/O-bound
  on subprocess calls).
- One polling thread that watches all in-flight kernels. Single-threaded
  polling keeps the Kaggle API polite and avoids stale-state races.
- Per-branch RunRecord tracked in RunStore so the partial-failure
  resume logic (P2) and cache logic (P5) can read it.
- Live progress is printed as a single self-overwriting line (no rich dep).
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from kagglepipe import credentials, kaggle_api, notebook as nb_mod, runner, slug
from kagglepipe.config import Config
from kagglepipe.polling import poll_kernel_status
from kagglepipe.slug import normalize_slug, resolve_template
from kagglepipe.state import RunRecord, RunStore
from kagglepipe.commands.feature import (
    GPU_INSTANCE_MAP,
    resolve_notebook_command,
    validate_branch,
)


def _render_and_push(
    cfg: Config, branch: str, *, gpu: str, data_dataset: str | None
) -> tuple[bool, str, str | None, str]:
    """Render the notebook and push the kernel.

    Returns (pushed_ok, kernel_slug, error, kernel_metadata_blob).
    """
    branch = validate_branch(cfg, branch)
    creds = credentials.load()
    src_slug = resolve_template(cfg.source.src_dataset_slug, username=creds.username)
    data_slug = (
        data_dataset
        or (resolve_template(cfg.data.dataset_slug, username=creds.username)
            if cfg.data.dataset_slug
            else "")
    )
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
        notebook_command=resolve_notebook_command(
            cfg.feature.notebook_command,
            username=creds.username,
            branch=branch,
            out_dir=cfg.feature.out_dir,
            src_mount=slug.resolve_template(
                cfg.feature.src_mount, username=creds.username, dataset=src_slug.split("/", 1)[-1]
            ),
            data_mount=(
                slug.resolve_template(
                    cfg.feature.data_mount, username=creds.username, dataset=data_slug.split("/", 1)[-1]
                )
                if data_slug
                else ""
            ),
        ),
        gpu=gpu_value,
    )
    nb_dir = (Path.cwd() / cfg.paths.notebooks_dir).resolve()
    nb_dir.mkdir(parents=True, exist_ok=True)
    nb_path = nb_dir / f"extract_{normalize_slug(branch)}.ipynb"
    nb_path.write_text(json_str := _to_json(nb), encoding="utf-8")
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
    (nb_dir / "kernel-metadata.json").write_text(_to_json(kernel_md), encoding="utf-8")
    result = runner.run(["kernels", "push", "-p", str(nb_dir)])
    if result.returncode != 0:
        return False, kernel_slug, (
            result.stderr.strip() or result.stdout.strip() or "kaggle kernels push failed"
        ), kernel_slug
    return True, kernel_slug, None, kernel_slug


def _to_json(obj) -> str:
    import json as _json
    return _json.dumps(obj, indent=2)


class ParallelRunner:
    """Run feature branches concurrently.

    Usage:
        runner = ParallelRunner(cfg, branches=[...], gpu="t4x2", workers=3)
        results = runner.run()
        # results is dict[branch, dict(status, artifact_path, error)]
    """

    def __init__(
        self,
        cfg: Config,
        branches: list[str],
        *,
        gpu: str = "t4x2",
        workers: int = 3,
        timeout_sec: int | None = None,
        data_dataset: str | None = None,
        features_dir: Path | None = None,
        quiet: bool = False,
        run_store: RunStore | None = None,
    ) -> None:
        self.cfg = cfg
        self.branches = list(branches)
        self.gpu = gpu
        self.workers = max(1, int(workers))
        self.timeout_sec = timeout_sec or cfg.feature.default_timeout_sec
        self.poll_interval_sec = cfg.feature.poll_interval_sec
        self.data_dataset = data_dataset
        self.features_dir = features_dir or (Path.cwd() / cfg.paths.features_dir).resolve()
        self.quiet = quiet
        self.run_store = run_store or RunStore()

    def run(self) -> dict[str, dict]:
        """Execute all branches. Returns dict[branch, {status, artifact, error}]."""
        results: dict[str, dict] = {}
        for b in self.branches:
            results[b] = {"status": "queued", "artifact": None, "error": None}

        # Phase 1: submit up to `workers` kernels in parallel.
        pending: dict[str, str] = {}  # branch -> kernel_slug, populated as we go
        submitted: dict[str, RunRecord] = {}
        futures: dict = {}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for branch in self.branches:
                futures[pool.submit(self._submit_one, branch)] = branch

            for fut in as_completed(futures):
                branch = futures[fut]
                ok, kernel_slug, error, _ = fut.result()
                if not ok:
                    results[branch] = {"status": "error", "artifact": None, "error": error}
                    self.run_store.add(
                        RunRecord(
                            branch=branch,
                            kernel_slug=kernel_slug,
                            state="error",
                            error=error,
                        )
                    )
                    if not self.quiet:
                        print(f"[{branch}] submit failed: {error}", file=sys.stderr)
                    continue
                pending[branch] = kernel_slug
                rec = RunRecord(branch=branch, kernel_slug=kernel_slug, state="running")
                self.run_store.add(rec)
                submitted[branch] = rec
                if not self.quiet:
                    print(f"[{branch}] submitted -> {kernel_slug}")

        # Phase 2: poll all pending kernels until they all finish (or timeout).
        if not pending:
            return results

        stop = threading.Event()
        poller_thread = threading.Thread(
            target=self._poll_all,
            args=(pending, submitted, results, stop),
            daemon=True,
        )
        poller_thread.start()
        poller_thread.join(timeout=self.timeout_sec)
        stop.set()
        if poller_thread.is_alive():
            for branch, slug_str in pending.items():
                if results[branch]["status"] == "running":
                    results[branch] = {
                        "status": "timeout",
                        "artifact": None,
                        "error": "global timeout",
                    }
                    self.run_store.update(
                        branch, slug_str, state="timeout", finished_at=time.time()
                    )

        return results

    def _submit_one(self, branch: str) -> tuple[bool, str, str | None, str]:
        return _render_and_push(
            self.cfg, branch, gpu=self.gpu, data_dataset=self.data_dataset
        )

    def _poll_all(
        self,
        pending: dict[str, str],
        submitted: dict[str, RunRecord],
        results: dict[str, dict],
        stop: threading.Event,
    ) -> None:
        """Single-threaded poller that watches all in-flight kernels."""
        last_print = 0.0
        while not stop.is_set() and pending:
            still_running: dict[str, str] = {}
            for branch, kernel_slug in pending.items():
                if results[branch]["status"] != "running":
                    continue
                state = _quick_status(kernel_slug)
                if state in ("complete", "error", "timeout"):
                    self._finish_branch(
                        branch, kernel_slug, state, results, submitted
                    )
                else:
                    still_running[branch] = kernel_slug
            pending = still_running
            now = time.time()
            if not self.quiet and now - last_print > self.poll_interval_sec / 2:
                self._print_progress(results)
                last_print = now
            if pending:
                stop.wait(self.poll_interval_sec)
        if not self.quiet:
            self._print_progress(results)

    def _finish_branch(
        self,
        branch: str,
        kernel_slug: str,
        state: str,
        results: dict[str, dict],
        submitted: dict[str, RunRecord],
    ) -> None:
        artifact_path: str | None = None
        error: str | None = None
        if state == "complete":
            try:
                artifact_path = self._download_artifact(branch, kernel_slug)
            except Exception as exc:
                state = "error"
                error = f"download failed: {exc}"
        elif state == "error":
            error = "kernel error"
        elif state == "timeout":
            error = "kernel timeout"
        results[branch] = {"status": state, "artifact": artifact_path, "error": error}
        self.run_store.update(
            branch,
            kernel_slug,
            state=state,
            artifact_path=artifact_path,
            error=error,
            finished_at=time.time(),
        )

    def _download_artifact(self, branch: str, kernel_slug: str) -> str:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            kaggle_api.download_kernel_output(kernel_slug, tmp)
            glob_pattern = self.cfg.feature.output_glob.format(branch=branch)
            src_artifact = kaggle_api.find_artifact(tmp, glob_pattern)
            self.features_dir.mkdir(parents=True, exist_ok=True)
            dest = self.features_dir / f"{normalize_slug(branch)}{src_artifact.suffix}"
            kaggle_api.copy_artifact(src_artifact, dest)
            return str(dest)

    def _print_progress(self, results: dict[str, dict]) -> None:
        if self.quiet:
            return
        done = sum(1 for r in results.values() if r["status"] in ("complete", "error", "timeout"))
        running = sum(1 for r in results.values() if r["status"] == "running")
        queued = sum(1 for r in results.values() if r["status"] == "queued")
        line = (
            f"  [{done}/{len(results)}] running={running} queued={queued} | "
            + " ".join(
                f"{b}={r['status'][:4]}"
                for b, r in results.items()
                if r["status"] in ("running", "queued")
            )
        )
        sys.stdout.write("\r" + line[:120].ljust(120))
        sys.stdout.flush()


def _quick_status(kernel_slug: str) -> str:
    """One-shot status check; returns 'running' on any non-terminal state."""
    try:
        s = kaggle_api.kernel_status(kernel_slug)
    except Exception:
        return "error"
    if "complete" in s:
        return "complete"
    if "error" in s or "fail" in s:
        return "error"
    return "running"
