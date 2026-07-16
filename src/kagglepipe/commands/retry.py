"""Retry / resume (P2).

`kagglepipe feature retry failed` re-runs every branch whose latest entry in
the RunStore is not 'complete' (or has no artifact). Optionally filtered by a
subcommand: `retry failed`, `retry error`, `retry timeout`, `retry all`.

`kagglepipe feature retry <branch>` retries a single branch.
"""

from __future__ import annotations

import sys

from kagglepipe.config import Config
from kagglepipe.state import RunStore


def _resolve_branches(runs: RunStore, cfg: Config, selector: str) -> list[str]:
    """Map the selector to a list of branches."""
    if selector == "failed":
        # Anything whose latest run isn't complete.
        return [
            r.branch
            for r in reversed(runs.all())
            if r.state != "complete" or r.artifact_path is None
        ]
    if selector == "error":
        return [r.branch for r in reversed(runs.all()) if r.state == "error"]
    if selector == "timeout":
        return [r.branch for r in reversed(runs.all()) if r.state == "timeout"]
    if selector == "incomplete":
        return [
            r.branch
            for r in reversed(runs.all())
            if r.state == "running"
        ]
    if selector == "all":
        return list({r.branch for r in runs.all()})
    # Single branch.
    return [selector]


def cmd_retry(
    cfg: Config,
    selector: str,
    *,
    gpu: str = "t4x2",
    parallel: int = 1,
    timeout_sec: int | None = None,
    quiet: bool = False,
) -> int:
    """Re-run branches identified by `selector` (failed | error | timeout | incomplete | all | <branch>)."""
    runs = RunStore()
    branches = _resolve_branches(runs, cfg, selector)
    if not branches:
        print(f"Nothing to retry for selector={selector!r}.", file=sys.stderr)
        return 0
    # De-dup, preserve order.
    seen: set[str] = set()
    unique = [b for b in branches if not (b in seen or seen.add(b))]
    if not quiet:
        print(f"Retrying {len(unique)} branch(es): {unique}")
    # Delegate to run_all with the same options.
    from kagglepipe.commands import feature
    return feature.run_all(
        cfg,
        branches=unique,
        gpu=gpu,
        timeout_sec=timeout_sec,
        quiet=quiet,
        parallel=parallel,
    )


def cmd_resume(
    cfg: Config,
    *,
    branches: list[str] | None = None,
    gpu: str = "t4x2",
    parallel: int = 1,
    timeout_sec: int | None = None,
    quiet: bool = False,
) -> int:
    """Resume a run, skipping branches whose latest entry in RunStore is complete."""
    from kagglepipe.commands import feature
    return feature.run_all(
        cfg,
        branches=branches,
        gpu=gpu,
        timeout_sec=timeout_sec,
        quiet=quiet,
        parallel=parallel,
        resume=True,
    )
