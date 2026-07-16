"""Experiment tracking (P6).

`kagglepipe experiments record` creates an experiment record capturing
git commit, dataset versions, feature branches, and (optionally) a
submission id + score.

`kagglepipe experiments list` / `experiments show <id>` query the store.
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from typing import Any

from kagglepipe.config import Config
from kagglepipe.state import ExperimentRecord, ExperimentStore


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def cmd_experiments_record(
    cfg: Config,
    *,
    id: str | None = None,
    notes: str = "",
    submission_id: str | None = None,
    score: float | None = None,
    feature_branches: list[str] | None = None,
) -> int:
    exp_id = id or f"exp-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    rec = ExperimentRecord(
        id=exp_id,
        git_commit=_git_commit(),
        dataset_versions=cfg.competition.get("dataset_versions", {}) or {},
        feature_branches=feature_branches or list(
            cfg.feature.heavy_branches or cfg.feature.branches
        ),
        submission_id=submission_id,
        score=score,
        notes=notes,
    )
    ExperimentStore().add(rec)
    print(f"Recorded experiment {exp_id}")
    return 0


def cmd_experiments_list(*, csv_output: bool = False, json_output: bool = False) -> int:
    recs = ExperimentStore().all()
    if json_output:
        print(json.dumps([_to_dict(r) for r in recs], indent=2))
        return 0
    if csv_output:
        if recs:
            keys = list(_to_dict(recs[0]).keys())
            print(",".join(keys))
            for r in recs:
                print(",".join(str(v) for v in _to_dict(r).values()))
        return 0
    if not recs:
        print("No experiments recorded. Use `kagglepipe experiments record`.")
        return 0
    print(f"{'ID':<40} {'COMMIT':<10} {'SCORE':<8} {'SUBMISSION'}")
    for r in recs:
        commit = (r.git_commit or "?")[:8]
        score = f"{r.score:.4f}" if r.score is not None else "-"
        sub = r.submission_id or "-"
        print(f"{r.id:<40} {commit:<10} {score:<8} {sub}")
    return 0


def cmd_experiments_show(exp_id: str) -> int:
    rec = ExperimentStore().get(exp_id)
    if rec is None:
        print(f"No experiment with id {exp_id!r}.", file=__import__("sys").stderr)
        return 1
    print(json.dumps(_to_dict(rec), indent=2))
    return 0


def _to_dict(r: ExperimentRecord) -> dict[str, Any]:
    from dataclasses import asdict
    return asdict(r)
