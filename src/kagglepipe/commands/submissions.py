"""Submission automation (P3).

`kagglepipe submit` trains + generates predictions + submits + records the
result. The actual "train" and "predict" steps are user-defined via the
notebook_command (or a separate [competition] section in kaggle.toml).

For typical Kaggle workflow:
1. A notebook is generated that trains a model and writes submission.csv
2. The submission CSV is downloaded locally
3. `kagglepipe competitions submit` uploads it
4. SubmissionStore records the submission locally
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from kagglepipe import credentials, kaggle_api, runner
from kagglepipe.config import Config
from kagglepipe.slug import normalize_slug, resolve_template
from kagglepipe.state import SubmissionRecord, SubmissionStore


def cmd_submit(
    cfg: Config,
    *,
    competition: str | None = None,
    file: Path | None = None,
    message: str | None = None,
    train: bool = False,
) -> int:
    """Submit to a competition.

    Args:
        competition: override the configured competition slug
        file: override the submission CSV path
        message: override the message
        train: if True, run a feature to generate the submission file first
               (uses `competition.train_command` from kaggle.toml)
    """
    comp_slug = competition or cfg.competition.get("slug", "")
    if not comp_slug:
        print(
            "No competition configured. Set [competition].slug in kaggle.toml or pass --competition.",
            file=sys.stderr,
        )
        return 1
    sub_path = file or Path(cfg.competition.get("submission_path", "submission.csv"))
    sub_message = message or cfg.competition.get("message", "kagglepipe submission")

    if train:
        cmd = cfg.competition.get("train_command", "")
        if not cmd:
            print(
                "train_command not configured. Set [competition].train_command in kaggle.toml.",
                file=sys.stderr,
            )
            return 1
        if not _run_train_command(cmd):
            return 1

    if not sub_path.exists():
        print(f"Submission file not found: {sub_path}", file=sys.stderr)
        return 1

    creds = credentials.load()
    try:
        kaggle_api.competitions_submit(comp_slug, sub_path, sub_message)
    except Exception as exc:
        print(f"Submission failed: {exc}", file=sys.stderr)
        SubmissionStore().add(
            SubmissionRecord(
                competition=comp_slug,
                file_path=str(sub_path),
                message=sub_message,
                status="failed",
            )
        )
        return 1
    rec = SubmissionRecord(
        competition=comp_slug,
        file_path=str(sub_path),
        message=sub_message,
        status="submitted",
    )
    SubmissionStore().add(rec)
    print(f"Submitted {sub_path} to {comp_slug}")
    return 0


def _run_train_command(cmd: str) -> bool:
    """Run the training command. In practice this is usually a local script;
    the user can also kick off a Kaggle kernel and wait for the output."""
    import subprocess
    print(f"$ {cmd}")
    rc = subprocess.run(cmd, shell=True).returncode
    return rc == 0


def cmd_submissions_list(
    *,
    competition: str | None = None,
    csv_output: bool = False,
    json_output: bool = False,
) -> int:
    store = SubmissionStore()
    recs = store.all()
    if competition:
        recs = [r for r in recs if r.competition == competition]
    if json_output:
        print(json.dumps([asdict(r) for r in recs], indent=2))
        return 0
    if csv_output:
        if recs:
            keys = list(asdict(recs[0]).keys())
            print(",".join(keys))
            for r in recs:
                print(",".join(str(getattr(r, k, "")) for k in keys))
        return 0
    if not recs:
        print("No submissions recorded. Use `kagglepipe submit` first.")
        return 0
    print(f"{'COMPETITION':<40} {'STATUS':<10} {'SUBMITTED AT':<22} {'FILE'}")
    for r in recs:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.submitted_at))
        print(f"{r.competition:<40} {r.status:<10} {ts:<22} {r.file_path}")
    return 0


def cmd_submissions_latest(competition: str | None = None) -> int:
    rec = SubmissionStore().latest(competition)
    if rec is None:
        print("No submissions recorded.")
        return 0
    print(json.dumps(asdict(rec), indent=2))
    return 0
