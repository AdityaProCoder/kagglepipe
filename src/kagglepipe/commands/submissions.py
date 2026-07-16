"""Submission automation (P3) + leaderboard tracking (P11) + submission provenance (P11.5).

`kagglepipe submit` trains + generates predictions + submits + records the
result. The actual "train" and "predict" steps are user-defined via the
notebook_command (or a separate [competition] section in kaggle.toml).

For typical Kaggle workflow:
1. A notebook is generated that trains a model and writes submission.csv
2. The submission CSV is downloaded locally
3. `kagglepipe competitions submit` uploads it
4. SubmissionStore records the submission locally with full provenance
   (P11.5): git commit, experiment id, feature branches, dataset versions,
   artifact hashes.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from kagglepipe import kaggle_api, runner
from kagglepipe.config import Config
from kagglepipe.provenance import build_provenance
from kagglepipe.state import SubmissionRecord, SubmissionStore


def cmd_submit(
    cfg: Config,
    *,
    competition: str | None = None,
    file: Path | None = None,
    message: str | None = None,
    train: bool = False,
    experiment_id: str | None = None,
) -> int:
    """Submit to a competition.

    Args:
        competition: override the configured competition slug
        file: override the submission CSV path
        message: override the message
        train: if True, run a feature to generate the submission file first
               (uses `competition.train_command` from kaggle.toml)
        experiment_id: optional experiment id to link this submission to
                       (recorded in P11.5 provenance)
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

    # Build the provenance record before submission so it captures state
    # even if the upload itself fails.
    provenance = build_provenance(
        experiment_id=experiment_id,
        feature_branches=cfg.competition.get("feature_branches"),
        dataset_versions=cfg.competition.get("dataset_versions"),
    )
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
                experiment_id=provenance.get("experiment_id"),
                git_commit=provenance.get("git_commit"),
                git_dirty=provenance.get("git_dirty"),
                feature_branches=provenance.get("feature_branches", []),
                dataset_versions=provenance.get("dataset_versions", {}),
                artifact_hashes=provenance.get("artifact_hashes", {}),
            )
        )
        return 1
    rec = SubmissionRecord(
        competition=comp_slug,
        file_path=str(sub_path),
        message=sub_message,
        status="submitted",
        experiment_id=provenance.get("experiment_id"),
        git_commit=provenance.get("git_commit"),
        git_dirty=provenance.get("git_dirty"),
        feature_branches=provenance.get("feature_branches", []),
        dataset_versions=provenance.get("dataset_versions", {}),
        artifact_hashes=provenance.get("artifact_hashes", {}),
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
    print(f"{'COMPETITION':<35} {'STATUS':<10} {'SCORE':<10} {'SUBMITTED AT':<22} {'EXPERIMENT'}")
    for r in recs:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.submitted_at))
        score = f"{r.score:.5f}" if r.score is not None else "-"
        exp = r.experiment_id or "-"
        print(f"{r.competition:<35} {r.status:<10} {score:<10} {ts:<22} {exp}")
    return 0


def cmd_submissions_latest(competition: str | None = None) -> int:
    rec = SubmissionStore().latest(competition)
    if rec is None:
        print("No submissions recorded.")
        return 0
    print(json.dumps(asdict(rec), indent=2))
    return 0


def cmd_submissions_best(
    competition: str | None = None,
    *,
    json_output: bool = False,
) -> int:
    """P11.5: print the highest-scoring submission with full provenance.

    Answers the question "what code produced my best leaderboard score?".
    """
    recs = SubmissionStore().all()
    if competition:
        recs = [r for r in recs if r.competition == competition]
    scored = [r for r in recs if r.score is not None]
    if not scored:
        print("No scored submissions recorded yet.", file=sys.stderr)
        return 1
    best = max(scored, key=lambda r: r.score)
    if json_output:
        print(json.dumps(asdict(best), indent=2))
        return 0
    print(f"Rank:        {best.rank if best.rank is not None else '-'}")
    print(f"Score:       {best.score:.5f}")
    print(f"Submission:  {best.submission_id or '-'}")
    print(f"Experiment:  {best.experiment_id or '-'}")
    print(f"Git commit:  {(best.git_commit or '-')[:12]}")
    if best.git_dirty is not None:
        print(f"Git state:   {'dirty' if best.git_dirty else 'clean'}")
    else:
        print("Git state:   (not a git repo)")
    print("Feature branches:")
    if best.feature_branches:
        for b in best.feature_branches:
            print(f"  - {b}")
    else:
        print("  (none recorded)")
    print("Dataset versions:")
    if best.dataset_versions:
        for k, v in best.dataset_versions.items():
            print(f"  - {k} v{v}")
    else:
        print("  (none recorded)")
    if best.artifact_hashes:
        print("Artifact hashes:")
        for k, v in best.artifact_hashes.items():
            print(f"  - {k[:12]}  {v[:12]}")
    return 0


def cmd_submissions_show(submission_id: str, *, json_output: bool = False) -> int:
    """P11.5: show full provenance for a single submission by id."""
    store = SubmissionStore()
    match = None
    for r in store.all():
        if r.submission_id == submission_id:
            match = r
            break
    if match is None:
        # Fallback: treat the argument as a local index (timestamp-based).
        try:
            idx = int(submission_id)
            recs = store.all()
            if 0 <= idx < len(recs):
                match = recs[idx]
        except ValueError:
            pass
    if match is None:
        print(f"No submission with id {submission_id!r}.", file=sys.stderr)
        return 1
    if json_output:
        print(json.dumps(asdict(match), indent=2))
        return 0
    print(json.dumps(asdict(match), indent=2))
    return 0


def _parse_leaderboard_rows(stdout: str) -> list[dict[str, str]]:
    import csv as _csv
    import io as _io
    if not stdout or not stdout.strip():
        return []
    return [dict(row) for row in _csv.DictReader(_io.StringIO(stdout))]


def cmd_submissions_watch(
    competition: str,
    *,
    current: str | None = None,
    poll_sec: int = 60,
    max_wait_sec: int = 1800,
    json_output: bool = False,
) -> int:
    """Watch a competition for new submissions and report when scored.

    Polls `kaggle competitions submissions` for the user's own submissions
    and prints any change in score. Backs off on each tick.

    `current`: an existing submission id; if None, picks the most recent
    one for this competition from the local SubmissionStore.
    """
    import time
    seen: dict[str, str | None] = {}  # submission_id -> last known score
    # Seed with local store.
    for r in SubmissionStore().all():
        if r.competition == competition and r.score is not None:
            seen[r.submission_id or f"local-{r.submitted_at}"] = str(r.score)
    if current:
        if current not in seen:
            seen[current] = None
    deadline = time.time() + max_wait_sec
    while time.time() < deadline:
        result = runner.run(["competitions", "submissions", "-c", competition])
        rows = _parse_leaderboard_rows(result.stdout or "")
        for row in rows:
            sid = row.get("teamId") or row.get("id") or row.get("ref")
            score = row.get("score") or row.get("publicScore") or row.get("privateScore")
            if sid and (sid not in seen or seen[sid] != score):
                seen[sid] = score
                msg = {
                    "competition": competition,
                    "submission_id": sid,
                    "score": score,
                    "watched_at": time.time(),
                }
                if json_output:
                    print(json.dumps(msg))
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] {sid} -> score={score}")
        time.sleep(poll_sec)
    return 0


def cmd_leaderboard_latest(competition: str, *, top: int = 20,
                            json_output: bool = False) -> int:
    """Show the latest competition leaderboard entries that have scored the
    user's most recent submission.

    Combines `kaggle competitions leaderboard` with the local SubmissionStore
    to surface the user's rank and score.
    """
    # Pull leaderboard
    result = runner.run(["competitions", "leaderboard", "-c", competition, "--csv"])
    rows = _parse_leaderboard_rows(result.stdout or "")
    rows = rows[:top]
    # Find this user's most recent submission
    my_latest = SubmissionStore().latest(competition)
    my_score = None
    if my_latest is not None and my_latest.score is not None:
        my_score = str(my_latest.score)
    if json_output:
        print(json.dumps({
            "competition": competition,
            "top": rows,
            "my_submission": asdict(my_latest) if my_latest else None,
            "my_score": my_score,
        }, indent=2))
        return 0
    print(f"Leaderboard (top {top}) for {competition}:")
    print(f"{'TEAM':<35}  {'SCORE'}")
    for r in rows:
        team = r.get("teamName") or r.get("teamId") or "?"
        score = r.get("score") or "?"
        print(f"{team:<35}  {score}")
    if my_latest is not None:
        print(f"\nYour most recent submission: {my_latest.submission_id or '-'} (score={my_score or 'pending'})")
    return 0
