"""competitions — list, files, submit, leaderboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from kagglepipe import kaggle_api


def list_competitions(*, csv_output: bool = False, json_output: bool = False) -> int:
    """List active competitions."""
    rows = kaggle_api.competitions_list()
    if json_output:
        print(json.dumps(rows, indent=2))
        return 0
    if csv_output:
        if rows:
            keys = list(rows[0].keys())
            print(",".join(keys))
            for r in rows:
                print(",".join(r.get(c, "") for c in keys))
        return 0
    if not rows:
        print("No active competitions.")
        return 0
    for r in rows:
        ref = r.get("ref", "?")
        title = r.get("title", "")
        deadline = r.get("deadline", "?")
        print(f"{ref:<40}  {title:<40}  {deadline}")
    return 0


def competition_files(slug: str) -> int:
    """List files in a competition."""
    # The kaggle CLI's `competitions files` doesn't support --csv; fall back
    # to plain output and pass through.
    from kagglepipe import runner

    result = runner.run(["competitions", "files", "-c", slug])
    if result.stdout:
        print(result.stdout, file=sys.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode


def competition_submit(slug: str, file: Path, message: str) -> int:
    """Submit a file to a competition."""
    if not file.exists():
        print(f"File not found: {file}", file=sys.stderr)
        return 1
    kaggle_api.competitions_submit(slug, file, message)
    print(f"Submitted {file} to {slug}")
    return 0


def competition_leaderboard(slug: str, *, top: int | None = 20,
                            csv_output: bool = False, json_output: bool = False) -> int:
    """Show the leaderboard for a competition."""
    rows = kaggle_api.competitions_leaderboard(slug, top=top)
    if json_output:
        print(json.dumps(rows, indent=2))
        return 0
    if csv_output:
        if rows:
            keys = list(rows[0].keys())
            print(",".join(keys))
            for r in rows:
                print(",".join(r.get(c, "") for c in keys))
        return 0
    if not rows:
        print("Leaderboard unavailable.", file=sys.stderr)
        return 1
    for r in rows:
        team = r.get("teamName", r.get("teamId", "?"))
        score = r.get("score", "?")
        print(f"{team:<40}  {score}")
    return 0
