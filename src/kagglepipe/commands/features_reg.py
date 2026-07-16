"""Feature registry (P7).

`kagglepipe features list` / `features show <name>` look at the local
registry of generated features. Each entry is recorded by the feature
runner on success.

A feature is "registered" whenever `feature run` completes successfully
and downloads an artifact. The registry is the canonical local answer to
"What features do I have and where do they live?".
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict

from kagglepipe.state import FeatureStore


def cmd_features_list(*, csv_output: bool = False, json_output: bool = False) -> int:
    recs = FeatureStore().all()
    if json_output:
        print(json.dumps([asdict(r) for r in recs], indent=2))
        return 0
    if csv_output:
        if recs:
            keys = list(asdict(recs[0]).keys())
            print(",".join(keys))
            for r in recs:
                print(",".join(str(v) for v in asdict(r).values()))
        return 0
    if not recs:
        print("No features registered. Run `kagglepipe feature run <branch>` to register one.")
        return 0
    print(f"{'NAME':<30} {'VERSION':<8} {'BRANCH':<20} {'DATASET SLUG'}")
    for r in recs:
        branch = r.branch or "-"
        print(f"{r.name:<30} {r.version:<8} {branch:<20} {r.dataset_slug}")
    return 0


def cmd_features_show(name: str) -> int:
    matches = FeatureStore().get(name)
    if not matches:
        print(f"No features registered as {name!r}.", file=sys.stderr)
        return 1
    print(f"{'VERSION':<8} {'ARTIFACT':<60} {'BRANCH':<20} {'CREATED'}")
    for r in matches:
        ts = __import__("time").strftime("%Y-%m-%d %H:%M:%S", __import__("time").localtime(r.created_at))
        print(f"{r.version:<8} {r.artifact_path:<60} {r.branch or '-':<20} {ts}")
    return 0
