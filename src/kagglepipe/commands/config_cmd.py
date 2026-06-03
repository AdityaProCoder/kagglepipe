"""config init / show / path commands."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from kagglepipe import config as cfg_mod


def init(
    path: Path | None = None,
    *,
    project_name: str | None = None,
    force: bool = False,
) -> int:
    """Scaffold a kaggle.toml in the current directory."""
    target = path or Path.cwd() / cfg_mod.CONFIG_FILENAME
    if target.exists() and not force:
        print(f"{target} already exists. Pass --force to overwrite.", file=sys.stderr)
        return 1
    if target.exists() and force:
        target.unlink()
    written = cfg_mod.scaffold(target, project_name=project_name)
    print(f"Wrote {written}")
    return 0


def show(*, json_output: bool = False) -> int:
    """Print the effective config (file + defaults + env overrides)."""
    cfg = cfg_mod.load()
    if json_output:
        print(json.dumps(cfg_mod.to_dict(cfg), indent=2))
        return 0
    # Pretty print
    d = cfg_mod.to_dict(cfg)
    if d.get("_config_path"):
        print(f"# Loaded from: {d['_config_path']}")
    for section, values in d.items():
        if section.startswith("_"):
            continue
        print(f"\n[{section}]")
        if not values:
            print("  (empty)")
            continue
        for k, v in values.items():
            print(f"  {k} = {v!r}")
    return 0


def path() -> int:
    """Print the path of the kaggle.toml that would be loaded."""
    target = Path.cwd() / cfg_mod.CONFIG_FILENAME
    if target.exists():
        print(target)
        return 0
    print(f"(no {cfg_mod.CONFIG_FILENAME} in {Path.cwd()}; using defaults)", file=sys.stderr)
    return 1
