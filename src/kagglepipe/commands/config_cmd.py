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
    username: str | None = None,
    force: bool = False,
    auto: bool = False,
) -> int:
    """Scaffold a kaggle.toml in the current directory.

    If --auto is set, also fill in the Kaggle username (from credentials)
    and auto-detect feature branches from the features/ directory.
    """
    target = path or Path.cwd() / cfg_mod.CONFIG_FILENAME
    if target.exists() and not force:
        print(f"{target} already exists. Pass --force to overwrite.", file=sys.stderr)
        return 1
    if target.exists() and force:
        target.unlink()

    name = project_name or Path.cwd().name.lower().replace(" ", "-")

    # Resolve username from credentials if auto and not provided
    if auto and not username:
        try:
            from kagglepipe import credentials as creds_mod
            c = creds_mod.load()
            username = c.username
        except Exception:
            pass

    written = cfg_mod.scaffold(target, project_name=name, username=username)

    if auto:
        from kagglepipe import credentials as creds_mod
        try:
            c = creds_mod.load()
            cfg_mod.fill(target, username=c.username, project_name=name)
            print(f"Filled {target} with username={c.username}")
        except Exception:
            pass

        detected = cfg_mod.detect_branches()
        if detected:
            print(f"Detected {len(detected)} branches: {', '.join(detected)}")
            print("(edit kaggle.toml to enable them — set branches = [...])")

    print(f"Wrote {written}")
    return 0


def fill(
    path: Path | None = None,
    *,
    username: str | None = None,
    project_name: str | None = None,
) -> int:
    """Patch an existing kaggle.toml: resolve {username} and {project_name}
    placeholders using real values. Call this after `kagglepipe login`."""
    target = path or Path.cwd() / cfg_mod.CONFIG_FILENAME
    if not target.exists():
        print(f"No kaggle.toml found at {target}. Run `kagglepipe config init` first.", file=sys.stderr)
        return 1
    if not username:
        print("--username is required", file=sys.stderr)
        return 1
    cfg_mod.fill(target, username=username, project_name=project_name)
    print(f"Filled {target}")
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
