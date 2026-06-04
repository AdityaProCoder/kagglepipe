"""Pre-flight validation (P10).

`kagglepipe validate` runs a battery of checks and reports any failures,
so users can fix config / credentials / templates before launching a real
(expensive) run.

Categories checked:
  1. Credentials (kaggle.json or env vars loadable)
  2. Config schema (kaggle.toml parses, defaults don't collide)
  3. Dependency graph (no cycles, all referenced branches known)
  4. Notebook template (renders without error)
  5. Source paths (every `source.include` entry exists)
  6. Output glob (parses, has a recognizable extension)
  7. Dataset configuration (data.dataset_slug, source.src_dataset_slug are well-formed)
  8. GPU setting (one of the supported values)
  9. State directory (writable)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable

from kagglepipe import credentials, notebook as nb_mod, slug
from kagglepipe.config import Config, load
from kagglepipe.state import state_dir


def _check_credentials() -> list[str]:
    issues: list[str] = []
    try:
        credentials.load()
    except credentials.CredentialsError as exc:
        issues.append(f"credentials: {exc}")
    return issues


def _check_paths(cfg: Config, root: Path) -> list[str]:
    issues: list[str] = []
    for entry in cfg.source.include:
        if not (root / entry).exists():
            issues.append(f"source.include: {entry!r} not found in {root}")
    return issues


def _check_output_glob(cfg: Config) -> list[str]:
    issues: list[str] = []
    pattern = cfg.feature.output_glob
    if not pattern:
        issues.append("feature.output_glob is empty")
        return issues
    # Must be a relative-style glob containing {branch} or a literal name.
    if "{branch}" not in pattern and "*" not in pattern:
        issues.append(
            f"feature.output_glob {pattern!r} should include {{branch}} or '*'"
        )
    if not re.search(r"\.[A-Za-z0-9]+$", pattern):
        issues.append(f"feature.output_glob {pattern!r} has no file extension")
    return issues


def _check_gpu(cfg: Config) -> list[str]:
    issues: list[str] = []
    valid = {"p100", "t4x2", "none"}
    if cfg.feature.default_gpu not in valid:
        issues.append(
            f"feature.default_gpu {cfg.feature.default_gpu!r} is not one of {sorted(valid)}"
        )
    return issues


def _check_dependency_graph(cfg: Config) -> list[str]:
    issues: list[str] = []
    from kagglepipe.commands.graph import build_plan
    for target in list(cfg.feature.dependencies.keys()) or list(cfg.feature.branches):
        try:
            build_plan(cfg, target)
        except Exception as exc:  # CycleError, KeyError
            issues.append(f"dependency graph: {target}: {exc}")
    return issues


def _check_notebook_template(cfg: Config, creds_username: str, src_slug: str, data_slug: str) -> list[str]:
    issues: list[str] = []
    try:
        nb = nb_mod.render(
            cfg.feature.notebook_template,
            branch="__validate__",
            src_dataset_slug=src_slug,
            src_version=1,
            src_mount=slug.resolve_template(
                cfg.feature.src_mount, username=creds_username, dataset=src_slug.split("/", 1)[-1]
            ),
            data_dataset_slug=data_slug,
            data_mount=(
                slug.resolve_template(
                    cfg.feature.data_mount, username=creds_username, dataset=data_slug.split("/", 1)[-1]
                )
                if data_slug else ""
            ),
            out_dir=cfg.feature.out_dir,
            notebook_command=cfg.feature.notebook_command,
            gpu="t4 x2",
        )
    except Exception as exc:
        issues.append(f"notebook template {cfg.feature.notebook_template!r}: {exc}")
        return issues
    # Must be a valid notebook dict.
    if not isinstance(nb, dict) or "cells" not in nb or "nbformat" not in nb:
        issues.append("notebook template did not produce a valid notebook dict")
    return issues


def _check_state_dir() -> list[str]:
    issues: list[str] = []
    try:
        d = state_dir()
        # Try to write a temp file inside it.
        test = d / ".validate_test"
        test.write_text("ok", encoding="utf-8")
        test.unlink()
    except Exception as exc:
        issues.append(f"state dir: {exc}")
    return issues


def _check_dataset_slugs(cfg: Config) -> list[str]:
    issues: list[str] = []
    src = cfg.source.src_dataset_slug
    if not src:
        issues.append("source.src_dataset_slug is empty")
    elif "{" in src and "username" not in src and "dataset" not in src:
        issues.append(
            f"source.src_dataset_slug {src!r} has braces but no {{username}} or {{dataset}} placeholder"
        )
    data = cfg.data.dataset_slug
    if data and "{" in data and "username" not in data and "dataset" not in data:
        issues.append(
            f"data.dataset_slug {data!r} has braces but no {{username}} or {{dataset}} placeholder"
        )
    return issues


def cmd_validate(*, json_output: bool = False) -> int:
    """Run all pre-flight checks. Returns 0 if everything passes, 1 otherwise."""
    cfg = load()
    root = Path.cwd()
    issues: list[tuple[str, list[str]]] = []
    creds = None
    try:
        creds = credentials.load()
        username = creds.username
    except credentials.CredentialsError as exc:
        username = "<unauthenticated>"
        issues.append(("credentials", [str(exc)]))

    src_slug = slug.resolve_template(cfg.source.src_dataset_slug, username=username)
    data_slug = (
        slug.resolve_template(cfg.data.dataset_slug, username=username)
        if cfg.data.dataset_slug
        else ""
    )

    if creds is not None:
        issues.append(("paths", _check_paths(cfg, root)))
        issues.append(("output_glob", _check_output_glob(cfg)))
        issues.append(("gpu", _check_gpu(cfg)))
        issues.append(("dependency_graph", _check_dependency_graph(cfg)))
        issues.append(("notebook_template", _check_notebook_template(cfg, username, src_slug, data_slug)))
        issues.append(("state_dir", _check_state_dir()))
        issues.append(("dataset_slugs", _check_dataset_slugs(cfg)))

    if json_output:
        import json as _json
        print(_json.dumps(
            [{"category": c, "issues": i} for c, i in issues if i],
            indent=2,
        ))
        return 0 if not any(i for _, i in issues) else 1

    print("Validating kagglepipe setup...")
    fail = False
    for category, items in issues:
        if not items:
            print(f"  [ok]    {category}")
            continue
        fail = True
        print(f"  [FAIL]  {category}")
        for it in items:
            print(f"          - {it}")
    if fail:
        print("\nValidation failed. Fix the issues above before running.")
        return 1
    print("\nAll checks passed.")
    return 0
