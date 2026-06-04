"""Pre-flight validation (P10).

`kagglepipe validate` runs a battery of checks and reports any failures,
so users can fix config / credentials / templates before launching a real
(expensive) run.

Categories checked:
  1. Credentials (kaggle.json or env vars loadable)
  2. Config schema (kaggle.toml parses, defaults don't collide)
  3. Source dataset existence (src_dataset_slug exists on Kaggle)
  4. Branch configuration (branches must not be empty)
  5. Dependency graph (no cycles, all referenced branches known)
  6. Notebook template (renders without error)
  7. Source paths (every `source.include` entry exists)
  8. Output glob (parses, has a recognizable extension)
  9. Dataset configuration (data.dataset_slug, source.src_dataset_slug are well-formed)
  10. GPU setting (one of the supported values)
  11. State directory (writable)
  12. Competition configuration (if competition.slug is set)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable

from kagglepipe import credentials, kaggle_api, notebook as nb_mod, runner, slug
from kagglepipe.config import Config, load
from kagglepipe.commands.feature import resolve_notebook_command
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
            notebook_command=resolve_notebook_command(
                cfg.feature.notebook_command,
                username=creds_username,
                branch="__validate__",
                out_dir=cfg.feature.out_dir,
                src_mount=slug.resolve_template(
                    cfg.feature.src_mount, username=creds_username, dataset=src_slug.split("/", 1)[-1]
                ),
                data_mount=(
                    slug.resolve_template(
                        cfg.feature.data_mount, username=creds_username, dataset=data_slug.split("/", 1)[-1]
                    )
                    if data_slug else ""
                ),
            ),
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


def _check_branches(cfg: Config) -> list[str]:
    """Fail if branches is empty — nothing can be run."""
    issues: list[str] = []
    if not cfg.feature.branches:
        issues.append(
            "feature.branches is empty. "
            "Add branches to [feature] in kaggle.toml, e.g.:\n"
            "  branches = [\"baseline\", \"user_features\"]"
        )
    return issues


def _check_src_dataset_exists(cfg: Config, username: str) -> list[str]:
    """Verify src_dataset_slug actually exists on the Kaggle account.

    Fails validation only if the dataset is definitively absent.
    If the API call itself fails (network, auth), this check is skipped
    so transient issues don't block validation.
    """
    issues: list[str] = []
    src = cfg.source.src_dataset_slug
    if not src:
        return issues
    slug_str = slug.resolve_template(src, username=username)
    result = runner.run(["datasets", "list", "--user", username, "--csv"])
    # If the API call itself failed, skip this check rather than block
    # the user on a potentially transient issue (auth, network, rate-limit).
    if result.returncode != 0:
        return issues
    refs: set[str] = set()
    for line in result.stdout.splitlines()[1:]:
        parts = line.split(",")
        if parts:
            refs.add(parts[0].strip('"'))
    if slug_str not in refs:
        issues.append(
            f"source dataset {slug_str!r} not found on Kaggle. "
            f"Run `kagglepipe src upload` first, or check source.src_dataset_slug in kaggle.toml."
        )
    return issues


def _check_competition(cfg: Config) -> list[str]:
    """If competition is configured, validate its required fields."""
    issues: list[str] = []
    comp = cfg.competition
    if not comp or not comp.get("slug"):
        return issues
    slug_str = comp.get("slug", "")
    if not slug_str:
        issues.append("competition.slug is empty")
        return issues
    # Verify the competition exists.
    result = runner.run(["competitions", "list", "--csv"])
    known: set[str] = set()
    for line in result.stdout.splitlines()[1:]:
        parts = line.split(",")
        if parts:
            # URL column: https://www.kaggle.com/competitions/slug
            url = parts[0].strip('"')
            if "/competitions/" in url:
                known.add(url.rsplit("/", 1)[-1])
    if slug_str not in known:
        issues.append(
            f"competition {slug_str!r} not found. "
            f"Check [competition].slug in kaggle.toml."
        )
    sub_path = comp.get("submission_path", "")
    if not sub_path:
        issues.append(
            "competition.submission_path is empty. "
            "Set a path like 'submission.csv' in kaggle.toml."
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
        issues.append(("branches", _check_branches(cfg)))
        issues.append(("src_dataset_exists", _check_src_dataset_exists(cfg, username)))
        issues.append(("output_glob", _check_output_glob(cfg)))
        issues.append(("gpu", _check_gpu(cfg)))
        issues.append(("dependency_graph", _check_dependency_graph(cfg)))
        issues.append(("notebook_template", _check_notebook_template(cfg, username, src_slug, data_slug)))
        issues.append(("state_dir", _check_state_dir()))
        issues.append(("dataset_slugs", _check_dataset_slugs(cfg)))
        issues.append(("competition", _check_competition(cfg)))

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
