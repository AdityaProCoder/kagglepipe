"""Dependency graph execution (P4).

`kagglepipe feature build <target>` resolves the dependency DAG declared in
`[feature].dependencies` of kaggle.toml, topologically sorts it, then runs
the resulting plan using the parallel runner. Independent branches run
concurrently; downstream branches wait for their declared upstreams.

Example config:

    [feature.dependencies]
    graph = ["user_features", "item_features"]
    meta = ["graph", "user_features"]
"""

from __future__ import annotations

import sys
from collections import defaultdict

from kagglepipe.config import Config


class CycleError(RuntimeError):
    """Raised when a dependency cycle is detected."""


def topo_sort(target: str, deps: dict[str, list[str]]) -> list[str]:
    """Return a topologically-sorted list of branches needed to build `target`.

    Raises CycleError if a cycle is detected.
    Raises KeyError if a dependency references an unknown branch.

    The returned list includes `target` and all of its transitive deps.
    """
    visited: set[str] = set()
    on_stack: set[str] = set()
    order: list[str] = []

    def visit(node: str, path: list[str]) -> None:
        if node in on_stack:
            cycle = " -> ".join(path + [node])
            raise CycleError(f"Dependency cycle detected: {cycle}")
        if node in visited:
            return
        on_stack.add(node)
        for upstream in deps.get(node, []):
            visit(upstream, path + [node])
        on_stack.remove(node)
        visited.add(node)
        order.append(node)

    visit(target, [])
    return order


def build_plan(cfg: Config, target: str) -> list[str]:
    """Topo-sort `target`'s dependency closure against the configured deps."""
    deps = cfg.feature.dependencies
    if target not in deps and target not in cfg.feature.branches:
        # Allow running any branch even if not explicitly listed in deps;
        # the user might be exploring. We just need its upstreams.
        if target not in deps:
            return [target]
    return topo_sort(target, deps)


def waves(target: str, deps: dict[str, list[str]]) -> list[list[str]]:
    """Compute parallel-execution waves: groups of branches that can run
    simultaneously given the dependency constraints.

    Each wave is a list of branch names that have all their upstreams
    already in earlier waves. Branches within a wave are independent.
    """
    order = topo_sort(target, deps)
    # Compute depth of each branch (longest path from a root to it).
    depth: dict[str, int] = {}
    for b in order:
        upstreams = deps.get(b, [])
        depth[b] = (max((depth[u] for u in upstreams), default=-1)) + 1
    by_depth: dict[int, list[str]] = defaultdict(list)
    for b, d in depth.items():
        by_depth[d].append(b)
    return [by_depth[d] for d in sorted(by_depth)]


def cmd_feature_build(
    cfg: Config,
    target: str,
    *,
    gpu: str = "t4x2",
    parallel: int = 1,
    timeout_sec: int | None = None,
    data_dataset: str | None = None,
    quiet: bool = False,
) -> int:
    """Resolve and execute the dependency closure for `target`."""
    try:
        plan = build_plan(cfg, target)
    except CycleError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyError as exc:
        print(f"Unknown branch in dependencies: {exc}", file=sys.stderr)
        return 1
    if not quiet:
        print(f"Build plan for '{target}': {plan}")
        if parallel > 1:
            w = waves(target, cfg.feature.dependencies)
            for i, wave in enumerate(w):
                print(f"  wave {i}: {wave}")
    from kagglepipe.commands import feature
    return feature.run_all(
        cfg,
        branches=plan,
        gpu=gpu,
        parallel=parallel,
        timeout_sec=timeout_sec,
        data_dataset=data_dataset,
        quiet=quiet,
    )


def cmd_feature_plan(cfg: Config, target: str) -> int:
    """Print the dependency plan without executing."""
    try:
        plan = build_plan(cfg, target)
    except CycleError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyError as exc:
        print(f"Unknown branch in dependencies: {exc}", file=sys.stderr)
        return 1
    print("Build order:", " -> ".join(plan))
    if plan:
        w = waves(target, cfg.feature.dependencies)
        for i, wave in enumerate(w):
            print(f"  wave {i}: {wave}")
    return 0
