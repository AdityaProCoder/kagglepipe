"""Tests for the dependency graph (P4)."""

from __future__ import annotations

import pytest

from kagglepipe.commands.graph import CycleError, build_plan, topo_sort, waves
from kagglepipe.config import Config, FeatureSection


def _cfg(deps: dict[str, list[str]]) -> Config:
    return Config(
        feature=FeatureSection(dependencies=deps, branches=list({b for upstream in deps.values() for b in upstream} | set(deps.keys())))
    )


def test_topo_sort_simple_chain() -> None:
    deps = {"b": ["a"]}
    order = topo_sort("b", deps)
    assert order.index("a") < order.index("b")


def test_topo_sort_diamond() -> None:
    deps = {
        "top": ["left", "right"],
        "left": ["base"],
        "right": ["base"],
        "base": [],
    }
    order = topo_sort("top", deps)
    assert order[0] == "base"
    assert order[-1] == "top"
    assert order.index("left") < order.index("top")
    assert order.index("right") < order.index("top")


def test_topo_sort_detects_cycle() -> None:
    deps = {"a": ["b"], "b": ["a"]}
    with pytest.raises(CycleError):
        topo_sort("a", deps)


def test_topo_sort_three_cycle() -> None:
    deps = {"a": ["b"], "b": ["c"], "c": ["a"]}
    with pytest.raises(CycleError):
        topo_sort("a", deps)


def test_build_plan() -> None:
    cfg = _cfg({"meta": ["graph"], "graph": ["user"]})
    plan = build_plan(cfg, "meta")
    assert plan == ["user", "graph", "meta"]


def test_waves() -> None:
    deps = {
        "top": ["left", "right"],
        "left": ["base"],
        "right": ["base"],
    }
    w = waves("top", deps)
    assert w[0] == ["base"]
    assert sorted(w[1]) == ["left", "right"]
    assert w[2] == ["top"]


def test_waves_independent_branches() -> None:
    """Two branches with no deps should land in the same wave."""
    deps = {"a": [], "b": []}
    w = waves("a", deps)  # a is the target
    # We may not see "b" in the plan if we only ask for "a"'s closure
    # so test with both
    deps = {"a": [], "b": [], "combo": ["a", "b"]}
    w = waves("combo", deps)
    assert w[0] == ["a", "b"] or sorted(w[0]) == ["a", "b"]
    assert w[1] == ["combo"]
