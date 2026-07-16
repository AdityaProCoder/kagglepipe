"""Dataset lineage (P8).

`kagglepipe lineage <feature>` prints the upstream chain of artifacts
that led to a given feature. Backed by `.kagglepipe/lineage.json` which
stores a directed graph: feature -> list of upstream feature names.

Lineage is best-effort. We track it when:
  * a feature has declared `dependencies` in kaggle.toml (P4)
  * a parent feature's artifact is in the local features dir
  * the user explicitly records a parent via `kagglepipe lineage add-parent <child> <parent>`
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from kagglepipe.state import state_dir

LINEAGE_FILE = "lineage.json"


@dataclass
class LineageEdge:
    child: str
    parents: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> LineageEdge:
        return cls(**d)


def _path() -> Path:
    return state_dir() / LINEAGE_FILE


def _load() -> dict[str, LineageEdge]:
    if not _path().exists():
        return {}
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    out: dict[str, LineageEdge] = {}
    if isinstance(data, dict):
        for child, parents in data.items():
            if isinstance(parents, list):
                out[child] = LineageEdge(child=child, parents=parents)
    return out


def _save(graph: dict[str, LineageEdge]) -> None:
    _path().write_text(
        json.dumps({e.child: e.parents for e in graph.values()}, indent=2),
        encoding="utf-8",
    )


def set_parents(child: str, parents: list[str]) -> None:
    """Set the parent list for a feature (overwrites)."""
    graph = _load()
    graph[child] = LineageEdge(child=child, parents=list(parents))
    _save(graph)


def add_parent(child: str, parent: str) -> None:
    graph = _load()
    edge = graph.get(child) or LineageEdge(child=child, parents=[])
    if parent not in edge.parents:
        edge.parents.append(parent)
    graph[child] = edge
    _save(graph)


def remove(child: str) -> None:
    graph = _load()
    graph.pop(child, None)
    # Also remove child from anyone else's parents list.
    for e in graph.values():
        if child in e.parents:
            e.parents = [p for p in e.parents if p != child]
    _save(graph)


def chain(feature: str) -> list[str]:
    """Return the upstream chain for a feature (oldest first).

    The result includes the feature itself and every transitive parent.
    """
    graph = _load()
    seen: set[str] = set()
    out: list[str] = []
    stack = [feature]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        out.append(node)
        for p in graph.get(node, LineageEdge(node, [])).parents:
            if p not in seen:
                stack.append(p)
    return list(reversed(out))


def cmd_lineage(feature: str, *, json_output: bool = False) -> int:
    c = chain(feature)
    if json_output:
        print(json.dumps({"feature": feature, "chain": c}, indent=2))
        return 0
    if not c:
        print(f"No lineage recorded for {feature!r}.")
        return 0
    print(" -> ".join(c))
    return 0


def cmd_lineage_add_parent(child: str, parent: str) -> int:
    add_parent(child, parent)
    print(f"Recorded parent: {parent} -> {child}")
    return 0


def cmd_lineage_remove(feature: str) -> int:
    remove(feature)
    print(f"Removed lineage entry for {feature!r}")
    return 0
