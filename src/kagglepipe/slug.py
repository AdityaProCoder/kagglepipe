"""Slug and template helpers."""

from __future__ import annotations

import re
from string import Template

# Kaggle normalizes slugs: lowercase, hyphens instead of underscores.
_NORMALIZE_RE = re.compile(r"[^a-z0-9-]+")


def normalize_slug(s: str) -> str:
    """Normalize a branch / kernel name to a Kaggle-safe slug.

    Kaggle's web UI maps arbitrary strings to lowercase-with-hyphens slugs
    (e.g., 'face_buffalo_l' -> 'face-buffalo-l'). Use this when looking up
    kernels by name so we match the on-disk form.
    """
    return _NORMALIZE_RE.sub("-", s.lower()).strip("-")


def resolve_template(template: str, *, username: str, **extra: str) -> str:
    """Resolve a `"{username}/foo"`-style template.

    Accepts `{name}`, `${name}`, and `$name` forms. Only `username` (always
    supplied) and the `**extra` keys are substituted; anything else is left
    intact so callers can layer their own substitution on top (e.g., a
    branch name).
    """
    import re

    variables: dict[str, str] = {"username": username, **extra}
    # Normalize `{name}` and `${name}` to `$name` for any *known* variable.
    def _maybe_replace(m: re.Match[str]) -> str:
        name = m.group(1) or m.group(2)
        if name in variables:
            return f"${name}"
        return m.group(0)

    normalized = re.sub(r"\$\{(\w+)\}|\{(\w+)\}", _maybe_replace, template)
    return Template(normalized).safe_substitute(variables)
