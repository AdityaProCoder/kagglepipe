"""Tests for slug normalization and template resolution."""

from __future__ import annotations

import pytest

from kagglepipe import slug


def test_normalize_slug_lowercases() -> None:
    assert slug.normalize_slug("DINOV3") == "dinov3"


def test_normalize_slug_replaces_underscores() -> None:
    assert slug.normalize_slug("face_buffalo_l") == "face-buffalo-l"


def test_normalize_slug_drops_other_chars() -> None:
    assert slug.normalize_slug("foo!@#bar") == "foo-bar"


def test_normalize_slug_strips_edges() -> None:
    assert slug.normalize_slug("--foo--") == "foo"


def test_resolve_template_username() -> None:
    assert slug.resolve_template("{username}/foo", username="alice") == "alice/foo"


def test_resolve_template_dollar_form() -> None:
    assert slug.resolve_template("$username/foo", username="alice") == "alice/foo"


def test_resolve_template_braces_form() -> None:
    assert slug.resolve_template("${username}/foo", username="alice") == "alice/foo"


def test_resolve_template_unknown_left_intact() -> None:
    # safe_substitute leaves unknowns alone rather than raising.
    assert slug.resolve_template("{nothing}/foo", username="alice") == "{nothing}/foo"
