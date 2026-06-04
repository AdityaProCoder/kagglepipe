"""Tests for `kagglepipe template init` scaffolding."""

from __future__ import annotations

from pathlib import Path

import pytest

from kagglepipe.commands.templates import cmd_template_init


@pytest.mark.parametrize("template_name", ["tabular", "cv", "nlp"])
def test_template_init_emits_single_brace_branch_placeholder(
    tmp_path: Path, template_name: str
) -> None:
    project_root = tmp_path / template_name
    project_root.mkdir()

    rc = cmd_template_init(template_name, project_name=f"{template_name}-demo", root=project_root)
    assert rc == 0

    kaggle_toml = (project_root / "kaggle.toml").read_text(encoding="utf-8")
    assert "configs/features/{branch}.yaml" in kaggle_toml
    assert "{{branch}}" not in kaggle_toml
