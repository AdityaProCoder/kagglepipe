"""Tests for the notebook renderer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kagglepipe import notebook as nb_mod


def test_render_produces_valid_ipynb_dict() -> None:
    nb = nb_mod.render(
        "kagglepipe.templates.notebook_default",
        branch="dinov3",
        src_dataset_slug="u/src",
        src_version=3,
        src_mount="/kaggle/input/src",
        notebook_command="python run.py --out {out_dir}",
        out_dir="/kaggle/working/features",
    )
    assert nb["nbformat"] == 4
    assert "cells" in nb
    # Without gpu passed, metadata should NOT have gpuInstanceConfig.
    assert "gpuInstanceConfig" not in nb["metadata"]


def test_render_includes_gpu_instance_when_provided() -> None:
    nb = nb_mod.render(
        "kagglepipe.templates.notebook_default",
        branch="x",
        src_dataset_slug="u/src",
        src_version=1,
        src_mount="/kaggle/input/src",
        notebook_command="python run.py",
        out_dir="/kaggle/working/features",
        gpu="t4 x2",
    )
    assert nb["metadata"]["gpuInstanceConfig"] == "t4 x2"
    assert nb["metadata"]["accelerator"] == "gpu"


def test_render_includes_dataset_sources() -> None:
    nb = nb_mod.render(
        "kagglepipe.templates.notebook_default",
        branch="x",
        src_dataset_slug="u/src",
        src_version=1,
        src_mount="/kaggle/input/src",
        data_dataset_slug="u/data",
        data_mount="/kaggle/input/data",
        notebook_command="echo hi",
        out_dir="/kaggle/working/features",
    )
    sources = nb["metadata"]["dataset_sources"]
    assert "u/src" in sources
    assert "u/data" in sources


def test_render_substitutes_branch_name() -> None:
    nb = nb_mod.render(
        "kagglepipe.templates.notebook_default",
        branch="my_branch",
        src_dataset_slug="u/src",
        src_version=1,
        src_mount="/kaggle/input/src",
        notebook_command="echo {branch}",
        out_dir="/kaggle/working/features",
    )
    text = json.dumps(nb)
    assert "my_branch" in text


def test_render_custom_template_from_file(tmp_path: Path) -> None:
    template = tmp_path / "my_template.py.j2"
    template.write_text(
        '{"cells": [{"cell_type": "code", "execution_count": null, "metadata": {}, '
        '"outputs": [], "source": ["print(\\"{{ branch }}\\")"]}], '
        '"metadata": {"dataset_sources": ["{{ src_dataset_slug }}"]}, '
        '"nbformat": 4, "nbformat_minor": 4}'
    )
    nb = nb_mod.render(
        str(template),
        branch="custom",
        src_dataset_slug="u/src",
        src_version=1,
        src_mount="/kaggle/input/src",
        notebook_command="echo",
        out_dir="/kaggle/working/features",
    )
    text = json.dumps(nb)
    assert "custom" in text
    assert "u/src" in text


def test_render_unknown_template_raises() -> None:
    with pytest.raises(FileNotFoundError):
        nb_mod.render(
            "/nonexistent/path.j2",
            branch="x",
            src_dataset_slug="u",
            src_version=1,
            src_mount="/kaggle/input/x",
            notebook_command="echo",
            out_dir="/kaggle/working/features",
        )


def test_write_kernel_metadata_contains_required_fields() -> None:
    md = nb_mod.write_kernel_metadata(
        kernel_slug="u/k",
        title="t",
        code_file="a.ipynb",
        dataset_sources=["u/d"],
        enable_internet=True,
        is_private=True,
        language="python",
        kernel_type="notebook",
    )
    assert md["id"] == "u/k"
    assert md["code_file"] == "a.ipynb"
    assert md["enable_gpu"] is False


def test_write_kernel_metadata_with_gpu() -> None:
    md = nb_mod.write_kernel_metadata(
        kernel_slug="u/k",
        title="t",
        code_file="a.ipynb",
        dataset_sources=[],
        enable_internet=True,
        is_private=True,
        language="python",
        kernel_type="notebook",
        gpu="t4 x2",
    )
    assert md["enable_gpu"] is True
    assert md["gpuInstanceConfig"] == "t4 x2"
    assert md["accelerator"] == "gpu"


def test_write_dataset_metadata_minimal() -> None:
    md = nb_mod.write_dataset_metadata(slug="u/foo")
    assert md["id"] == "u/foo"
    assert md["title"] == "foo"
    assert md["licenses"][0]["name"] == "CC0-1.0"
