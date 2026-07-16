"""Render parameterized Jupyter notebooks for kagglepipe.

Notebooks are produced by rendering a Jinja2 template with a context dict
(branch, src/data slugs, mounts, etc.). The default template is shipped at
`kagglepipe/templates/notebook_default.py.j2`; users can override the path
in `kaggle.toml` (`feature.notebook_template`).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template

DEFAULT_TEMPLATE_PACKAGE = "kagglepipe.templates.notebook_default"


def _load_template(template_ref: str) -> Template:
    """Load a Jinja2 template from either a package dotted path or a file path.

    Accepted forms:
      - "some.pkg.template"        -> loads some/pkg/template.{name}.j2
        (where {name} is the last segment, e.g. "notebook_default")
      - "/abs/path/to/foo.j2"      -> loads directly
      - "relative/foo.j2"          -> loads relative to CWD
    """
    # Try package resource first.
    if "." in template_ref and not Path(template_ref).exists():
        try:
            module_name, _, _ = template_ref.rpartition(".")
            module = importlib.import_module(module_name)
            # Look for a .j2 file alongside the module.
            module_file = Path(module.__file__).parent
            last = template_ref.rsplit(".", 1)[-1]
            for ext in (".py.j2", ".j2"):
                candidate = module_file / f"{last}{ext}"
                if candidate.exists():
                    env = Environment(
                        loader=FileSystemLoader(str(module_file)),
                        undefined=StrictUndefined,
                        autoescape=False,
                    )
                    return env.get_template(candidate.name)
        except (ImportError, AttributeError):
            pass
    # Fall back to a direct file path.
    direct = Path(template_ref)
    if direct.is_absolute() or direct.exists():
        if not direct.is_file():
            raise FileNotFoundError(
                f"Could not find notebook template {template_ref!r} as a package resource or a file."
            )
        env = Environment(
            loader=FileSystemLoader(str(direct.parent)),
            undefined=StrictUndefined,
            autoescape=False,
        )
        return env.get_template(direct.name)
    raise FileNotFoundError(
        f"Could not find notebook template {template_ref!r} as a package resource or a file."
    )


def render(
    template_ref: str,
    *,
    branch: str,
    src_dataset_slug: str,
    src_version: int,
    src_mount: str,
    data_dataset_slug: str = "",
    data_mount: str = "",
    out_dir: str = "/kaggle/working/features",
    notebook_command: str = "",
    gpu: str | None = None,
    date: str | None = None,
) -> dict[str, Any]:
    """Render the notebook JSON for a feature run.

    Returns a dict with keys `cells`, `metadata`, `nbformat`, `nbformat_minor`
    suitable for serialization via `json.dumps`.
    """
    template = _load_template(template_ref)
    when = date or datetime.now(UTC).strftime("%Y-%m-%d")
    rendered = template.render(
        branch=branch,
        src_dataset_slug=src_dataset_slug,
        src_version=src_version,
        src_mount=src_mount,
        data_dataset_slug=data_dataset_slug,
        data_mount=data_mount,
        out_dir=out_dir,
        notebook_command=notebook_command,
        date=when,
    )
    nb = json.loads(rendered)
    # Apply runtime metadata that depends on flags the template can't see.
    nb.setdefault("metadata", {})
    if gpu is not None:
        nb["metadata"]["gpuInstanceConfig"] = gpu
        nb["metadata"]["accelerator"] = "gpu"
    # Build dataset_sources from whatever the template included plus the
    # canonical {src, data} slugs.
    sources = list(nb["metadata"].get("dataset_sources", []))
    for s in (src_dataset_slug, data_dataset_slug):
        if s and s not in sources:
            sources.append(s)
    nb["metadata"]["dataset_sources"] = sources
    return nb


def write_kernel_metadata(
    *,
    kernel_slug: str,
    title: str,
    code_file: str,
    dataset_sources: list[str],
    enable_internet: bool,
    is_private: bool,
    language: str,
    kernel_type: str,
    gpu: str | None = None,
) -> dict[str, Any]:
    """Build the `kernel-metadata.json` payload for `kaggle kernels push`."""
    md: dict[str, Any] = {
        "id": kernel_slug,
        "title": title,
        "code_file": code_file,
        "language": language,
        "kernel_type": kernel_type,
        "is_private": is_private,
        "enable_gpu": gpu is not None,
        "enable_internet": enable_internet,
        "dataset_sources": [s for s in dataset_sources if s],
    }
    if gpu is not None:
        md["accelerator"] = "gpu"
        md["gpuInstanceConfig"] = gpu
    return md


def write_dataset_metadata(*, slug: str, title: str | None = None) -> dict[str, Any]:
    """Build a minimal `dataset-metadata.json` for `kaggle datasets create`."""
    return {
        "title": title or slug.split("/")[-1],
        "id": slug,
        "licenses": [{"name": "CC0-1.0"}],
    }
