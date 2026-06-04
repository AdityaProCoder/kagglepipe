"""Configuration loader for kagglepipe.

Reads `kaggle.toml` from the current working directory (or a path supplied by
the caller) into a typed `Config` dataclass. Missing fields are filled from
`_DEFAULTS`. Env vars (prefix `KAGGLEPIPE_`) override file values.
"""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    pass  # tomllib is in stdlib
else:  # pragma: no cover - we require 3.11+
    import tomli as tomllib  # type: ignore[no-redef]


CONFIG_FILENAME = "kaggle.toml"


# --- dataclass model -----------------------------------------------------


@dataclass
class ProjectSection:
    name: str = "kagglepipe"


@dataclass
class SourceSection:
    include: list[str] = field(
        default_factory=lambda: ["src", "configs", "scripts", "pyproject.toml", "README.md"]
    )
    exclude_dirs: list[str] = field(
        default_factory=lambda: [
            ".venv",
            "data",
            "models",
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            ".mypy_cache",
            "kaggle_notebooks",
            "submissions",
        ]
    )
    exclude_exts: list[str] = field(
        default_factory=lambda: [".parquet", ".lgb", ".pt", ".pth", ".bin"]
    )
    src_dataset_slug: str = "{username}/kagglepipe-src"


@dataclass
class DataSection:
    dataset_slug: str = ""  # empty = no data dataset


@dataclass
class FeatureSection:
    branches: list[str] = field(default_factory=list)
    heavy_branches: list[str] = field(default_factory=list)
    default_gpu: str = "none"  # "p100" | "t4x2" | "none"
    kernel_slug_template: str = "{username}/kagglepipe-{branch}"
    kernel_title_prefix: str = "kagglepipe"
    notebook_template: str = "kagglepipe.templates.notebook_default"
    notebook_command: str = "python scripts/run.py --out {out_dir}"
    # Kaggle mounts datasets at /kaggle/input/datasets/<username>/<name> (current
    # API behavior). The placeholders {username} and {dataset} are substituted
    # at render time; the actual values come from the source/data slugs.
    data_mount: str = "/kaggle/input/datasets/{username}/{dataset}"
    src_mount: str = "/kaggle/input/datasets/{username}/{dataset}"
    out_dir: str = "/kaggle/working/features"
    output_glob: str = "{branch}.parquet"
    default_timeout_sec: int = 1800
    poll_interval_sec: int = 30
    # P1: parallel execution. 0 / 1 = sequential (default for backward compat).
    parallel: int = 1
    # P4: dependency graph. branch -> list of upstream branch names.
    dependencies: dict[str, list[str]] = field(default_factory=dict)
    # P5: artifact caching. 0 = disabled, 1 = enabled.
    cache: int = 0


@dataclass
class KernelsSection:
    is_private: bool = True
    enable_internet: bool = True
    language: str = "python"
    kernel_type: str = "notebook"


@dataclass
class PathsSection:
    notebooks_dir: str = "kaggle_notebooks"
    features_dir: str = "features_kaggle"


@dataclass
class Config:
    project: ProjectSection = field(default_factory=ProjectSection)
    source: SourceSection = field(default_factory=SourceSection)
    data: DataSection = field(default_factory=DataSection)
    feature: FeatureSection = field(default_factory=FeatureSection)
    kernels: KernelsSection = field(default_factory=KernelsSection)
    paths: PathsSection = field(default_factory=PathsSection)
    competition: dict[str, Any] = field(default_factory=dict)
    config_path: Path | None = None


# --- loading --------------------------------------------------------------


def _coerce_section(cls: type, raw: dict[str, Any]) -> Any:
    """Coerce a raw dict into a dataclass of the given type, dropping unknowns."""
    if not isinstance(raw, dict):
        return cls()
    known = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in raw.items() if k in known}
    return cls(**kwargs)


def load(path: Path | None = None) -> Config:
    """Load config from `path` (default: ./kaggle.toml) + env overrides.

    Missing file is not an error — defaults are returned. Env vars take the
    form `KAGGLEPIPE_<SECTION>__<FIELD>` (double underscore as the section
    separator, e.g., `KAGGLEPIPE_FEATURE__DEFAULT_GPU=t4x2`).
    """
    cfg_path = path.expanduser() if path else Path.cwd() / CONFIG_FILENAME
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        try:
            raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"Invalid TOML in {cfg_path}: {exc}") from exc

    config = Config(
        project=_coerce_section(ProjectSection, raw.get("project", {})),
        source=_coerce_section(SourceSection, raw.get("source", {})),
        data=_coerce_section(DataSection, raw.get("data", {})),
        feature=_coerce_section(FeatureSection, raw.get("feature", {})),
        kernels=_coerce_section(KernelsSection, raw.get("kernels", {})),
        paths=_coerce_section(PathsSection, raw.get("paths", {})),
        competition=raw.get("competition", {}) or {},
        config_path=cfg_path if cfg_path.exists() else None,
    )
    return _apply_env_overrides(config)


_ENV_PREFIX = "KAGGLEPIPE_"


def _apply_env_overrides(cfg: Config) -> Config:
    """Override config fields from `KAGGLEPIPE_<SECTION>__<FIELD>` env vars."""
    sections: dict[str, Any] = {
        "project": cfg.project,
        "source": cfg.source,
        "data": cfg.data,
        "feature": cfg.feature,
        "kernels": cfg.kernels,
        "paths": cfg.paths,
    }
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(_ENV_PREFIX):
            continue
        suffix = env_key[len(_ENV_PREFIX) :]
        if "__" not in suffix:
            continue
        section, field_name = suffix.split("__", 1)
        section = section.lower()
        if section not in sections:
            continue
        field_name = field_name.lower()
        target = sections[section]
        if field_name not in {f.name for f in fields(target)}:
            continue
        current = getattr(target, field_name)
        # Coerce strings to the existing field's type (e.g., "1800" -> 1800).
        if isinstance(current, bool):
            coerced: Any = env_val.lower() in ("1", "true", "yes", "on")
        elif isinstance(current, int):
            coerced = int(env_val)
        elif isinstance(current, list):
            coerced = [s.strip() for s in env_val.split(",") if s.strip()]
        else:
            coerced = env_val
        setattr(target, field_name, coerced)
    return cfg


def to_dict(cfg: Config) -> dict[str, Any]:
    """Return a JSON-serializable dict (used by `config show --json`)."""
    out: dict[str, Any] = {
        "project": asdict(cfg.project),
        "source": asdict(cfg.source),
        "data": asdict(cfg.data),
        "feature": asdict(cfg.feature),
        "kernels": asdict(cfg.kernels),
        "paths": asdict(cfg.paths),
        "competition": cfg.competition,
    }
    if cfg.config_path is not None:
        out["_config_path"] = str(cfg.config_path)
    return out


# --- scaffold -------------------------------------------------------------


DEFAULT_CONFIG_TEMPLATE = """# kaggle.toml — configuration for kagglepipe.
# See https://github.com/<you>/kagglepipe for the full schema and docs.

[project]
name = "{project_name}"

[source]
include = ["src", "configs", "scripts", "pyproject.toml", "README.md"]
exclude_dirs = [".venv", "data", "models", ".git", "__pycache__"]
exclude_exts = [".parquet", ".lgb", ".pt", ".pth", ".bin"]
src_dataset_slug = "{{username}}/{project_name}-src"

[data]
dataset_slug = "{{username}}/{project_name}-data"

[feature]
branches = []
heavy_branches = []
default_gpu = "t4x2"  # "p100" | "t4x2" | "none"
kernel_slug_template = "{{username}}/{project_name}-{{branch}}"
kernel_title_prefix = "{project_name}"
notebook_command = "python scripts/run.py --out {{out_dir}}"
data_mount = "/kaggle/input/{project_name}-data"
src_mount = "/kaggle/input/{project_name}-src"
out_dir = "/kaggle/working/features"
output_glob = "{{branch}}.parquet"
default_timeout_sec = 1800
poll_interval_sec = 30

[kernels]
is_private = true
enable_internet = true
language = "python"
kernel_type = "notebook"

[paths]
notebooks_dir = "kaggle_notebooks"
features_dir = "features_kaggle"
"""


def scaffold(path: Path | None = None, project_name: str | None = None) -> Path:
    """Write a starter kaggle.toml. Returns the path."""
    target = path.expanduser() if path else Path.cwd() / CONFIG_FILENAME
    if target.exists():
        raise FileExistsError(f"{target} already exists; remove it first or pass --force.")
    name = project_name or Path.cwd().name.lower().replace(" ", "-")
    target.write_text(DEFAULT_CONFIG_TEMPLATE.format(project_name=name), encoding="utf-8")
    return target
