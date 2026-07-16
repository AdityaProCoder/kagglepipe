"""Configuration loader for kagglepipe.

Reads `kaggle.toml` from the current working directory (or a path supplied by
the caller) into a typed `Config` dataclass. Missing fields are filled from
`_DEFAULTS`. Env vars (prefix `KAGGLEPIPE_`) override file values.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

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
data_mount = "/kaggle/input/datasets/{{username}}/{{dataset}}"
src_mount = "/kaggle/input/datasets/{{username}}/{{dataset}}"
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


def scaffold(
    path: Path | None = None,
    project_name: str | None = None,
    username: str | None = None,
) -> Path:
    """Write a starter kaggle.toml. Returns the path.

    Args:
        path:       Target file path (default: ./kaggle.toml).
        project_name: Name for the project (default: cwd dirname).
        username:   Kaggle username (default: leave as {username} placeholder).
    """
    target = path.expanduser() if path else Path.cwd() / CONFIG_FILENAME
    if target.exists():
        raise FileExistsError(f"{target} already exists; remove it first or pass --force.")
    name = project_name or Path.cwd().name.lower().replace(" ", "-")
    text = DEFAULT_CONFIG_TEMPLATE.format(project_name=name)
    if username:
        text = text.replace("{username}", username)
        text = text.replace("$name", name)
    target.write_text(text, encoding="utf-8")
    return target


def fill(cfg_path: Path, username: str, project_name: str | None = None) -> Path:
    """Patch an existing kaggle.toml so {username}/{project_name} placeholders
    are resolved using the real values. Does a read→substitute→write round so
    no other settings are disturbed."""
    raw = cfg_path.read_text(encoding="utf-8")
    name = project_name or _read_project_name(cfg_path) or Path.cwd().name.lower().replace(" ", "-")

    # Fields that use {username} or $name / {project_name} placeholders
    # {username} is replaced with the actual username.
    # $name and {project_name} are replaced with the project directory name.
    fixed = raw.replace("{username}", username)
    fixed = fixed.replace("$name", name)
    fixed = fixed.replace("{project_name}", name)

    # Also fix the [project].name if it still reads the default placeholder
    fixed = _patch_table_field(fixed, "project", "name", name)

    # Fix slug templates so they use the real username (they now just contain
    # the literal placeholder which we already replaced above, but also fix
    # any that use the older "$name" form in a string literal context).
    fixed = _patch_table_field(
        fixed, "source", "src_dataset_slug", f"{username}/{name}-src"
    )
    fixed = _patch_table_field(
        fixed, "data", "dataset_slug", f"{username}/{name}-data"
    )
    fixed = _patch_table_field(
        fixed, "feature", "kernel_slug_template", f"{username}/{name}-{{branch}}"
    )
    fixed = _patch_table_field(fixed, "feature", "kernel_title_prefix", name)
    fixed = _patch_table_field(fixed, "feature", "data_mount", "/kaggle/input/datasets/{username}/{dataset}")
    fixed = _patch_table_field(fixed, "feature", "src_mount", "/kaggle/input/datasets/{username}/{dataset}")

    cfg_path.write_text(fixed, encoding="utf-8")
    return cfg_path


def detect_branches(features_dir: Path | str = "features") -> list[str]:
    """Return list of feature branch names found in the features directory.

    Looks for files/dirs that look like feature implementations:
    - features/<name>.py   → branch name = <name>
    - features/<name>/    → branch name = <name>
    Excludes __init__.py, .gitkeep, and hidden files.
    """
    root = Path(features_dir)
    if not root.is_dir():
        return []
    branches: set[str] = set()
    for item in root.iterdir():
        name = item.name
        if name.startswith(".") or name in ("__init__.py", ".gitkeep"):
            continue
        if item.suffix == ".py":
            branches.add(name[:-3])
        elif item.is_dir():
            branches.add(name)
    return sorted(branches)


def _read_project_name(cfg_path: Path) -> str | None:
    """Extract the current project.name from a kaggle.toml without full parse."""
    import re
    raw = cfg_path.read_text(encoding="utf-8")
    m = re.search(r'^\s*name\s*=\s*"([^"]+)"', raw, re.MULTILINE)
    return m.group(1) if m else None


def _patch_table_field(text: str, table: str, key: str, value: str) -> str:
    """Replace the value of `key` inside [{table}] in a TOML string."""
    import re
    table_pat = rf'^\[\s*{re.escape(table)}\s*\]'
    m = re.search(table_pat, text, re.MULTILINE)
    if not m:
        return text
    rest = text[m.end():]
    next_table = re.search(r'^\[', rest, re.MULTILINE)
    section = rest[: next_table.start() if next_table else len(rest)]
    key_pat = rf'^\s*{re.escape(key)}\s*='
    key_m = re.search(key_pat, section, re.MULTILINE)
    if not key_m:
        return text
    line_end = re.search(r'\r?\n', section[key_m.end():])
    end_pos = key_m.end() + (line_end.start() if line_end else len(section[key_m.end():]))
    before = text[: m.start() + m.end() + key_m.start()]
    after = text[m.start() + m.end() + end_pos:]
    return f"{before}{key} = {value!r}{after}"
