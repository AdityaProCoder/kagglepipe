# kagglepipe

Full terminal control over [Kaggle](https://www.kaggle.com). Thin, configurable
orchestrator on top of the official `kaggle` CLI.

```
pip install -e .            # from repo root
kagglepipe --help

Commands:
  whoami                          verify your credentials
  login                           bootstrap ~/.kaggle/kaggle.json
  config  init|show|path           manage kaggle.toml
  src     upload                   package & push source as a Kaggle Dataset
  feature run <branch>|all         run feature branches on Kaggle GPUs
  status                          list your kernels
  kernels  list|status|output|logs|stop|push
  datasets list|versions|get|create|version
  competitions list|files|submit|leaderboard
```

## Install

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"   # Windows
.venv/bin/python -m pip install -e ".[dev]"             # Linux/macOS
kagglepipe --version          # -> kagglepipe 0.1.0
```

## Quick start

```bash
# 1. Configure credentials (or set KAGGLE_USERNAME / KAGGLE_KEY env vars)
kagglepipe login
kagglepipe whoami

# 2. Add kaggle.toml to your project
cd ~/my-ml-project
kagglepipe config init
$EDITOR kaggle.toml

# 3. Upload your source as a Kaggle Dataset
kagglepipe src upload

# 4. Run a feature branch on a Kaggle GPU
kagglepipe feature run my-branch --gpu t4x2

# 5. Run multiple branches sequentially
kagglepipe feature all --gpu t4x2

# 6. Check kernel status
kagglepipe status
```

## Configuration

`kaggle.toml` is loaded from the current directory (`--config` overrides).
Every field accepts env-var overrides (`KAGGLEPIPE_<SECTION>__<FIELD>`).

```toml
[project]
name = "myproj"

[source]
include = ["src", "scripts", "pyproject.toml", "README.md"]
exclude_dirs = [".venv", "data", "models", ".git", "__pycache__"]
exclude_exts = [".parquet", ".lgb", ".pt", ".bin"]
src_dataset_slug = "{username}/myproj-src"

[data]
dataset_slug = "{username}/myproj-data"

[feature]
branches = ["branch-a", "branch-b"]
heavy_branches = ["branch-a"]
default_gpu = "t4x2"
kernel_slug_template = "{username}/myproj-{branch}"
kernel_title_prefix = "myproj"
notebook_command = "python scripts/run.py --out {out_dir}"
output_glob = "{branch}.parquet"
default_timeout_sec = 1800
poll_interval_sec = 30

[kernels]
is_private = true
enable_internet = true
language = "python"
kernel_type = "notebook"

[paths]
notebooks_dir = "kaggle_notebooks"
features_dir  = "features_kaggle"
```

Key detail: Kaggle mounts datasets at `/kaggle/input/datasets/<username>/<name>/`
(the default mount paths already encode this).

## Project layout

```
src/kagglepipe/          # core modules + commands/*
tests/                   # 80 unit tests + 1 live integration test
docs/quickstart.md       # step-by-step guide
pyproject.toml           # pip install -e .
LICENSE
```

See `docs/quickstart.md` for the full walkthrough.

## License

MIT
