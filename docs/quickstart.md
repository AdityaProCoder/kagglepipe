# Quickstart

A full end-to-end walkthrough: take a brand-new local project, configure
kagglepipe, push it to Kaggle, run a feature, and pull the result back.

## 0. Install

```bash
git clone <repo> kagglepipe && cd kagglepipe
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"     # Windows
.venv/bin/python -m pip install -e ".[dev]"             # macOS / Linux
```

## 1. Configure credentials

```bash
kagglepipe login                       # interactive
# or set KAGGLE_USERNAME and KAGGLE_KEY in your shell
kagglepipe whoami                      # verify
```

## 2. Scaffold a config

In the project you want to run on Kaggle:

```bash
cd ~/my-ml-project
kagglepipe config init
$EDITOR kaggle.toml
```

A minimal `kaggle.toml` looks like:

```toml
[project]
name = "myproj"

[source]
include = ["src", "scripts", "pyproject.toml", "README.md"]
exclude_dirs = [".venv", "data", "models", ".git", "__pycache__"]
exclude_exts = [".parquet", ".lgb", ".pt", ".bin"]
src_dataset_slug = "{username}/myproj-src"

[feature]
branches = ["fast", "accurate"]
heavy_branches = ["accurate"]
default_gpu = "t4x2"
kernel_slug_template = "{username}/myproj-{branch}"
kernel_title_prefix = "myproj"
notebook_command = "python scripts/run.py --out {out_dir}"
output_glob = "{branch}.parquet"

[kernels]
is_private = true
enable_internet = true

[paths]
notebooks_dir = "kaggle_notebooks"
features_dir  = "features_kaggle"
```

Inspect the merged result:

```bash
kagglepipe config show
kagglepipe config show --json   # machine-readable
```

## 3. Upload source

```bash
kagglepipe src upload
# Packaging . -> you/myproj-src v1
# Built tarball: /tmp/src.tar.gz (1234 bytes)
# Uploaded: you/myproj-src v1
```

The next run auto-bumps to v2 (then v3, ...).

## 4. Run a feature

```bash
kagglepipe feature run accurate --gpu t4x2
# Wrote notebook: kaggle_notebooks/extract_accurate.ipynb
# Pushed kernel: you/myproj-accurate
# Kernel state: complete
# Downloaded: features_kaggle/accurate.parquet
```

Multiple branches, sequentially:

```bash
kagglepipe feature all --gpu t4x2
# === fast ===
# ...
# === accurate ===
# ...
# === Summary (412s) ===
# Total: 2, OK: 2, Failed: 0
```

## 5. See your kernels

```bash
kagglepipe status
# you/myproj-accurate                          complete     2026-06-03 14:23
# you/myproj-fast                              complete     2026-06-03 14:30
```

## 6. Other useful commands

```bash
kagglepipe datasets list --user you
kagglepipe datasets versions you/myproj-src
kagglepipe kernels logs you/myproj-accurate
kagglepipe kernels stop you/myproj-accurate      # cancel a running kernel
kagglepipe kernels output you/myproj-accurate   # download output dir
kagglepipe competitions list
kagglepipe competitions leaderboard titanic --top 10
```

## 7. Advanced: env-var overrides

Useful for CI:

```bash
KAGGLEPIPE_FEATURE__DEFAULT_GPU=p100 \
KAGGLEPIPE_FEATURE__DEFAULT_TIMEOUT_SEC=3600 \
KAGGLEPIPE_FEATURE__BRANCHES=ci-only \
kagglepipe feature run ci-only
```

## 8. Advanced: custom notebook templates

`feature.notebook_template` accepts either:

- a Python dotted path: `mypkg.templates.notebook` (kagglepipe looks for
  `mypkg/templates/notebook.py.j2` or `mypkg/templates/notebook.j2`)
- a path to a `.j2` file: `/abs/path/to/my_template.j2`

The default template at
`kagglepipe/templates/notebook_default.py.j2` is the canonical reference
for the variables kagglepipe passes to render (`branch`, `src_dataset_slug`,
`src_mount`, `data_mount`, `out_dir`, `notebook_command`, `gpu`, `date`).

## 9. Advanced: `templates/<...>` override for the freuid pattern

If you want kagglepipe to behave like the original freuid
`scripts/kaggle_run.py` (uploads `freuid-src` and `freuid-sample`,
extracts from a `configs/features/{branch}.yaml` config), copy
`examples/freuid/kaggle.toml` from this repo into the freuid checkout
and run:

```bash
pip install kagglepipe
kagglepipe src upload
kagglepipe feature run dinov3
kagglepipe feature all
```
