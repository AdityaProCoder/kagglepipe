# Quickstart

End-to-end: scaffold a project, configure kagglepipe, push source to Kaggle,
run a feature branch, and pull the result back.

---

## 0. Install

```bash
git clone https://github.com/AdityaProCoder/kagglepipe && cd kagglepipe
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"   # Windows
.venv/bin/python -m pip install -e ".[dev]"             # Linux/macOS
```

---

## 1. Credentials

```bash
kagglepipe auth login         # interactive (username + API key prompts)
# or manually:
mkdir -p ~/.kaggle
echo '{"username":"you","key":"yourkey"}' > ~/.kaggle/kaggle.json
# or env vars:
#   export KAGGLE_USERNAME=you KAGGLE_KEY=yourkey

kagglepipe auth whoami        # verify
```

---

## 2. Scaffold config

```bash
cd ~/my-ml-project
kagglepipe config init --name myproj
```

Edit `kaggle.toml` — here's a working minimum:

```toml
[project]
name = "myproj"

[source]
include = ["src", "scripts", "pyproject.toml"]
exclude_dirs = [".venv", "data", "models", ".git", "__pycache__"]
exclude_exts = [".parquet", ".lgb", ".pt"]
src_dataset_slug = "{username}/myproj-src"

[data]
dataset_slug = "{username}/myproj-data"

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

Preview the resolved config:

```bash
kagglepipe config show
kagglepipe config show --json
```

---

## 3. Upload source

```bash
kagglepipe src upload
# -> you/myproj-src v1
# Next run auto-bumps to v2, then v3 ...
```

---

## 4. Run a feature

```bash
kagglepipe feature run accurate --gpu t4x2
# Wrote notebook: kaggle_notebooks/extract_accurate.ipynb
# Pushed kernel: you/myproj-accurate
# Kernel state: complete
# Downloaded: features_kaggle/accurate.parquet
```

Multiple branches, one after another:

```bash
kagglepipe feature all --gpu t4x2
# === fast ===
# === accurate ===
# Summary (412s) — Total: 2, OK: 2, Failed: 0
```

Override branches or GPU per-invocation:

```bash
kagglepipe feature run accurate --gpu none --timeout 600
kagglepipe feature all --branches fast,accurate --gpu t4x2
```

---

## 5. Check results

```bash
kagglepipe status                    # your kernels matching kernel_title_prefix
kagglepipe status --all             # all your kernels
kagglepipe status --csv             # CSV output

kagglepipe kernels logs you/myproj-accurate
kagglepipe kernels output you/myproj-accurate
kagglepipe kernels stop you/myproj-accurate

# interactive dashboard; use --once for CI or logs
kagglepipe monitor
kagglepipe monitor --once
```

---

## 6. Other commands

```bash
kagglepipe datasets list --user you
kagglepipe datasets versions you/myproj-src
kagglepipe datasets get you/myproj-data ./data/

kagglepipe competitions list
kagglepipe competitions leaderboard titanic --top 10
kagglepipe competitions submit titanic submission.csv -m "v1 baseline"
```

---

## 7. Env-var overrides (for CI)

```bash
KAGGLEPIPE_FEATURE__DEFAULT_GPU=p100 \
KAGGLEPIPE_FEATURE__DEFAULT_TIMEOUT_SEC=3600 \
kagglepipe feature run accurate
```

Format: `KAGGLEPIPE_<SECTION>__<FIELD>` (double underscore). Lists are
comma-separated; booleans accept `1`/`true`/`yes`/`on`.

---

## 8. Custom notebook templates

`feature.notebook_template` in `kaggle.toml` accepts:

- **Python dotted path** — `mypkg.templates.my_template` → looks for
  `mypkg/templates/my_template.py.j2`
- **Absolute file path** — `/abs/path/to/my_template.j2`

Variables passed to every template:

| Variable | Description |
|---|---|
| `branch` | Feature branch name |
| `src_dataset_slug` | Source dataset slug |
| `src_version` | Source dataset version |
| `src_mount` | Kaggle mount path for source |
| `data_dataset_slug` | Data dataset slug (empty if none) |
| `data_mount` | Kaggle mount path for data |
| `out_dir` | Kernel output directory |
| `notebook_command` | Command to run in the notebook |
| `gpu` | GPU instance (e.g. `t4 x2`, `p100`, or `None`) |
| `date` | Render timestamp (YYYY-MM-DD) |

The default template ships at `kagglepipe/templates/notebook_default.py.j2`
and is a good reference.
