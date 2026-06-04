# KagglePipe

**Workflow orchestration for Kaggle.** Manage feature pipelines, remote GPU jobs,
and artifact lifecycles from your local codebase — without touching the browser.

```bash
pip install -e . && kagglepipe --help
```

Kaggle CLI manages Kaggle **resources**.
KagglePipe manages Kaggle **workflows**.

---

## The Problem

Most serious Kaggle competitors eventually end up with the same mess:

- A local codebase with feature engineering scripts
- Multiple feature branches — each run differently
- GPU training jobs scattered across manual notebooks
- Notebook generation hell: copy-paste-edit-repeat per branch
- Dataset versioning by hand: `src-v1`, `src-v2`, `src-v3...`
- Source code that needs to be synced to Kaggle before every run
- Waiting for kernels to finish — then checking the web UI
- Downloading outputs, renaming files, organizing artifacts
- `kernel-metadata.json` that needs to stay in sync with your local config
- The same 12-step workflow repeated every time you want to iterate

It's not a Kaggle problem. It's a **workflow problem**. And everyone solves it
the same way: custom scripts, Makefiles, CI pipelines, shell aliases — eventually
building their own internal tooling.

KagglePipe is that tooling, built for everyone.

---

## What KagglePipe Does

Instead of manually:

1. Packaging source code into a tarball
2. Uploading it as a Kaggle Dataset
3. Generating a parameterized notebook per branch
4. Creating `kernel-metadata.json`
5. Pushing a kernel
6. Polling `kaggle kernels status` until it completes
7. Downloading output artifacts
8. Organizing everything into a feature store

You run:

```bash
kagglepipe feature run user_features
```

And KagglePipe orchestrates the entire pipeline — end to end, from your terminal.

---

## Git analogy

```
Git                        ~  Kaggle CLI
GitHub Actions             ~  KagglePipe

GitHub Actions builds on Git to add workflow orchestration.
KagglePipe builds on the Kaggle CLI to add workflow orchestration.

GitHub Actions doesn't replace Git — it sits on top of it.
KagglePipe doesn't replace the Kaggle CLI — it sits on top of it.
```

Kaggle CLI gives you **primitives**. KagglePipe gives you **workflows**.

---

## Core Workflows

### Source Dataset Management

Package your local codebase and upload it as a versioned Kaggle Dataset.
KagglePipe auto-detects whether to `create` (v1) or `version` (v2+).

```bash
kagglepipe src upload
# Packaging . -> user/myproj-src v3
# Built tarball: /tmp/src.tar.gz
# Uploaded: user/myproj-src v3
```

### Single Feature Branch Execution

Render a parameterized notebook, push it as a kernel, poll until complete,
download the output artifact — in one command.

```bash
kagglepipe feature run user_features --gpu t4x2
# Wrote notebook: kaggle_notebooks/extract_user_features.ipynb
# Pushed kernel: user/myproj-user_features
# Kernel state: complete
# Downloaded: features_kaggle/user_features.parquet
```

### Full Feature Pipeline Execution

Run all configured branches sequentially, with a summary.

```bash
kagglepipe feature all --gpu t4x2
# === baseline ===
# === dinov3_features ===
# === siglip_features ===
# === Summary (1,240s) ===
# Total: 3, OK: 3, Failed: 0
```

---

## End-to-End Flow

```
Local codebase                  Kaggle infrastructure
────────────────────           ─────────────────────────────────────
│                              │
src/                       ──► │  Kaggle Dataset (versioned source)
configs/                         │
scripts/                         │
                                 │
kagglepipe feature run <branch>  │  Kernel (GPU) executes the pipeline
                                 │
                                 ▼
features_kaggle/             ◄── │  Output artifacts downloaded
  branch-a.parquet
  branch-b.parquet
```

---

## Kaggle CLI vs KagglePipe

| Task | Kaggle CLI | KagglePipe |
|---|---|---|
| Upload source code | `datasets create` / `datasets version` | `kagglepipe src upload` |
| Detect next version | Manual | Auto (queries existing versions) |
| Generate a notebook | Manual (copy-paste-edit) | Template rendering (Jinja2) |
| Push a kernel | `kernels push` | `kagglepipe feature run` |
| Poll for completion | `kaggle kernels status` (manual loop) | Auto (configurable interval + timeout) |
| Download outputs | `kaggle kernels output` | Auto (glob-matched, placed in features dir) |
| Run multiple branches | Sequential manual calls | `kagglepipe feature all` |
| Orchestrate the whole pipeline | DIY scripts + Makefiles | `kagglepipe feature run <branch>` |

Kaggle CLI is the **engine**. KagglePipe is the **vehicle**.

---

## Who Should Use KagglePipe?

**Good fit:**
- Serious Kaggle competitors running multi-branch feature pipelines
- Competition teams with shared feature engineering codebases
- Users running GPU-heavy feature extraction on Kaggle's free hardware
- ML engineers who want to develop locally and execute remotely

**Not necessary:**
- Casual Kaggle users who submit a few notebooks manually
- People who only use Kaggle's web editor
- Simple single-submission workflows

---

## Design Philosophy

- **Thin layer over the official Kaggle CLI** — no API magic, just better UX
- **Configuration-driven** — `kaggle.toml` encodes your workflow, not your code
- **Reproducible workflows** — same config, same result every run
- **Local-first development** — iterate on your code, push when ready
- **Remote execution on Kaggle infrastructure** — free GPU time, no local hardware needed

---

## Install

```bash
git clone https://github.com/AdityaProCoder/kagglepipe && cd kagglepipe
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"   # Windows
.venv/bin/python -m pip install -e ".[dev]"           # Linux/macOS

kagglepipe --version        # -> kagglepipe 0.1.0
kagglepipe whoami           # verify credentials
```

Credentials via `~/.kaggle/kaggle.json`, or set `KAGGLE_USERNAME` / `KAGGLE_KEY`.

---

## Configure

```bash
cd ~/my-kaggle-project
kagglepipe config init --name myproj
$EDITOR kaggle.toml
```

```toml
[project]
name = "myproj"

[source]
include = ["src", "configs", "scripts", "pyproject.toml"]
exclude_dirs = [".venv", "data", "models", ".git", "__pycache__"]
exclude_exts = [".parquet", ".lgb", ".pt", ".bin"]
src_dataset_slug = "{username}/myproj-src"

[data]
dataset_slug = "{username}/myproj-data"

[feature]
branches = ["baseline", "dinov3", "siglip"]
heavy_branches = ["dinov3", "siglip"]
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

Every field accepts env-var overrides: `KAGGLEPIPE_<SECTION>__<FIELD>`
(e.g. `KAGGLEPIPE_FEATURE__DEFAULT_GPU=p100`).

---

## Full Command Reference

| Command | Description |
|---|---|
| `kagglepipe whoami` | Print verified username |
| `kagglepipe login` | Bootstrap `~/.kaggle/kaggle.json` |
| `kagglepipe config init` | Scaffold `kaggle.toml` |
| `kagglepipe config show [--json]` | Print effective config |
| `kagglepipe src upload [--version N]` | Package & push source dataset (auto-versions) |
| `kagglepipe feature run <branch>` | Render notebook → push → poll → download artifact |
| `kagglepipe feature all` | Run all configured branches sequentially |
| `kagglepipe status [--all] [--csv]` | List your kernels |
| `kagglepipe kernels list` | List kernels with filters |
| `kagglepipe kernels status <slug>` | Live kernel status |
| `kagglepipe kernels output <slug>` | Download kernel output directory |
| `kagglepipe kernels logs <slug>` | Print logs URL |
| `kagglepipe kernels stop <slug>` | Cancel a running kernel |
| `kagglepipe datasets list` | List your datasets |
| `kagglepipe datasets get <slug> <path>` | Download a dataset |
| `kagglepipe datasets create <dir>` | Create a new dataset |
| `kagglepipe datasets version <dir> -m "msg"` | New version of existing dataset |
| `kagglepipe competitions list` | Active competitions |
| `kagglepipe competitions submit <comp> <file> -m "msg"` | Submit to a competition |
| `kagglepipe competitions leaderboard <comp>` | Competition leaderboard |

Run `kagglepipe <cmd> --help` for all flags.

---

## Project Layout

```
src/kagglepipe/
  cli.py              argparse root + dispatch
  config.py           kaggle.toml loader + env overrides
  credentials.py      ~/.kaggle/kaggle.json + KAGGLE_USERNAME/KEY
  runner.py           subprocess wrapper (UTF-8 safe, python -X utf8 -m kaggle)
  slug.py            {username}/{branch} template resolver
  tarball.py         build_tarball(include, exclude_dirs, exclude_exts)
  notebook.py        render Jinja2 notebook + kernel-metadata.json
  polling.py         poll_kernel_status(...)
  kaggle_api.py      high-level wrappers around the kaggle CLI
  commands/           one module per command group
  templates/          default notebook template
tests/               80 unit tests + 1 live integration test
docs/quickstart.md   step-by-step walkthrough
```

---

## License

MIT
