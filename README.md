# kagglepipe

Full terminal control over [Kaggle](https://www.kaggle.com). A thin, configurable
orchestrator on top of the official `kaggle` CLI. No more clicking in the
Kaggle web UI — manage datasets, kernels, competitions, and end-to-end
ML pipelines entirely from your shell.

```
$ kagglepipe --help
usage: kagglepipe [-h] [--version] [--config CONFIG] COMMAND ...

Full terminal control over Kaggle. Thin, configurable orchestrator on top of
the official kaggle CLI.

positional arguments:
  COMMAND
    whoami         Print the current Kaggle username
    login          Bootstrap ~/.kaggle/kaggle.json
    config         Manage kaggle.toml
    src            Source dataset operations
    feature        Run feature branches on Kaggle
    status         List kernels matching the configured prefix
    kernels        Kernel operations
    datasets       Dataset operations
    competitions   Competition operations
```

---

## Why

The official `kaggle` CLI is a great low-level REST wrapper, but it is dumb on
purpose. `kagglepipe` adds the high-level workflow primitives that real
projects need:

- **Per-project configuration** in `kaggle.toml` — no hard-coded slugs, branch
  lists, or tarball rules.
- **Parameterized notebook generation** (Jinja2) — one template, many branches.
- **Source dataset auto-versioning** — `kagglepipe src upload` knows whether
  to `create` or `version`.
- **End-to-end feature pipeline** — `kagglepipe feature run <branch>` renders
  a notebook, pushes a kernel, polls until complete, downloads the output
  artifact.
- **Sequential multi-branch orchestration** — `kagglepipe feature all`.
- **Cross-platform UTF-8 hardening** — Windows consoles using `cp1252` no
  longer crash on the upstream CLI's box-drawing output.
- **A unified command tree** for `kernels`, `datasets`, and `competitions` —
  so you can do 100% of your Kaggle work from the terminal.

The single non-obvious piece of Kaggle trivia this tool handles: the
Kaggle runtime mounts attached datasets at
`/kaggle/input/datasets/<username>/<dataset-name>/`, not at
`/kaggle/input/<dataset-name>/` (as older docs and many tutorials show).
The default `kaggle.toml` already encodes this, so you don't have to.

---

## Install

```bash
# From the project root:
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"     # Windows
.venv/bin/python -m pip install -e ".[dev]"             # macOS / Linux
```

Verify:

```bash
kagglepipe --version
# kagglepipe 0.1.0
```

---

## Configure credentials

`kagglepipe` reads the standard `~/.kaggle/kaggle.json` (or `KAGGLE_USERNAME` +
`KAGGLE_KEY` env vars). Bootstrap interactively:

```bash
kagglepipe login
```

Or manually:

```bash
mkdir -p ~/.kaggle
echo '{"username":"yourname","key":"yourkey"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json     # on macOS / Linux
```

Verify it works:

```bash
kagglepipe whoami
# yourname
```

---

## Quickstart

```bash
# 1. Scaffold a kaggle.toml in your project
cd ~/my-ml-project
kagglepipe config init
$EDITOR kaggle.toml

# 2. Verify auth
kagglepipe whoami

# 3. Package your source as a Kaggle Dataset (auto version bumps)
kagglepipe src upload

# 4. Run a feature branch on a Kaggle GPU
kagglepipe feature run dinov3 --gpu t4x2

# 5. See all your kernels
kagglepipe status
```

See `docs/quickstart.md` for a full end-to-end walkthrough.

---

## Full command reference

Every subcommand supports `-h` / `--help` for its own flags. Below is the
exhaustive list.

### `kagglepipe [global]`

| Flag | Description |
|---|---|
| `--version` | Print the kagglepipe version and exit. |
| `--config PATH` | Path to `kaggle.toml` (default: `./kaggle.toml`). |
| `-h`, `--help` | Show help and exit. |

---

### `kagglepipe whoami`

Print the current Kaggle username (verified by hitting `kernels list`).

```bash
kagglepipe whoami             # prints: yourname
kagglepipe whoami --json      # prints: {"username": "yourname", "auth_ok": true}
```

(No additional flags.)

---

### `kagglepipe login`

Bootstrap `~/.kaggle/kaggle.json`.

| Flag | Description |
|---|---|
| `--username NAME` | Kaggle username (omit to prompt). |
| `--key KEY` | Kaggle API key (omit to prompt via `getpass`). |
| `--path PATH` | Override credentials file path (default: `~/.kaggle/kaggle.json`). |

---

### `kagglepipe config`

Manage `kaggle.toml`.

#### `kagglepipe config init`

Scaffold a starter `kaggle.toml` in the current directory.

| Flag | Description |
|---|---|
| `--path PATH` | Where to write the file (default: `./kaggle.toml`). |
| `--name NAME` | Project name (default: basename of cwd). |
| `--force` | Overwrite an existing `kaggle.toml`. |

#### `kagglepipe config show`

Print the effective config (defaults + file + env overrides). Use
`--json` for machine-readable output.

| Flag | Description |
|---|---|
| `--json` | Emit JSON instead of the human-readable sectioned form. |

#### `kagglepipe config path`

Print the path `kagglepipe` would load. Exits 1 if no `kaggle.toml` in
`$PWD` (and prints a hint to stderr).

---

### `kagglepipe src`

Source dataset operations.

#### `kagglepipe src upload`

Build a tarball of the configured `source.include` entries and upload it as
a Kaggle Dataset. Auto-detects whether to `create` (version 1) or `version`
(2+).

| Flag | Description |
|---|---|
| `--version N` | Override the auto-detected version number. |
| `--src-root DIR` | Project root to package (default: `cwd`). |
| `--slug SLUG` | Override dataset slug (default: `<username>/<src_dataset_slug>` from `kaggle.toml`). |

---

### `kagglepipe feature`

Run feature branches on Kaggle.

#### `kagglepipe feature run <branch>`

Render a parameterized notebook, push it as a kernel, poll until complete,
download the output artifact.

| Flag | Description |
|---|---|
| `branch` (positional) | Branch name (e.g., `dinov3`). Must be in `[feature].branches` if that whitelist is set. |
| `--gpu {p100,t4x2,none}` | GPU instance. `none` runs on CPU. Default: `[feature].default_gpu`. |
| `--timeout SEC` | Total seconds to wait for `complete` (default: 1800). |
| `--src-dataset SLUG` | Source dataset slug (default: `<username>/<src_dataset_slug>`). |
| `--data-dataset SLUG` | Data dataset slug (default: `<username>/<data_dataset_slug>` if set). |
| `--src-version N` | Source dataset version (default: auto-detect latest). |
| `--features-dir DIR` | Where to drop the downloaded parquet (default: `[paths].features_dir`). |
| `--notebooks-dir DIR` | Where to write the rendered `.ipynb` (default: `[paths].notebooks_dir`). |
| `--no-download` | Render and push only; don't download the output. |

Exit codes: `0` on success, `1` on kernel error / timeout / missing output.

#### `kagglepipe feature all`

Run `[feature].heavy_branches` (or all of `[feature].branches` if no heavy
list is set) sequentially.

| Flag | Description |
|---|---|
| `--branches CSV` | Comma-separated override for the branch list. |
| `--gpu {p100,t4x2,none}` | GPU instance for every run. |
| `--timeout SEC` | Per-branch timeout. |
| `--data-dataset SLUG` | Override data dataset slug for every run. |
| `--features-dir DIR` | Override output dir. |
| `--notebooks-dir DIR` | Override notebook staging dir. |

Exit codes: `0` if every branch succeeded, `1` if any failed. Prints a
summary with timings at the end.

---

### `kagglepipe status`

List kernels owned by the current user that match `[feature].kernel_title_prefix`.

| Flag | Description |
|---|---|
| `--all` | Show all of your kernels, not just those matching the title prefix. |
| `--csv` | Emit CSV instead of the default aligned columns. |

---

### `kagglepipe kernels`

Kernel operations.

#### `kagglepipe kernels list`

| Flag | Description |
|---|---|
| `--user USER` | Show kernels for a specific user (default: yourself). |
| `--search TEXT` | Free-text search. |
| `--page-size N` | Page size (default: 20). |
| `--csv` | Emit CSV. |
| `--json` | Emit JSON. |

#### `kagglepipe kernels status <slug>`

Print the live status of a kernel. Exits 0 if `complete`/`running`/`queued`,
1 otherwise.

#### `kagglepipe kernels output <slug>`

Download the kernel's output directory.

| Flag | Description |
|---|---|
| `--path DIR` | Where to put the output (default: `./<slug>_output/`). |

#### `kagglepipe kernels logs <slug>`

Print the canonical logs URL (`https://www.kaggle.com/<slug>/logs`).

#### `kagglepipe kernels stop <slug>`

Cancel a running kernel.

#### `kagglepipe kernels push <dir>`

Push a directory containing a `kernel-metadata.json` (low-level escape
hatch; most users won't need this).

---

### `kagglepipe datasets`

Dataset operations.

#### `kagglepipe datasets list`

| Flag | Description |
|---|---|
| `--user USER` | Show datasets for a specific user. |
| `--search TEXT` | Free-text search. |
| `--csv` | Emit CSV. |
| `--json` | Emit JSON. |

#### `kagglepipe datasets versions <slug>`

Print whether a dataset exists. (The upstream kaggle CLI doesn't expose a
clean per-dataset version listing; visit the web UI for that.)

#### `kagglepipe datasets get <slug> <path>`

Download a dataset to `<path>`. (Positional: `<slug> <path>`.)

#### `kagglepipe datasets create <path>`

Create a new dataset from a directory containing `dataset-metadata.json`.

| Flag | Description |
|---|---|
| `--public` | Make the dataset public (default: private). |

#### `kagglepipe datasets version <path>`

Create a new version of an existing dataset.

| Flag | Description |
|---|---|
| `--message`, `-m MSG` | Version notes (required). |
| `--dir-mode`, `-r {zip,tar}` | Upload mode (default: `zip`). |

---

### `kagglepipe competitions`

Competition operations.

#### `kagglepipe competitions list`

| Flag | Description |
|---|---|
| `--csv` | Emit CSV. |
| `--json` | Emit JSON. |

#### `kagglepipe competitions files <comp>`

List files in a competition. Pass-through to `kaggle competitions files`.

#### `kagglepipe competitions submit <comp> <file>`

| Flag | Description |
|---|---|
| `--message`, `-m MSG` | Submission description (required). |

#### `kagglepipe competitions leaderboard <comp>`

| Flag | Description |
|---|---|
| `--top N` | Show only the top N entries (default: 20). |
| `--csv` | Emit CSV. |
| `--json` | Emit JSON. |

---

## Configuration

`kagglepipe` loads `kaggle.toml` from the current directory (or `--config`).
Missing file is fine — defaults kick in. See the generated file from
`kagglepipe config init` for a template. Top-level sections:

```toml
[project]    name, slug prefixes
[source]     include / exclude rules + source dataset slug
[data]       optional data dataset slug
[feature]    branches, GPUs, notebook template, mount paths
[kernels]    default kernel-metadata.json fields
[paths]      where to write notebooks / features
```

Every field can be overridden by env vars of the form
`KAGGLEPIPE_<SECTION>__<FIELD>` (double underscore), e.g.
`KAGGLEPIPE_FEATURE__DEFAULT_GPU=t4x2`. Lists are comma-separated; booleans
are `1`/`true`/`yes`/`on`.

See `docs/quickstart.md` for the full schema and a worked example.

---

## Project layout

```
kagglepipe/
├── pyproject.toml
├── README.md                       (this file)
├── LICENSE
├── src/kagglepipe/
│   ├── cli.py                      # argparse root, dispatches to commands/*
│   ├── config.py                   # kaggle.toml loader + env overrides
│   ├── credentials.py              # ~/.kaggle/kaggle.json + env
│   ├── runner.py                   # subprocess wrapper around `kaggle` CLI
│   ├── slug.py                     # {username}/{branch} template resolver
│   ├── tarball.py                  # build_tarball(include, exclude_*)
│   ├── notebook.py                 # render Jinja2 notebook + write kernel-metadata.json
│   ├── polling.py                  # poll_kernel_status(...)
│   ├── kaggle_api.py               # high-level wrappers around the kaggle CLI
│   ├── commands/                   # one module per command group
│   └── templates/
│       └── notebook_default.py.j2  # default parameterized notebook
├── tests/                          # 80 unit tests + 1 live integration test
├── examples/                       # reference kaggle.toml configs
│   └── freuid/                     # drop-in for the freuid 2026 project
└── docs/quickstart.md
```

---

## License

MIT. See [LICENSE](LICENSE).
