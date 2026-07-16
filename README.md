# KagglePipe

[![CI](https://github.com/AdityaProCoder/kagglepipe/actions/workflows/ci.yml/badge.svg)](https://github.com/AdityaProCoder/kagglepipe/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

KagglePipe is a reproducible command-line workflow for Kaggle. It packages your source, runs parameterized kernels, retrieves artifacts, and records the provenance behind every run.

```bash
kagglepipe feature run user_features --gpu t4x2
```

It is deliberately built on top of the official Kaggle CLI: Kaggle owns remote resources; KagglePipe handles the repeatable workflow around them.

## Why use it?

Kaggle competitors often repeat the same manual loop: archive code, version a dataset, edit and push a notebook, wait, download files, and try to remember what produced a score. KagglePipe makes that loop explicit and recoverable.

- Source datasets are versioned automatically.
- Feature kernels are rendered from a configurable notebook template.
- Runs record configuration, Git state, dataset versions, and artifact hashes.
- Dry runs, validation, retry, resume, and local monitoring make remote work safer.
- The state stays in your project under `.kagglepipe/` and is never uploaded by KagglePipe.

## Install

```bash
python -m pip install kagglepipe
kagglepipe --version
```

For development:

```bash
git clone https://github.com/AdityaProCoder/kagglepipe.git
cd kagglepipe
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"  # Linux/macOS
# Windows: .venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Five-minute workflow

### 1. Authenticate

Create an API token at [Kaggle account settings](https://www.kaggle.com/settings/account), then run:

```bash
kagglepipe auth login
kagglepipe auth whoami
```

`whoami` and `login` remain supported as top-level aliases for existing scripts. You can also provide `KAGGLE_USERNAME` and `KAGGLE_KEY` as environment variables.

### 2. Initialize a project

```bash
cd ~/my-kaggle-project
kagglepipe config init --name myproj --auto
kagglepipe validate
```

`--auto` fills the username when credentials are available and reports detected feature branches. `validate` reports local configuration errors even before authentication is configured.

The generated `kaggle.toml` is the entire workflow contract. The important settings are:

```toml
[source]
include = ["src", "configs", "scripts", "pyproject.toml"]
src_dataset_slug = "{username}/myproj-src"

[data]
dataset_slug = "{username}/myproj-data"

[feature]
branches = ["baseline", "user_features"]
heavy_branches = ["user_features"]
default_gpu = "t4x2"
kernel_slug_template = "{username}/myproj-{branch}"
notebook_command = "python scripts/run.py --out {out_dir}"
output_glob = "{branch}.parquet"
```

### 3. Upload source and run a feature

```bash
kagglepipe src upload
kagglepipe feature run user_features --gpu t4x2
```

KagglePipe writes the rendered notebook to `kaggle_notebooks/`, submits it, waits for a terminal kernel state, downloads the requested output to `features_kaggle/`, and records the run.

Use a dry run before spending Kaggle quota:

```bash
kagglepipe src upload --dry-run
kagglepipe feature run user_features --dry-run
```

### 4. Monitor and recover

```bash
kagglepipe monitor                 # interactive dashboard
kagglepipe monitor --once          # portable text snapshot for CI/logs
kagglepipe feature retry failed
kagglepipe feature resume
```

## Core commands

| Command | Purpose |
| --- | --- |
| `kagglepipe validate` | Check credentials, project files, templates, config, and remote source state. |
| `kagglepipe src upload` | Package and create/version the source dataset. |
| `kagglepipe feature run BRANCH` | Run one feature kernel and download its artifact. |
| `kagglepipe feature all` | Run configured heavy branches. |
| `kagglepipe feature plan TARGET` | Inspect a feature dependency plan. |
| `kagglepipe monitor` | Inspect local run, artifact, experiment, and submission state. |
| `kagglepipe submissions best` | Show the highest-scoring recorded submission. |
| `kagglepipe run export BRANCH` | Export a portable run bundle. |

Run `kagglepipe --help` or `kagglepipe <command> --help` for the complete CLI.

## Safety and reproducibility

- `--dry-run` does not create or modify remote Kaggle resources.
- `validate` separates local diagnostics from remote checks, so first-time setup is actionable.
- Credentials are read from environment variables or `~/.kaggle/kaggle.json`; they are never written to project configuration.
- Run manifests capture the config hash, notebook hash, Git revision and dirty state, dataset versions, and artifact hash when available.
- Integration tests are opt-in because they create real Kaggle resources.

## Configuration and templates

Every config setting can be overridden for CI with `KAGGLEPIPE_<SECTION>__<FIELD>`, for example:

```bash
KAGGLEPIPE_FEATURE__DEFAULT_GPU=p100 kagglepipe feature run user_features
```

The default notebook template is packaged as `kagglepipe.templates.notebook_default`. Point `feature.notebook_template` at a dotted Python path or a local `.j2` file to use your own template. It receives the branch name, source/data dataset slugs and mount paths, output directory, resolved command, and run date.

See the [quickstart](docs/quickstart.md) for a fuller walkthrough.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for local setup, test commands, and integration-test safety. Please report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE)
