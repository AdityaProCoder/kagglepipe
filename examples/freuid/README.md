# Using kagglepipe with the FREUID 2026 project

This is a **drop-in** replacement for the legacy `scripts/kaggle_run.py` in
the freuid checkout. It reproduces the same behavior using the
`kagglepipe` CLI.

## What the old script did

`scripts/kaggle_run.py` (freuid) had four subcommands:

| Old | New |
|---|---|
| `python scripts/kaggle_run.py upload-src` | `kagglepipe src upload` |
| `python scripts/kaggle_run.py run-feature <branch>` | `kagglepipe feature run <branch>` |
| `python scripts/kaggle_run.py status` | `kagglepipe status` |
| `python scripts/kaggle_run.py all` | `kagglepipe feature all` |

## How to switch

1. From the freuid repo root: `pip install kagglepipe`
2. Copy `kaggle.toml` from this directory to the freuid project root.
3. Run:
   ```bash
   kagglepipe src upload
   kagglepipe feature run dinov3
   kagglepipe feature all
   kagglepipe status
   ```
4. (Optional) Delete `scripts/kaggle_run.py` and
   `tests/test_kaggle_run.py` from the freuid project.

## What kagglepipe fixes (vs. the old script)

1. **Mount path is correct for the current Kaggle runtime.** The old
   script used `/kaggle/input/freuid-sample`, which is wrong on today's
   Kaggle — datasets are mounted at
   `/kaggle/input/datasets/<username>/freuid-sample/`. The
   `kaggle.toml` shipped here uses the correct path, so
   `feature run` actually completes instead of dying with
   `FileNotFoundError: '/kaggle/input/freuid-sample'`.
2. **No hard-coded branch list.** The old script's
   `ALLOWED_BRANCHES = {"dinov3", "siglip2", ...}` is now driven by
   `feature.branches` in `kaggle.toml`.
3. **Configurable GPU instance.** `--gpu p100 | t4x2 | none` is a
   first-class flag instead of a hard-coded tuple.
4. **Cross-platform UTF-8 safety.** The old script had the right
   `os.environ["PYTHONIOENCODING"] = "utf-8"` trick in one place;
   kagglepipe centralizes it in `runner.py` so every command gets it
   for free.
5. **`upload-src` actually fails loudly** when the title is already in
   use. The upstream `kaggle datasets create` returns `rc=0` even on
   that error; the old script printed "Uploaded: ... v1" and moved on.
   kagglepipe detects the failure and exits non-zero.
6. **`dataset_exists` is reliable.** The old code used
   `kaggle datasets list --search <slug>` to detect existing
   datasets, which doesn't find by full slug. kagglepipe lists the
   user's own datasets and filters by exact `ref`.

## The kaggle.toml in this directory

```toml
[project]
name = "freuid"

[source]
include = ["src", "configs", "scripts", "pyproject.toml", "README.md"]
exclude_dirs = [".venv", "data", "models", ".git", "__pycache__",
                ".pytest_cache", ".ruff_cache", ".mypy_cache",
                "kaggle_notebooks", "submissions",
                "the-freuid-challenge-2026-ijcai-ecai"]
exclude_exts = [".parquet", ".lgb", ".pt", ".pth", ".bin"]
src_dataset_slug = "{username}/freuid-src"

[data]
dataset_slug = "{username}/freuid-sample"

[feature]
branches = ["dinov3", "siglip2", "paddleocr_vl",
            "layout_pp_doclayoutv3", "face_buffalo_l", "forensics"]
heavy_branches = ["dinov3", "siglip2", "paddleocr_vl", "face_buffalo_l"]
default_gpu = "t4x2"
kernel_slug_template = "{username}/freuid-extract-{branch}"
kernel_title_prefix = "freuid-extract"
notebook_command = "python scripts/extract_features.py --config configs/features/{branch}.yaml --data-root /kaggle/input/datasets/{username}/freuid-sample --out-dir /kaggle/working/features"
data_mount = "/kaggle/input/datasets/{username}/freuid-sample"
src_mount = "/kaggle/input/datasets/{username}/freuid-src"
out_dir = "/kaggle/working/features"
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
features_dir = "features_kaggle"
```

Override anything per-invocation with flags:

```bash
kagglepipe feature run dinov3 --gpu p100 --timeout 3600
kagglepipe feature all --branches dinov3,face_buffalo_l --gpu none
```

Or per-environment with env vars:

```bash
KAGGLEPIPE_FEATURE__DEFAULT_GPU=none kagglepipe feature all
```
