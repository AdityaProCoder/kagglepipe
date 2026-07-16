"""Template library (P12).

`kagglepipe template init <type>` scaffolds a starter project for common
Kaggle competition archetypes:

  - tabular: gradient-boosted features on a tabular CSV dataset
  - cv:      computer-vision features (image embeddings)
  - nlp:     text / transformer features

Generates: kaggle.toml, a scripts/ skeleton, an example feature
definition, a sample notebook-command, and a starter feature config.

The templates contain two kinds of braces:
  - `{{name}}`    -- literal `{name}` in the output (kaggle.toml template syntax)
  - `{name}`     -- substituted with the project name
We use `string.Template` ($name syntax) for substitution, so any literal
`$` in the body must be doubled as `$$`. The bodies are also dedented.
"""

from __future__ import annotations

import sys
from pathlib import Path
from string import Template
from textwrap import dedent


def _write(path: Path, body: str, *, name: str, force: bool = False) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use string.Template to substitute only `$name`. This avoids conflicts
    # with Python f-string / dict-style `{...}` braces in the bodies.
    rendered = Template(dedent(body).lstrip()).substitute(name=name)
    path.write_text(rendered, encoding="utf-8")


_TEMPLATES: dict[str, dict[str, str]] = {
    "tabular": {
        "kaggle.toml": '''\
            [project]
            name = "$name"

            [source]
            include = ["scripts", "configs", "README.md"]
            exclude_dirs = [".venv", "data", "models", ".git", "__pycache__",
                            ".pytest_cache", ".ruff_cache", "kaggle_notebooks",
                            "submissions"]
            exclude_exts = [".parquet", ".lgb", ".pt", ".bin"]
            src_dataset_slug = "{username}/$name-src"

            [data]
            dataset_slug = "{username}/$name-data"

            [feature]
            branches = ["baseline", "user_features"]
            heavy_branches = ["user_features"]
            default_gpu = "none"
            kernel_slug_template = "{username}/$name-{branch}"
            kernel_title_prefix = "$name"
            notebook_command = "python scripts/run.py --config configs/features/{branch}.yaml --data-root /kaggle/input/datasets/{username}/$name-data --out-dir /kaggle/working/features"
            data_mount = "/kaggle/input/datasets/{username}/$name-data"
            src_mount = "/kaggle/input/datasets/{username}/$name-src"
            out_dir = "/kaggle/working/features"
            output_glob = "{branch}.parquet"
            default_timeout_sec = 1800
            poll_interval_sec = 30

            [kernels]
            is_private = true
            enable_internet = false
            language = "python"
            kernel_type = "notebook"

            [paths]
            notebooks_dir = "kaggle_notebooks"
            features_dir = "features_kaggle"

            [competition]
            slug = ""
            submission_path = "submission.csv"
            message = "kagglepipe submission"
            train_command = "python scripts/train.py --out submission.csv"
            ''',
        "scripts/run.py": '''\
            #!/usr/bin/env python3
            """Tabular feature runner: reads config + CSV, writes a parquet."""
            import argparse
            import os
            import yaml
            import pandas as pd

            p = argparse.ArgumentParser()
            p.add_argument("--config", required=True)
            p.add_argument("--data-root", required=True)
            p.add_argument("--out-dir", required=True)
            args = p.parse_args()

            cfg = yaml.safe_load(open(args.config))
            train_csv = os.path.join(args.data_root, cfg["data"]["train_csv"])
            df = pd.read_csv(train_csv)
            os.makedirs(args.out_dir, exist_ok=True)
            out = os.path.join(args.out_dir, f"{cfg['name']}.parquet")
            df.to_parquet(out, index=False)
            print("wrote", out, "rows=", len(df))
            ''',
        "configs/features/baseline.yaml": '''\
            name: baseline
            data:
              train_csv: train.csv
            features:
              - length
              - word_count
            ''',
        "configs/features/user_features.yaml": '''\
            name: user_features
            data:
              train_csv: train.csv
            features:
              - tfidf
              - user_aggregations
            ''',
        "README.md": '''\
            # $name

            Tabular Kaggle project scaffolded with `kagglepipe template init tabular`.

            ## First run

            ```bash
            kagglepipe config show
            kagglepipe src upload
            kagglepipe feature run baseline
            kagglepipe feature all
            ```
            ''',
    },
    "cv": {
        "kaggle.toml": '''\
            [project]
            name = "$name"

            [source]
            include = ["scripts", "configs", "README.md"]
            exclude_dirs = [".venv", "data", "models", ".git", "__pycache__",
                            ".pytest_cache", ".ruff_cache", "kaggle_notebooks",
                            "submissions"]
            exclude_exts = [".parquet", ".pt", ".bin"]
            src_dataset_slug = "{username}/$name-src"

            [data]
            dataset_slug = "{username}/$name-data"

            [feature]
            branches = ["baseline", "dinov3", "siglip"]
            heavy_branches = ["dinov3", "siglip"]
            default_gpu = "t4x2"
            kernel_slug_template = "{username}/$name-{branch}"
            kernel_title_prefix = "$name"
            notebook_command = "python scripts/extract_features.py --config configs/features/{branch}.yaml --data-root /kaggle/input/datasets/{username}/$name-data --out-dir /kaggle/working/features"
            data_mount = "/kaggle/input/datasets/{username}/$name-data"
            src_mount = "/kaggle/input/datasets/{username}/$name-src"
            out_dir = "/kaggle/working/features"
            output_glob = "{branch}.parquet"
            default_timeout_sec = 3600
            poll_interval_sec = 60

            [kernels]
            is_private = true
            enable_internet = true
            language = "python"
            kernel_type = "notebook"

            [paths]
            notebooks_dir = "kaggle_notebooks"
            features_dir = "features_kaggle"
            ''',
        "scripts/extract_features.py": '''\
            #!/usr/bin/env python3
            """CV feature extractor: produces image embeddings per sample."""
            import argparse
            import os
            import yaml
            import pandas as pd

            p = argparse.ArgumentParser()
            p.add_argument("--config", required=True)
            p.add_argument("--data-root", required=True)
            p.add_argument("--out-dir", required=True)
            args = p.parse_args()

            cfg = yaml.safe_load(open(args.config))
            os.makedirs(args.out_dir, exist_ok=True)
            rows = []
            img_dir = os.path.join(args.data_root, cfg["data"]["image_dir"])
            if os.path.isdir(img_dir):
                for f in sorted(os.listdir(img_dir))[:50]:
                    full = os.path.join(img_dir, f)
                    if os.path.isfile(full):
                        rows.append({"file": f, "size": os.path.getsize(full)})
            df = pd.DataFrame(rows or [{"file": "empty", "size": 0}])
            out = os.path.join(args.out_dir, f"{cfg['name']}.parquet")
            df.to_parquet(out, index=False)
            print("wrote", out, "rows=", len(df))
            ''',
        "configs/features/baseline.yaml": '''\
            name: baseline
            data:
              image_dir: train_sample
            model: histogram
            ''',
        "configs/features/dinov3.yaml": '''\
            name: dinov3
            data:
              image_dir: train_sample
            model: dinov3-vitb16
            embedding_dim: 768
            ''',
        "configs/features/siglip.yaml": '''\
            name: siglip
            data:
              image_dir: train_sample
            model: siglip-base-patch16-224
            embedding_dim: 768
            ''',
        "README.md": '''\
            # $name

            Computer-vision Kaggle project scaffolded with `kagglepipe template init cv`.

            ## First run

            ```bash
            kagglepipe src upload
            kagglepipe feature run dinov3 --gpu t4x2
            kagglepipe feature all --parallel 2
            ```
            ''',
    },
    "nlp": {
        "kaggle.toml": '''\
            [project]
            name = "$name"

            [source]
            include = ["scripts", "configs", "README.md"]
            exclude_dirs = [".venv", "data", "models", ".git", "__pycache__",
                            ".pytest_cache", ".ruff_cache", "kaggle_notebooks",
                            "submissions"]
            exclude_exts = [".parquet", ".bin", ".safetensors"]
            src_dataset_slug = "{username}/$name-src"

            [data]
            dataset_slug = "{username}/$name-data"

            [feature]
            branches = ["baseline", "tfidf", "transformer"]
            heavy_branches = ["transformer"]
            default_gpu = "t4x2"
            kernel_slug_template = "{username}/$name-{branch}"
            kernel_title_prefix = "$name"
            notebook_command = "python scripts/run.py --config configs/features/{branch}.yaml --data-root /kaggle/input/datasets/{username}/$name-data --out-dir /kaggle/working/features"
            data_mount = "/kaggle/input/datasets/{username}/$name-data"
            src_mount = "/kaggle/input/datasets/{username}/$name-src"
            out_dir = "/kaggle/working/features"
            output_glob = "{branch}.parquet"
            default_timeout_sec = 3600
            poll_interval_sec = 60

            [kernels]
            is_private = true
            enable_internet = true
            language = "python"
            kernel_type = "notebook"

            [paths]
            notebooks_dir = "kaggle_notebooks"
            features_dir = "features_kaggle"
            ''',
        "scripts/run.py": '''\
            #!/usr/bin/env python3
            """NLP feature runner."""
            import argparse
            import os
            import yaml
            import pandas as pd

            p = argparse.ArgumentParser()
            p.add_argument("--config", required=True)
            p.add_argument("--data-root", required=True)
            p.add_argument("--out-dir", required=True)
            args = p.parse_args()

            cfg = yaml.safe_load(open(args.config))
            os.makedirs(args.out_dir, exist_ok=True)
            text_csv = os.path.join(args.data_root, cfg["data"]["text_csv"])
            df = pd.read_csv(text_csv) if os.path.exists(text_csv) else pd.DataFrame()
            out = os.path.join(args.out_dir, f"{cfg['name']}.parquet")
            df.to_parquet(out, index=False)
            print("wrote", out, "rows=", len(df))
            ''',
        "configs/features/baseline.yaml": '''\
            name: baseline
            data:
              text_csv: train.csv
            model: bag_of_words
            ''',
        "configs/features/tfidf.yaml": '''\
            name: tfidf
            data:
              text_csv: train.csv
            model: tfidf
            ngram_range: [1, 2]
            max_features: 50000
            ''',
        "configs/features/transformer.yaml": '''\
            name: transformer
            data:
              text_csv: train.csv
            model: distilbert-base-uncased
            max_length: 256
            ''',
        "README.md": '''\
            # $name

            NLP Kaggle project scaffolded with `kagglepipe template init nlp`.

            ## First run

            ```bash
            kagglepipe src upload
            kagglepipe feature run transformer --gpu t4x2
            kagglepipe feature all
            ```
            ''',
    },
}


def cmd_template_init(template: str, *, project_name: str | None = None,
                      root: Path | None = None, force: bool = False) -> int:
    """Scaffold a starter project for the given template name."""
    if template not in _TEMPLATES:
        print(
            f"Unknown template {template!r}. Available: {sorted(_TEMPLATES)}",
            file=sys.stderr,
        )
        return 1
    base = (root or Path.cwd()).resolve()
    name = project_name or base.name.lower().replace(" ", "-")
    files = _TEMPLATES[template]
    written: list[Path] = []
    skipped: list[Path] = []
    for relpath, body in files.items():
        path = base / relpath
        if path.exists() and not force:
            skipped.append(path)
            continue
        _write(path, body, name=name, force=force)
        written.append(path)
    print(f"[{template}] wrote {len(written)} file(s) to {base}")
    for p in written:
        print(f"  + {p.relative_to(base)}")
    for p in skipped:
        print(f"  = {p.relative_to(base)} (skipped, exists; pass --force to overwrite)")
    print("\nNext steps:")
    print("  kagglepipe config show        # review the generated kaggle.toml")
    print("  kagglepipe validate           # check config + credentials + paths")
    print("  kagglepipe src upload         # sync source to Kaggle")
    return 0


def cmd_template_list() -> int:
    print("Available templates:")
    for name in sorted(_TEMPLATES):
        print(f"  - {name}")
    return 0
