#!/usr/bin/env python3
"""Minimal smoke test for kagglepipe.

This is the simplest possible kagglepipe project. It:
1. Creates a kaggle.toml with one branch
2. Creates a scripts/run.py that reads a CSV and writes a parquet
3. Runs `kagglepipe validate` to check the setup
4. Runs `kagglepipe src upload` to push the source
5. Runs `kagglepipe feature run demo` to execute on Kaggle

Expected output after a successful run:
- features_kaggle/demo.parquet exists locally
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}")
    rc = subprocess.run(cmd).returncode
    print(f"  exit={rc}")
    return rc

def main() -> int:
    root = Path(__file__).parent.resolve()
    os.chdir(root)

    # Create minimal kaggle.toml
    (root / "kaggle.toml").write_text("""\
[project]
name = "minimal"

[source]
include = ["scripts", "train.csv"]
exclude_dirs = [".venv", ".git", "__pycache__"]
exclude_exts = [".parquet", ".lgb", ".pt"]
src_dataset_slug = "{username}/minimal-src"

[data]
dataset_slug = ""

[feature]
branches = ["demo"]
default_gpu = "none"
kernel_slug_template = "{username}/minimal-{branch}"
kernel_title_prefix = "minimal"
notebook_command = "python scripts/run.py --data-root /kaggle/input/datasets/{username}/minimal-src --out-dir /kaggle/working/features"
data_mount = ""
src_mount = "/kaggle/input/datasets/{username}/minimal-src"
out_dir = "/kaggle/working/features"
output_glob = "{branch}.parquet"
default_timeout_sec = 300
poll_interval_sec = 20

[kernels]
is_private = true
enable_internet = false

[paths]
notebooks_dir = "kaggle_notebooks"
features_dir = "features_kaggle"
""")

    # Create scripts/run.py
    (root / "scripts").mkdir(exist_ok=True)
    (root / "scripts" / "run.py").write_text("""\
#!/usr/bin/env python3
import argparse, os, pandas as pd
p = argparse.ArgumentParser()
p.add_argument("--data-root", required=True)
p.add_argument("--out-dir", required=True)
args = p.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
csv = os.path.join(args.data_root, "train.csv")
df = pd.read_csv(csv)
out = os.path.join(args.out_dir, "demo.parquet")
df.to_parquet(out, index=False)
print("wrote", out, "rows=", len(df))
""")

    # Create a tiny train.csv
    (root / "train.csv").write_text("id,x,y\\n1,0.5,0.2\\n2,0.8,0.9\\n")

    print("\\n=== Step 1: validate ===")
    if run([sys.executable, "-m", "kagglepipe", "validate"]) != 0:
        print("FAILED: validate")
        return 1

    print("\\n=== Step 2: src upload ===")
    if run([sys.executable, "-m", "kagglepipe", "src", "upload"]) != 0:
        print("FAILED: src upload")
        return 1

    print("\\n=== Step 3: feature run ===")
    if run([sys.executable, "-m", "kagglepipe", "feature", "run", "demo", "--timeout", "300"]) != 0:
        print("FAILED: feature run")
        return 1

    artifact = root / "features_kaggle" / "demo.parquet"
    if not artifact.exists():
        print(f"FAILED: artifact not found at {artifact}")
        return 1

    df = pd.read_parquet(artifact)
    print(f"\\n=== SUCCESS: {artifact} ({len(df)} rows) ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())
