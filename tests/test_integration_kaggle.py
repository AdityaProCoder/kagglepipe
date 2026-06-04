"""End-to-end integration test against the real Kaggle API.

Gated by `KAGGLEPIPE_RUN_INTEGRATION=1` so unit-test runs don't hit Kaggle.

Pipeline:
  1. Ensure `holamigohello/sample-data` (13 images, 3.2 MB) is on the account.
  2. Upload a tiny source dataset (`kagglepipe-itest-src`).
  3. Push a tiny kernel that reads the data, writes a parquet.
  4. Poll until complete.
  5. Download the parquet to local features_kaggle/.
  6. Verify the parquet exists and has rows.

Cleanup: the test creates new datasets/kernels prefixed with `kagglepipe-itest-`.
You can delete them via `kagglepipe datasets list` / `kagglepipe kernels list`.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from kagglepipe import credentials, runner
from kagglepipe.commands import feature, src

pytestmark = pytest.mark.integration


def _has_integration_env() -> bool:
    return os.environ.get("KAGGLEPIPE_RUN_INTEGRATION") == "1"


pytestmark = pytest.mark.skipif(
    not _has_integration_env(),
    reason="set KAGGLEPIPE_RUN_INTEGRATION=1 to run end-to-end Kaggle tests",
)


@pytest.fixture(scope="module")
def kaggle_creds() -> credentials.Credentials:
    return credentials.load()


def _existing_dataset_refs(creds: credentials.Credentials) -> set[str]:
    result = runner.run(["datasets", "list", "--user", creds.username, "--csv"])
    refs: set[str] = set()
    for line in result.stdout.splitlines()[1:]:
        parts = line.split(",")
        if parts:
            refs.add(parts[0].strip('"'))
    return refs


def test_end_to_end_pipeline(tmp_path, kaggle_creds):
    """Full upload + kernel + download + verify cycle."""
    project = tmp_path / "itest_project"
    project.mkdir()
    (project / "scripts").mkdir()
    (project / "scripts" / "run.py").write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import pandas as pd\n"
        "out_dir = '/kaggle/working/features'\n"
        "os.makedirs(out_dir, exist_ok=True)\n"
        "rows = []\n"
        "data_root = '/kaggle/input/datasets/holamigohello/sample-data'\n"
        "if os.path.isdir(data_root):\n"
        "    for root, dirs, files in os.walk(data_root):\n"
        "        for f in sorted(files):\n"
        "            full = os.path.join(root, f)\n"
        "            rel = os.path.relpath(full, data_root)\n"
        "            rows.append({'file': rel, 'size': os.path.getsize(full)})\n"
        "df = pd.DataFrame(rows or [{'file': 'empty', 'size': 0}])\n"
        "out_path = os.path.join(out_dir, 'itest.parquet')\n"
        "df.to_parquet(out_path, index=False)\n"
        "print('Wrote:', out_path, 'rows=', len(df))\n"
    )

    # Config points the test at the user's existing 13-image dataset
    # as the data source.
    cfg_text = f"""
[project]
name = "kagglepipe-itest"

[source]
include = ["scripts"]
exclude_dirs = []
exclude_exts = []
src_dataset_slug = "{kaggle_creds.username}/kagglepipe-itest-src"

[data]
dataset_slug = "{kaggle_creds.username}/sample-data"

[feature]
branches = ["itest"]
heavy_branches = ["itest"]
default_gpu = "none"
kernel_slug_template = "{kaggle_creds.username}/kagglepipe-itest-{{branch}}"
kernel_title_prefix = "kagglepipe-itest"
notebook_command = "python scripts/run.py"
data_mount = "/kaggle/input/datasets/{{username}}/sample-data"
src_mount = "/kaggle/input/datasets/{{username}}/kagglepipe-itest-src"
out_dir = "/kaggle/working/features"
output_glob = "itest.parquet"
default_timeout_sec = 300
poll_interval_sec = 20

[kernels]
is_private = true
enable_internet = false
language = "python"
kernel_type = "notebook"

[paths]
notebooks_dir = "kaggle_notebooks"
features_dir = "features_kaggle"
"""
    (project / "kaggle.toml").write_text(cfg_text, encoding="utf-8")

    # Verify the source dataset (sample-data) is on the account.
    refs = _existing_dataset_refs(kaggle_creds)
    sample = f"{kaggle_creds.username}/sample-data"
    assert sample in refs, (
        f"Expected {sample} to exist; cannot run integration test. "
        f"Upload 13 images to your Kaggle account first."
    )
    print(f"Found source data: {sample}")

    # Run upload + feature
    os.chdir(project)
    from kagglepipe.config import load
    cfg = load(project / "kaggle.toml")

    print(">>> Running src.upload ...")
    rc = src.upload(cfg, src_root=project)
    assert rc == 0, f"src.upload failed (rc={rc})"

    # Give Kaggle a few seconds to index the new dataset for kernel mounting.
    # In our first run the kernel pushed immediately and the source mount
    # path didn't exist yet (FileNotFoundError on /kaggle/input/...).
    print(">>> Waiting 30s for dataset to propagate ...")
    import time
    time.sleep(30)

    print(">>> Running feature.run_feature ...")
    rc = feature.run_feature(cfg, "itest", gpu="none", timeout_sec=300, quiet=False)
    assert rc == 0, f"feature.run_feature failed (rc={rc})"

    output = project / "features_kaggle" / "itest.parquet"
    assert output.exists(), f"expected output at {output}"
    df = pd.read_parquet(output)
    assert len(df) > 0, "output parquet has no rows"
    print(f">>> OK: got {len(df)} rows")
    print(f">>> First few files: {df['file'].tolist()[:5]}")
    print(f">>> Total size: {df['size'].sum()} bytes")
