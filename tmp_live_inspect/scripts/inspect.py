#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path("/kaggle/input")
    checks = [
        "/kaggle/input/datasets/holamigohello/sample-data",
        "/kaggle/input/datasets/holamigohello/kagglepipe-inspect-src",
        "/kaggle/input/holamigohello/sample-data",
        "/kaggle/input/holamigohello/kagglepipe-inspect-src",
        "/kaggle/input/sample-data",
        "/kaggle/input/kagglepipe-inspect-src",
    ]
    info = {
        "input_exists": root.exists(),
        "input_dirs": sorted(p.name for p in root.iterdir()) if root.exists() else [],
        "checks": {},
    }
    for c in checks:
        info["checks"][c] = {
            "exists": os.path.exists(c),
            "isdir": os.path.isdir(c),
            "files": sorted(os.listdir(c))[:20] if os.path.isdir(c) else [],
        }
    matches = []
    if root.exists():
        for p in root.rglob("*"):
            s = str(p).lower()
            if "sample-data" in s or "kagglepipe-inspect-src" in s:
                matches.append(str(p))
    info["matches"] = matches[:200]
    (out_dir / "inspect.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("wrote", out_dir / "inspect.json")
    print("input_dirs", info["input_dirs"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
