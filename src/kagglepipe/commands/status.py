"""status — list kernels matching the configured prefix."""

from __future__ import annotations

from kagglepipe import credentials, kaggle_api
from kagglepipe.config import Config
from kagglepipe.slug import normalize_slug


def status(cfg: Config, *, all_kernels: bool = False, csv_output: bool = False) -> int:
    """List kernels owned by the current user matching the title prefix."""
    creds = credentials.load()
    kernels = kaggle_api.kernels_list(user=creds.username, page_size=100)
    if all_kernels:
        matched = kernels
    else:
        prefix_norm = normalize_slug(cfg.feature.kernel_title_prefix)
        matched = [k for k in kernels if prefix_norm in k.get("ref", "").lower()]
    if csv_output:
        if matched:
            keys = list(matched[0].keys())
            print(",".join(keys))
            for k in matched:
                print(",".join(k.get(c, "") for c in keys))
        return 0
    if not matched:
        print("No kernels matched.")
        return 0
    matched.sort(key=lambda k: k.get("ref", ""))
    for k in matched:
        ref = k.get("ref", "?")
        state = k.get("status", "?")
        last = k.get("lastRunTime", "?")
        print(f"{ref:<60}  {state:<10}  {last}")
    return 0
