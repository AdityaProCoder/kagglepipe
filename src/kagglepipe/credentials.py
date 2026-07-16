"""Kaggle credentials discovery.

Order of precedence:
  1. KAGGLE_USERNAME + KAGGLE_KEY env vars
  2. ~/.kaggle/kaggle.json (chmod 600)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


class CredentialsError(RuntimeError):
    """Raised when no Kaggle credentials can be located."""


@dataclass(frozen=True)
class Credentials:
    username: str
    key: str

    def masked(self) -> str:
        if len(self.key) <= 4:
            return f"{self.username}:****"
        return f"{self.username}:{self.key[:2]}***{self.key[-2:]}"


def default_path() -> Path:
    """Return the canonical credentials path (~/.kaggle/kaggle.json)."""
    return Path.home() / ".kaggle" / "kaggle.json"


def load(path: Path | None = None) -> Credentials:
    """Load credentials from env or file. Raises CredentialsError if absent."""
    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")
    if env_user and env_key:
        return Credentials(username=env_user, key=env_key)

    cfg = (path or default_path()).expanduser()
    if not cfg.exists():
        raise CredentialsError(
            f"Kaggle credentials not found at {cfg}. "
            "Set KAGGLE_USERNAME and KAGGLE_KEY, or run `kagglepipe login`."
        )
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CredentialsError(f"Could not read Kaggle credentials at {cfg}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CredentialsError(f"Invalid JSON in {cfg}: {exc}") from exc
    try:
        return Credentials(username=data["username"], key=data["key"])
    except KeyError as exc:
        raise CredentialsError(f"{cfg} missing required field: {exc}") from exc


def write(username: str, key: str, path: Path | None = None) -> Path:
    """Write credentials to ~/.kaggle/kaggle.json. Returns the path."""
    cfg = (path or default_path()).expanduser()
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        json.dumps({"username": username, "key": key}, indent=2) + "\n",
        encoding="utf-8",
    )
    # chmod is a no-op on Windows; on POSIX, lock it down.
    try:
        os.chmod(cfg, 0o600)
    except OSError:
        pass
    return cfg
