"""Live-rendering monitor loop.

Wraps a Rich `Live` context around the `MonitorSnapshot` so the
dashboard updates without flicker. Uses `Console.screen` semantics
implicitly through `Live`.

Kept separate from `commands/monitor.py` so the panel builders
can be unit-tested in isolation.
"""

from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console
from rich.live import Live

from kagglepipe.commands.monitor import build_layout
from kagglepipe.monitor import collect_snapshot


class MonitorApp:
    """A small live-rendering loop around `MonitorSnapshot`.

    Why not Textual? Textual's App.run() owns the terminal. For a
    read-only dashboard that we just want to refresh, `rich.live.Live`
    is simpler, faster to start, and works on every terminal that
    Rich supports. The dashboard IS the app.
    """

    def __init__(
        self,
        *,
        refresh_seconds: int = 5,
        project_root: Path | None = None,
    ) -> None:
        if refresh_seconds < 1:
            refresh_seconds = 1
        self._refresh_seconds = refresh_seconds
        self._root = project_root or Path.cwd()
        self._console = Console()
        # Track running state so SIGINT / KeyboardInterrupt exits cleanly.
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> int:
        """Block until interrupted, refreshing the dashboard each tick."""
        # Friendly startup. If anything fails on the first frame
        # (e.g., not a tty), we fall through to one-shot mode.
        if not self._console.is_terminal:
            self._print_once()
            return 0

        try:
            with Live(
                self._render_once(),
                console=self._console,
                refresh_per_second=2,  # cap animation rate
                screen=False,
                transient=False,
            ) as live:
                # Initial paint
                while self._running:
                    live.update(self._render_once(), refresh=True)
                    time.sleep(self._refresh_seconds)
        except KeyboardInterrupt:
            pass
        return 0

    def _render_once(self):
        """Return a fresh `rich.layout.Layout` for the current snapshot."""
        snapshot = collect_snapshot(self._root)
        return build_layout(snapshot)

    def _print_once(self) -> None:
        """One-shot print for non-TTY environments."""
        self._console.print(self._render_once())


def run_monitor(refresh_seconds: int, project_root: Path | None = None) -> int:
    """Top-level entry used by the CLI. Kept as a tiny shim so callers
    don't need to import the class directly.
    """
    return MonitorApp(
        refresh_seconds=refresh_seconds,
        project_root=project_root,
    ).run()
