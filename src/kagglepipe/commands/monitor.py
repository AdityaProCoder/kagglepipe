"""Textual-based monitor dashboard for kagglepipe.

Renders a 2x3 grid of Rich panels with live data from `MonitorSnapshot`.
The dashboard is read-only — no forms, no editing, no mutations.

Architecture:
    MonitorSnapshot   (kagglepipe.monitor)
        ↓
    build_layout(snapshot)  -> rich.layout.Layout
        ↓
    MonitorApp.on_mount()    -> refresh every N seconds
"""

from __future__ import annotations

from rich.align import Align
from rich.console import Group
from rich.layout import Layout
from rich.padding import Padding
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from kagglepipe.monitor import (
    MonitorSnapshot,
)

# ---- state -> style mapping --------------------------------------------


# Color choices use a small palette so the dashboard feels consistent.
# These map to standard Rich colors which respect the user's terminal theme.
_STATE_STYLES = {
    "complete": ("DONE", "bold bright_green", "●"),
    "running": ("RUN", "bold bright_yellow", "◐"),
    "queued": ("QUE", "dim white", "○"),
    "error": ("ERR", "bold bright_red", "✕"),
    "timeout": ("TO", "bold red", "⌛"),
    "skipped": ("SKIP", "dim cyan", "✓"),
    "unknown": ("—", "dim white", "?"),
}


def _state_badge(state: str) -> Text:
    label, style, glyph = _STATE_STYLES.get(state, _STATE_STYLES["unknown"])
    return Text(f"{glyph}{label}", style=style)


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60):02d}m"


def _format_gpu(gpu: str | None) -> Text:
    if not gpu:
        return Text("n/a", style="dim")
    g = gpu.lower()
    if g in {"none", "cpu"}:
        return Text("cpu", style="dim")
    if "t4" in g:
        return Text(g, style="bold magenta")
    if "p100" in g:
        return Text(g, style="bold cyan")
    return Text(g, style="cyan")


def _format_cache(cache: str) -> Text:
    if cache == "HIT":
        return Text(" HIT ", style="bold bright_green on grey11")
    if cache == "MISS":
        return Text(" MISS ", style="bold bright_yellow on grey11")
    return Text("  —  ", style="dim on grey11")


# ---- panel builders -----------------------------------------------------


def _build_jobs_panel(snapshot: MonitorSnapshot) -> Panel:
    """LEFT panel: Active Jobs."""
    if not snapshot.jobs:
        body = Padding(
            Align.center(Text("No active jobs", style="dim italic"), vertical="middle"),
            (1, 2),
        )
        return Panel(
            body,
            title="[bold]Active Jobs[/bold]",
            border_style="blue",
            padding=(0, 0),
        )

    table = Table(
        show_header=True,
        header_style="bold dim",
        expand=True,
        box=None,
        padding=(0, 0),
    )
    # Concise table: branch, state badge, elapsed time. GPU and cache
    # are appended inline into the state cell so the row is one
    # readable line on any panel width.
    table.add_column("branch", style="bold", no_wrap=True, overflow="ellipsis", ratio=3)
    table.add_column("status", no_wrap=True, ratio=2)

    for j in snapshot.jobs:
        # Build a single status cell that combines state, elapsed, GPU
        # and cache. Keeps the row readable even on narrow panels.
        gpu_str = j.gpu if j.gpu else "n/a"
        # Compact: just state · gpu (drop cache and elapsed from inline;
        # elapsed is on its own implicit via the state badge).
        status = Text.assemble(
            _state_badge(j.state),
            " · ",
            gpu_str,
        )
        table.add_row(j.branch, status)

    return Panel(table, title="[bold]Active Jobs[/bold]", border_style="blue", padding=(0, 0))


def _build_overview_panel(snapshot: MonitorSnapshot) -> Panel:
    """CENTER panel: Pipeline Overview with progress bar and counters."""
    if snapshot.total_branches == 0:
        body = Padding(
            Align.center(
                Text("No branches configured", style="dim italic"),
                vertical="middle",
            ),
            (1, 2),
        )
        return Panel(
            body,
            title="[bold]Pipeline Overview[/bold]",
            border_style="blue",
            padding=(0, 0),
        )

    # A single progress bar with explicit completed/total to keep the visual
    # tight. We also include the percentage rendered inline.
    bar = ProgressBar(
        total=snapshot.total_branches,
        completed=snapshot.completed,
        complete_style="bright_green",
        finished_style="bright_green",
        pulse_style="bright_yellow",
    )
    pct = f"{snapshot.percent_complete:5.1f}%"
    # Text.assemble can't accept a ProgressBar, so we render bar + label
    # as separate items in a Group below.
    bar_label = Text(f"  {pct} complete", style="bold bright_white")

    # Counters laid out in a compact 2x2 grid
    counters = Table.grid(padding=(0, 2))
    counters.add_column(justify="right", style="bold")
    counters.add_column(justify="left")
    counters.add_row("Total    ", Text(str(snapshot.total_branches), style="white"))
    counters.add_row("Complete ", Text(str(snapshot.completed), style="bright_green"))
    counters.add_row("Running  ", Text(str(snapshot.running), style="bright_yellow"))
    counters.add_row("Failed   ", Text(str(snapshot.failed), style="bright_red"))
    counters.add_row("Queued   ", Text(str(snapshot.queued), style="dim white"))

    body = Group(
        Padding(bar, (1, 1)),
        Padding(bar_label, (0, 1)),
        Padding(counters, (1, 2)),
    )
    return Panel(
        body,
        title="[bold]Pipeline Overview[/bold]",
        border_style="blue",
        padding=(0, 0),
    )


def _build_artifacts_panel(snapshot: MonitorSnapshot) -> Panel:
    """RIGHT panel: Latest Artifacts."""
    if not snapshot.artifacts:
        body = Padding(
            Align.center(
                Text("No artifacts yet", style="dim italic"),
                vertical="middle",
            ),
            (1, 2),
        )
        return Panel(
            body,
            title="[bold]Latest Artifacts[/bold]",
            border_style="blue",
            padding=(0, 0),
        )

    table = Table(
        show_header=True,
        header_style="bold dim",
        expand=True,
        box=None,
        padding=(0, 1),
    )
    table.add_column("artifact", style="cyan", overflow="ellipsis")
    table.add_column("size", justify="right", style="green")
    table.add_column("when", justify="right", style="dim")

    for a in snapshot.artifacts:
        table.add_row(a.name, a.size_human, a.timestamp_human)
    return Panel(
        table,
        title="[bold]Latest Artifacts[/bold]",
        border_style="blue",
        padding=(0, 0),
    )


def _build_latest_submission_panel(snapshot: MonitorSnapshot) -> Panel:
    """BOTTOM LEFT: Latest Submission."""
    s = snapshot.latest_submission
    if s is None:
        body = Padding(
            Align.center(
                Text("No submissions recorded", style="dim italic"),
                vertical="middle",
            ),
            (1, 2),
        )
        return Panel(
            body,
            title="[bold]Latest Submission[/bold]",
            border_style="blue",
            padding=(0, 0),
        )

    score_text = (
        Text(f"{s.score:.5f}", style="bold bright_green")
        if s.score is not None
        else Text("pending", style="dim")
    )
    rank_text = (
        Text(f"#{s.rank}", style="bold cyan")
        if s.rank is not None
        else Text("—", style="dim")
    )

    rows = Table.grid(padding=(0, 2))
    rows.add_column(justify="right", style="bold dim")
    rows.add_column(justify="left")
    rows.add_row("Competition ", Text(s.competition, style="cyan"))
    rows.add_row("Score       ", score_text)
    rows.add_row("Rank        ", rank_text)
    rows.add_row("Submitted   ", Text(s.timestamp_human, style="dim"))
    rows.add_row("ID          ", Text(s.submission_id or "—", style="dim"))

    return Panel(
        Padding(rows, (1, 2)),
        title="[bold]Latest Submission[/bold]",
        border_style="blue",
        padding=(0, 0),
    )


def _build_best_submission_panel(snapshot: MonitorSnapshot) -> Panel:
    """BOTTOM CENTER: Best Submission Ever — flagship section."""
    b = snapshot.best_submission
    if b is None:
        body = Padding(
            Align.center(
                Text("No scored submissions yet", style="dim italic"),
                vertical="middle",
            ),
            (1, 2),
        )
        return Panel(
            body,
            title="[bold]Best Submission Ever[/bold]",
            border_style="bright_magenta",
            padding=(0, 0),
        )

    # The score is the centerpiece. Make it big and bold.
    score_text = (
        Text(f"{b.score:.5f}", style="bold bright_green")
        if b.score is not None
        else Text("—", style="dim")
    )
    rank_text = (
        Text(f"#{b.rank}", style="bold bright_cyan")
        if b.rank is not None
        else Text("—", style="dim")
    )

    # Truncate git commit for display
    commit = b.git_commit or "—"
    if len(commit) > 10:
        commit = commit[:10]

    rows = Table.grid(padding=(0, 2))
    rows.add_column(justify="right", style="bold dim")
    rows.add_column(justify="left")
    rows.add_row("Score        ", score_text)
    rows.add_row("Rank         ", rank_text)
    rows.add_row("Git commit   ", Text(commit, style="yellow"))
    rows.add_row("Experiment   ", Text(b.experiment_id or "—", style="magenta"))
    rows.add_row("Competition  ", Text(b.competition, style="cyan"))
    rows.add_row("Achieved     ", Text(b.timestamp_human, style="dim"))

    return Panel(
        Padding(rows, (1, 2)),
        title="[bold bright_magenta]★ Best Submission Ever[/bold bright_magenta]",
        border_style="bright_magenta",
        padding=(0, 0),
    )


def _build_experiment_summary_panel(snapshot: MonitorSnapshot) -> Panel:
    """BOTTOM RIGHT: Experiment Summary."""
    rows = Table.grid(padding=(0, 2))
    rows.add_column(justify="right", style="bold dim")
    rows.add_column(justify="left")

    rows.add_row("Experiments  ", Text(str(snapshot.total_experiments), style="bright_cyan"))
    rows.add_row("Manifests    ", Text(str(snapshot.total_manifests), style="cyan"))
    rows.add_row("Cache hits   ", Text(str(snapshot.cache_hits), style="bright_green"))
    rows.add_row("Cache miss   ", Text(str(snapshot.cache_misses), style="bright_yellow"))
    rows.add_row("Features     ", Text(str(snapshot.registered_features), style="cyan"))

    return Panel(
        Padding(rows, (1, 2)),
        title="[bold]Experiment Summary[/bold]",
        border_style="blue",
        padding=(0, 0),
    )


# ---- the full layout ----------------------------------------------------


def build_layout(snapshot: MonitorSnapshot) -> Layout:
    """Build the complete Rich layout for the snapshot.

    Returns a Layout that can be printed to any Console (interactive
    Textual or one-shot stdout).
    """
    layout = Layout()

    # Top bar (header) — fixed height, full width. We use a Panel with
    # the default box (HEAVY) rather than box=None, which Rich renders
    # more reliably across terminal widths.
    header_text = Text.assemble(
        ("  KagglePipe Monitor  ", "bold bright_white"),
        ("  Project: ", "dim"),
        (snapshot.project_name, "bold cyan"),
        ("    User: ", "dim"),
        (snapshot.user, "bold cyan"),
        ("    ", " "),
        ("  Last refresh: ", "dim"),
        (snapshot.collected_at_human, "dim italic"),
    )
    header = Panel(
        Align.center(header_text),
        border_style="bright_blue",
        padding=(0, 0),
    )
    layout.split_column(
        Layout(header, name="header", size=3),
        Layout(name="body"),
    )

    # Body: 2x3 grid. The left column gets more width because the
    # job table has more columns (branch, state, elapsed, GPU, cache).
    layout["body"].split_column(
        Layout(name="row1"),
        Layout(name="row2"),
    )
    layout["body"]["row1"].split_row(
        Layout(name="r1c1", ratio=3),
        Layout(name="r1c2", ratio=2),
        Layout(name="r1c3", ratio=2),
    )
    layout["body"]["row2"].split_row(
        Layout(name="r2c1", ratio=2),
        Layout(name="r2c2", ratio=2),
        Layout(name="r2c3", ratio=2),
    )

    layout["body"]["row1"]["r1c1"].update(_build_jobs_panel(snapshot))
    layout["body"]["row1"]["r1c2"].update(_build_overview_panel(snapshot))
    layout["body"]["row1"]["r1c3"].update(_build_artifacts_panel(snapshot))
    layout["body"]["row2"]["r2c1"].update(_build_latest_submission_panel(snapshot))
    layout["body"]["row2"]["r2c2"].update(_build_best_submission_panel(snapshot))
    layout["body"]["row2"]["r2c3"].update(_build_experiment_summary_panel(snapshot))

    return layout


# ---- the CLI command -----------------------------------------------------


def cmd_monitor(*, refresh: int = 5, once: bool = False, project_root: str | None = None) -> int:
    """Entry point for `kagglepipe monitor [--refresh N] [--once]`.

    - `refresh`: seconds between auto-refresh (default 5, min 1).
    - `once`: render a single snapshot to stdout and exit. Useful for
      CI logs, scripts, or for capturing a snapshot into a file.
    - `project_root`: optional project directory; defaults to cwd.

    Interactive mode uses `rich.live.Live` for flicker-free auto-refresh.
    Non-interactive mode (e.g., piped to `less`) prints a single
    snapshot. The `--once` flag forces non-interactive rendering.
    """
    from pathlib import Path

    from kagglepipe.monitor import collect_snapshot

    root = Path(project_root) if project_root else Path.cwd()

    # Always render at least one snapshot. If --once or stdout isn't a
    # TTY, render once and exit.
    import sys
    is_tty = sys.stdout.isatty()
    if once or not is_tty:
        snapshot = collect_snapshot(root)
        # A compact ASCII snapshot is deliberate here: it is reliable in CI,
        # log collectors, redirected output, and legacy Windows code pages.
        # The full Rich dashboard remains available in an interactive TTY.
        print(_plain_snapshot(snapshot))
        return 0

    # Interactive: live-rendering loop with the same layout.
    from kagglepipe.monitor_app import run_monitor

    return run_monitor(refresh_seconds=refresh, project_root=root)


def _plain_snapshot(snapshot: MonitorSnapshot) -> str:
    """Return a portable, machine-log-friendly one-shot monitor summary."""
    lines = [
        f"KagglePipe Monitor | Project: {snapshot.project_name} | User: {snapshot.user or '-'}",
        (
            "Runs: "
            f"{snapshot.total_branches} total, {snapshot.completed} complete, "
            f"{snapshot.running} running, {snapshot.failed} failed, {snapshot.queued} queued"
        ),
    ]
    if snapshot.jobs:
        lines.append("Jobs:")
        lines.extend(f"  {job.branch}: {job.state}" for job in snapshot.jobs)
    else:
        lines.append("Jobs: none")
    if snapshot.artifacts:
        lines.append(f"Artifacts: {len(snapshot.artifacts)}")
    else:
        lines.append("Artifacts: none")
    return "\n".join(lines)
