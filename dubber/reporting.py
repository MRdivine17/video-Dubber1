"""How the pipeline reports what it is doing.

The pipeline only talks to a `Reporter`: stage started / finished, log lines and
progress bars. `CliReporter` draws them in the terminal with rich; the web server
installs its own reporter that streams the same events to the browser.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, ContextManager, Iterator, Protocol

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console(highlight=False)

# (key, title, tool) for every pipeline step, in order. Shared with the web UI.
STAGES: list[tuple[str, str, str]] = [
    ("download", "Download video", "yt-dlp"),
    ("separate", "Separate speech from music & effects", "Demucs htdemucs"),
    ("transcribe", "Transcribe speech", "faster-whisper large-v3"),
    ("speakers", "Identify speakers", "ECAPA embeddings + clustering"),
    ("translate", "Translate to English", "IndicTrans2 / NLLB-200 + LLM polish"),
    ("synthesize", "Synthesize English voice", "Coqui XTTS v2 voice cloning"),
    ("mix", "Time-align & mix", "rubberband time-fit + loudness match"),
    ("mux", "Replace audio & save video", "ffmpeg, video stream copied"),
]
STAGE_TITLES = {key: title for key, title, _ in STAGES}


class ProgressLike(Protocol):
    """The subset of rich.progress.Progress the pipeline modules use."""

    def add_task(self, description: str, total: float | None = None, **kwargs: Any) -> int: ...
    def update(self, task_id: int, **kwargs: Any) -> None: ...
    def advance(self, task_id: int, advance: float = 1) -> None: ...


class Reporter:
    """Base reporter: ignores everything. Subclass and override what you need."""

    def stage_started(self, key: str) -> None: ...
    def stage_finished(self, key: str, seconds: float, cached: bool) -> None: ...
    def stage_failed(self, key: str, error: BaseException) -> None: ...
    def video_info(self, meta: dict) -> None: ...
    def log(self, text: str, level: str = "info") -> None: ...
    def run_finished(self, report: dict) -> None: ...

    def progress(self) -> ContextManager[ProgressLike]:
        raise NotImplementedError


_reporter: Reporter | None = None


def set_reporter(reporter: Reporter) -> None:
    global _reporter
    _reporter = reporter


def log(text: str, level: str = "info") -> None:
    """Log a line through the active reporter (terminal or browser)."""
    (_reporter or CliReporter()).log(text, level)


def fmt_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


# ── stage timing ─────────────────────────────────────────────────────────────────

@dataclass
class StageTiming:
    key: str
    seconds: float
    cached: bool


@dataclass
class StageContext:
    progress: ProgressLike
    cached: bool = False


@dataclass
class StageTimer:
    """Times each step and tells the reporter when it starts / ends / fails."""

    reporter: Reporter
    timings: list[StageTiming] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)

    @contextmanager
    def stage(self, key: str) -> Iterator[StageContext]:
        self.reporter.stage_started(key)
        t0 = time.perf_counter()
        try:
            with self.reporter.progress() as progress:
                ctx = StageContext(progress)
                yield ctx
        except BaseException as exc:
            self.reporter.stage_failed(key, exc)
            raise
        elapsed = time.perf_counter() - t0
        self.timings.append(StageTiming(key, elapsed, ctx.cached))
        self.reporter.stage_finished(key, elapsed, ctx.cached)

    @property
    def total_seconds(self) -> float:
        return time.perf_counter() - self.started


# ── terminal reporter ────────────────────────────────────────────────────────────

class CliReporter(Reporter):
    STYLES = {"info": "", "warn": "yellow", "error": "bold red", "success": "green"}

    def stage_started(self, key: str) -> None:
        index = [k for k, _, _ in STAGES].index(key) + 1
        console.rule(f"[bold cyan]Step {index}/{len(STAGES)} · {STAGE_TITLES[key]}")

    def stage_finished(self, key: str, seconds: float, cached: bool) -> None:
        note = " [yellow](reused cached result)[/]" if cached else ""
        console.print(f"[green]done[/] {STAGE_TITLES[key]} in {fmt_seconds(seconds)}{note}")

    def stage_failed(self, key: str, error: BaseException) -> None:
        if not isinstance(error, KeyboardInterrupt):
            console.print(f"[bold red]failed[/] {STAGE_TITLES[key]}: {error}")

    def video_info(self, meta: dict) -> None:
        console.print(f"[bold]{meta['title']}[/]  ({meta['id']})")

    def log(self, text: str, level: str = "info") -> None:
        style = self.STYLES.get(level, "")
        console.print(f"[{style}]{text}[/]" if style else text)

    def progress(self) -> ContextManager[ProgressLike]:
        return Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(bar_width=34),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("[dim]eta"),
            TimeRemainingColumn(),
            console=console,
        )

    def run_finished(self, report: dict) -> None:
        console.rule("[bold green]Finished")
        table = Table(title="Processing time", title_style="bold")
        table.add_column("Step")
        table.add_column("Time", justify="right")
        table.add_column("Note", style="dim")
        for s in report["stages"]:
            table.add_row(STAGE_TITLES[s["key"]], fmt_seconds(s["seconds"]), "cached" if s["cached"] else "")
        table.add_section()
        table.add_row("[bold]Total", f"[bold]{report['processing_time']}", "")
        table.add_row("Video length", fmt_seconds(report["video_seconds"]),
                      f"{report['realtime_factor']:.2f}x real time")
        console.print(table)
        t = report["timing"]
        console.print(f"lines sped up: {t['sped_up_pct']}% (avg {t['avg_speed']}x, max {t['max_speed']}x) · "
                      f"max delay {t['max_late_s']}s · median delay {t['median_late_s']}s")
        for check in report["checks"]:
            console.print(f"[green]check[/] {check}")
        if any(s["cached"] for s in report["stages"]):
            console.print("[yellow]Some steps reused cached results; run with --force for a clean end-to-end timing.[/]")
        console.print(f"\n[bold]Output:[/] {report['output']}\n[bold]Report:[/] {report['report']}")
