"""Background job runner for the web UI.

Jobs run one at a time on a single worker thread (one GPU). While a job runs, a
`WebReporter` records every pipeline event into the job's state; the API streams
snapshots of that state to the browser.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from dubber.config import Options
from dubber.media import free_gpu
from dubber.reporting import STAGES, Reporter

# Rough share of total run time per step, used for the overall progress bar.
STAGE_WEIGHTS = {"download": 6, "separate": 10, "transcribe": 14, "speakers": 4,
                 "translate": 8, "synthesize": 44, "mix": 10, "mux": 4}
TERMINAL = {"done", "failed", "cancelled"}


class JobCancelled(Exception):
    pass


@dataclass
class TaskState:
    """One progress bar inside a step."""

    description: str
    total: float | None
    completed: float = 0.0
    started: float = field(default_factory=time.time)

    def snapshot(self) -> dict:
        fraction = min(1.0, self.completed / self.total) if self.total else None
        eta = None
        if fraction and 0.02 < fraction < 1:
            elapsed = time.time() - self.started
            eta = round(elapsed * (1 - fraction) / fraction)
        return {"description": self.description, "fraction": fraction, "eta_seconds": eta}


@dataclass
class StageState:
    key: str
    title: str
    tool: str
    status: str = "pending"            # pending | running | done | cached | failed | cancelled
    started: float | None = None
    seconds: float | None = None
    tasks: list[TaskState] = field(default_factory=list)

    def fraction(self) -> float:
        if self.status in ("done", "cached"):
            return 1.0
        known = [t.completed / t.total for t in self.tasks if t.total]
        return min(1.0, sum(known) / len(known)) if known else 0.0

    def snapshot(self) -> dict:
        elapsed = self.seconds if self.seconds is not None else (
            time.time() - self.started if self.started else None)
        return {"key": self.key, "title": self.title, "tool": self.tool, "status": self.status,
                "seconds": round(elapsed, 1) if elapsed is not None else None,
                "fraction": round(self.fraction(), 4), "tasks": [t.snapshot() for t in self.tasks]}


class Job:
    def __init__(self, options: Options, preview: dict | None):
        self.id = uuid.uuid4().hex[:10]
        self.options = options
        self.status = "queued"
        self.created = time.time()
        self.started: float | None = None
        self.finished: float | None = None
        self.error: str | None = None
        self.video: dict | None = preview
        self.stages = {key: StageState(key, title, tool) for key, title, tool in STAGES}
        self.current: str | None = None
        self.logs: deque[dict] = deque(maxlen=500)
        self.report: dict | None = None
        self.cancel_requested = False
        self.version = 0
        self.lock = threading.RLock()

    def touch(self) -> None:
        self.version += 1

    def add_log(self, text: str, level: str = "info") -> None:
        with self.lock:
            self.logs.append({"t": round(time.time(), 2), "level": level, "text": text})
            self.touch()

    def overall_fraction(self) -> float:
        if self.status == "done":
            return 1.0
        total = sum(STAGE_WEIGHTS.values())
        return round(sum(STAGE_WEIGHTS[k] * s.fraction() for k, s in self.stages.items()) / total, 4)

    def summary(self) -> dict:
        return {"id": self.id, "status": self.status, "url": self.options.url, "video": self.video,
                "created": self.created, "max_minutes": self.options.max_minutes,
                "progress": self.overall_fraction(), "current": self.current}

    def snapshot(self, decorate_report) -> dict:
        with self.lock:
            now = self.finished or time.time()
            return {
                **self.summary(),
                "options": {"max_minutes": self.options.max_minutes, "voice": self.options.voice,
                            "speakers": self.options.speakers, "translator": self.options.translator,
                            "polish": self.options.polish,
                            "llm_model": self.options.llm_model, "language": self.options.language},
                "started": self.started,
                "finished": self.finished,
                "elapsed": round(now - self.started, 1) if self.started else 0,
                "error": self.error,
                "stages": [s.snapshot() for s in self.stages.values()],
                "logs": list(self.logs),
                "report": decorate_report(self.report) if self.report else None,
                "version": self.version,
            }


class JobProgress:
    """Duck-types the bits of rich.progress.Progress the pipeline uses, writing into a job step."""

    def __init__(self, job: Job, stage: StageState):
        self.job, self.stage = job, stage
        self.tasks: dict[int, TaskState] = {}

    def __enter__(self) -> "JobProgress":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def _check_cancel(self) -> None:
        if self.job.cancel_requested:
            raise JobCancelled("cancelled by user")

    def add_task(self, description: str, total: float | None = None, **_: Any) -> int:
        self._check_cancel()
        with self.job.lock:
            task = TaskState(description, total)
            self.stage.tasks.append(task)
            task_id = len(self.tasks)
            self.tasks[task_id] = task
            self.job.touch()
        return task_id

    def update(self, task_id: int, *, total: float | None = None, completed: float | None = None,
               advance: float | None = None, description: str | None = None, **_: Any) -> None:
        self._check_cancel()
        with self.job.lock:
            task = self.tasks[task_id]
            if total is not None:
                task.total = total
            if completed is not None:
                task.completed = completed
            if advance is not None:
                task.completed += advance
            if description is not None:
                task.description = description
            self.job.touch()

    def advance(self, task_id: int, advance: float = 1) -> None:
        self.update(task_id, advance=advance)


class WebReporter(Reporter):
    def __init__(self, job: Job):
        self.job = job

    def stage_started(self, key: str) -> None:
        if self.job.cancel_requested:
            raise JobCancelled("cancelled by user")
        with self.job.lock:
            stage = self.job.stages[key]
            stage.status, stage.started, stage.tasks = "running", time.time(), []
            self.job.current = key
            self.job.touch()
        self.log(f"Step started: {stage.title}")

    def stage_finished(self, key: str, seconds: float, cached: bool) -> None:
        with self.job.lock:
            stage = self.job.stages[key]
            stage.status, stage.seconds = ("cached" if cached else "done"), seconds
            self.job.touch()
        self.log(f"Step finished: {stage.title} ({seconds:.1f}s{', cached' if cached else ''})", "success")

    def stage_failed(self, key: str, error: BaseException) -> None:
        with self.job.lock:
            stage = self.job.stages[key]
            stage.status = "cancelled" if isinstance(error, JobCancelled) else "failed"
            stage.seconds = time.time() - (stage.started or time.time())
            self.job.touch()

    def video_info(self, meta: dict) -> None:
        with self.job.lock:
            self.job.video = {**(self.job.video or {}), **meta}
            self.job.touch()

    def log(self, text: str, level: str = "info") -> None:
        self.job.add_log(text, level)
        print(f"[job {self.job.id}] {text}", flush=True)

    def progress(self) -> JobProgress:
        return JobProgress(self.job, self.job.stages[self.job.current])

    def run_finished(self, report: dict) -> None:
        with self.job.lock:
            self.job.report = report
            self.job.touch()


class JobManager:
    """Owns all jobs and the single GPU worker thread."""

    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._queue: queue.Queue[Job] = queue.Queue()
        self._worker = threading.Thread(target=self._work, name="gpu-worker", daemon=True)
        self._worker.start()

    def submit(self, options: Options, preview: dict | None) -> Job:
        for job in self.jobs.values():   # same video + same run settings already queued or running
            if (job.status in ("queued", "running") and job.options.url == options.url
                    and job.options.max_minutes == options.max_minutes):
                return job
        job = Job(options, preview)
        self.jobs[job.id] = job
        job.add_log("Queued. Waiting for the GPU." if self._busy() else "Queued.")
        self._queue.put(job)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list(self) -> list[dict]:
        return [j.summary() for j in sorted(self.jobs.values(), key=lambda j: -j.created)]

    def cancel(self, job_id: str) -> Job | None:
        job = self.jobs.get(job_id)
        if job and job.status in ("queued", "running"):
            job.cancel_requested = True
            if job.status == "queued":
                with job.lock:
                    job.status, job.finished = "cancelled", time.time()
                    job.touch()
            job.add_log("Cancel requested.", "warn")
        return job

    def _busy(self) -> bool:
        return any(j.status == "running" for j in self.jobs.values())

    def _work(self) -> None:
        from dubber.pipeline import run   # heavy import, done on the worker thread

        while True:
            job = self._queue.get()
            if job.status == "cancelled":
                continue
            with job.lock:
                job.status, job.started = "running", time.time()
                job.touch()
            try:
                run(job.options, WebReporter(job))
                status = "done"
            except JobCancelled:
                status = "cancelled"
                job.add_log("Job cancelled. Cached steps are kept; starting again resumes from them.", "warn")
            except Exception as exc:
                status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.add_log(job.error, "error")
                job.add_log(traceback.format_exc(limit=6), "error")
            finally:
                free_gpu()
            with job.lock:
                job.status, job.finished = status, time.time()
                if job.current and job.stages[job.current].status == "running":
                    job.stages[job.current].status = status if status != "done" else "done"
                job.touch()
