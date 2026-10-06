"""HTTP API for the dubbing studio (React UI in web/).

    GET  /api/system                      GPU + model status
    GET  /api/preview?url=                video title / thumbnail / length before dubbing
    POST /api/jobs                        start a dub  -> job snapshot
    GET  /api/jobs                        all jobs this session
    GET  /api/jobs/{id}                   job snapshot
    GET  /api/jobs/{id}/events            live snapshots (Server-Sent Events)
    POST /api/jobs/{id}/cancel            stop a queued / running job
    GET  /api/library                     finished dubs found in output/
    GET  /api/library/{report}/transcript line-by-line source / English / timing
    /media/output/*, /media/work/*        videos, subtitles (HTTP range requests for seeking)
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dubber import llm
from dubber.config import MODELS_DIR, OUTPUT_DIR, ROOT, WORK_DIR, Options
from dubber.download import fetch_info, is_youtube_url
from dubber.gpu import GPUUnavailable, require_gpu
from dubber.media import has_filter, read_json
from dubber.reporting import STAGES
from dubber.translate import INDICTRANS_MODEL, NLLB_MODEL

from .jobs import TERMINAL, JobManager

app = FastAPI(title="Dubbing Studio API", version="1.0")
manager = JobManager()
WEB_DIST = ROOT / "web" / "dist"

try:
    GPU_INFO: dict | None = require_gpu()
    GPU_ERROR: str | None = None
except GPUUnavailable as exc:
    GPU_INFO, GPU_ERROR = None, str(exc)


# ── helpers ─────────────────────────────────────────────────────────────────────

def media_url(path: str | Path | None) -> str | None:
    """Map a file under output/ or work/ to the URL it is served from."""
    if not path:
        return None
    p = Path(path).resolve()
    for root, prefix in ((OUTPUT_DIR, "/media/output/"), (WORK_DIR, "/media/work/")):
        try:
            return prefix + quote(p.relative_to(root.resolve()).as_posix())
        except ValueError:
            continue
    return None


def with_media(report: dict) -> dict:
    """A run report plus browser URLs for its files."""
    return {**report, "media": {
        "video": media_url(report.get("output")),
        "original": media_url(report.get("source_video")),
        "srt": media_url(report.get("subtitles")),
        "vtt": media_url(report.get("subtitles_vtt")),
        "report": media_url(report.get("report")),
        "report_name": Path(report["report"]).name if report.get("report") else None,
        "speaker_refs": {name: media_url(spk.get("reference")) for name, spk in report.get("speakers", {}).items()},
    }}


def _model_status() -> dict:
    """Which model weights are fully on disk (partial downloads don't count)."""
    from huggingface_hub import try_to_load_from_cache

    def hf_cached(repo: str, filename: str, cache_dir: Path | None = None) -> bool:
        found = try_to_load_from_cache(repo, filename, cache_dir=str(cache_dir) if cache_dir else None)
        return isinstance(found, str) and Path(found).exists()

    xtts = list((MODELS_DIR / "tts").rglob("model.pth"))
    return {
        "whisper": hf_cached("Systran/faster-whisper-large-v3", "model.bin", MODELS_DIR / "whisper"),
        "demucs": any((MODELS_DIR / "torch").rglob("*.th")),
        "xtts": bool(xtts) and xtts[0].stat().st_size > 1_000_000_000,
        "nllb": hf_cached(NLLB_MODEL, "config.json"),
        "indictrans2": hf_cached(INDICTRANS_MODEL, "config.json"),
    }


# The machine's LLM (any provider, see dubber/llm.py) is detected and checked in the background so
# the UI can show which model the polish pass will use. POST /api/llm/refresh re-runs it.
LLM_STATUS: dict = {"state": "checking", "detail": "", "provider": None, "model": None, "local": False, "models": []}


def _check_llm() -> None:
    llm.detect.cache_clear()
    LLM_STATUS.update(state="checking", detail="")
    cfg = llm.detect()
    if cfg is None:
        LLM_STATUS.update(state="missing", provider=None, model=None, local=False, models=[],
                          detail="No API key or local model server found")
        return
    ok, detail = llm.check(cfg)
    LLM_STATUS.update(state="ok" if ok else "invalid", detail=detail, provider=cfg.provider,
                      model=cfg.model, local=cfg.local, models=cfg.models[:50])


threading.Thread(target=_check_llm, daemon=True).start()


# ── API ─────────────────────────────────────────────────────────────────────────

class JobRequest(BaseModel):
    url: str
    max_minutes: float | None = Field(None, gt=0, le=600)
    voice: Literal["clone", "edge"] = "clone"
    speakers: int | None = Field(None, ge=1, le=8)
    language: str | None = Field(None, max_length=5)
    translator: Literal["auto", "speech", "text"] = "auto"
    polish: bool = True
    llm_model: str | None = Field(None, max_length=200)   # None = auto-detected model
    force: bool = False


@app.post("/api/llm/refresh")
def refresh_llm() -> dict:
    """Re-detect keys / local model servers (e.g. after starting Ollama or editing .env)."""
    _check_llm()
    return LLM_STATUS


@app.get("/api/system")
def system() -> dict:
    gpu = None
    if GPU_INFO:
        import torch
        free, total = torch.cuda.mem_get_info()
        gpu = {**GPU_INFO, "free_gb": round(free / 1024 ** 3, 1)}
    return {
        "gpu": gpu,
        "gpu_error": GPU_ERROR,
        "models": _model_status(),
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "rubberband": has_filter("rubberband"),
        "llm": LLM_STATUS,
        "hf_token": bool(os.environ.get("HF_TOKEN")),
        "stages": [{"key": k, "title": t, "tool": tool} for k, t, tool in STAGES],
        "busy": any(j["status"] == "running" for j in manager.list()),
    }


@app.get("/api/preview")
def preview(url: str = Query(..., min_length=10)) -> dict:
    if not is_youtube_url(url):
        raise HTTPException(422, "That doesn't look like a YouTube video link.")
    try:
        return fetch_info(url)
    except Exception as exc:
        raise HTTPException(502, f"Could not read this video: {str(exc)[:200]}")


@app.post("/api/jobs")
def create_job(req: JobRequest) -> dict:
    if GPU_ERROR:
        raise HTTPException(503, GPU_ERROR)
    url = req.url.strip()
    if not is_youtube_url(url):
        raise HTTPException(422, "Paste a YouTube video link (youtube.com/watch?v=... or youtu.be/...).")
    options = Options(url=url, max_minutes=req.max_minutes, voice=req.voice, speakers=req.speakers,
                      language=req.language or None, translator=req.translator,
                      polish=req.polish, llm_model=req.llm_model or None,
                      force=req.force)
    job = manager.submit(options, preview=None)
    return job.snapshot(with_media)


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return manager.list()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job.snapshot(with_media)


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request) -> StreamingResponse:
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    async def stream():
        last_version, idle = -1, 0.0
        while True:
            if await request.is_disconnected():
                return
            if job.version != last_version:
                last_version = job.version
                yield f"data: {json.dumps(job.snapshot(with_media))}\n\n"
                idle = 0.0
                if job.status in TERMINAL:
                    return
            elif idle >= 15:
                yield ": keep-alive\n\n"
                idle = 0.0
            await asyncio.sleep(0.25)
            idle += 0.25

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    job = manager.cancel(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job.snapshot(with_media)


@app.get("/api/library")
def library() -> list[dict]:
    items = []
    for path in OUTPUT_DIR.glob("*.report.json"):
        try:
            report = read_json(path)
        except (OSError, ValueError):
            continue
        if Path(report.get("output", "")).exists():
            items.append(with_media(report))
    return sorted(items, key=lambda r: r.get("finished_at", ""), reverse=True)


@app.get("/api/library/{report_name}/transcript")
def transcript(report_name: str) -> list[dict]:
    path = (OUTPUT_DIR / report_name).resolve()
    if path.parent != OUTPUT_DIR.resolve() or not path.name.endswith(".report.json") or not path.exists():
        raise HTTPException(404, "Report not found")
    work = Path(read_json(path)["work_dir"])
    rows = read_json(work / "translation.json")
    placements = {p["id"]: p for p in read_json(work / "placements.json")}
    return [{**row, "dub_start": placements.get(row["id"], {}).get("start"),
             "dub_end": placements.get(row["id"], {}).get("end"),
             "speed": placements.get(row["id"], {}).get("speed")} for row in rows]


# ── files ───────────────────────────────────────────────────────────────────────

app.mount("/media/output", StaticFiles(directory=OUTPUT_DIR), name="output")
app.mount("/media/work", StaticFiles(directory=WORK_DIR), name="work")

if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        file = (WEB_DIST / path).resolve()
        if path and file.is_file() and WEB_DIST.resolve() in file.parents:
            return FileResponse(file)
        return FileResponse(WEB_DIST / "index.html")
